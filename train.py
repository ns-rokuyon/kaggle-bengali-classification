import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from argparse import ArgumentParser
try:
    from apex import amp
except ImportError:
    print('Apex is not available')
    amp = None

import loss as L
import model as M
from dataset import bengali_dataset
from data import get_logger, Workspace, get_current_lr
from model import create_init_model, set_batchnorm_eval
from preprocessing import (
    create_transformer_v1,
    create_testing_transformer_v1,
    create_augmentor_v1,
    create_augmentor_v2,
    create_augmentor_v3,
    SOURCE_IMAGE_HEIGHT,
    SOURCE_IMAGE_WIDTH
)
from evaluation import hierarchical_macro_averaged_recall
from optim import CosineLRWithRestarts, Ranger, RAdam
from loss import get_criterion
from sampler import PKSampler, LowFreqSampleMixinBatchSampler
from config import Config
from lib.cutmix import cutmix, cutmix_criterion
from lib.mixup import mixup, mixup_criterion


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--conf', required=True, type=Path,
                        help='Path to config file')
    return parser.parse_args()


def main():
    args = parse_args()
    conf = Config(args.conf)

    data_dir = conf.data_dir
    fold_id = conf.fold_id

    workspace = Workspace(conf.run_id).setup()
    workspace.save_conf(args.conf)
    workspace.log(f'{conf.to_dict()}')

    torch.cuda.set_device(0)

    if conf.use_augmentor:
        if conf.augmentor_type == 'v1':
            augmentor = create_augmentor_v1(
                enable_random_morph=conf.enable_random_morph
            )
        elif conf.augmentor_type == 'v2':
            augmentor = create_augmentor_v2(
                enable_random_morph=conf.enable_random_morph,
                invert_color=conf.invert_color
            )
        elif conf.augmentor_type == 'v3':
            if conf.input_size_tuple:
                input_size = tuple(conf.input_size_tuple)
            else:
                input_size = (conf.input_size, conf.input_size) if conf.input_size else \
                             (SOURCE_IMAGE_HEIGHT, SOURCE_IMAGE_WIDTH)
            augmentor = create_augmentor_v3(
                input_size,
                enable_random_morph=conf.enable_random_morph,
                invert_color=conf.invert_color
            )
        else:
            raise ValueError(conf.augmentor_type)
        workspace.log(f'Use augmentor: {conf.augmentor_type}')
    else:
        augmentor = None

    if not conf.input_size_tuple and conf.input_size == 0:
        train_transformer = create_transformer_v1(augmentor=augmentor)
        val_transformer = create_testing_transformer_v1()
        workspace.log('Input size: default')
    else:
        if conf.input_size_tuple:
            input_size = tuple(conf.input_size_tuple)
        else:
            input_size = (conf.input_size, conf.input_size)
        train_transformer = create_transformer_v1(input_size=input_size,
                                                  augmentor=augmentor)
        val_transformer = create_testing_transformer_v1(input_size=input_size)
        workspace.log(f'Input size: {input_size}')

    train_dataset, val_dataset = bengali_dataset(data_dir,
                                                 fold_id=fold_id,
                                                 train_transformer=train_transformer,
                                                 val_transformer=val_transformer,
                                                 invert_color=conf.invert_color,
                                                 n_channel=conf.n_channel,
                                                 use_grapheme_code=conf.use_grapheme_code,
                                                 logger=workspace.logger)
    workspace.log(f'#train={len(train_dataset)}, #val={len(val_dataset)}')
    train_dataset.set_low_freq_groups(n_class=conf.n_class_low_freq)

    if conf.sampler_type == 'pk':
        sampler = PKSampler(train_dataset,
                            n_iter_per_epoch=conf.n_iter_per_epoch,
                            p=conf.batch_p, k=conf.batch_k)
        train_loader = DataLoader(train_dataset,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True,
                                  batch_sampler=sampler)
        workspace.log(f'{sampler} is enabled')
        workspace.log(f'Real batch_size={sampler.batch_size}')
    elif conf.sampler_type == 'random+append':
        batch_sampler = LowFreqSampleMixinBatchSampler(
            train_dataset,
            conf.batch_size,
            n_low_freq_samples=conf.n_low_freq_samples,
            drop_last=True
        )
        train_loader = DataLoader(train_dataset,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True,
                                  batch_sampler=batch_sampler)
        workspace.log(f'{batch_sampler} is enabled')
        workspace.log(f'Real batch_size={batch_sampler.batch_size}')
    elif conf.sampler_type == 'random':
        train_loader = DataLoader(train_dataset,
                                  batch_size=conf.batch_size,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=True)
    else:
        raise ValueError(f'Invalid sampler_type: {conf.sampler_type}')

    val_loader = DataLoader(val_dataset,
                            batch_size=conf.batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)

    workspace.log(f'Create init model: arch={conf.arch}')
    model = create_init_model(conf.arch,
                              pretrained=True,
                              pooling=conf.pooling_type,
                              dim=conf.feat_dim,
                              use_maxblurpool=conf.use_maxblurpool,
                              remove_last_stride=conf.remove_last_stride,
                              n_channel=conf.n_channel)
    if conf.weight_file:
        pretrained_weight = torch.load(conf.weight_file, map_location='cpu')
        result = model.load_state_dict(pretrained_weight)
        workspace.log(f'Pretrained weights were loaded: {conf.weight_file}')
        workspace.log(result)

    model = model.cuda()

    sub_models = []

    criterion_g = get_criterion(
        conf.loss_type_g,
        weight=train_dataset.get_class_weights_g(),
        rate=conf.ohem_rate
    )
    workspace.log(f'Loss type (g): {conf.loss_type_g}')

    criterion_v = get_criterion(
        conf.loss_type_v,
        weights=train_dataset.get_class_weights_v(),
        rate=conf.ohem_rate
    )
    workspace.log(f'Loss type (v): {conf.loss_type_v}')

    criterion_c = get_criterion(
        conf.loss_type_c,
        weights=train_dataset.get_class_weights_c(),
        rate=conf.ohem_rate
    )
    workspace.log(f'Loss type (c): {conf.loss_type_c}')

    if conf.loss_type_feat_g != 'none':
        assert isinstance(model, (M.BengaliResNet34V3,
                                  M.BengaliResNet34V4,
                                  M.BengaliResNet34AGeMV4,
                                  M.BengaliSEResNeXt50V4,
                                  M.BengaliEfficientNetB0V4,
                                  M.BengaliEfficientNetB3V4))
        criterion_feat_g = get_criterion(
            conf.loss_type_feat_g,
            dim=model.multihead.head_g.dim, n_class=168,
            s=conf.af_scale_g
        )
        workspace.log(f'Loss type (fg): {conf.loss_type_feat_g}')
        if conf.loss_type_feat_g in ('af',):
            sub_models.append(criterion_feat_g)
            workspace.log('Add criterion_feat_g to sub model')
    else:
        criterion_feat_g = None

    if conf.loss_type_feat_v != 'none':
        assert isinstance(model, (M.BengaliResNet34V3,
                                  M.BengaliResNet34V4,
                                  M.BengaliResNet34AGeMV4,
                                  M.BengaliSEResNeXt50V4,
                                  M.BengaliEfficientNetB0V4,
                                  M.BengaliEfficientNetB3V4))
        criterion_feat_v = get_criterion(
            conf.loss_type_feat_v,
            dim=model.multihead.head_v.dim, n_class=11,
            s=conf.af_scale_v
        )
        workspace.log(f'Loss type (fv): {conf.loss_type_feat_v}')
        if conf.loss_type_feat_v in ('af',):
            sub_models.append(criterion_feat_v)
            workspace.log('Add criterion_feat_v to sub model')
    else:
        criterion_feat_v = None

    if conf.loss_type_feat_c != 'none':
        assert isinstance(model, (M.BengaliResNet34V3,
                                  M.BengaliResNet34V4,
                                  M.BengaliResNet34AGeMV4,
                                  M.BengaliSEResNeXt50V4,
                                  M.BengaliEfficientNetB0V4,
                                  M.BengaliEfficientNetB3V4))
        criterion_feat_c = get_criterion(
            conf.loss_type_feat_c,
            dim=model.multihead.head_c.dim, n_class=7,
            s=conf.af_scale_c
        )
        workspace.log(f'Loss type (fc): {conf.loss_type_feat_c}')
        if conf.loss_type_feat_c in ('af',):
            sub_models.append(criterion_feat_c)
            workspace.log('Add criterion_feat_c to sub model')
    else:
        criterion_feat_c = None

    if conf.use_grapheme_code:
        workspace.log('Use grapheme code classifier')
        grapheme_classifier = nn.Sequential(
            nn.BatchNorm1d(168 + 11 + 7),
            nn.Linear(168 + 11 + 7, 1295)
        )
        grapheme_classifier = grapheme_classifier.cuda()
        grapheme_classifier.train()
        sub_models.append(grapheme_classifier)
        criterion_grapheme = L.OHEMCrossEntropyLoss().cuda()
    else:
        grapheme_classifier = None
        criterion_grapheme = None

    parameters = [{'params': model.parameters()}] + \
                 [{'params': sub_model.parameters()} for sub_model in sub_models]

    if conf.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(parameters,
                                     lr=conf.lr)
    elif conf.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(parameters,
                                    lr=conf.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)
    elif conf.optimizer_type == 'ranger':
        optimizer = Ranger(parameters,
                           lr=conf.lr,
                           weight_decay=1e-4)
    elif conf.optimizer_type == 'radam':
        optimizer = RAdam(parameters,
                          lr=conf.lr,
                          weight_decay=1e-4)
    else:
        raise ValueError(conf.optimizer_type)
    workspace.log(f'Optimizer type: {conf.optimizer_type}')

    if conf.use_apex:
        workspace.log('Apex initialization')
        _models, optimizer = amp.initialize([model] + sub_models, optimizer, opt_level=conf.apex_opt_level)
        if len(_models) == 1:
            model = _models[0]
        else:
            model = _models[0]
            criterion_feat_g = _models[1]
            criterion_feat_v = _models[2]
            criterion_feat_c = _models[3]
        workspace.log('Initialized by Apex')
        workspace.log(f'{optimizer.__class__.__name__}')
        for m in _models:
            workspace.log(f'{m.__class__.__name__}')

    if conf.scheduler_type == 'cosanl':
        scheduler = CosineLRWithRestarts(
            optimizer, conf.batch_size, len(train_dataset),
            restart_period=conf.cosanl_restart_period,
            t_mult=conf.cosanl_t_mult
        )
        workspace.log(f'restart_period={scheduler.restart_period}')
        workspace.log(f't_mult={scheduler.t_mult}')
    elif conf.scheduler_type == 'rop':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=conf.rop_patience, mode='max',
            factor=conf.rop_factor, min_lr=1e-6, verbose=True
        )
    else:
        raise ValueError(conf.scheduler_type)

    train(model, train_loader, val_loader,
          optimizer,
          criterion_g,
          criterion_v,
          criterion_c,
          criterion_feat_g,
          criterion_feat_v,
          criterion_feat_c,
          workspace,
          scheduler=scheduler,
          n_epoch=conf.n_epoch,
          cutmix_prob=conf.cutmix_prob,
          mixup_prob=conf.mixup_prob,
          freeze_bn_epochs=conf.freeze_bn_epochs,
          feat_loss_weight=conf.feat_loss_weight,
          use_apex=conf.use_apex,
          decrease_ohem_rate=conf.decrease_ohem_rate,
          use_grapheme_code=conf.use_grapheme_code,
          grapheme_classifier=grapheme_classifier,
          criterion_grapheme=criterion_grapheme,
          final_ft=conf.final_ft)


def train(model, train_loader, val_loader,
          optimizer: torch.optim.Optimizer,
          criterion_g,
          criterion_v,
          criterion_c,
          criterion_feat_g,
          criterion_feat_v,
          criterion_feat_c,
          workspace: Workspace,
          scheduler=None,
          n_epoch=30,
          cutmix_prob=0,
          mixup_prob=0,
          freeze_bn_epochs=None,
          feat_loss_weight=1.0,
          use_apex=False,
          decrease_ohem_rate=False,
          use_grapheme_code=False,
          grapheme_classifier=None,
          criterion_grapheme=None,
          final_ft=False):
    score = evaluate(model, val_loader)
    workspace.log(f'Score={score}', epoch=0)
    workspace.plot_score('val/score', score, 0)

    freeze_bn_epochs = freeze_bn_epochs or []
    global_step = -1

    if final_ft:
        workspace.log('Freeze backbone')
        M.freeze_backbone(model)
        M.freeze_multihead(model)

    for epoch in range(1, n_epoch + 1):
        model.train()
        if grapheme_classifier is not None:
            grapheme_classifier.train()

        if epoch in freeze_bn_epochs:
            model.apply(set_batchnorm_eval)
            workspace.log(f'Freeze BN', epoch=epoch)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()
            workspace.log(f'Scheduler.step()', epoch=epoch)

        if decrease_ohem_rate:
            if isinstance(criterion_g, L.OHEMCrossEntropyLoss):
                r_before, r_after = criterion_g.adjust_rate(epoch)
                workspace.log(f'OHEM(g).rate: {r_before} -> {r_after}')
            if isinstance(criterion_v, L.OHEMCrossEntropyLoss):
                r_before, r_after = criterion_v.adjust_rate(epoch)
                workspace.log(f'OHEM(v).rate: {r_before} -> {r_after}')
            if isinstance(criterion_c, L.OHEMCrossEntropyLoss):
                r_before, r_after = criterion_c.adjust_rate(epoch)
                workspace.log(f'OHEM(c).rate: {r_before} -> {r_after}')
            if isinstance(criterion_grapheme, L.OHEMCrossEntropyLoss):
                r_before, r_after = criterion_grapheme.adjust_rate(epoch)
                workspace.log(f'OHEM(grapheme).rate: {r_before} -> {r_after}')
        else:
            if isinstance(criterion_g, L.OHEMCrossEntropyLoss):
                workspace.log(f'OHEM(g).rate: {criterion_g.rate}')
            if isinstance(criterion_v, L.OHEMCrossEntropyLoss):
                workspace.log(f'OHEM(v).rate: {criterion_v.rate}')
            if isinstance(criterion_c, L.OHEMCrossEntropyLoss):
                workspace.log(f'OHEM(c).rate: {criterion_c.rate}')
            if isinstance(criterion_grapheme, L.OHEMCrossEntropyLoss):
                workspace.log(f'OHEM(grapheme).rate: {criterion_grapheme.rate}')

        for iteration, data_tuple in enumerate(train_loader):
            global_step += 1
            if use_grapheme_code:
                (x, tg, tv, tc, tgrapheme) = data_tuple
            else:
                (x, tg, tv, tc) = data_tuple

            if global_step == 0:
                workspace.log(f'Check tensor size: x={x.size()}, '
                              f'tg={tg.size()}, tv={tv.size()}, tc={tc.size()}')

            r = np.random.rand(1)
            if r < cutmix_prob:
                use_cutmix = True
                use_mixup = False
            elif r < cutmix_prob + mixup_prob:
                use_cutmix = False
                use_mixup = True
            else:
                use_cutmix = False
                use_mixup = False

            x = x.cuda()
            (tg, tv, tc) = (tg.cuda(), tv.cuda(), tc.cuda())
            
            if use_grapheme_code:
                tgrapheme = tgrapheme.cuda()

            loss_feat_g = 0
            loss_feat_v = 0
            loss_feat_c = 0
            loss_grapheme = 0

            if use_cutmix or use_mixup:
                if use_cutmix:
                    x, rand_index, lam = cutmix(x, beta=1.0)
                    mix_criterion = cutmix_criterion
                elif use_mixup:
                    x, rand_index, lam = mixup(x, alpha=1.0)
                    mix_criterion = mixup_criterion

                tga, tgb = tg, tg[rand_index]
                tva, tvb = tv, tv[rand_index]
                tca, tcb = tc, tc[rand_index]

                if isinstance(model, (M.BengaliResNet34JPUAF,)):
                    logit_g, logit_v, logit_c = model(x, tg=tg, tv=tv, tc=tc)
                elif isinstance(model, (M.BengaliResNet34V3,
                                        M.BengaliResNet34V4,
                                        M.BengaliResNet34AGeMV4,
                                        M.BengaliSEResNeXt50V4,
                                        M.BengaliEfficientNetB0V4,
                                        M.BengaliEfficientNetB3V4)):
                    (feat,
                     feat_g, logit_g,
                     feat_v, logit_v,
                     feat_c, logit_c) = model(x)
                else:
                    logit_g, logit_v, logit_c = model(x)

                loss_g = mix_criterion(
                    logit_g, tga, tgb, lam, criterion=criterion_g
                )
                loss_v = mix_criterion(
                    logit_v, tva, tvb, lam, criterion=criterion_v
                )
                loss_c = mix_criterion(
                    logit_c, tca, tcb, lam, criterion=criterion_c
                )
            else:
                if isinstance(model, (M.BengaliResNet34JPUAF,)):
                    logit_g, logit_v, logit_c = model(x, tg=tg, tv=tv, tc=tc)
                elif isinstance(model, (M.BengaliResNet34V3,
                                        M.BengaliResNet34V4,
                                        M.BengaliResNet34AGeMV4,
                                        M.BengaliSEResNeXt50V4,
                                        M.BengaliEfficientNetB0V4,
                                        M.BengaliEfficientNetB3V4)):
                    (feat,
                     feat_g, logit_g,
                     feat_v, logit_v,
                     feat_c, logit_c) = model(x)

                    if criterion_feat_g is None:
                        pass
                    else:
                        loss_feat_g = criterion_feat_g(feat_g, tg)

                    if criterion_feat_v is None:
                        pass
                    else:
                        loss_feat_v = criterion_feat_v(feat_v, tv)

                    if criterion_feat_c is None:
                        pass
                    else:
                        loss_feat_c = criterion_feat_c(feat_c, tc)
                else:
                    logit_g, logit_v, logit_c = model(x)

                loss_g = criterion_g(logit_g, tg)
                loss_v = criterion_v(logit_v, tv)
                loss_c = criterion_c(logit_c, tc)

                if use_grapheme_code:
                    logit_grapheme = grapheme_classifier(
                        torch.cat([logit_g, logit_v, logit_c], dim=1)
                    )
                    loss_grapheme = criterion_grapheme(logit_grapheme, tgrapheme)

            loss_feat = loss_feat_g + loss_feat_v + loss_feat_c
            loss = loss_g + loss_v + loss_c + loss_grapheme + feat_loss_weight * loss_feat

            if global_step % 20 == 0:
                if loss_feat == 0:
                    if loss_grapheme == 0:
                        workspace.log(f'Iteration={iteration}, Loss={loss}',
                                      epoch=epoch)
                    else:
                        workspace.log(f'Iteration={iteration}, Loss={loss}, LossGrapheme={loss_grapheme}',
                                      epoch=epoch)
                else:
                    if loss_grapheme == 0:
                        workspace.log(f'Iteration={iteration}, Loss={loss}, FeatLoss={loss_feat}',
                                      epoch=epoch)
                    else:
                        workspace.log(f'Iteration={iteration}, Loss={loss}, LossGrapheme={loss_grapheme}, FeatLoss={loss_feat}',
                                      epoch=epoch)
                workspace.plot_score('train/loss', float(loss.item()),
                                     global_step)
                workspace.plot_score('train/lr',
                                     float(get_current_lr(optimizer)),
                                     global_step)

            optimizer.zero_grad()

            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            if isinstance(scheduler, CosineLRWithRestarts):
                scheduler.batch_step()

        score = evaluate(model, val_loader)

        workspace.log(f'Score={score}', epoch=epoch)
        workspace.plot_score('val/score', score, epoch)
        saved = workspace.save_bestmodel(model, epoch, score)

        if saved:
            checkpoint = {
                'optimizer': optimizer.state_dict(),
                'amp': None if not use_apex else amp.state_dict()
            }
            if scheduler is not None:
                checkpoint['scheduler'] = scheduler.state_dict()
            if isinstance(criterion_feat_g, nn.Module):
                checkpoint['criterion_feat_g'] = criterion_feat_g.state_dict()
            if isinstance(criterion_feat_v, nn.Module):
                checkpoint['criterion_feat_v'] = criterion_feat_v.state_dict()
            if isinstance(criterion_feat_c, nn.Module):
                checkpoint['criterion_feat_c'] = criterion_feat_c.state_dict()
            workspace.save_checkpoint(epoch, name='best', **checkpoint)
    workspace.save_model(model, n_epoch)


def evaluate(model, val_loader, device=None):
    model.eval()

    pred_g, pred_v, pred_c = [], [], []
    true_g, true_v, true_c = [], [], []
    with torch.no_grad():
        for x, tg, tv, tc in tqdm.tqdm(val_loader):
            if device:
                x = x.to(device)
            else:
                x = x.cuda()
            logit_g, logit_v, logit_c = model(x)
            pred_g.append(torch.argmax(logit_g, dim=1).cpu().numpy())
            pred_v.append(torch.argmax(logit_v, dim=1).cpu().numpy())
            pred_c.append(torch.argmax(logit_c, dim=1).cpu().numpy())
            true_g.append(tg.numpy())
            true_v.append(tv.numpy())
            true_c.append(tc.numpy())

    pred_g = np.concatenate(pred_g)
    pred_v = np.concatenate(pred_v)
    pred_c = np.concatenate(pred_c)
    true_g = np.concatenate(true_g)
    true_v = np.concatenate(true_v)
    true_c = np.concatenate(true_c)

    score = hierarchical_macro_averaged_recall(
        pred_g, true_g,
        pred_v, true_v,
        pred_c, true_c
    )
    return score


if __name__ == '__main__':
    main()