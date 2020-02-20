import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from argparse import ArgumentParser

import loss as L
from dataset import bengali_dataset
from data import get_logger, Workspace, get_current_lr
from model import create_init_model, set_batchnorm_eval
from preprocessing import (
    create_transformer_v1,
    create_testing_transformer_v1,
    create_augmentor_v1
)
from evaluation import hierarchical_macro_averaged_recall
from optim import CosineLRWithRestarts, Ranger
from loss import get_criterion
from sampler import PKSampler
from config import Config
from lib.cutmix import cutmix, cutmix_criterion


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
        augmentor = create_augmentor_v1()
        workspace.log(f'Use augmentor: {conf.augmentor_type}')
    else:
        augmentor = None

    if conf.input_size == 0:
        train_transformer = create_transformer_v1(augmentor=augmentor)
        val_transformer = create_testing_transformer_v1()
        workspace.log('Input size: default')
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
                                                 logger=workspace.logger)
    workspace.log(f'#train={len(train_dataset)}, #val={len(val_dataset)}')

    if conf.sampler_type == 'pk':
        sampler = PKSampler(train_dataset,
                            n_iter_per_epoch=conf.n_iter_per_epoch,
                            p=24, k=4)
        train_loader = DataLoader(train_dataset,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True,
                                  batch_sampler=sampler)
        workspace.log(f'{sampler} is enabled')
        workspace.log(f'Real batch_size={sampler.batch_size}')
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
                              use_maxblurpool=conf.use_maxblurpool)
    model = model.cuda()

    criterion = get_criterion(conf.loss_type)
    workspace.log(f'Loss type: {conf.loss_type}')

    if conf.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=conf.lr)
    elif conf.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=conf.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4)
    elif conf.optimizer_type == 'ranger':
        optimizer = Ranger(model.parameters(),
                           lr=conf.lr,
                           weight_decay=1e-4)
    else:
        raise ValueError(conf.optimizer_type)
    workspace.log(f'Optimizer type: {conf.optimizer_type}')

    if conf.scheduler_type == 'cosanl':
        scheduler = CosineLRWithRestarts(
            optimizer, conf.batch_size, len(train_dataset),
            restart_period=6, t_mult=1.0
        )
    elif conf.scheduler_type == 'rop':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=conf.rop_patience, mode='max',
            factor=conf.rop_factor, min_lr=1e-7, verbose=True
        )
    else:
        raise ValueError(conf.scheduler_type)

    train(model, train_loader, val_loader,
          optimizer,
          criterion,
          workspace,
          scheduler=scheduler,
          n_epoch=conf.n_epoch,
          cutmix_prob=conf.cutmix_prob,
          freeze_bn_epochs=conf.freeze_bn_epochs)


def train(model, train_loader, val_loader,
          optimizer: torch.optim.Optimizer,
          criterion,
          workspace: Workspace,
          scheduler=None,
          n_epoch=30,
          cutmix_prob=0,
          freeze_bn_epochs=None):
    score = evaluate(model, val_loader)
    workspace.log(f'Score={score}', epoch=0)
    workspace.plot_score('val/score', score, 0)

    freeze_bn_epochs = freeze_bn_epochs or []
    global_step = -1

    for epoch in range(1, n_epoch + 1):
        model.train()
        if epoch in freeze_bn_epochs:
            model.apply(set_batchnorm_eval)
            workspace.log(f'Freeze BN', epoch=epoch)

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(score)
            else:
                scheduler.step()
            workspace.log(f'Scheduler.step()', epoch=epoch)

        for iteration, (x, tg, tv, tc) in enumerate(train_loader):
            global_step += 1
            r = np.random.rand(1)
            use_cutmix = r < cutmix_prob

            x = x.cuda()
            (tg, tv, tc) = (tg.cuda(), tv.cuda(), tc.cuda())

            if use_cutmix:
                x, rand_index, lam = cutmix(x, beta=1.0)
                tga, tgb = tg, tg[rand_index]
                tva, tvb = tv, tv[rand_index]
                tca, tcb = tc, tc[rand_index]

                logit_g, logit_v, logit_c = model(x)

                loss_g = cutmix_criterion(
                    logit_g, tga, tgb, lam, criterion=criterion
                )
                loss_v = cutmix_criterion(
                    logit_v, tva, tvb, lam, criterion=criterion
                )
                loss_c = cutmix_criterion(
                    logit_c, tca, tcb, lam, criterion=criterion
                )
            else:
                logit_g, logit_v, logit_c = model(x)

                loss_g = criterion(logit_g, tg)
                loss_v = criterion(logit_v, tv)
                loss_c = criterion(logit_c, tc)

            loss = loss_g + loss_v + loss_c

            if global_step % 20 == 0:
                workspace.log(f'Iteration={iteration}, Loss={loss}',
                              epoch=epoch)
                workspace.plot_score('train/loss', float(loss.item()),
                                     global_step)
                workspace.plot_score('train/lr',
                                     float(get_current_lr(optimizer)),
                                     global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if isinstance(scheduler, CosineLRWithRestarts):
                scheduler.batch_step()

        score = evaluate(model, val_loader)

        workspace.log(f'Score={score}', epoch=epoch)
        workspace.plot_score('val/score', score, epoch)

        workspace.save_bestmodel(model, epoch, score)
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