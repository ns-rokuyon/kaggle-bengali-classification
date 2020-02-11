import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tqdm
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from argparse import ArgumentParser

from dataset import bengali_dataset
from data import get_logger, Workspace, get_current_lr
from model import create_init_model
from preprocessing import (
    create_transformer_v1,
    create_testing_transformer_v1
)
from evaluation import hierarchical_macro_averaged_recall
from optim import CosineLRWithRestarts
from loss import get_criterion
from lib.cutmix import cutmix, cutmix_criterion


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--run-id', required=True,
                        help='Run id')
    parser.add_argument('--data-dir', default='data', type=Path,
                        help='Path to data dir')
    parser.add_argument('--fold-id', default=0, type=int,
                        help='Kfold id')
    parser.add_argument('--n-epoch', default=30, type=int,
                        help='Number of epoch')
    parser.add_argument('--arch', default='BengaliSEResNeXt50',
                        help='Networks arch')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='Minibatch size')
    parser.add_argument('--input-size', default=0, type=int,
                        help='Input image size')
    parser.add_argument('--loss-type', default='ce',
                        choices=('ce', 'ohem'),
                        help='Loss type')
    parser.add_argument('--pooling-type', default='gap',
                        choices=('gap', 'gemp'),
                        help='Global pooling type')
    parser.add_argument('--cutmix-prob', default=0.0, type=float,
                        help='Probability of cutmix')
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    fold_id = args.fold_id

    workspace = Workspace(args.run_id).setup()
    workspace.log(f'Args: {args}')

    torch.cuda.set_device(0)

    if args.input_size == 0:
        train_transformer = create_transformer_v1()
        val_transformer = create_testing_transformer_v1()
        workspace.log('Input size: default')
    else:
        input_size = (args.input_size, args.input_size)
        train_transformer = create_transformer_v1(input_size=input_size)
        val_transformer = create_testing_transformer_v1(input_size=input_size)
        workspace.log(f'Input size: {input_size}')

    train_dataset, val_dataset = bengali_dataset(data_dir,
                                                 fold_id=fold_id,
                                                 train_transformer=train_transformer,
                                                 val_transformer=val_transformer,
                                                 logger=workspace.logger)
    workspace.log(f'#train={len(train_dataset)}, #val={len(val_dataset)}')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)

    workspace.log(f'Create init model: arch={args.arch}')
    model = create_init_model(args.arch,
                              pretrained=True,
                              pooling=args.pooling_type)
    model = model.cuda()

    criterion = get_criterion(args.loss_type)
    workspace.log(f'Loss type: {args.loss_type}')

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = CosineLRWithRestarts(
        optimizer, args.batch_size, len(train_dataset),
        restart_period=6, t_mult=1.0
    )

    train(model, train_loader, val_loader,
          optimizer,
          criterion,
          workspace,
          scheduler=scheduler,
          n_epoch=args.n_epoch,
          cutmix_prob=args.cutmix_prob)


def train(model, train_loader, val_loader,
          optimizer: torch.optim.Optimizer,
          criterion,
          workspace: Workspace,
          scheduler=None,
          n_epoch=30,
          cutmix_prob=0):
    score = evaluate(model, val_loader)
    workspace.log(f'Score={score}', epoch=0)
    workspace.plot_score('val/score', score, 0)

    global_step = -1

    for epoch in range(1, n_epoch + 1):
        model.train()

        if scheduler:
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

            if scheduler:
                scheduler.batch_step()

        score = evaluate(model, val_loader)

        workspace.log(f'Score={score}', epoch=epoch)
        workspace.plot_score('val/score', score, epoch)

        workspace.save_bestmodel(model, epoch, score)


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