import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from argparse import ArgumentParser

from dataset import bengali_dataset
from data import get_logger
from model import create_init_model
from preprocessing import (
    create_transformer_v1,
    create_testing_transformer_v1
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data-dir', default='data', type=Path,
                        help='Path to data dir')
    parser.add_argument('--fold-id', default=0, type=int,
                        help='Kfold id')
    parser.add_argument('--n-epoch', default=30, type=int,
                        help='Number of epoch')
    parser.add_argument('--arch', default='BengaliResNet34',
                        help='Networks arch')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Minibatch size')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger(__name__)
    data_dir = args.data_dir
    fold_id = args.fold_id

    torch.cuda.set_device(0)

    train_transformer = create_transformer_v1()
    val_transformer = create_testing_transformer_v1()

    train_dataset, val_dataset = bengali_dataset(data_dir,
                                                 fold_id=fold_id,
                                                 train_transformer=train_transformer,
                                                 val_transformer=val_transformer,
                                                 logger=logger)
    logger.info(f'#train={len(train_dataset)}, #val={len(val_dataset)}')

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            pin_memory=True,
                            drop_last=True)

    logger.info(f'Create init model: arch={args.arch}')
    model = create_init_model(args.arch, pretrained=True)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters())

    train(model, train_loader, val_loader,
          optimizer,
          logger,
          n_epoch=args.n_epoch)


def train(model, train_loader, val_loader,
          optimizer: torch.optim.Optimizer,
          logger,
          n_epoch=30):
    for epoch in range(n_epoch):
        model.train()

        for iteration, (x, tg, tv, tc) in enumerate(train_loader):
            x = x.cuda()
            (tg, tv, tc) = (tg.cuda(), tv.cuda(), tc.cuda())

            logit_g, logit_v, logit_c = model(x)
            loss_g = F.cross_entropy(logit_g, tg)
            loss_v = F.cross_entropy(logit_v, tv)
            loss_c = F.cross_entropy(logit_c, tc)

            loss = loss_g + loss_v + loss_c

            if iteration % 10 == 0:
                logger.info(f'Iteration={iteration}, Epoch={epoch} '
                            f'Loss={loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()