from pathlib import Path
from argparse import ArgumentParser
from dataset import bengali_dataset
from data import get_logger


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data-dir', default='data', type=Path,
                        help='Path to data dir')
    parser.add_argument('--fold-id', default=0, type=int,
                        help='Kfold id')
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger(__name__)
    data_dir = args.data_dir
    fold_id = args.fold_id

    train_dataset, val_dataset = bengali_dataset(data_dir,
                                                 fold_id=fold_id,
                                                 logger=logger)
    logger.info(f'#train={len(train_dataset)}, #val={len(val_dataset)}')


if __name__ == '__main__':
    main()