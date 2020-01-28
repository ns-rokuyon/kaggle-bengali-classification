import pandas as pd
from torch.utils.data.dataset import Dataset

from data import read_image, load_kfolds


def bengali_dataset(data_dir, fold_id=0,
                    train_transformer=None,
                    val_transformer=None,
                    logger=None):
    """Load Bengali dataset (train, val)
    """
    train_csv = data_dir / 'train.csv'
    train_image_data = [
        data_dir / f'train_image_data_{i}.parquet'
        for i in range(4)
    ]
    if logger:
        logger.info('load train_csv ...')
    df = pd.read_csv(train_csv)

    if logger:
        logger.info('load image_dfs ...')
    image_dfs = [pd.read_parquet(p) for p in train_image_data]

    if logger:
        logger.info('load kfolds ...')
    kfolds = load_kfolds(data_dir)

    if logger:
        logger.info('Concat image_dfs')
    image_df = pd.concat(image_dfs).reset_index(drop=True)

    train_ids, val_ids = kfolds[fold_id]
    train_dataset = BengaliSubsetDataset(df, image_df, train_ids,
                                         transformer=train_transformer)
    val_dataset = BengaliSubsetDataset(df, image_df, val_ids,
                                       transformer=val_transformer)
    return train_dataset, val_dataset


class BengaliDataset(Dataset):
    def __init__(self, df, image_df, transformer=None):
        self.df = df
        self.image_df = image_df
        self.transformer = transformer
        assert len(self.df) == len(self.image_df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        _, im = read_image(self.image_df, i, to_pil=True)
        if self.transformer:
            im = self.transformer(im)
        labels = self.get_multilabels(i)
        return (im, *labels)

    def get_multilabels(self, i):
        row = self.df.iloc[i]
        grapheme = int(row.grapheme_root)
        vowel = int(row.vowel_diacritic)
        consonant = int(row.consonant_diacritic)
        return (grapheme, vowel, consonant)


class BengaliSubsetDataset(BengaliDataset):
    def __init__(self, df, image_dfs, active_ids,
                 transformer=None):
        super().__init__(df, image_dfs, transformer=transformer)
        self.active_ids = active_ids

    def __len__(self):
        return len(self.active_ids)

    def __getitem__(self, i):
        j = self.active_ids[i]
        return super().__getitem__(j)