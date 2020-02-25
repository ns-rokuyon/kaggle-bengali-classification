import pandas as pd
from pathlib import Path
from torch.utils.data.dataset import Dataset

from data import read_image, load_kfolds


def bengali_dataset(data_dir, fold_id=0,
                    train_transformer=None,
                    val_transformer=None,
                    logger=None):
    """Load Bengali dataset (train, val)
    """
    data_dir = Path(data_dir)
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
        logger.info('Concat image_dfs ...')
    image_df = pd.concat(image_dfs).reset_index(drop=True)

    image_ids = image_df.iloc[:, 0].values
    images = image_df.iloc[:, 1:].values

    train_ids, val_ids = kfolds[fold_id]

    if logger:
        logger.info('Create datasets ...')
    train_dataset = BengaliSubsetDataset(df, image_ids, images, train_ids,
                                         transformer=train_transformer)
    val_dataset = BengaliSubsetDataset(df, image_ids, images, val_ids,
                                       transformer=val_transformer)
    return train_dataset, val_dataset


class BengaliDataset(Dataset):
    def __init__(self, df, image_ids, images, transformer=None):
        self.df = df
        self.images = images
        self.ids = image_ids
        self.transformer = transformer
        assert len(self.df) == len(self.images)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        _, im = read_image(self, i, to_pil=True)
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
    def __init__(self, df, image_ids, images, active_ids,
                 transformer=None):
        super().__init__(df, image_ids, images, transformer=transformer)
        self.active_ids = active_ids

    def __len__(self):
        return len(self.active_ids)

    def __getitem__(self, i):
        j = self.active_ids[i]
        return super().__getitem__(j)

    def get_grapheme_root_labels(self):
        return self.df.grapheme_root[self.active_ids]

    def get_class_weights_g(self):
        g_counts = self.df.grapheme_root.value_counts()
        return g_counts[list(range(len(g_counts)))]

    def get_class_weights_v(self):
        v_counts = self.df.vowel_diacritic.value_counts()
        return v_counts[list(range(len(v_counts)))]

    def get_class_weights_c(self):
        c_counts = self.df.consonant_diacritic.value_counts()
        return c_counts[list(range(len(c_counts)))]