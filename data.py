import numpy as np
import tqdm
import pickle
import logging
from PIL import Image


SOURCE_IMAGE_HEIGHT = 137
SOURCE_IMAGE_WIDTH = 236



def read_image(df, i, to_pil=True):
    """Read an image in dataframe as grayscale PIL image
    """
    image_id = df.iloc[i, 0]
    image = df.iloc[i, 1:].values.reshape(137, 236).astype(np.uint8)
    if to_pil:
        return image_id, Image.fromarray(image, 'L')
    return image_id, image


def load_kfolds(data_dir):
    with open(data_dir / 'kfolds.pickle', 'rb') as fp:
        kfolds = pickle.load(fp)
    return kfolds


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel('DEBUG')
    formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger