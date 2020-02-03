import numpy as np
import tqdm
import pickle
import logging
import torch
from pathlib import Path
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


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


def get_logger(name, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel('DEBUG')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class Workspace:
    def __init__(self, run_id, root_dir=None):
        self.run_id = run_id
        self.root_dir = Path(root_dir or 'workspace')
        self.logger = None
        self.best_score = 0
        self.best_epoch = 1
        self.tb_writer = None

    def setup(self):
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.tb_root_log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__, log_file=self.log_file)
        self.tb_writer = SummaryWriter(log_dir=str(self.tb_log_dir))
        return self

    @property
    def model_dir(self):
        return self.root_dir / 'model' / self.run_id

    @property
    def tb_root_log_dir(self):
        return self.root_dir / 'tb'

    @property
    def tb_log_dir(self):
        return self.tb_root_log_dir / self.run_id

    @property
    def log_dir(self):
        return self.root_dir / 'log'

    @property
    def log_file(self):
        return self.log_dir / f'{self.run_id}.log'

    def log(self, message, epoch=None):
        if epoch is not None:
            message = f'Epoch({epoch}): {message}'
        self.logger.info(message)

    def save_bestmodel(self, model: torch.nn.Module, epoch: int, score: float):
        if score >= self.best_score:
            best_model_path = self.model_dir / f'{self.run_id}_best.model'
            torch.save(model.state_dict(), best_model_path)
            self.log('Saved best model', epoch=epoch)
            self.log(f'Best score {self.best_score} -> {score}', epoch=epoch)
            self.best_score = score
            self.best_epoch = epoch

    def plot_score(self, tag, value, global_step):
        self.tb_writer.add_scalar(tag, value,
                                  global_step=global_step)