import numpy as np
import tqdm
import pickle
import logging
import torch
import shutil
from pathlib import Path
from PIL import Image
from torch.utils.tensorboard import SummaryWriter


SOURCE_IMAGE_HEIGHT = 137
SOURCE_IMAGE_WIDTH = 236


def read_image(dataset, i, to_pil=True, invert_color=False, n_channel=1):
    """Read an image in dataframe as grayscale PIL image
    """
    image_id = dataset.ids[i]
    image = dataset.images[i, :].reshape(137, 236).astype(np.uint8)
    if invert_color:
        image = 255 - image
    if n_channel == 3:
        image = np.tile(image, (3, 1, 1)).transpose((1, 2, 0))
    if to_pil:
        img_type = 'RGB' if n_channel == 3 else 'L'
        return image_id, Image.fromarray(image, img_type)
    return image_id, image


def read_image_df(df, i, to_pil=True, invert_color=False, n_channel=1):
    """Read an image in dataframe as grayscale PIL image
    """
    image_id = df.iloc[i, 0]
    image = df.iloc[i, 1:].values.reshape(137, 236).astype(np.uint8)
    if invert_color:
        image = 255 - image
    if n_channel == 3:
        image = np.tile(image, (3, 1, 1)).transpose((1, 2, 0))
    if to_pil:
        img_type = 'RGB' if n_channel == 3 else 'L'
        return image_id, Image.fromarray(image, img_type)
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


def get_current_lr(optimizer, head=True):
    lrs = [g['lr'] for g in optimizer.param_groups]
    if head:
        return lrs[0]
    return lrs


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
        self.conf_dir.mkdir(parents=True, exist_ok=True)
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
    def conf_dir(self):
        return self.root_dir / 'conf'

    @property
    def conf_file(self):
        return self.conf_dir / f'{self.run_id}.yaml'

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

    def save_conf(self, config_file: Path):
        shutil.copy(config_file, self.conf_file)

    def save_bestmodel(self, model: torch.nn.Module, epoch: int, score: float):
        if score >= self.best_score:
            best_model_path = self.model_dir / f'{self.run_id}_best.model'
            torch.save(model.state_dict(), best_model_path)
            self.log('Saved best model', epoch=epoch)
            self.log(f'Best score {self.best_score} -> {score}', epoch=epoch)
            self.best_score = score
            self.best_epoch = epoch
            return True
        return False

    def save_model(self, model: torch.nn.Module, epoch: int):
        model_path = self.model_dir / f'{self.run_id}_epoch{epoch}.model'
        torch.save(model.state_dict(), model_path)
        self.log(f'Saved model: {model_path}', epoch=epoch)

    def save_checkpoint(self, epoch: int, name=None, **kwargs):
        name = name or f'epoch_{epoch}'
        checkpoint_path = self.model_dir / f'{self.run_id}_{name}.checkpoint'
        checkpoint = kwargs
        torch.save(checkpoint, checkpoint_path)
        self.log(f'Saved checkpoint: {checkpoint_path}', epoch=epoch)

    def plot_score(self, tag, value, global_step):
        self.tb_writer.add_scalar(tag, value,
                                  global_step=global_step)