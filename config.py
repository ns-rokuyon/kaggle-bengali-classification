import yaml
from pathlib import Path


defs = {
    'run_id': (str, None),
    'data_dir': (Path, 'data'),
    'fold_id': (int, 0),
    'n_epoch': (int, 30),
    'n_iter_per_epoch': (int, 1600),
    'arch': (str, 'BengaliSEResNeXt50'),
    'use_maxblurpool': (bool, False),
    'batch_size': (int, 64),
    'input_size': (int, 128),
    'lr': (float, 1e-3),
    'freeze_bn_epochs': (lambda xs: list(map(int, xs)), []),
    'loss_type': (str, 'ce'),   # ce | ohem
    'pooling_type': (str, 'gap'),   # gap | gemp
    'optimizer_type': (str, 'adam'),    # adam | sgd
    'scheduler_type': (str, 'cosanl'),  # cosanl | rop
    'augmentor_type': (str, 'v1'),  # v1
    'sampler_type': (str, 'random'),    # random | pk
    'use_augmentor': (bool, False),
    'cutmix_prob': (float, 0.0),
    'feat_dim': (int, 64),
    'rop_patience': (int, 3),
    'rop_factor': (float, 0.1),
    'enable_random_morph': (bool, False)
}


class Config:
    def __init__(self, config_file):
        self.config_file = config_file
        with open(config_file, 'r') as fp:
            self._config = yaml.safe_load(fp)

    def __getattr__(self, key):
        T, default = defs.get(key)
        v = self._config.get(key) or default
        if v is None:
            raise KeyError(key)
        return T(v)

    def to_dict(self):
        return {
            k: getattr(self, k)
            for k in defs.keys()
        }