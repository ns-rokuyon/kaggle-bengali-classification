import yaml
from pathlib import Path


defs = {
    'run_id': (str, None),
    'data_dir': (Path, 'data'),
    'fold_id': (int, 0),
    'n_epoch': (int, 30),
    'arch': (str, 'BengaliSEResNeXt50'),
    'batch_size': (int, 64),
    'input_size': (int, 128),
    'loss_type': (str, 'ce'),   # ce | ohem
    'pooling_type': (str, 'gap'),   # gap | gemp
    'scheduler_type': (str, 'cosanl'),  # cosanl | rop
    'cutmix_prob': (float, 0.0)
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