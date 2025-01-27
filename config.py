import yaml
from pathlib import Path


defs = {
    'run_id': (str, None),
    'data_dir': (Path, 'data'),
    'fold_id': (int, 0),
    'use_apex': (bool, False),
    'apex_opt_level': (str, 'O1'),
    'n_epoch': (int, 30),
    'n_iter_per_epoch': (int, 1600),
    'arch': (str, 'BengaliSEResNeXt50'),
    'n_channel': (int, 1),
    'use_maxblurpool': (bool, False),
    'remove_last_stride': (bool, False),
    'weight_file': (str, ''),
    'batch_size': (int, 64),
    'batch_p': (int, 64),
    'batch_k': (int, 2),
    'n_low_freq_samples': (int, 10),
    'n_class_low_freq': (int, 60),
    'input_size': (int, 128),
    'input_size_tuple': (lambda xs: list(map(int, xs)), []),
    'lr': (float, 1e-3),
    'freeze_bn_epochs': (lambda xs: list(map(int, xs)), []),
    'loss_type_g': (str, 'ce'),   # ce | weighted_ce | ohem | focal | reduced_focal
    'loss_type_v': (str, 'ce'),   # ce | weighted_ce | ohem | focal | reduced_focal
    'loss_type_c': (str, 'ce'),   # ce | weighted_ce | ohem | focal | reduced_focal
    'loss_type_feat_g': (str, 'none'),
    'loss_type_feat_v': (str, 'none'),
    'loss_type_feat_c': (str, 'none'),
    'feat_loss_weight': (float, 1.0),
    'pooling_type': (str, 'gap'),   # gap | gemp
    'optimizer_type': (str, 'adam'),    # adam | sgd | ranger | radam
    'scheduler_type': (str, 'cosanl'),  # cosanl | rop
    'augmentor_type': (str, 'v1'),  # v1 | v2 | v3
    'sampler_type': (str, 'random'),    # random | pk
    'use_augmentor': (bool, False),
    'cutmix_prob': (float, 0.0),
    'mixup_prob': (float, 0.0),
    'feat_dim': (int, 64),
    'rop_patience': (int, 3),
    'rop_factor': (float, 0.1),
    'cosanl_t_mult': (float, 1.0),
    'cosanl_restart_period': (int, 10),
    'enable_random_morph': (bool, False),
    'invert_color': (bool, False),
    'decrease_ohem_rate': (bool, False),
    'ohem_rate': (float, 0.7),
    'af_scale_g': (float, 8),
    'af_scale_v': (float, 3),
    'af_scale_c': (float, 3),
    'use_grapheme_code': (bool, False),
    'final_ft': (bool, False)
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