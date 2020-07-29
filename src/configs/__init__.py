
import os
import pathlib
import yaml


__all__ = ['get_config']

curr_path = pathlib.Path(__file__).parent.absolute()
DEFAULT_FILE = os.path.join(curr_path, 'default.yaml')


def get_config(filename):
    r""" Merges specified config (or none) with default cfg file. """
    
    print(f" > Loading config ({filename})..", end='')

    with open(DEFAULT_FILE, 'r') as f:
        default_cfg = yaml.safe_load(f)
    
    if not filename:
        experiment_cfg = {}
    else:
        if os.path.isfile(filename):
            cfg_file = filename
        else: 
            dir_path = pathlib.Path(__file__).parent.absolute()
            cfg_file = os.path.join(dir_path, filename)
            assert os.path.isfile(cfg_file), \
                f"{filename} not in configs ({dir_path})"
        with open(cfg_file, 'r') as f:
            experiment_cfg = yaml.safe_load(f)

    experiment_cfg = _merge(default_cfg, experiment_cfg)
    print(f" done.")
    return experiment_cfg


def _merge(default_d, experiment_d):
    merged_cfg = dict(default_d)
    for k, v in experiment_d.items():
        if isinstance(v, dict) and k in default_d:
            v = _merge(default_d[k], v)
        merged_cfg[k] = v
    return merged_cfg
