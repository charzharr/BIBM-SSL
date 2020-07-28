
from .isic17_baseline import emain as isic17_baseline

EXPERIMENTS = {
    'isic17_base': isic17_baseline,
}

def get_module(exp_name):
    if exp_name in EXPERIMENTS:
        return EXPERIMENTS[exp_name]
    else:
        for exp in EXPERIMENTS:
            if exp in exp_name:
                return EXPERIMENTS[exp]
