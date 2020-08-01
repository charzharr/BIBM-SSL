""" esetup.py (By: Charley Zhang, July 2020)
Does the heavy lifting for setting up the experiment.
  - Gathers the necessary resources, modules, and utilities.
  - Configures all essential training components 
    (model architecture + params, criterion, optimizer, schedulers)
  - Initializes stat trackers and experiment trackers
  - Defines dataset and data collection for batches
"""

import sys, os
import torch

import pathlib
from PIL import Image
import numpy as np
import torch

import lib
from lib.utils import schedulers, statistics
from lib.modules import losses, unets
from lib.data import transforms, isic_2017


NUM_WORKERS = 16
CURR_PATH = pathlib.Path(__file__).parent


def setup(cfg, checkpoint):
    """
    1. Verify configuration
    2. Load major training components
        - Data  - Model  - Optimizer  - Criterion  - Recording
    """

    # Print Settings
    print(f"[Experiment Settings (@esetup.py)]")
    print(f" > Prepping train config..")
    print(f"\t - experiment:  {cfg['experiment']['project']} - "
            f"{cfg['experiment']['name']}, id({cfg['experiment']['id']})")
    print(f"\t - batch_size {cfg['train']['batch_size']}, "
          f"\t - start epoch: {cfg['train']['start_epoch']}/"
            f"{cfg['train']['epochs']},")
    print(f"\t - Optimizer ({cfg['train']['optimizer']['name']}): "
          f"\t - lr {cfg['train']['optimizer']['lr']}, "
          f"\t - wt_decay {cfg['train']['optimizer']['wt_decay']}, "
          f"\t - mom {cfg['train']['optimizer']['momentum']}, ")
    print(f"\t - Scheduler ({cfg['train']['scheduler']['name']}): "
          f"\t - factor {cfg['train']['scheduler']['factor']}, ")

    # Flags
    use_wandb = cfg if not cfg['experiment']['debug']['mode'] and \
                cfg['experiment']['debug']['wandb'] else None

    # Load Model Components
    data = get_data(cfg)
    device = torch.device(cfg['experiment']['device'])
    criterion = get_criterion(cfg)
    model = get_model(cfg)

    if checkpoint:
        resume_dict = torch.load(checkpoint)
        cfg['train']['start_epoch'] = resume_dict['epoch']
        
        state_dict = resume_dict['state_dict']
        print(' > ' + str(model.load_state_dict(state_dict, strict=False)))
        
        optimizer = resume_dict['optimizer']
        
        if 'scheduler' in resume_dict:
            scheduler = resume_dict['scheduler']
        else:
            scheduler = get_scheduler(cfg, optimizer)

        if 'tracker' in resume_dict:
            tracker = resume_dict['tracker']
            tracker.use_wandb = use_wandb
            # TODO: resuming log
        else:
            tracker = utils.statistics.ExperimentTracker(wandb=use_wandb)
            
    else:
        optimizer = get_optimizer(cfg, model.parameters())
        scheduler = get_scheduler(cfg, optimizer)
        tracker = statistics.ExperimentTracker(wandb=use_wandb)

    return {
        'device': device,
        'data': data,
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'scheduler': scheduler,
        'tracker': tracker
    }


### ======================================================================== ###
### * ### * ### * ### *       Training Components        * ### * ### * ### * ###
### ======================================================================== ###


def get_model(cfg):
    model = lib.modules.get_model(cfg)
    return model


def get_scheduler(cfg, optimizer):
    sched = cfg['train']['scheduler']['name']
    t = cfg['train']['start_epoch']
    T = cfg['train']['epochs']
    factor = cfg['train']['scheduler']['factor']
    rampup_rates = cfg['train']['scheduler']['rampup_rates']
    
    if 'plateau' in sched:
        scheduler = schedulers.ReduceOnPlateau(
            optimizer,
            factor=factor,
            patience=cfg['train']['scheduler']['plateau']['patience'],
            lowerbetter=True,
            rampup_rates=rampup_rates
        )
    elif 'step' in sched:
        scheduler = schedulers.StepDecay(
            optimizer,
            factor=factor,
            T=T,
            steps=cfg['train']['scheduler']['step']['steps'],
            rampup_rates=rampup_rates
        )
    elif 'cos' in sched:
        scheduler = schedulers.CosineDecay(
            optimizer,
            T=T,
            t=t,
            rampup_rates=rampup_rates
        )
    else:
        scheduler = schedulers.Uniform(optimizer, rampup_rates=rampup_rates)
    
    return scheduler


def get_optimizer(cfg, params):
    opt = cfg['train']['optimizer']['name']
    lr = cfg['train']['optimizer']['lr']
    mom = cfg['train']['optimizer']['momentum']
    wdecay = cfg['train']['optimizer']['wt_decay']

    if 'adam' in opt:
        optimizer = torch.optim.Adam(
            params, 
            lr=lr, 
            weight_decay=wdecay,
            betas=cfg['train']['optimizer']['adam']['betas']
        )
    elif 'nesterov' in opt:
        optimizer = torch.optim.SGD(params, 
            lr=lr, 
            momentum=mom, 
            weight_decay=wdecay,
            nesterov=True
        )
    else:  # sgd
        optimizer = torch.optim.SGD(
            params, 
            lr=lr, 
            momentum=mom, 
            weight_decay=wdecay
        )

    return optimizer


def get_criterion(cfg):
    
    crit_cfg = cfg['criterion']
    crit_name = crit_cfg['name']
    kwargs = crit_cfg[crit_name] if crit_name in crit_cfg else {}
    if 'weights' in kwargs:
        kwargs['weights'] = torch.tensor(kwarts['weights'])

    if crit_name == 'softdice' or crit_name == 'dice':
        criterion = losses.SoftDiceLoss(**kwargs)
    elif crit_name == 'softjaccard' or crit_name == 'softiou' or crit_name == 'iou':
        criterion = losses.SoftJaccardLoss(**kwargs)
    elif crit_name == 'ce' or 'cross_entropy' in crit_name:
        criterion = torch.nn.CrossEntropyLoss(**kwargs)
    elif crit_name == 'mse' or 'mean_square' in crit_name:
        criterion = torch.nn.MSELoss(**kwargs)
    else:
        raise ValueError(f"Criterion {critname} is not supported.")

    return criterion


### ======================================================================== ###
### * ### * ### * ### *           Data Handling          * ### * ### * ### * ###
### ======================================================================== ###


def get_data(cfg, num_workers=NUM_WORKERS):
    ret = {}
    ret['df'] = df = isic_2017.get_df(os.path.join(
        CURR_PATH.parent.parent.parent.absolute(), 'datasets', 'ISIC_2017'
    ))
    
    ret['train_dataset'] = ISIC17(
        df[df['subsetname'] == 'train'],
        transforms.GeneralTransform(cfg['train']['transforms'])
    )
    ret['train_loader'] = torch.utils.data.DataLoader(
        ret['train_dataset'],
        batch_size=cfg['train']['batch_size'],
        shuffle=cfg['train']['shuffle'],
        num_workers=num_workers, pin_memory=False  # non_block not useful here
    )

    ret['test_dataset'] = ISIC17(
        df[df['subsetname'] == 'test'],
        transforms.GeneralTransform(cfg['test']['transforms'])
    )
    ret['test_loader'] = torch.utils.data.DataLoader(
        ret['test_dataset'],
        batch_size=cfg['test']['batch_size'],
        shuffle=False,
        num_workers=num_workers, pin_memory=False  # non_block not useful here
    )

    return ret



class ISIC17(torch.utils.data.Dataset):
    
    def __init__(self, df, transforms):
        self.df = df
        self.T = transforms
        
        self.images, self.masks = [], []
        for _, s in df.iterrows():
            self.images.append(s['image'])
            self.masks.append(s['mask'])
        for im, mask in zip(self.images, self.masks):  # sanity check
            assert im.split('/')[-1][:-4] + '_segmentation' == \
                mask.split('/')[-1][:-4], f"ERROR: {im}, {mask} mismatch."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        impath, maskpath = self.images[idx], self.masks[idx]
        X, tokx = self.T.transform(Image.open(impath).convert('RGB'), 
            shake=True, token=True
        )
        
        mask = np.array(Image.open(maskpath).convert('L')).astype(np.uint8)
        Y, toky = self.T.transform(Image.fromarray(mask, mode='L'), 
            label=True, token=True
        )
        
        # import matplotlib.pyplot as plt
        # fig = plt.figure(figsize=(8,8))
        # ax = fig.add_subplot(2, 2, 1)
        # ax.imshow(Image.open(impath))
        # ax = fig.add_subplot(2, 2, 2)
        # ax.imshow(self.T.reverse(X, tokx))
        # ax = fig.add_subplot(2, 2, 3)
        # ax.imshow(Image.open(maskpath))
        # ax = fig.add_subplot(2, 2, 4)
        # ax.imshow(self.T.reverse(Y, toky))
        # plt.show()
        # import IPython; IPython.embed(); exit(1)

        return X, Y
