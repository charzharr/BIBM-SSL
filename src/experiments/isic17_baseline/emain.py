"""
Customized for this experiment:
  Characteristics:
    - Single frame examples
    - Batches with certain ratio of inputs with labels
  Custom Functions
    - create_batches(examples, batch_size, has_label_ratio)
"""

import sys, os
import time
import pathlib
import collections
import pprint
import random, math
import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch, torchvision
from torchvision import transforms
from torchsummary import summary

import lib
from lib import metrics
from lib.utils import statistics, images, devices, timers

from . import esetup


WATCH = timers.StopWatch()
SAVE_BEST_METRICS = {  # metric (str): max_better (bool)
    'test_ep_dice': True,
}

def run(cfg, checkpoint=None):

    def it_metrics(pred, targ, loss, meter, tracker, test=False):
        mets = metrics.big5_metrics(pred, targ)
        itmets = {'loss': float(loss)}
        itmets['dice'] = mets['DI']
        itmets['jaccard'] = mets['JA']
        itmets['accuracy'] = mets['AC']
        itmets['sensitivity'] = mets['SE']
        itmets['specificity'] = mets['SP']
        itmets['TP'] = mets['TP']
        itmets['FP'] = mets['FP']
        itmets['FN'] = mets['FN']
        itmets['TN'] = mets['TN']
        meter.update(itmets)
        
        pre = 'test_' if test else 'train_' 
        tracker.update({
            pre + 'it_loss': itmets['loss'],
            pre + 'it_dice': itmets['dice'],
            pre + 'it_jaccard': itmets['jaccard']
        })
        return itmets
    
    # Training Components
    debug = cfg['experiment']['debug']
    comps = esetup.setup(cfg, checkpoint)

    data = comps['data']
    trainloader = data['train_loader']
    testloader = data['test_loader']

    device = comps['device']
    model = comps['model'].to(device)
    criterion = comps['criterion'].to(device)
    optimizer = comps['optimizer']
    scheduler = comps['scheduler']
    tracker = comps['tracker']

    if debug['overfitbatch']:
        batches = []
        for i, batch in enumerate(trainloader):
            if i in (3,4,5):
                batches.append(batch)
            elif i > 5: break
        trainloader = batches

    for epoch in range(cfg['train']['start_epoch'], cfg['train']['epochs']+1):
        print("\n======================")
        print(f"Starting Epoch {epoch} (lr: {scheduler.lr})")
        print("======================")
        WATCH.tic(name='epoch')

        model.train()
        epmeter = statistics.EpochMeters()
        WATCH.tic(name='iter')
        for it, batch in enumerate(trainloader):
            
            X = batch[0].to(device)
            Y = batch[1].to(device)

            out = model(X)  # list of out_ds
    
            optimizer.zero_grad()
            loss = criterion(out, Y)
            loss.backward()
            optimizer.step()

            X, Y, out = X.detach(), Y.detach(), out.detach()

            itmets = it_metrics((out > 0).int(), Y, 
                loss.item(), epmeter, tracker)
            _print_itmets(cfg, it+1, len(trainloader), itmets,
                WATCH.toc(name='iter', disp=False))
            
            if debug['break_train_iter']: break
            WATCH.tic(name='iter')

        # Test + Epoch Metrics
        if epoch % debug['test_every_n_epochs'] == 0:
            print(f"\nTesting..\n")
            with torch.no_grad():
                model.eval()
                testmeter = statistics.EpochMeters()
                WATCH.tic(name='iter')
                for it, batch in enumerate(testloader):
                    X = batch[0].to(device)
                    Y = batch[1].to(device)

                    out = model(X)  # list of out_ds
                    loss = criterion(out, Y)

                    X, Y, out = X.detach(), Y.detach(), out.detach()

                    itmets = it_metrics((out > 0).int(), Y, 
                        loss.item(), testmeter, tracker)
                    _print_itmets(cfg, it+1, len(testloader), itmets,
                        WATCH.toc(name='iter', disp=False))
                    
                    if debug['break_test_iter'] or debug['overfitbatch']: 
                        break
                    WATCH.tic(name='iter')
            epmets = _epoch_mets( 
                cfg, tracker,
                epmeter.avg(no_avg=['TP', 'FP', 'FN', 'TN']), 
                testmeter.avg(no_avg=['TP', 'FP', 'FN', 'TN'])
            )
        else:
            epmets = _epoch_mets(
                cfg, tracker, 
                epmeter.avg(no_avg=['TP', 'FP', 'FN', 'TN'])
            )
        
        WATCH.toc(name='epoch')
        scheduler.step(epoch=epoch, value=0)

        if debug['save']:
            _save_model(epmets, model.state_dict(), criterion, optimizer, 
                tracker,cfg)



### ======================================================================== ###
### * ### * ### * ### *              Helpers             * ### * ### * ### * ###
### ======================================================================== ###


def _print_itmets(cfg, iter_num, iter_tot, it_mets, duration):
    dev = cfg['experiment']['device']
    mem = devices.get_gpu_memory_map()[int(dev[-1])]/1000 if 'cuda' in dev else -1
    print(
        f"\n    Iter {iter_num}/{iter_tot} ({duration:.1f} sec, {mem:.1f} GB) - "
        f"loss {float(it_mets['loss']):.3f}, "
        f"dice {float(it_mets['dice']):.3f}, "
        f"iou {float(it_mets['jaccard']):.3f}, "
    )

# dict_keys: loss, mAP, F1, F1s, TPs, FPs, FNs
def _epoch_mets(cfg, tracker, *dicts):
    classnames = cfg['data'][cfg['data']['name']]
    merged = {}
    for i, d in enumerate(dicts):
        pre = 'test_ep_' if i else 'train_ep_'
        class_sep = False if len(classnames) == 1 else True
        for k, v in d.items():
            if k not in ['TP', 'FP', 'FN', 'TN']: 
                merged[pre + k] = v
        merged[pre + 'TP'] = d['TP']
        merged[pre + 'FP'] = d['FP']
        merged[pre + 'FN'] = d['FN']
        merged[pre + 'TN'] = d['TN']
        merged[pre + 'cumjaccard'] = d['TP']/(d['TP']+d['FP']+d['FN']+10**-5)
        merged[pre + 'cumdice'] = 2*d['TP']/(2*d['TP']+d['FP']+d['FN']+10**-5)
                    
    tracker.update(merged, wandb=True)
    
    print("\nEpoch Stats\n-----------")
    for k, v in merged.items():
        if isinstance(v, float):
            print(f"  {k: <21} {v:.4f}")
        else:
            print(f"  {k: <21} {v:d}")
    return merged
    
    
def _save_model(epmets, state, crit, opt, tracker, cfg):
    print(f"Saving model ", end='')
    end = 'last'
    for met, max_gud in SAVE_BEST_METRICS.items():
        if met in epmets and \
        tracker.best(met, max_better=max_gud) == epmets[met]:
            end = f"best-{met.split('_')[-1]}"
            print(f"({end}: {epmets[met]:.2f}) ", end='')
            break
    filename = f"{cfg['experiment']['id']}_{cfg['experiment']['name']}_" + end
    print(f"-> {filename}")

    curr_path = pathlib.Path(__file__).parent.absolute()
    save_path = os.path.join(curr_path, filename + '.pth')
    torch.save({
        'state_dict': state,
        'criterion': crit,
        'optimizer': opt,
        'tracker': tracker,
        'config': cfg
        },
        save_path
    )



