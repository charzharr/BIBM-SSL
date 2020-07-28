
import torchsummary
import torch

from .unet import UNet
from .nestedunet import NestedUNet
# from .bigunet import BigUNet


def get_model(cfg, pretrained=False):

    model_name = cfg['model']['name']
    ds_name = cfg['data']['name']
    bilinear = cfg['model'][model_name]['bilinear']

    out_channels = len(cfg['data'][ds_name]['classnames'])
    in_channels = 0
    for entry in cfg['train']['transforms']:
        if entry[0] == 'normmeanstd':
            in_channels = len(entry[1][0])
    assert in_channels > 0, f"Could not get input dim from normmeanstd transform"
    
    if model_name == 'unet_2d':
        model = UNet(in_channels, out_channels, bilinear=bilinear)
    elif model_name == 'bigunet_2d':
        model = BigUNet(in_channels, out_channels, bilinear=bilinear)
    elif model_name == 'nestedunet_2d':
        print(f"Confirmed")
        model = NestedUNet(
            out_channels, 
            input_channels=in_channels,
            deep_supervision=cfg['model']['nestedunet_2d']['deepsup']
        )
    else:
        raise ValueError(f"invalid model name {model_name}.")
    
    # torch.cuda.set_device('cuda:1')
    # torchsummary.summary(model.cuda(), (1,384,384), batch_size=-1, device='cuda')
    # sys.exit(0)
    model.apply(init_weights)
    return model

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_uniform_(
            m.weight, mode='fan_in', nonlinearity='relu'
        )
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    
                