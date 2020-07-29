
from .unets import unet, bigunet, nestedunet


__all__ = ['get_model']

MODELS = {
    'unet': unet,
    'bigunet': bigunet,
    'nestedunet': nestedunet
}


def get_model(cfg):
    name = cfg['model']['name']
    settings = cfg['model'][name]

    ds_name = cfg['data']['name']
    out_channels = len(cfg['data'][ds_name]['classnames'])
    in_channels = 0
    for entry in cfg['train']['transforms']:
        if entry[0] == 'normmeanstd':
            in_channels = len(entry[1][0])
    assert in_channels > 0, f"Could not get input dim from normmeanstd transform"

    print(f"  > Fetching {name} model..", end='')
    if name == 'unet':
        bilinear = settings['bilinear']
        model = unet.UNet(in_channels, out_channels, bilinear=bilinear)
    elif name == 'bigunet':
        bilinear = settings['bilinear']
        model = bigunet.BigUNet(in_channels, out_channels, bilinear=bilinear)
    elif name == 'nestedunet':
        model = nestedunet.NestedUNet(
            out_channels, 
            input_channels=in_channels,
            deep_supervision=cfg['model']['nestedunet_2d']['deepsup']
        )
    else:
        raise ValueError(f"  > Model({name}) not found..")

    return model