
from .unets import unet, bigunet, nestedunet, r2attentionunet, denseunet

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
    elif 'bigunet' in name:
        bilinear = settings['bilinear']
        base_size = settings['base_size']
        model = bigunet.BigUNet(in_channels, out_channels, bilinear=bilinear,
                    base_size=base_size)
    elif name == 'nestedunet':
        model = nestedunet.NestedUNet(
            out_channels, 
            input_channels=in_channels,
            deep_supervision=cfg['model']['nestedunet']['deepsup']
        )
    elif name == 'r2attentionunet':
        model = r2attentionunet.R2Attention_UNet(
            in_ch=in_channels, out_ch=out_channels, t=2
        )
    elif name == 'denseunet':
        model = denseunet.get_model(
            str(settings['layers']), 
            num_classes=out_channels
        )
    else:
        raise ValueError(f"  > Model({name}) not found..")

    if name != 'denseunet':
        print(f"\t{name} initialized ({model.param_counts[0]} params).")
    return model