
import sys
import torch


def big5_metrics(pred, targ, eps=10e-6):
    r""" 
    Returns JA (Jaccard), DI (Dice), AC (accuracy), SE (Sensitivity), 
    and SP (Specificity) as a dictionary of floats.
    Parameters
        pred (Bx1xHxW binary tensor)  
        targ (Bx1xHxW binary tensor) 
    """
    B = pred.shape[0]
    preds = pred.view(-1)
    targs = targ.view(-1)

    TP = torch.sum(preds.eq(1) & targs.eq(1)).item()
    FP = torch.sum(preds.eq(1) & targs.eq(0)).item()
    FN = torch.sum(preds.eq(0) & targs.eq(1)).item()
    TN = torch.sum(preds.eq(0) & targs.eq(0)).item()

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'JA': TP / (TP + FN + FP + eps),
        'DI': 2 * TP / (2 * TP + FN + FP + eps),
        'AC': (TP + TN) / (TP + FP + TN + FN + eps),
        'SE': TP / (TP + FN + eps),
        'SP': TN / (TN + FP + eps)
    }


### ======================================================================== ###
### * ### * ### * ### *      More Generic 2D Metrics     * ### * ### * ### * ###
### ======================================================================== ###


def dice(pred, targ, per_channel=False, eps=10e-6):
    """ Computes segmentation dice coefficient for a batch.
        Dice = 2* |P ∩ T| / (|P| |T|) <- averaged over all C if not per_channel
    Parameters
        pred (4D tensor) - BxCxHxW model output
        targ (4D tensor) - BxCxHxW target label
    """
    assert pred.shape == targ.shape, f"Pred: {pred.shape}, Targ: {targ.shape}."

    pred = transpose_and_flatten(pred).float()
    targ = transpose_and_flatten(targ).float()

    intersect = (pred * targ).sum(-1)  # Cx1
    pred_card = pred.sum(-1)  # Cx1 |P| 
    targ_card = targ.sum(-1)  # Cx1 |T|
    dice_per_channel = 2 * intersect / (pred_card + targ_card + eps)
    
    if per_channel:
        return dice_per_channel
    return torch.mean(dice_per_channel)


def IoU(pred, targ, per_channel=False, eps=10e-6):
    assert pred.shape == targ.shape, f"Pred: {pred.shape}, Targ: {targ.shape}."
    
    intersect = (pred & targ).float()
    union = (pred | targ).float()

    if per_channel:
        return intersect.sum(-1) / (union.sum(-1) + eps)
    return intersect.sum() / (union.sum() + eps)


### ======================================================================== ###
### * ### * ### * ### *             Helpers              * ### * ### * ### * ###
### ======================================================================== ###


def transpose_and_flatten(tens):
    """
    NxCxHxW -> CxN*H*W for easy metric calculation per channel.
    """
    assert tens.ndim == 4, f"Tensor: {tens.shape}."
    
    C = tens.shape[1]
    trans = tens.permute(1, 0, 2, 3).contiguous()
    flat = trans.view(C, -1)
    return flat
    


