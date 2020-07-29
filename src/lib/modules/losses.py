
import sys
import torch, torch.nn as nn


class FocalLoss(nn.Module):
    """ L(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
          p_t: prediction prob for target class, t
    Paper: Focal Loss for Dense Object Detection
        https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, preds, targs, eps=10**-6):
        B = pred.shape[0]
        preds = self.softmax(preds)       
        preds = inputs.view(B, -1)
        targets = targets.view(B, -1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP)**self.gamma * BCE
                       
        return focal_loss


class SoftDiceLoss(nn.Module):

    def __init__(self, square_denom=False):
        super(SoftDiceLoss, self).__init__()
        self.square_denom = square_denom
        self.softmax = nn.Softmax(dim=1)
        print('\tSoftDice initialized.')

    def forward(self, pred, targ, eps=10**-6):
        """
        Parameters
            pred: BxCxHxW logits as a float tensor on the same device
            targ: BxCxHxW binary labels as a float tensor
        """
        assert pred.shape == targ.shape, f"Pred: {pred.shape}, Targ: {targ.shape}"
        assert pred.dtype == torch.float32
        
        B, C = pred.shape[0], pred.shape[1]
        pred_probs = pred.sigmoid() if C == 1 else self.softmax(pred)
        preds = pred_probs.view(B, -1)
        targs = targ.view(B, -1)
        
        intersect = preds * targs
        if self.square_denom:
            dice = (2. * intersect.sum(1) + eps) / (preds.square().sum(1) + \
                    targs.square().sum(1) + eps)
        else:
            dice = (2. * intersect.sum(1) + eps) / (preds.sum(1) + \
                    targs.sum(1) + eps)
        
        return 1 - dice.mean()


class SoftJaccardLoss(nn.Module):

    def __init__(self, log_loss=False):
        """
        log_loss = -ln(IOU) as presented in this paper:
            https://arxiv.org/pdf/1608.01471.pdf
        """
        super(SoftJaccardLoss, self).__init__()
        self.log_loss = log_loss
        self.softmax = nn.Softmax(dim=1)
        print('\tSoftJaccard initialized.')

    def forward(self, pred, targ, eps=10**-6):
        """
        Parameters
            pred: BxCxHxW logits as a float tensor on the same device
            targ: BxCxHxW binary labels as a float tensor
        """
        assert pred.shape == targ.shape, f"Pred: {pred.shape}, Targ: {targ.shape}"
        assert pred.dtype == torch.float32
        
        B, C = pred.shape[0], pred.shape[1]
        pred_probs = pred.sigmoid() if C == 1 else self.softmax(pred)
        preds = pred_probs.view(B, -1)
        targs = targ.view(B, -1)
        
        intersect = torch.sum(preds * targs, 1)
        union = torch.sum(preds + targs, 1) - intersect  # B x C tensor
        iou = torch.mean((intersect + eps) / (union + eps))
        return 1 - iou if not self.log_loss else -torch.log(iou)


class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pred, targ, eps=1):
        
        B, C = pred.shape[0], pred.shape[1]
        pred_probs = pred.sigmoid() if C == 1 else self.softmax(pred)       
        preds = inputs.view(B, -1)
        targets = targets.view(B, -1)
        
        TP = (preds * targets).sum(1)    
        FP = ((1 - targets) * preds).sum(1)
        FN = (targets * (1 - preds)).sum(1)
       
        tversky = (TP + eps) / (TP + self.alpha * FP + self.beta * FN + eps)  
        return 1 - torch.mean(tversky)
