import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module) :
    def __init__(self, gamma=0, alpha=None, size_average=True) :
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)) :
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list) :
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target) :
        if input.dim() > 2 :
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target[:, 1 :].contiguous()
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target.to(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None :
            if self.alpha.type() != input.data.type() :
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1).to(torch.int64))
            logpt = logpt * Variable(at)

        loss = -(1 - pt) ** self.gamma * logpt
        if self.size_average :
            return loss.mean()
        else :
            return loss.sum()


def dice_loss(prediction, target) :
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    prediction = torch.softmax(prediction, dim=1)[:, 1:].contiguous()
    target = target[:, 1:].contiguous()

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, ce_weight=0.5) :
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        ce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """

    focal_loss = FocalLoss(gamma=2, alpha=torch.FloatTensor([1., 1.]))
    ce = focal_loss(prediction, target)

    dice = dice_loss(prediction, target)

    loss = ce * ce_weight + dice * (1 - ce_weight)

    return loss


def dice_score(prediction, target) :
    prediction = torch.sigmoid(prediction)
    smooth = 1.0
    i_flat = prediction.view(-1)
    t_flat = target.view(-1)
    intersection = (i_flat * t_flat).sum()
    return (2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth)


def prediction_map_distillation(y, teacher_scores, T=4) :
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param y: student score map
    :param teacher_scores: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    p = p.view(-1, 2)
    q = q.view(-1, 2)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return l_kl


def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))


def importance_maps_distillation(s, t, exp=4):
    """
    importance_maps_distillation KD loss, based on "Paying More Attention to Attention:
    Improving the Performance of Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    :param exp: exponent
    :param s: student feature maps
    :param t: teacher feature maps
    :return: imd loss value
    """
    if s.shape[2] != t.shape[2]:
        s = F.interpolate(s, t.size()[-2:], mode='bilinear')
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()


def region_contrast(x, gt):
    """
    calculate region contrast value
    :param x: feature
    :param gt: mask
    :return: value
    """
    smooth = 1.0
    mask0 = gt[:, 0].unsqueeze(1)
    mask1 = gt[:, 1].unsqueeze(1)

    region0 = torch.sum(x * mask0, dim=(2, 3)) / torch.sum(mask0, dim=(2, 3))
    region1 = torch.sum(x * mask1, dim=(2, 3)) / (torch.sum(mask1, dim=(2, 3)) + smooth)
    return F.cosine_similarity(region0, region1, dim=1)


def region_affinity_distillation(s, t, gt):
    """
    region affinity distillation KD loss
    :param s: student feature
    :param t: teacher feature
    :return: loss value
    """
    gt = F.interpolate(gt, s.size()[2:])
    return (region_contrast(s, gt) - region_contrast(t, gt)).pow(2).mean()
