"""Calculating the loss
You can build the loss function of BsiNet by combining multiple losses
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = torch.sigmoid(prediction)
    dice = dice_loss(prediction, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss



class log_cosh_dice_loss(nn.Module):
    def __init__(self, num_classes=1, smooth=1, alpha=0.7):
        super(log_cosh_dice_loss, self).__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, outputs, targets):
        x = self.dice_loss(outputs, targets)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)

    def dice_loss(self, y_pred, y_true):
        """[function to compute dice loss]
        Args:
            y_true ([float32]): [ground truth image]
            y_pred ([float32]): [predicted image]
        Returns:
            [float32]: [loss value]
        """
        smooth = 1.
        y_true = torch.flatten(y_true)
        y_pred = torch.flatten(y_pred)
        intersection = torch.sum((y_true * y_pred))
        coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
        return (1. - coeff)


def focal_loss(predict, label, alpha=0.6, beta=2):
    probs = torch.sigmoid(predict)
    # 交叉熵Loss
    ce_loss = nn.BCELoss()
    ce_loss = ce_loss(probs,label)
    alpha_ = torch.ones_like(predict) * alpha
    # 正label 为alpha, 负label为1-alpha
    alpha_ = torch.where(label > 0, alpha_, 1.0 - alpha_)
    probs_ = torch.where(label > 0, probs, 1.0 - probs)
    # loss weight matrix
    loss_matrix = alpha_ * torch.pow((1.0 - probs_), beta)
    # 最终loss 矩阵，为对应的权重与loss值相乘，控制预测越不准的产生更大的loss
    loss = loss_matrix * ce_loss
    loss = torch.sum(loss)
    return loss



class Loss:
    def __init__(self, dice_weight=0.0, class_weights=None, num_classes=1, device=None):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None
        self.nll_loss = nn.NLLLoss2d(weight=nll_weight)
        self.dice_weight = dice_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)
        if self.dice_weight:
            eps = 1e-7
            cls_weight = self.dice_weight / self.num_classes
            for cls in range(self.num_classes):
                dice_target = (targets == cls).float()
                dice_output = outputs[:, cls].exp()
                intersection = (dice_output * dice_target).sum()
                # union without intersection
                uwi = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - intersection / uwi) * cls_weight
            loss /= (1 + self.dice_weight)
        return loss



class LossMulti:
    def __init__(
            self, jaccard_weight=0.0, class_weights=None, num_classes=1, device=None
    ):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None

        self.nll_loss = nn.NLLLoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):

        targets = targets.squeeze(1)

        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-7  # 原先是1e-7
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (
                        torch.log((intersection + eps) / (union - intersection + eps))
                        * self.jaccard_weight
                )
        return loss



class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class weighted_bce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bd_pre, target):
        n, c, h, w = bd_pre.size()
        log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
        target_t = target.view(1, -1)

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)

        weight = torch.zeros_like(log_p)
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

        return loss
class LossF(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_seg = BCEDiceLoss()
        self.loss_edge = LossMulti(num_classes=2)
        self.loss_dist = nn.MSELoss()

    def forward(self, outputs, targets):
        if len(outputs) not in (3, 7):
            raise ValueError(f"LossF expects 3 outputs without deep supervision or 7 outputs with deep supervision, got {len(outputs)}.")
        targets1, targets2, targets3 = targets
        outputs1, outputs2, outputs3 = outputs[:3]

        targets1 = targets1.float()
        targets2 = targets2.long()
        targets3 = targets3.float()

        loss_sem = self.loss_seg(outputs1, targets1)
        if len(outputs) == 7:
            outputs4, outputs5, outputs6, outputs7 = outputs[3:]
            loss_sem = (
                loss_sem
              + 0.1*self.loss_seg(outputs4, targets1)
              + 0.2*self.loss_seg(outputs5, targets1)
              + 0.4*self.loss_seg(outputs6, targets1)
              + 0.6*self.loss_seg(outputs7, targets1)
            )

        loss_edge = self.loss_edge(outputs2, targets2)
        loss_dist = self.loss_dist(outputs3, targets3)

        return {
            "sem": loss_sem,
            "edge": loss_edge,
            "dist": loss_dist
        }


class LossFHBG(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_seg = BCEDiceLoss()
        self.loss_edge = LossMulti(num_classes=2)
        self.loss_dist = nn.MSELoss()

    def forward(self,outputs, targets):
        targets1, targets2, targets3 = targets
        outputs1, outputs2, outputs3 = outputs

        targets1 = targets1.float()
        targets2 = targets2.long()
        targets3 = targets3.float()

        loss_sem = self.loss_seg(outputs1, targets1)
        loss_edge = self.loss_edge(outputs2, targets2)
        loss_dist = self.loss_dist(outputs3, targets3)

        return {
            "sem": loss_sem,
            "edge": loss_edge,
            "dist": loss_dist
        }

class LossFBSI(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_seg = LossMulti(num_classes=2)   #mask_loss
        self.loss_edge = LossMulti(num_classes=2)   #contour_loss
        self.loss_dist = nn.MSELoss()               ##distance_loss
        

    def __call__(self,outputs, targets):
        targets1, targets2, targets3 = targets
        outputs1, outputs2, outputs3 = outputs

        targets1 = targets1.float()
        targets2 = targets2.long()
        targets3 = targets3.float()
        #
        loss_sem = self.loss_seg(outputs1, targets1)
        loss_edge = self.loss_edge(outputs2, targets2)
        loss_dist = self.loss_dist(outputs3, targets3)

        return {
            "sem": loss_sem,
            "edge": loss_edge,
            "dist": loss_dist
        }
class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2):
        super().__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
class AWLLossWrapper(nn.Module):
    def __init__(self, task_names):
        super().__init__()
        self.task_names = task_names
        self.awl = AutomaticWeightedLoss(len(task_names))

    def forward(self, task_losses):
        losses = []
        log_dict = {}

        for name in self.task_names:
            if name not in task_losses or task_losses[name] is None:
                raise ValueError(f"task_losses 缺少任务 {name}，当前只有: {list(task_losses.keys())}")
            losses.append(task_losses[name])
            log_dict[name] = task_losses[name]

        total_loss = self.awl(*losses)
        log_dict["total"] = total_loss

        with torch.no_grad():
            for i, name in enumerate(self.task_names):
                log_dict[f"{name}_weight"] = 0.5 / (self.awl.params[i] ** 2)
                log_dict[f"{name}_param"] = self.awl.params[i]

        return total_loss, log_dict
class LossFBFI(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_seg = BCEDiceLoss()   #mask_loss
        self.loss_edge= weighted_bce()   #contour_loss
    
    def __call__(self, outputs, targets):
        targets1, targets2 = targets
        outputs1, outputs2 = outputs

        targets1 = targets1.float()
        targets2 = targets2.float()

        #
        loss_sem = self.loss_seg(outputs1, targets1)
        loss_edge = self.loss_edge(outputs2, targets2)
        
        return {
            "sem": loss_sem,
            "edge": loss_edge
        }

# loss for Edge Detection
class WBCE(nn.Module):
    def __init__(self, balance=1.1):
        super().__init__()
        self.balance = balance

    def forward(self, output, target):
        device = output.device
        n, c, h, w = output.size()
        weights = torch.zeros(size=(n, c, h, w), device=device)
        for i in range(n):
            t = target[i, :, :, :]
            pos = (t == 1).sum()
            neg = (t == 0).sum()
            valid = neg + pos
            weights[i, t == 1] = neg / valid
            weights[i, t == 0] = pos * self.balance / valid
        loss_bce = nn.BCELoss(weights, reduction='sum')(output, target)
        return loss_bce / n



class LossFREAU(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_seg = BCEDiceLoss()
    
    def __call__(self, outputs, targets):
        out,out7, out6, out5, out4, out3, out2, out1= outputs

        targets = targets.float()

        loss_sem = (
            0.8*self.loss_seg(out1, targets)
          + 0.7*self.loss_seg(out2, targets)
          + 0.6*self.loss_seg(out3, targets)

          + 0.5*self.loss_seg(out4, targets)
          + 0.4*self.loss_seg(out5, targets)
          + 0.3*self.loss_seg(out6, targets)
          + 0.2*self.loss_seg(out7, targets)
          + 1.0*self.loss_seg(out, targets)
        )
        
        
        return {
            "sem": loss_sem
        }


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """
    Uncertainty-based weighting for multi-task learning
    Supports arbitrary task losses
    """

    def __init__(self, task_names):
        super().__init__()

        self.task_names = task_names

        # log(σ^2) is more numerically stable
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in task_names
        })

    def forward(self, losses: dict):
        """
        losses: dict {task_name: scalar loss}
        """
        total_loss = 0.0
        loss_dict = {}

        for name, loss in losses.items():
            log_var = self.log_vars[name]

            precision = torch.exp(-log_var)
            weighted_loss = precision * loss + log_var

            total_loss += weighted_loss
            loss_dict[name] = loss.item()
            loss_dict[f"{name}_weight"] = precision.item()

        loss_dict["total"] = total_loss

        return total_loss, loss_dict

import torch
import torch.nn as nn

class AWLLossWrapper(nn.Module):
    def __init__(self, task_names):
        super().__init__()
        self.task_names = task_names
        self.awl = AutomaticWeightedLoss(len(task_names))

    def forward(self, task_losses):
        losses = []
        log_dict = {}

        for name in self.task_names:
            if name not in task_losses or task_losses[name] is None:
                raise ValueError(f"task_losses 缺少任务 {name}，当前只有: {list(task_losses.keys())}")
            losses.append(task_losses[name])
            log_dict[name] = task_losses[name]

        total_loss = self.awl(*losses)
        log_dict["total"] = total_loss

        # 记录可视化用的“有效权重”
        with torch.no_grad():
            for i, name in enumerate(self.task_names):
                log_dict[f"{name}_weight"] = 0.5 / (self.awl.params[i] ** 2)
                log_dict[f"{name}_param"] = self.awl.params[i]

        return total_loss, log_dict

class LossF3(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_seg = BCEDiceLoss()
        self.loss_edge = LossMulti(num_classes=2)
        self.loss_dist = nn.MSELoss()

    def forward(self, outputs, targets):
        targets1, targets2, targets3 = targets
        outputs1, outputs2, outputs3 = outputs

        targets1 = targets1.float()
        targets2 = targets2.long()
        targets3 = targets3.float()

        loss_sem = self.loss_seg(outputs1, targets1)
        loss_edge = self.loss_edge(outputs2, targets2)
        loss_dist = self.loss_dist(outputs3, targets3)

        return {
            "sem": loss_sem,
            "edge": loss_edge,
            "dist": loss_dist
        }
class LossF2(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss_seg = BCEDiceLoss()
        self.loss_edge = LossMulti(num_classes=2)
        

    def forward(self, outputs, targets):
        targets1, targets2 = targets
        outputs1, outputs2= outputs

        targets1 = targets1.float()
        targets2 = targets2.long()
        

        loss_sem = self.loss_seg(outputs1, targets1)
        loss_edge = self.loss_edge(outputs2, targets2)
        
        return {
            "sem": loss_sem,
            "edge": loss_edge
        }
