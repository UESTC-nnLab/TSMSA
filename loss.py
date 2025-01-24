import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()
    def forward(self, preds, gt_masks,data=None):
        if isinstance(preds, list) or isinstance(preds, tuple):
            loss_total = 0
            for i in range(len(preds)):
                pred = preds[i]
                smooth = 1
                intersection = pred * gt_masks
                loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
                loss = 1 - loss.mean()
                loss_total = loss_total + loss
            return loss_total / len(preds)
        else:
            pred = preds
            smooth = 1
            intersection = pred * gt_masks
            loss = (intersection.sum() + smooth) / (pred.sum() + gt_masks.sum() -intersection.sum() + smooth)
            loss = 1 - loss.mean()
            return loss

class ISNetLoss(nn.Module):
    def __init__(self):
        super(ISNetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = Get_gradient_nopadding()
        
    def forward(self, preds, gt_masks,data=None):
        edge_gt = self.grad(gt_masks.clone())
        
        ### img loss
        loss_img = self.softiou(preds[0], gt_masks)
        
        ### edge loss
        loss_edge = 10 * self.bce(preds[1], edge_gt)+ self.softiou(preds[1].sigmoid(), edge_gt)
        
        return loss_img + loss_edge

class RPCANetLoss(nn.Module):
    def __init__(self):
        super(RPCANetLoss, self).__init__()
        self.softiou = SoftIoULoss()
        self.mse = nn.MSELoss()
        # self.device = torch.device("cuda:{}".format("1") if torch.cuda.is_available() else "cpu")
    def forward(self,preds,gt_masks,data=None):
        if isinstance(preds, list) or isinstance(preds, tuple):
            D = preds[0]
            T = preds[1]
            loss_softiou = self.softiou(T, gt_masks)
            loss_mse = self.mse(D, data)
            gamma = torch.Tensor([0.1]).to(device='cuda')
            loss_all = loss_softiou + torch.mul(gamma, loss_mse)
            return loss_all
        else:
            loss_softiou = self.softiou(T, gt_masks)
            return loss_softiou


class FocalIoULoss(nn.Module):
    def __init__(self):
        super(FocalIoULoss, self).__init__()
    def forward(self, inputs, targets,data=None):
        "Non weighted version of Focal Loss"
        # def __init__(self, alpha=.25, gamma=2):
        #     super(WeightedFocalLoss, self).__init__()
        # targets =
        # inputs = torch.relu(inputs)
        [b,c,h,w] = inputs.size()

        # inputs = torch.nn.Sigmoid()(inputs)
        inputs = 0.999*(inputs-0.5)+0.5  ## 张量缩放和平移到[0.001, 0.999]的范围内
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets) #reduction='none'参数确保损失按元素计算，得到与输入形状相同的张量。
        intersection = torch.mul(inputs, targets) #计算inputs和targets张量的逐元素乘积，得到形状相同的张量。
        smooth = 1 #通过使用平滑因子smooth对交集和并集的计算进行平滑处理。平滑因子的作用是避免分母为零的情况，并增加计算的稳定性。

        IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)

        alpha = 0.75
        gamma = 2
        num_classes = 2
        # alpha_f = torch.tensor([alpha, 1 - alpha]).cuda()
        # alpha_f = torch.tensor([alpha, 1 - alpha])
        gamma = gamma
        size_average = True


        pt = torch.exp(-BCE_loss)

        F_loss =  torch.mul(((1-pt) ** gamma) ,BCE_loss) ##FocalLoss

        at = targets*alpha+(1-targets)*(1-alpha)

        F_loss = (1-IoU)*(F_loss)**(IoU*0.5+0.5)

        F_loss_map = at * F_loss

        F_loss_sum = F_loss_map.sum()

        # return F_loss_map,F_loss_sum
        return F_loss_sum

