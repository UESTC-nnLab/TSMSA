import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .context import CPM, AGCB_Element, AGCB_Patch
from .fusion import *
from .ShuffleNetv2 import *

__all__ = ['agpcnet']


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.5):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(drop),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)

import pdb

class AGPCNet(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',
                 drop=0.1):
        super(AGPCNet, self).__init__()
        assert backbone in ['resnet18', 'resnet34','suffleNetv2']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=False)
        elif backbone=='suffleNetv2':
            self.backbone = ShuffleNetV2(3,1,1.0)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(464, 232, 232)  #(512,256,256)
        self.fuse12 = AsymFusionModule(232, 116, 116)   #(256, 128, 128) 

        self.head = _FCNHead(116, 1, drop=drop) #128

        self.context = CPM(planes=464, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,att_mode=gca_att)#512
        self.conv = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=1,stride=1) #1   5

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape
        x = self.conv(x)
        # print(x.shape)
        c1, c2, c3 = self.backbone(x) ## torch.Size([8, 256, 128, 128]) torch.Size([8, 256, 128, 128])   [8,512,64,64]
        #pdb.set_trace()
        out = self.context(c3) # [8,512,64,64]

        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)## [8,512,128,128]
        out = self.fuse23(out, c2) 

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return torch.sigmoid(out)


class AGPCNet_Pro(nn.Module):
    def __init__(self, backbone='resnet18', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',
                 drop=0.1):
        super(AGPCNet_Pro, self).__init__()
        assert backbone in ['resnet18', 'resnet34']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=False)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=False)
        else:
            raise NotImplementedError

        self.fuse23 = AsymFusionModule(512, 256, 256)
        self.fuse12 = AsymFusionModule(256, 128, 128)

        self.head = _FCNHead(128, 1, drop=drop)

        self.context = CPM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)

        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, hei, wid = x.shape

        c1, c2, c3 = self.backbone(x)

        out = self.context(c3)

        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

        return out


def agpcnet(backbone, scales, reduce_ratios, gca_type, gca_att, drop):
    return AGPCNet(backbone=backbone, scales=scales, reduce_ratios=reduce_ratios, gca_type=gca_type, gca_att=gca_att, drop=drop)
