import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F

import numpy as np

__all__ = ['RPCANet']
import pdb
class _FCNHead(nn.Module):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, momentum, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels #// 4
        self.block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,kernel_size=3, padding=1, bias=False),
        norm_layer(inter_channels, momentum=momentum),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(in_channels=inter_channels, out_channels=channels,kernel_size=1)
        )
    # pylint: disable=arguments-differ
    def forward(self, x):
        return self.block(x)
class RPCANet(nn.Module):
    def __init__(self, stage_num=6, slayers=6, llayers=3, mlayers=3, channel=32, mode='test'):
        super(RPCANet, self).__init__()
        self.stage_num = stage_num
        self.decos = nn.ModuleList()
        self.mode = mode
        for _ in range(stage_num):
            self.decos.append(DecompositionModule(slayers=slayers, llayers=llayers,
                                                  mlayers=mlayers, channel=channel))
        # 迭代循环初始化参数
        for m in self.modules():
            # 也可以判断是否为conv2d，使用相应的初始化方式
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.head = _FCNHead(in_channels=3, channels=1, momentum=0.9)
    def forward(self, D):
        # pdb.set_trace()
        if D.size(1)==1:
            D=D.repeat(1,3,1,1)
        T = torch.zeros(D.shape).to(D.device)
        for i in range(self.stage_num):
            D, T = self.decos[i](D, T)
            # print(T.shape)
        # pdb.set_trace()
        if self.mode == 'train':
            return D,T
        else:
            return self.head(T)

class DecompositionModule(object):
    pass


class DecompositionModule(nn.Module):
    def __init__(self, slayers=6, llayers=3, mlayers=3, channel=32):
        super(DecompositionModule, self).__init__()
        self.lowrank = LowrankModule(channel=channel, layers=llayers)
        self.sparse = SparseModule(channel=channel, layers=slayers)
        self.merge = MergeModule(channel=channel, layers=mlayers)

    def forward(self, D, T):
        B = self.lowrank(D, T)
        T = self.sparse(D, B, T)
        D = self.merge(B, T)
        return D, T


class LowrankModule(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(LowrankModule, self).__init__()

        convs = [nn.Conv2d(3, channel, kernel_size=3, padding=1, stride=1), #3
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        #self.relu = nn.ReLU()
        #self.gamma = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, D, T):
        x = D - T
        B = x + self.convs(x)
        return B


class SparseModule(nn.Module):
    def __init__(self, channel=32, layers=6) -> object:
        super(SparseModule, self).__init__()
        convs = [nn.Conv2d(3, channel, kernel_size=3, padding=1, stride=1), #3
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.convs = nn.Sequential(*convs)
        self.epsilon = nn.Parameter(torch.Tensor([0.01]), requires_grad=True)

    def forward(self, D, B, T):
        # print(D.shape)
        x = T + D - B
        T = x - self.epsilon * self.convs(x)
        return T


class MergeModule(nn.Module):
    def __init__(self, channel=32, layers=3):
        super(MergeModule, self).__init__()
        convs = [nn.Conv2d(3, channel, kernel_size=3, padding=1, stride=1),
                 nn.BatchNorm2d(channel),
                 nn.ReLU(True)]
        for i in range(layers):
            convs.append(nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1))
            convs.append(nn.BatchNorm2d(channel))
            convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(channel, 1, kernel_size=3, padding=1, stride=1))
        self.mapping = nn.Sequential(*convs)

    def forward(self, B, T):
        
        x = B + T
        D = self.mapping(x)
        return D

if __name__ == '__main__':
    D = torch.randn(1, 3, 32, 32)
    model = RPCANet()
    T = model(D) #D, 
    print( T.shape) #D.shape,