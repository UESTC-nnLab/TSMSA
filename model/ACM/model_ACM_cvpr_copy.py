import torch
import torch.nn as nn
import  numpy as np
from torch.nn import BatchNorm2d
# from  torchvision.models.resnet import BasicBlock

import torch.nn.functional as F

from typing import Any, Callable, List, Optional, Type, Union
# class BasicBlock(nn.Module):
#     expansion: int = 1

#     def __init__(
#         self,
#         inplanes: int,
#         planes: int,
#         stride: int = 1,
#         downsample: Optional[nn.Module] = None,
#         groups: int = 1,
#         base_width: int = 64,
#         dilation: int = 1,
#         norm_layer: Optional[Callable[..., nn.Module]] = None,
#     ) -> None:
#         super().__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError("BasicBlock only supports groups=1 and base_width=64")
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)

#         return out

# def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
#     """3x3 convolution with padding"""
#     return nn.Conv2d(
#         in_planes,
#         out_planes,
#         kernel_size=3,
#         stride=stride,
#         padding=dilation,
#         groups=groups,
#         bias=False,
#         dilation=dilation,
#     )


# def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Sequential(
                nn.Conv2d(inplanes,inplanes,kernel_size=3,stride=stride,padding=1,groups=inplanes),
                nn.Conv2d(inplanes,planes,kernel_size=1),
               ) #conv3x3(inplanes, planes, stride) nn.LeakyReLU()
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Sequential(
                nn.Conv2d(planes,planes,kernel_size=3,padding=1,groups=planes),
               )
        #conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

       
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class AsymBiChaFuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AsymBiChaFuse, self).__init__()
        self.channels = channels
        self.bottleneck_channels = int(channels // r)


        self.topdown = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=self.channels, out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels,  kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
        nn.Conv2d(in_channels=self.channels,out_channels=self.bottleneck_channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.bottleneck_channels,momentum=0.9),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=self.bottleneck_channels,out_channels=self.channels, kernel_size=1, stride=1,padding=0),
        nn.BatchNorm2d(self.channels,momentum=0.9),
        nn.Sigmoid()
        )

        self.post = nn.Sequential(
        nn.Conv2d(in_channels=channels,out_channels=channels, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.BatchNorm2d(channels,momentum=0.9),
        nn.ReLU(inplace=True)
        )

    def forward(self, xh, xl):

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * torch.mul(xl, topdown_wei) + 2 * torch.mul(xh, bottomup_wei)
        xs = self.post(xs)
        return xs
import pdb
class ASKCResUNet(nn.Module):
    def __init__(self, in_channels=1, layers=[1,1,1], channels=[8,16,32,64], fuse_mode='AsymBi', tiny=False, classes=1,
                 norm_layer=BatchNorm2d,groups=1, norm_kwargs=None, **kwargs):
        super(ASKCResUNet, self).__init__()
        self.layer_num = len(layers)
        self.tiny = tiny
        self._norm_layer = norm_layer
        self.groups = groups
        self.momentum=0.9
        stem_width = int(channels[0])  ##channels: 8 16 32 64
        # self.stem.add(norm_layer(scale=False, center=False,**({} if norm_kwargs is None else norm_kwargs)))
        if tiny:  # 默认是False
            self.stem = nn.Sequential(
            norm_layer(in_channels,self.momentum),
            nn.Conv2d(in_channels, out_channels=stem_width * 2, kernel_size=3, stride=1,padding=1, bias=False),
            norm_layer(stem_width * 2, momentum=self.momentum),
            nn.ReLU(inplace=True)
            )
        else:
            self.stem = nn.Sequential(
            # self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=2,
            #                          padding=1, use_bias=False))
            # self.stem.add(norm_layer(in_channels=stem_width*2))
            # self.stem.add(nn.Activation('relu'))
            # self.stem.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            norm_layer(in_channels, momentum=self.momentum),
            nn.Conv2d(in_channels=in_channels,out_channels=stem_width, kernel_size=3, stride=2,padding=1, bias=False),
            norm_layer(stem_width,momentum=self.momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_width,out_channels=stem_width, kernel_size=3, stride=1,padding=1, bias=False),
            norm_layer(stem_width,momentum=self.momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_width,out_channels=stem_width * 2, kernel_size=3, stride=1,padding=1, bias=False),
            norm_layer(stem_width * 2,momentum=self.momentum),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ).cuda()

        self.layer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                       out_channels=channels[1],
                                       in_channels=channels[1], stride=1).cuda()

        self.layer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                       out_channels=channels[2], stride=2,
                                       in_channels=channels[1]).cuda()
        #
        self.layer3 = self._make_layer(block=BasicBlock, blocks=layers[2],
                                       out_channels=channels[3], stride=2,
                                       in_channels=channels[2]).cuda()

        self.deconv2 = nn.ConvTranspose2d(in_channels=channels[3] ,out_channels=channels[2], kernel_size=(4, 4),     ##channels: 8 16 32 64
                                          stride=2, padding=1).cuda()
        self.uplayer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                         out_channels=channels[2], stride=1,
                                         in_channels=channels[2]).cuda()
        self.fuse2 = self._fuse_layer(fuse_mode, channels=channels[2]).cuda()

        self.deconv1 = nn.ConvTranspose2d(in_channels=channels[2] ,out_channels=channels[1], kernel_size=(4, 4),
                                          stride=2, padding=1).cuda()
        self.uplayer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                         out_channels=channels[1], stride=1,
                                         in_channels=channels[1]).cuda()
        self.fuse1 = self._fuse_layer(fuse_mode, channels=channels[1]).cuda()

        self.head = _FCNHead(in_channels=channels[1], channels=classes, momentum=self.momentum).cuda()

        self.output_0 = nn.Conv2d(32, 1, 1)
        self.output_1 = nn.Conv2d(16, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        # self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        # self.final = nn.Conv2d(4, 1, 3, 1, 1)
        self.final = nn.Conv2d(2, 1, 3, 1, 1)

    def _make_layer(self, block, out_channels, in_channels, blocks, stride):

        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or out_channels != in_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels , stride),
                norm_layer(out_channels * block.expansion, momentum=self.momentum),
            )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample, self.groups, norm_layer=norm_layer))
        self.inplanes = out_channels  * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, self.groups, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _fuse_layer(self, fuse_mode, channels):

        if fuse_mode == 'AsymBi':
          fuse_layer = AsymBiChaFuse(channels=channels)
        else:
            raise ValueError('Unknown fuse_mode')
        return fuse_layer

    def forward(self,x,train=False):
        _, C, height, width = x.size()
        if C>1:
            x=x.mean(dim=1, keepdim=True)
        _, _, hei, wid = x.shape               #[8, 1, 480, 480]
       
        #pdb.set_trace()
        x = self.stem(x)      # (4,16,120,120)  tiny:[8, 16, 480, 480]
        c1 = self.layer1(x)   # (4,16,120,120)  tiny:[8, 16, 480, 480]
        c2 = self.layer2(c1)  # (4,32, 60, 60)   tiny: [8, 32, 240, 240]
        c3 = self.layer3(c2)  # (4,64, 30, 30)    tiny: [8, 64, 120, 120]

        deconvc2 = self.deconv2(c3)        # (4,32, 60, 60)  tiny: [8, 32, 240, 240]
        fusec2 = self.fuse2(deconvc2, c2)  # (4,32, 60, 60)   tiny: [8, 32, 240, 240]
        upc2 = self.uplayer2(fusec2)       # (4,32, 60, 60)   tiny: [8, 32, 240, 240]
       
        '''mask1'''
        mask1= self.output_0(upc2) 
       
        deconvc1 = self.deconv1(upc2)        # (4,16,120,120)  tiny: [8, 32, 480, 480]
        fusec1 = self.fuse1(deconvc1, c1)    # (4,16,120,120)  tiny: [8, 32, 480, 480]
        upc1 = self.uplayer1(fusec1)         # (4,16,120,120)  tiny: [8, 32, 480, 480]
        '''mask2'''
        mask2= self.output_1(upc1)
        #pdb.set_trace()
        if train:
            output = self.final(torch.cat([mask2, self.up(mask1)], dim=1))
            pred= F.interpolate(output, scale_factor=4, mode='bilinear')
            mask1=F.interpolate(mask1, scale_factor=4, mode='bilinear') #240,240
            mask2=F.interpolate(mask2, scale_factor=4, mode='bilinear') #480.480
            return [mask2, mask1], pred
        else:
            mask2= self.output_1(upc1)
            mask2=F.interpolate(mask2, scale_factor=4, mode='bilinear')
            return mask2  
            
                     # (4,1,120,120)   tiny: [8, 32, 480, 480]

       

        # if self.tiny:
        #     out = pred
        # else:
        #     # out = F.contrib.BilinearResize2D(pred, height=hei, width=wid)  # down 4
        #     out = F.interpolate(pred, scale_factor=4, mode='bilinear')  # down 4             # (4,1,480,480)

        # return out

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)


class _FCNHead(nn.Module):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, momentum, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
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

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class LightWeightNetwork(nn.Module):
    def __init__(self,):
        super(LightWeightNetwork, self).__init__()
       
        #pdb.set_trace()
        
        self.model = ASKCResUNet()
       
        
    def forward(self, img):
        return self.model(img)

#########################################################
###2.测试ASKCResUNet
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    layers = [3] * 3
    channels = [x * 1 for x in [8, 16, 32, 64]]
    in_channels = 1
    model= ASKCResUNet()

    model=model.cuda()
    DATA = torch.randn(8,1,480,480).to(DEVICE)

    output=model.forward(DATA,True)
    pdb.set_trace()
    print("output:",np.shape(output))
##########################################################