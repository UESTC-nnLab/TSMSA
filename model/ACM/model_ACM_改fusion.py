import torch
import torch.nn as nn
import  numpy as np
from torch.nn import BatchNorm2d
from  torchvision.models.resnet import BasicBlock
from .fusion import AsymBiChaFuse
import torch.nn.functional as F
# from model.utils import init_weights, count_param
import pdb


class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()
        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )##512

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )#512

        ##############add spatial attention ###Cross UtU############
        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),
            # nn.Sigmoid(),

            SpatialAttention(kernel_size=3),
            # nn.Conv2d(self.bottleneck_channels, 2, 3, 1, 0),
            # nn.Conv2d(2, 1, 1, 1, 0),
            #nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )#512

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl * topdown_wei)
        xs1 = 2 * xl * topdown_wei  #1
        out1 = self.post(xs1)

        xs2 = 2 * xh * bottomup_wei    #1
        out2 = self.post(xs2)
        return out1,out2
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x



class ASKCResUNet(nn.Module):
    def __init__(self, in_channels=1, layers=[3,3,3], channels=[8,16,32,64], fuse_mode='UIUNet', tiny=False, classes=1,
                 norm_layer=BatchNorm2d,groups=1, norm_kwargs=None, **kwargs): #[8,16,32,64]
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
            )

        self.layer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                       out_channels=channels[1],
                                       in_channels=channels[1], stride=1)

        self.layer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                       out_channels=channels[2], stride=2,
                                       in_channels=channels[1])
        #
        self.layer3 = self._make_layer(block=BasicBlock, blocks=layers[2],
                                       out_channels=channels[3], stride=2,
                                       in_channels=channels[2])

        self.deconv2 = nn.ConvTranspose2d(in_channels=channels[3] ,out_channels=channels[2], kernel_size=(4, 4),     ##channels: 8 16 32 64
                                          stride=2, padding=1)
        self.uplayer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                         out_channels=channels[2], stride=1,
                                         in_channels=channels[2])
        self.fuse2 = self._fuse_layer(fuse_mode, channels=channels[2])
        self.conv2_1=nn.Conv2d(64,32,3,1,1)
        self.conv1_1=nn.Conv2d(32,16,3,1,1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=channels[2] ,out_channels=channels[1], kernel_size=(4, 4),
                                          stride=2, padding=1)
        self.uplayer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                         out_channels=channels[1], stride=1,
                                         in_channels=channels[1])
        self.fuse1 = self._fuse_layer(fuse_mode, channels=channels[1])

        self.head = _FCNHead(in_channels=channels[1], channels=classes, momentum=self.momentum)
        # self.Rep1=RepBlock(16,16)
        # self.Rep2=RepBlock(16,16)
        # self.Rep3=RepBlock(32,32)
        # self.Rep4=RepBlock(64,64)
        # self.Rep5=RepBlock(32,32)
        # self.Rep6=RepBlock(16,16)



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
        #   pdb.set_trace()
          fuse_layer = AsymBiChaFuse(channels=channels)
        elif fuse_mode=='UIUNet':
            # pdb.set_trace()
            fuse_layer=AsymBiChaFuseReduce(channels,channels,out_channels=channels)
        else:
            raise ValueError('Unknown fuse_mode')
        return fuse_layer

    def forward(self,  x):

        _, _, hei, wid = x.shape

        x = self.stem(x)      # (4,16,120,120)
        #x=self.Rep1(x)
        c1 = self.layer1(x)   # (4,16,120,120)
        #c1=self.Rep2(c1)
        c2 = self.layer2(c1)  # (4,32, 60, 60)
        #c2=self.Rep3(c2)
        c3 = self.layer3(c2)  # (4,64, 30, 30)
        #c3=self.Rep4(c3)
        deconvc2 = self.deconv2(c3) 
        #pdb.set_trace()       # (4,32, 60, 60)
        out1,out2 = self.fuse2(deconvc2, c2)  # (4,32, 60, 60)
        fusec2=torch.cat([out1,out2],1)
        fusec2=self.conv2_1(fusec2)
        upc2 = self.uplayer2(fusec2)       # (4,32, 60, 60)
        #upc2=self.Rep5(upc2)
        #pdb.set_trace()
        deconvc1 = self.deconv1(upc2)        # (4,16,120,120)

        out1,out2 = self.fuse1(deconvc1, c1)    # (4,16,120,120)
        fusec1=torch.cat([out1,out2],1)
        fusec1=self.conv1_1(fusec1)
        upc1 = self.uplayer1(fusec1)         # (4,16,120,120)
        #upc1=self.Rep6(upc1)
        pred = self.head(upc1)               # (4,1,120,120)

        if self.tiny:
            out = pred
        else:
            # out = F.contrib.BilinearResize2D(pred, height=hei, width=wid)  # down 4
            out = F.interpolate(pred, scale_factor=4, mode='bilinear')  # down 4             # (4,1,480,480)
        #out=out.sigmoid()
        #out=torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))
        #pdb.set_trace()
        return out.sigmoid()

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
'''我加的'''
class ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return x
class RepBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 channelAttention_reduce=4):
        super().__init__()

        self.C = in_channels
        self.O = out_channels

        assert in_channels == out_channels
        self.ca = ChannelAttention(input_channels=in_channels, internal_neurons=in_channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(in_channels,in_channels,kernel_size=5,padding=2,groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,7),padding=(0,3),groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(7,1),padding=(3,0),groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,11),padding=(0,5),groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(11,1),padding=(5,0),groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels,in_channels,kernel_size=(1,21),padding=(0,10),groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels,in_channels,kernel_size=(21,1),padding=(10,0),groups=in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=(1,1),padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        channel_att_vec = self.ca(inputs)
        inputs = channel_att_vec * inputs #(1,64,32,32)
        #pdb.set_trace()
        x_init = self.dconv5_5(inputs) #(1,64,32,32)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x) #(1,64,32,32)
        out = spatial_att * inputs
        out = self.conv(out) #(1,64,32,32)
        return out
'''end'''
#########################################################
###2.测试ASKCResUNet
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    layers = [3] * 3
    channels = [x * 1 for x in [8, 16, 32, 64]]
    in_channels = 3
    model= ASKCResUNet(in_channels, layers=layers, channels=channels, fuse_mode='UIUNet',tiny=False, classes=1)

    model=model.cuda()
    DATA = torch.randn(8,3,480,480).to(DEVICE)

    output=model(DATA)
    print("output:",np.shape(output))
##########################################################