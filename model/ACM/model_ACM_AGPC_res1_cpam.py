import torch
import torch.nn as nn
import  numpy as np
from torch.nn import BatchNorm2d
# from  torchvision.models.resnet import BasicBlock
import pdb
import torch.nn.functional as F
# from model.utils import init_weights, count_param
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, List, Optional, Type, Union
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
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

__all__ = ['NonLocalBlock', 'GCA_Channel', 'GCA_Element', 'AGCB_Element', 'AGCB_Patch', 'CPM']

class AsymFusionModule(nn.Module):
    def __init__(self, planes_high, planes_low, planes_out):
        super(AsymFusionModule, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.BatchNorm2d(planes_low//4),
            nn.ReLU(True),

            nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.plus_conv = nn.Sequential(
            nn.Conv2d(planes_high, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.ReLU(True)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(planes_low, planes_low//4, kernel_size=1),
            nn.BatchNorm2d(planes_low//4),
            nn.ReLU(True),

            nn.Conv2d(planes_low//4, planes_low, kernel_size=1),
            nn.BatchNorm2d(planes_low),
            nn.Sigmoid(),
        )
        self.end_conv = nn.Sequential(
            nn.Conv2d(planes_low, planes_out, 3, 1, 1),
            nn.BatchNorm2d(planes_out),
            nn.ReLU(True),
        )

    def forward(self, x_high, x_low):
        x_high = self.plus_conv(x_high)
        pa = self.pa(x_low)
        ca = self.ca(x_high)

        feat = x_low + x_high
        feat = self.end_conv(feat)
        feat = feat * ca
        feat = feat * pa
        return feat
    

class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + x
        return out


class GCA_Channel(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(GCA_Channel, self).__init__()
        assert att_mode in ['origin', 'post']

        self.att_mode = att_mode
        if att_mode == 'origin':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
            self.sigmoid = nn.Sigmoid()
        elif att_mode == 'post':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
            self.conv_att = nn.Sequential(
                nn.Conv2d(planes, planes // 4, kernel_size=1),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(True),

                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.att_mode == 'origin':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.sigmoid(gca)
        elif self.att_mode == 'post':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.conv_att(gca)
        else:
            raise NotImplementedError
        return gca


class GCA_Element(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(GCA_Element, self).__init__()
        assert att_mode in ['origin', 'post']

        self.att_mode = att_mode
        if att_mode == 'origin':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
            self.sigmoid = nn.Sigmoid()
        elif att_mode == 'post':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
            self.conv_att = nn.Sequential(
                nn.Conv2d(planes, planes // 4, kernel_size=1),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(True),

                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
            )
            self.sigmoid = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        batch_size, C, height, width = x.size()

        if self.att_mode == 'origin':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        elif self.att_mode == 'post':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.conv_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        else:
            raise NotImplementedError
        return gca


class AGCB_Patch(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Patch, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Channel(planes, scale, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x)

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            attention = gca[:, :, attention_ind[i], attention_ind[i+1]].view(batch_size, C, 1, 1)
            context_list.append(self.non_local(block) * attention)

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class AGCB_Element(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Element, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Element(planes, scale, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x)

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            # attention = gca[:, :, attention_ind[i], attention_ind[i+1]].view(batch_size, C, 1, 1)
            context_list.append(self.non_local(block))

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = context * gca
        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class AGCB_NoGCA(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32):
        super(AGCB_NoGCA, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            context_list.append(self.non_local(block))

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class CPM(nn.Module):
    def __init__(self, planes, block_type, scales=(3,5,6,10), reduce_ratios=(4,8), att_mode='origin'):
        super(CPM, self).__init__()
        assert block_type in ['patch', 'element']
        assert att_mode in ['origin', 'post']

        inter_planes = planes // reduce_ratios[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, inter_planes, kernel_size=1),
            nn.BatchNorm2d(inter_planes),
            nn.ReLU(True),
        )

        if block_type == 'patch':
            self.scale_list = nn.ModuleList(
                [AGCB_Patch(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        elif block_type == 'element':
            self.scale_list = nn.ModuleList(
                [AGCB_Element(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        else:
            raise NotImplementedError

        channels = inter_planes * (len(scales) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        reduced = self.conv1(x)

        blocks = []
        for i in range(len(self.scale_list)):
            blocks.append(self.scale_list[i](reduced))
        out = torch.cat(blocks, 1)
        out = torch.cat((reduced, out), 1)
        out = self.conv2(out)
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

        # self.layer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
        #                                out_channels=channels[1],
        #                                in_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                       out_channels=channels[2], stride=2,
                                       in_channels=channels[1]).cuda()
        #
        self.layer3 = self._make_layer(block=BasicBlock, blocks=layers[2],
                                       out_channels=channels[3], stride=2,
                                       in_channels=channels[2]).cuda()
        self.context = CPM(planes=64, scales=(10, 6, 5, 3), reduce_ratios=(16, 4), block_type='patch',att_mode='post').cuda()#512
        self.fuse23 = AsymFusionModule(64, 32, 32).cuda()
        self.fuse12= AsymFusionModule(32, 16, 16).cuda()
        # self.deconv2 = nn.ConvTranspose2d(in_channels=channels[3] ,out_channels=channels[2], kernel_size=(4, 4),     ##channels: 8 16 32 64
        #                                   stride=2, padding=1)
        # self.uplayer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
        #                                  out_channels=channels[2], stride=1,
        #                                  in_channels=channels[2])
        # self.fuse2 = self._fuse_layer(fuse_mode, channels=channels[2])

        # self.deconv1 = nn.ConvTranspose2d(in_channels=channels[2] ,out_channels=channels[1], kernel_size=(4, 4),
        #                                   stride=2, padding=1)
        # self.uplayer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
        #                                  out_channels=channels[1], stride=1,
        #                                  in_channels=channels[1])
        # self.fuse1 = self._fuse_layer(fuse_mode, channels=channels[1])

        self.head = _FCNHead(in_channels=channels[1], channels=classes, momentum=self.momentum).cuda()


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

    def forward(self,  x):
        _, C, height, width = x.size()
        if C>1:
            x=x.mean(dim=1, keepdim=True)
        _, _, hei, wid = x.shape

        c1 = self.stem(x)      # (4,16,120,120)
        # c1 = self.layer1(x)   # (4,16,120,120)
        c2 = self.layer2(c1)  # (4,32, 60, 60)
        c3 = self.layer3(c2)  # (4,64, 30, 30)
        out=c3
        out=self.context(c3)
        out = F.interpolate(out, size=[hei // 8, wid // 8], mode='bilinear', align_corners=True) #4
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)#2
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)

         
        # deconvc2 = self.deconv2(c3)        # (4,32, 60, 60)
        # fusec2 = self.fuse2(deconvc2, c2)  # (4,32, 60, 60)
        # upc2 = self.uplayer2(fusec2)       # (4,32, 60, 60)

        # deconvc1 = self.deconv1(upc2)        # (4,16,120,120)
        # fusec1 = self.fuse1(deconvc1, c1)    # (4,16,120,120)
        # upc1 = self.uplayer1(fusec1)         # (4,16,120,120)

        # pred = self.head(upc1)               # (4,1,120,120)

        # if self.tiny:
        #     out = pred
        # else:
        #     # out = F.contrib.BilinearResize2D(pred, height=hei, width=wid)  # down 4
        #     out = F.interpolate(pred, scale_factor=4, mode='bilinear')  # down 4             # (4,1,480,480)

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


class LightWeightNetwork(nn.Module):
    def __init__(self,):
        super(LightWeightNetwork, self).__init__()
       
        #pdb.set_trace()
        
        self.model = ASKCResUNet()
       
        
    def forward(self, img):
        return self.model(img)

#########################################################
##2.测试ASKCResUNet
if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
    layers = [3] * 3
    channels = [x * 1 for x in [8, 16, 32, 64]]
    in_channels = 3
    model=LightWeightNetwork()

    model=model.cuda()
    DATA = torch.randn(8,3,480,480).to(DEVICE)

    output=model(DATA)
    print("output:",np.shape(output))
##########################################################