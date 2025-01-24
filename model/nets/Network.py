import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
#from darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

class Feature_Extractor(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("stem","dark2","dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act) #512 256
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width), # 512
            int(in_channels[1] * width),
            round(3 * depth), #3*0.33->1
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )


    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat_stem,feat0,feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]
        #[4,128,64,64]  [4,256,32,32]  [4,512,16,16]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        #pdb.set_trace()
        P5          = self.lateral_conv0(feat3) #(4,256,16,16)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5) #(4, 256, 32, 32)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1) #(4, 512, 32, 32)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample) #(4, 256, 32, 32)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) #(4, 128, 32, 32)
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4)   #(4,128,64,64)
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1)  #(4, 256, 64, 64)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)   #(4, 128, 64, 64)
        
        
        return feat_stem,feat0,P3_out

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs


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


import pdb
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
class Network(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=5):
        super(Network, self).__init__()
        self.num_frame = num_frame
        self.backbone = Feature_Extractor(0.33,0.50).cuda()

        #-----------------------------------------#
        #   尺度感知模块
        #-----------------------------------------#
        # self.neck = Motion_coupling_Neck(channels=[128], num_frame=num_frame)
        #----------------------------------------------------------#
        #   head
        #----------------------------------------------------------#
        # self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")
        
        # self.deconv1 = nn.ConvTranspose2d(in_channels=128 ,out_channels=64, kernel_size=(4, 4),
        #                                   stride=2, padding=1)
        # self.deconv2 = nn.ConvTranspose2d(in_channels=64 ,out_channels=32, kernel_size=(4, 4),
        #                                   stride=2, padding=1)
        # self.deconv3 = nn.ConvTranspose2d(in_channels=32 ,out_channels=16, kernel_size=(4, 4),
        #                                   stride=2, padding=1)
        # self.C1 = CSPLayer(
        #     64,
        #     64,
        #     round(1),
        #     False,
        #     depthwise =False,
        #     act ='silu',
        # )
        # self.C2 = CSPLayer(
        #     32,
        #     32,
        #     round(1),
        #     False,
        #     depthwise =False,
        #     act ='silu',
        # )
        # self.C3 = CSPLayer(
        #     16,
        #     16,
        #     round(1),
        #     False,
        #     depthwise =False,
        #     act ='silu',
        # )
        self.head =_FCNHead(in_channels=32,channels=1,momentum=0.9).cuda()
        
        self.context = CPM(planes=128, scales=(10, 6), reduce_ratios=(8, 8), block_type='patch',att_mode='origin').cuda()
        self.fuse23 = AsymFusionModule(128, 64, 64).cuda()  #(512,256,256)
        self.fuse12 = AsymFusionModule(64, 32, 32).cuda()   #(256, 128, 128) 
        
    def forward(self, inputs): #4, 3, 5, 512, 512
        _, _, _,hei, wid = inputs.shape
        #inputs:[4, 3, 5, 512, 512]
        feat = []
        #pdb.set_trace()
        for i in range(self.num_frame):
            feat.append(self.backbone(inputs[:,:,i,:,:])) #[4, 128, 64, 64]  #8倍下采样  len(feat):5
        pdb.set_trace()
        """[b,128,32,32][b,256,16,16][b,512,8,8]"""
        stem,feat0,feat=self.backbone(inputs)  
        
        feat=self.context(feat)
        #pdb.set_trace()
        feat = F.interpolate(feat, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)## [8,512,128,128]
        feat = self.fuse23(feat, feat0) 
       
        feat = F.interpolate(feat, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)
        feat = self.fuse12(feat,stem)

        feat = self.head(feat)
        feat = F.interpolate(feat, size=[hei, wid], mode='bilinear', align_corners=True)
        
        #pdb.set_trace()
        # feat=self.deconv1(feat)
        # feat= self.C1(feat)
        # feat=self.deconv2(feat)
        # feat=self.C2(feat)
        # feat=self.deconv3(feat)
        # feat=self.C3(feat)
        # feat=self.head(feat)
        #pdb.set_trace()
        #feat_all=feat_all.append(feat)
        #outputs  = self.head(feat_all)#[4,128,64,64]->[4, 6, 64, 64]: 一个点监督8个像素
        
        # if self.training:
        #     return  outputs, motion_loss  
        # else:
        return  feat.sigmoid()
            






    
   
  



def print_model_size(model):
    # 计算模型参数的总数量
    param_num = sum(p.numel() for p in model.parameters())
    # 计算模型参数所占的总内存大小（假设参数为float32，即4字节）
    param_size = param_num * 4 / (1024**2)  # 单位转换为MB
    print(f"模型参数数量：{param_num}")
    print(f"模型大小：{param_size:.2f} MB")

# 假设你已经定义并初始化了一个模型
# model = YourModelClass()  # 请替换为你的模型类

# 使用函数打印模型的大小

import torch

def get_model_complexity(model, input_size=(3,5,512,512)):
    """计算并打印模型的FLOPs和Params"""
    # 计算FLOPs和Params
    input = torch.randn(1, *input_size)  # 假设输入是一个batch size为1的张量，根据您的模型调整尺寸
    flops, params = profile(model, inputs=(input,))
    
    # 将FLOPs转换为GFLOPs
    gflops = flops / 1e9
    
    # 将Params转换为MParams
    mparams = params / 1e6
    
    return gflops, mparams

# 假设model是您的模型实例
#model = your_model()  # 请替换为实际的模型定义或实例化

    
if __name__ == "__main__":
    from thop import profile
    import torch
    
    from yolo_training import YOLOLoss
    net = Network(num_classes=1, num_frame=5)
    # gflops, mparams = get_model_complexity(net)

    # print(f"GFLOPs: {gflops:.2f}")
    # print(f"MParams: {mparams:.2f}")  




    '''    
    params = sum(p.numel() for p in net.parameters())
    print("Params: ", params)



    #model = your_model()
    input = torch.randn(1, 3, 5, 512,512) # 以一个batch size为1，3通道，224x224大小的图片为例
    flops, params = profile(net, inputs=(input,))
    print("FLOPs: ", flops, "Params: ", params)

    print_model_size(net)
    '''
    bs = 4
    a = torch.randn(bs, 3, 5, 512, 512).cuda()
    out = net(a)
    pdb.set_trace()
    for item in out:
        print(item.size())
        
    yolo_loss    = YOLOLoss(num_classes=1, fp16=False, strides=[16])

    target = torch.randn([bs, 1, 5]).cuda()
    target = nn.Softmax()(target)
    target = [item for item in target]

    loss = yolo_loss(out, target)
    print(loss)
