import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
from torch.nn import BatchNorm2d
import pdb
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

'''self-attention'''


        
class GCBlock(nn.Module):
    def __init__(self,inplanes=16,ratio=1/16,pooling_type='att',
                 fusion_types=('channel_add', )):
        super(GCBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']

        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv1d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv1d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv1d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, Q,K):
        batch, channel, embed= Q.size()
        if self.pooling_type == 'att':
            input_x = Q #(8,16,1344)
            # pdb.set_trace()
            # [N, C, H * W]
            # input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            # input_x = input_x.unsqueeze(2)
            # pdb.set_trace()
            # [N, 1, H, W]
            context_mask = self.conv_mask(K)#(8,1,1344)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, embed) #height * width
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            # context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask.transpose(1,2))
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(Q)
        return context

    def forward(self, Q,K,V):
        # [N, C, 1, 1]
        context = self.spatial_pool(Q,K)
        # pdb.set_trace()
        out = V
        # pdb.set_trace()
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # pdb.set_trace()
            # [N, C, 1, 1]
            # channel_add_term = self.channel_add_conv(context)
            out =  context *out
        return out
'''end'''
class Cross_Slice_ConvLSTM_Node(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(Cross_Slice_ConvLSTM_Node, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias) 
        self.conv2 = nn.Conv2d(in_channels=4 * self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias) 
        self.motion = nn.Sequential(nn.Conv2d(in_channels=3 * self.input_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias))

        self.mm1 = nn.Conv2d(2 * self.input_dim, 1 * self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.mm2 = nn.Conv2d(2 * self.input_dim, 1 * self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.past_attention = GCBlock(inplanes=hidden_dim,ratio=1/16)#nn.MultiheadAttention(embed_dim=1344, num_heads=8,dropout=0.1, batch_first=True) #1344
        self.future_attention = GCBlock(inplanes=hidden_dim,ratio=1/16)#nn.MultiheadAttention(embed_dim=1344, num_heads=8,dropout=0.1, batch_first=True)#1344
        # self.pool = nn.AdaptiveMaxPool2d(32)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)  

    def forward(self, input_tensor, input_head, all_state, cur_state, multi_head):  #input_tensor=cur_layer_input[:, t, :, :, :], input_head = head_input[:, t, :, :, :], all_state = head_input, cur_state=[h, c], multi_head=[m_h, m_c]
        # pdb.set_trace()
        h_cur, c_cur = cur_state  #[4,64,128,128] [4,64,128,128]

        combined = torch.cat([input_tensor, h_cur], dim=1)  #input_tensor[4,64,128,128] h_cur[4,64,128,128]  combined[4,128,128,128]
        combined_conv = self.conv(combined)    #combined_conv[4,256,128,128]
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) #[4, 64, 128, 128]
  
        m_h, m_c = multi_head  #[4,64,128,128] [4,64,128,128]

        combined2 = torch.cat([input_tensor, h_cur, input_head, m_h], dim=1)  #[4, 256, 128, 128]
        combined_conv2 = self.conv2(combined2)  #[4, 256, 128, 128]

        mm_i, mm_f, mm_o, mm_g = torch.split(combined_conv2, self.hidden_dim, dim=1)
        
        m_i = torch.sigmoid(mm_i+cc_i) 
        m_f = torch.sigmoid(mm_f+cc_f)
        m_o = torch.sigmoid(mm_o+cc_o)
        m_g = torch.tanh(mm_g+cc_g)
        
        c_next = m_f * c_cur + m_i * m_g       
        h_next = m_o * torch.tanh(c_next) 
        # pdb.set_trace()
        _,_,h,w=m_h.shape
        mh_feat = self.pool(m_h).flatten(start_dim=2,end_dim=3)#[4,64,32,32]->[4,64,1024] #m_h 运动变换的原型 #32,42
        input_tensor_feat =  self.pool(all_state[:,0,:,:,:]).flatten(start_dim=2,end_dim=3)#第一帧 [4,64,1024]
        input_head_feat = self.pool(all_state[:,-1,:,:,:]).flatten(start_dim=2,end_dim=3)#最后一帧 [4,64,1024]
        
        mh_feat= self.past_attention(input_tensor_feat, mh_feat, mh_feat) #4, 64, 1024 ,, _ 
        input_feat= self.future_attention(input_head_feat, mh_feat, mh_feat) #multihead_attn(query, key, value) , _ 
        # pdb.set_trace()
        mh_feat = mh_feat.view(all_state.shape[0],self.hidden_dim,int(h/4),int(w/4))#[4,64,1024]->  [4,64,32,32]  32,32
        # pdb.set_trace()
        mh_feat = F.interpolate(mh_feat, size=[all_state.shape[3], all_state.shape[4]], mode='bilinear', align_corners=True)#[4,64,128,128] 3,3

        input_feat = input_feat.view(all_state.shape[0],self.hidden_dim,int(h/4),int(w/4))   #32 32
        input_feat = F.interpolate(input_feat, size=[all_state.shape[3], all_state.shape[4]], mode='bilinear', align_corners=True)#3,3
        # pdb.set_trace()
        motion1 = torch.cat([torch.sigmoid(mh_feat)+m_h, m_h],1) #m_h存储的运动信息与第一帧的自注意力 [4,128,128,128]
        motion1 = self.mm1(motion1) #[4,64,128,128]
        motion2 = torch.cat([torch.sigmoid(input_feat)+m_h, m_h],1)
        motion2 = self.mm2(motion2) #m_h存储的运动信息与最后一帧的自注意力 #[4,64,128,128]

        motion_feat = torch.cat([input_head, motion1, motion2], 1) #[4,192,128,128]
        motion = self.motion(motion_feat) #[4,256,128,128]
        motion_i, motion_f, motion_g, motion_o = torch.split(motion, self.hidden_dim, dim=1)#通过注意力机制的motion来更新lstm

        motion_i = torch.sigmoid(motion_i)
        motion_f = torch.sigmoid(motion_f)
        motion_o = torch.sigmoid(motion_o)
        motion_g = torch.tanh(motion_g)

        m_c_next = motion_f * m_c + motion_i * motion_g + c_next
        m_h_next = motion_o * torch.tanh(m_c_next) + h_next
        
        return h_next, c_next, m_h_next, m_c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
    
   
  

class SSTNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, kernel_size, num_slices, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(SSTNet, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        self.num_slices = num_slices
        cell_list = {}
       
        for i in range(0, self.num_slices):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            for j in range(0, self.num_layers):
                
                cell_list.update({'%d%d'%(i,j): Cross_Slice_ConvLSTM_Node(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias)}) 
                                          
        self.cell_list = nn.ModuleDict(cell_list)
        

    def forward(self, input_tensor, hidden_state=None):
        #pdb.set_trace() #input_tensor[4,5,64,128,128]
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w)) #[4,64,128,128]  [4,64,128,128]
            
            deep_state = self._init_motion_hidden(batch_size=b,
                                             image_size=(h, w), t_len = input_tensor.shape[1]) #dict_keys(['00', '01', '02', '03', '04'])  #[4,64,128,128]  [4,64,128,128] zero

        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)#5
        cur_layer_input = input_tensor #4,5,64,128,128
        head_input = input_tensor #4,5,64,128,128
        
        input_deep_h = {}
        input_deep_c = {}
        
        for deep_idx in range(self.num_slices):   

            for layer_idx in range(self.num_layers):

                output_inner = []

                h, c  = hidden_state['%d%d'%(deep_idx,layer_idx)]  #4,64,128,128 4,64,128,128
                
                for t in range(seq_len): 

                    if deep_idx == 0:
                        m_h, m_c = deep_state['%d%d'%(layer_idx, t)] #4,64,128,128 4,64,128,128
                    else:
                        m_h = input_deep_h['%d%d%d'%(deep_idx-1,layer_idx, t)]
                        m_c = input_deep_c['%d%d%d'%(deep_idx-1,layer_idx, t)]
                    
                    h, c, m_h, m_c = self.cell_list['%d%d'%(deep_idx,layer_idx)](input_tensor=cur_layer_input[:, t, :, :, :], input_head = head_input[:, t, :, :, :], all_state = head_input, cur_state=[h, c], multi_head=[m_h, m_c]) 
                    
                    output_inner.append(h)

                    input_deep_h.update({'%d%d%d'%(deep_idx,layer_idx,t): m_h}) 
                    input_deep_c.update({'%d%d%d'%(deep_idx,layer_idx,t): m_c}) 

                layer_output = torch.stack(output_inner, dim=1) #[4,5,64,128,128]
                head_output = torch.stack(([input_deep_h['%d%d%d'%(deep_idx, layer_idx, t)] for t in range (seq_len)]), dim=1) #[4,5,64,128,128]
                # pdb.set_trace()
                cur_layer_input = layer_output
                head_input = head_output 
            
                layer_output_list.append(layer_output)
                last_state_list.append([h, c]) # [4, 64, 128, 128] [4, 64, 128, 128]

            if not self.return_all_layers:
                layer_output_list = layer_output_list[-1:] #[4, 5, 64, 128, 128]
                last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    
    def _init_hidden(self, batch_size, image_size):
        
        init_states = {}
        for i in range(0, self.num_slices):
            for j in range(0,self.num_layers):
                init_states.update({'%d%d'%(i,j): self.cell_list['%d%d'%(i,j)].init_hidden(batch_size, image_size)}) 
        return init_states
        
    def _init_motion_hidden(self, batch_size, image_size, t_len):
        
        init_states = {}
        for i in range(0,self.num_layers):
            for j in range(0,t_len):
                init_states.update({'%d%d'%(i,j): self.cell_list['00'].init_hidden(batch_size, image_size)}) 
        return init_states


    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
class feature_embedding(nn.Module):
    def __init__(self,):
        super(feature_embedding, self).__init__()
    def forward(self, x):

        return x
    
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

        self.connect_conv=nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
    def forward(self, x):
        identity = x
        identity = self.connect_conv(identity)

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

class feature_embedding(nn.Module):
    def __init__(self, in_channels=1,layers=[4,4,4], channels=[8,16,32,64],norm_layer=BatchNorm2d,t_frame=5,groups=1):
        super(feature_embedding, self).__init__()
        self.momentum=0.9
        self.t_frame=t_frame
        stem_width=channels[0]
        self._norm_layer = norm_layer
        self.groups = groups
        self.feature1=nn.Sequential(
            # self.stem.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=2,
            #                          padding=1, use_bias=False))
            # self.stem.add(norm_layer(in_channels=stem_width*2))
            # self.stem.add(nn.Activation('relu'))
            # self.stem.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            norm_layer(in_channels, momentum=self.momentum),
            nn.Conv2d(in_channels=in_channels,out_channels=stem_width, kernel_size=3, stride=1,padding=1, bias=False),#2
            norm_layer(stem_width,momentum=self.momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=stem_width,out_channels=stem_width, kernel_size=3, stride=1,padding=1, bias=False),
            norm_layer(stem_width,momentum=self.momentum),
            nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=stem_width,out_channels=stem_width * 2, kernel_size=3, stride=1,padding=1, bias=False),
            # norm_layer(stem_width * 2,momentum=self.momentum),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ).cuda()
        # self.feature1=nn.Sequential(
        #     norm_layer(in_channels, momentum=self.momentum),
        #     nn.Conv2d(in_channels=in_channels,out_channels=stem_width, kernel_size=3, stride=1,padding=1, bias=False),
        #     norm_layer(stem_width,momentum=self.momentum),
        #     nn.ReLU(inplace=True)
        #     )
        # self.feature2=BasicBlock(inplanes=stem_width,planes=stem_width,stride=2)
        # self.feature3=BasicBlock(inplanes=stem_width,planes=stem_width*2,stride=2)

        
        self.layer1 = self._make_layer(block=BasicBlock, blocks=layers[0],
                                        out_channels=channels[1],
                                        in_channels=channels[0], stride=2)

        self.layer2 = self._make_layer(block=BasicBlock, blocks=layers[1],
                                        out_channels=channels[2], stride=2,
                                        in_channels=channels[1])
        #
        self.layer3 = self._make_layer(block=BasicBlock, blocks=layers[2],
                                        out_channels=channels[3], stride=2,
                                        in_channels=channels[2])
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
    
    def forward(self, x):
        x1=self.feature1(x)
        x2=self.layer1(x1)
        x3=self.layer2(x2)
        x4=self.layer3(x3)
        
        return [x1,x2,x3,x4]

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "sigmoid":
        module = nn.Sigmoid()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module
class Feature_Fusion(nn.Module):
    def __init__(self, in_channels =[32,64],act = "silu"):
        super().__init__()
       

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[1]), int(in_channels[0]), 1, 1, act=act)
        self.lateral_conv1  = BaseConv(int(2*in_channels[0]), int(in_channels[0]), 1, 1, act=act)




    def forward(self, hr,lr):
        
        lr=self.lateral_conv0(lr)
        lr_upsample = self.upsample(lr)
        hr_upsample = torch.cat([lr_upsample, hr], 1)
        out=self.lateral_conv1(hr_upsample)
  
        
        
        return  out #P4_upsample
class Feature_enhancement(nn.Module):
    def __init__(self,channels=32,num_frame=5):
        super(Feature_enhancement, self).__init__()
        self.frames=num_frame
        self.Sconv1 = BasicBlock(inplanes=channels*self.frames,planes=channels,stride=1)
        self.Sconv2=BasicBlock(inplanes=channels*2,planes=channels,stride=1)
        self.Sconv3=BasicBlock(inplanes=channels,planes=channels,stride=1)


        self.weight = nn.ParameterList(torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True) for _ in range(num_frame))
        self.Tconv1 = nn.Sequential(
            BasicBlock(inplanes=channels*2,planes=channels,stride=1)
        )
        self.Tconv2 = nn.Sequential(
            BasicBlock(inplanes=channels,planes=channels,stride=1)
        )
        
        self.conv_mix = nn.Sequential(
            BasicBlock(inplanes=2*channels,planes=channels,stride=1)
        )
    def forward(self,key_s,motion_feat):
        ###空间特征增强
        feat_SF=torch.cat([motion_feat[:,i,:,:,:] for i in range(self.frames)],dim=1)
        feat_SF=self.Sconv1(feat_SF)
        feat_SF=self.Sconv2(torch.cat([key_s,feat_SF],dim=1))
        key_ss=self.Sconv3(key_s)
        key_ss=key_ss+feat_SF

        ###时序特征增强
        feat_TF = torch.stack([self.Tconv1(torch.cat([motion_feat[:,i,:,:,:], key_s], dim=1))*self.weight[i] for i in range(self.frames)], dim=0)
        feat_TF= self.Tconv2(torch.sum(feat_TF, dim=0))
        
        feat= self.conv_mix(torch.cat([key_ss,feat_TF],dim=1))

  
        return feat

    
class SSTUNet(nn.Module):
    def __init__(self, in_channels=1, layers=[3,3,3], channels=[8,16,32,64], fuse_mode='AsymBi', tiny=False, classes=1,
                 norm_layer=BatchNorm2d,groups=1, num_frame=5,norm_kwargs=None, **kwargs):
        super(SSTUNet, self).__init__()
        self.layer_num = len(layers)
        self.tiny = tiny
        self._norm_layer = norm_layer
        self.groups = groups
        self.momentum=0.9
        stem_width = int(channels[0])  ##channels: 8 16 32 64
        self.num_frame=num_frame
        # self.stem.add(norm_layer(scale=False, center=False,**({} if norm_kwargs is None else norm_kwargs)))
        if tiny:  # 默认是False
            self.stem = nn.Sequential(
            norm_layer(in_channels,self.momentum),
            nn.Conv2d(in_channels, out_channels=stem_width * 2, kernel_size=3, stride=1,padding=1, bias=False),
            norm_layer(stem_width * 2, momentum=self.momentum),
            nn.ReLU(inplace=True)
            )
        else:
            self.backbone = feature_embedding(in_channels=in_channels,layers=layers, channels=channels,norm_layer=norm_layer)
        self.mapping1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, stride=4, padding=1, bias=False),
                nn.LeakyReLU(),
                nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LeakyReLU()) 
        self.SSTCell=SSTNet(input_dim=channels[2], hidden_dim=[channels[2]], kernel_size=(3, 3), num_slices=1, num_layers=1)
        self.feature_enhance=Feature_enhancement(channels=channels[2],num_frame=5)
        self.fusion1=Feature_Fusion(in_channels=[channels[2],channels[3]],act = "silu")
    
        self.fusion2 = Feature_Fusion(in_channels=[channels[1],channels[2]],act = "silu")
        self.fusion3=Feature_Fusion(in_channels=[channels[0],channels[1]],act = "silu")

        self.head = _FCNHead(in_channels=channels[0], channels=classes, momentum=self.momentum).cuda()


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

    # def _fuse_layer(self, fuse_mode, channels):

    #     if fuse_mode == 'AsymBi':
    #       fuse_layer = AsymBiChaFuse(channels=channels)
    #     else:
    #         raise ValueError('Unknown fuse_mode')
    #     return fuse_layer

    def forward(self,  x):
        # pdb.set_trace()  x:[8, 1, 5, 768, 992]
        x=x.unsqueeze(1)
        B, C, T,height, width = x.size()
        if C>1:
            x=x.mean(dim=1, keepdim=True)
        _, _, T,hei, wid = x.shape
        '''获取关键帧的多级空间信息'''

        key_s=x[:,:,-1]
        ###key_s feature
        key_s=self.backbone(key_s)
        # pdb.set_trace()
        '''获取多帧的时序信息'''
        feat_S = torch.cat([self.mapping1(x[:,:,i,:,:]).unsqueeze(1) for i in range(self.num_frame)], 1)#[8,5,32,192,248]
        temporal,_=self.SSTCell(feat_S)
        motion_feat = temporal[-1] #motion feat里面有5个时间步的信息和空间信息
        #feat_SF=[motion_feat[:,i,:,:,:] for i in range(self.num_frame)]
        '''关键帧与时序的多帧信息融合'''
        
        out= self.fusion1(key_s[2],key_s[3])  # (1,32,128,128) #五个时间步的信息和空间key帧信息
        
      
        '''neck fusion, feature enhancement neck'''  #1.通过时序来补合key_frame的特征，2.通过空间加强来补key_frame的特征
        ##空间特征增强,五个帧一起看具有整体的信息
        
        pdb.set_trace()
        out=self.feature_enhance(out,motion_feat)

        

        '''逐步上采样'''
        out=self.fusion2(key_s[1],out)
        # pdb.set_trace()
        out=self.fusion3(key_s[0],out)
       




        out = self.head(out)    
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

if __name__=="__main__":
    model=SSTUNet().cuda().eval()
    a=torch.randn(1,5,512,512).cuda()
    b=model(a)
    print(b.shape)


    # print(123)
