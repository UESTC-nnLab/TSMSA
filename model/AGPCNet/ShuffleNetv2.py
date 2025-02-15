import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from collections import OrderedDict

def conv3x3(in_channels,out_channels,stride=1,padding=1,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=padding,bias=bias,groups=groups)
def conv1x1(in_channels,out_channels,bias=True,groups=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=bias,groups=groups)
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x
def channel_split(x,splits=[24,24]):
    return torch.split(x,splits,dim=1)

class ParimaryModule(nn.Module):
    def __init__(self,in_channels=3,out_channels=24):
        super(ParimaryModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.ParimaryModule = nn.Sequential(
                                            OrderedDict(
                                                        [
                                                         ('ParimaryConv',conv3x3(in_channels,out_channels,1,1,True,1)),
                                                         ('ParimaryBN',nn.BatchNorm2d(out_channels)),
                                                         ('ParimaryMaxPool',nn.MaxPool2d(kernel_size=3,stride=1,padding=1)) #1
                                                        ]
                                                        )
                                            )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self,x):
        x = self.ParimaryModule(x)
        return x

class FinalModule(nn.Module):
    def __init__(self,in_channels=464,out_channels=1024,num_classes=1000):
        super(FinalModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        
        self.FinalConv = nn.Sequential(
                                       OrderedDict(
                                                   [
                                                    ('FinalConv',conv1x1(in_channels,out_channels,True,1)),
                                                    ('FinalConvBN',nn.BatchNorm2d(out_channels)),
                                                    ('FinalConvReLU',nn.ReLU())
                                                   ]
                                                   )
                                       )
        self.FC = nn.Sequential(
                                OrderedDict(
                                            [
                                             ('FC',conv1x1(out_channels,num_classes,True,1))
                                            ]
                                            )
                                )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self,x):
        x = self.FinalConv(x)
        x = F.avg_pool2d(x, x.data.size()[-2:])
        x = self.FC(x)
        return x

class ShuffleNetV2Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,splits_left=2):
        super(ShuffleNetV2Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.splits_left = splits_left
        
        if stride==2:
            self.Left = nn.Sequential(
                                      OrderedDict(
                                                  [
                                                   ('DepthwiseConv3x3',conv3x3(in_channels,in_channels,stride,1,True,in_channels)),
                                                   ('DepthwiseConv3x3BN',nn.BatchNorm2d(in_channels)),
                                                   ('UnCompressConv1x1',conv1x1(in_channels,out_channels//2,True,1)),
                                                   ('UnCompressConv1x1BN',nn.BatchNorm2d(out_channels//2)),
                                                   ('UnCompressConv1x1ReLU',nn.ReLU())
                                                  ]
                                                  )
                                      )
            self.Right = nn.Sequential(
                                       OrderedDict(
                                                   [
                                                    ('NoCompressConv1x1',conv1x1(in_channels,in_channels,True,1)),
                                                    ('NoCompressConv1x1BN',nn.BatchNorm2d(in_channels)),
                                                    ('NoCompressConv1x1ReLU',nn.ReLU()),
                                                    ('DepthwiseConv3x3',conv3x3(in_channels,in_channels,stride,1,True,in_channels)),
                                                    ('DepthwiseConv3x3BN',nn.BatchNorm2d(in_channels)),
                                                    ('UnCompressConv1x1',conv1x1(in_channels,out_channels//2,True,1)),
                                                    ('UnCompressConv1x1BN',nn.BatchNorm2d(out_channels//2)),
                                                    ('UnCompressConv1x1ReLU',nn.ReLU())
                                                   ]
                                                   )
                                       )
        elif stride==1:
            in_channels = in_channels - in_channels//splits_left
            self.Right = nn.Sequential(
                                       OrderedDict(
                                                   [
                                                    ('NoCompressConv1x1',conv1x1(in_channels,in_channels,True,1)),
                                                    ('NoCompressConv1x1BN',nn.BatchNorm2d(in_channels)),
                                                    ('NoCompressConv1x1ReLU',nn.ReLU()),
                                                    ('DepthwiseConv3x3',conv3x3(in_channels,in_channels,stride,1,True,in_channels)),
                                                    ('DepthwiseConv3x3BN',nn.BatchNorm2d(in_channels)),
                                                    ('UnCompressConv1x1',conv1x1(in_channels,in_channels,True,1)),
                                                    ('UnCompressConv1x1BN',nn.BatchNorm2d(in_channels)),
                                                    ('UnCompressConv1x1ReLU',nn.ReLU())
                                                   ]
                                                   )
                                       )
        else:
            raise ValueError('stride must be 1 or 2')
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self,x):
        if self.stride==2:
            x_left,x_right = x,x
            x_left = self.Left(x_left)
            x_right = self.Right(x_right)
        elif self.stride==1:
            x_split = channel_split(x,[self.in_channels//self.splits_left,self.in_channels-self.in_channels//self.splits_left])
            x_left,x_right = x_split[0],x_split[1]
            x_right = self.Right(x_right)
        
        x = torch.cat((x_left,x_right),dim=1)
        x = channel_shuffle(x,2)
        return x

class ShuffleNetV2(nn.Module):
    def __init__(self,in_channels=3,num_classes=1000,net_scale=1.0,splits_left=2):
        super(ShuffleNetV2, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net_scale = net_scale
        self.splits_left = splits_left
        
        if net_scale==0.5:
            self.out_channels = [24,48,96,192,1024]
        elif net_scale==1.0:
            self.out_channels = [24,116,232,464,1024]
        elif net_scale==1.5:
            self.out_channels = [24,176,352,704,1024]
        elif net_scale==2.0:
            self.out_channels = [24,244,488,976,2048]
        else:
            raise ValueError('net_scale must be 0.5,1.0,1.5 or 2.0')
        
        self.ParimaryModule = ParimaryModule(in_channels,self.out_channels[0])
        
        self.Stage1 = self.Stage(1,[1,3])
        self.Stage2 = self.Stage(2,[1,7])
        self.Stage3 = self.Stage(3,[1,3])
        
        self.FinalModule = FinalModule(self.out_channels[3],self.out_channels[4],num_classes)
        
    def Stage(self,stage=1,BlockRepeat=[1,3]):
        modules = OrderedDict()
        name = 'ShuffleNetV2Stage{}'.format(stage)
        
        if BlockRepeat[0]==1:
            modules[name+'_0'] = ShuffleNetV2Block(self.out_channels[stage-1],self.out_channels[stage],2,self.splits_left)
        else:
            raise ValueError('stage first block must only repeat 1 time')
        
        for i in range(BlockRepeat[1]):
            modules[name+'_{}'.format(i+1)] = ShuffleNetV2Block(self.out_channels[stage],self.out_channels[stage],1,self.splits_left)
        
        return nn.Sequential(modules)
    
    def forward(self,x):
        x = self.ParimaryModule(x)
        # print(x.shape)
        stage1 = self.Stage1(x)
        # print(stage1.shape)
        stage2 = self.Stage2(stage1)
        # print(stage2.shape)
        stage3 = self.Stage3(stage2)
        # print(stage3.shape)
        # x = self.FinalModule(x)
        # print(x.shape)
        return stage1,stage2,stage3
        
        
if __name__=='__main__':
    net = ShuffleNetV2(3,1,1.0)
    print(net)
    input = torch.randn(1,3,512,512)
    output1,output2,output3 = net(input)
    # 测试网络结构和前向传播
    #print(output)
    
    params = list(net.parameters())
    num = 0
    for i in params:
        l=1
        #print('Size:{}'.format(list(i.size())))
        for j in i.size():
            l *= j
        num += l
    print('All Parameters:{}'.format(num))
    
    torch.save(net.state_dict(),'ShuffleNetV2.pth')