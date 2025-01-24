from .models.yolo import SegmentationModel
import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
cfg='asf-yolo.yaml'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = .to(device)

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
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # self.num_frame = num_frame
        self.backbone = SegmentationModel(cfg, ch=1, nc=1) 


        #-----------------------------------------#
        #   尺度感知模块
        #-----------------------------------------#
        # self.neck = Motion_coupling_Neck(channels=[128], num_frame=num_frame)
        #----------------------------------------------------------#
        #   head
        self.head =_FCNHead(in_channels=256,channels=1,momentum=0.9)
        
        
        
     
        
        
    def forward(self, inputs): #4, 3, 5, 512, 512
        """[b,128,32,32][b,256,16,16][b,512,8,8]"""
        feat=self.backbone(inputs)
    
        feat=self.head(feat)
        out = F.interpolate(feat, scale_factor=8, mode='bilinear')  # down 4             # (4,1,480,480)
        #out=out.sigmoid()
        #out=torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))
        #pdb.set_trace()
        return out.sigmoid()
       
if __name__=='__main__':
    model=Network().cuda()
    a=torch.randn(1,1,640,640).cuda()
    pdb.set_trace( )
    pred=model(a)