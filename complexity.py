from thop import profile
import torch
# from net import LightWeightNetwork
from model.ACM.model_ACM import ASKCResUNet as ACM
# from model.ACM.model_ACM import ASKCResUNet as ACM
from model.ACM.model_ALCnet import ASKCResNetFPN as ALCNet
from model.SCTransNet.SCTransNet import SCTransNet as SCTransNet
from model.Proposed.proposed_triple import SSTUNet as SSTUNet
from model.AGPCNet import  get_segmentation_model as AGPCNet
from model.DNANet.model_DNANet import DNANet
from model.RDIAN.model_RDIAN import RDIAN as RDIAN
from model.Unet.model_Unet import U_Net

if __name__ == '__main__':

    input_img = torch.rand(1,1,512,512).cuda()
    net = U_Net().cuda() #LightWeightNetwork().cuda() #AGPCNet('agpcnet_1')
    flops, params = profile(net, inputs=(input_img, ))
    print('Params: %2fM' % (params/1e6))
    print('FLOPs: %2fGFLOPs' % (flops/1e9))

