from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
# from model import ACM
# from skimage.feature.tests.test_orb import img
# from model.ACM.model_ACM_AGPC_res1_cpam import ASKCResUNet as ACM
from model.ACM.model_ACM import ASKCResUNet as ACM
from model.ACM.model_ACM_cvpr import ASKCResUNet as ours
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from model.SA_nets.SANet import SANet
from model.UCF.UCF import UCFNet
from model.AGPCNet import get_segmentation_model
from model.nets.Network import Network
from model.DNANet.model_DNANet import DNANet
from model.Unet.model_Unet import U_Net
from model.ACM.model_ALCnet import  ASKCResNetFPN as ALCNet
from model.SCTransNet.SCTransNet import SCTransNet as SCTransNet
from model.Proposed.proposed_triple import SSTUNet as SSTUNet
from model.RISTDnet.model_RISTDnet import RISTDnet as RISTDnet
from model.RDIAN.model_RDIAN import RDIAN as RDIAN
from model.RPCANet.RPCANet import  RPCANet as RPCANet
from model.MSHNet.MSH import MSHNet as MSHNet
from model.SA_nets.SANet import SANet as SANet
# from ASF_models.model import Network as ASFYolo
class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        #pdb.set_trace()
        self.cal_loss = FocalIoULoss()#SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        elif model_name == 'DNANet_BY':
            if mode == 'train':
                self.model = DNAnet_BY(mode='train')
            else:
                self.model = DNAnet_BY(mode='test')  
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ours':
            self.model=ours()
        elif model_name== 'SSTNet':
            self.model=Network(num_classes=1, num_frame=5)#num_classes=1, num_frame=5 ASFYolo()#
        elif model_name=='SCTransNet':
            self.model=SCTransNet()
        elif model_name=='proposed_triple':
            self.model=SSTUNet()
        elif model_name=='UCF':
            self.model=UCFNet(theta_r=0, theta_0=0.7, theta_1=0, theta_2=0.7, n_blocks=7)
        elif model_name =='SANet':
            self.model=SANet(0.33,0.5)
        elif model_name=='AGPCNet':
            self.model=get_segmentation_model('agpcnet_1')
            
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            if mode == 'train':
                self.model = ISNet(mode='train')
            else:
                self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'UNet':
            self.model = U_Net()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()
        elif model_name=="RPCANet":
            self.model=RPCANet()
        elif model_name=="MSHNet":
            self.model=MSHNet()
        elif model_name=="SANet":
            self.model=SANet()
    def forward(self, img,train=False):
        return self.model(img)#train

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
