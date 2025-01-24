import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time
import pdb
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F
        pred=pred[:,:,:size[0],:size[1]]
        #pdb.set_trace()
        gt_mask = gt_mask[:,:,:size[0],:size[1]]
      
        eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
       
        eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)   
        
        ### save img
        if opt.save_img == True:
            img_save = transforms.ToPILImage()((pred[0,0,:,:]).cpu())
            if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
            img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')  
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')