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
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['ACM'], type=list, 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")#['ACM', 'ALCNet','DNANet', 'ISNet', 'RDIAN', 'ISTDU-Net']
parser.add_argument("--pth_dirs", default=['PRCV2024_old/ACM_230.pth.tar'], type=list, help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='/home/public/', type=str, help="train_dataset_dir") #'NUAA-SIRST/ACM_200.pth.tar'
parser.add_argument("--dataset_names", default=['PRCV2024'], type=list, 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")#, 'NUDT-SIRST', 'IRSTD-1K'
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.25)

def downsample_if_needed(img, size_limit=512):
    """如果图像尺寸超过限制，进行下采样"""
    _,_,h, w = img.shape
    if max(h, w) > size_limit:
        scale_factor = size_limit / max(h, w)
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        img=F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        #img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        return img, h,w
    else:
        return img, h,w

global opt
opt = parser.parse_args()

# def slice_tensor(tensor, slice_size=(256, 256), stride=128):
#     """
#     对Tensor图像进行切片处理。
#     """
#     width, height = tensor.shape[-2:]
#     slices = []
#     for y in range(0, height, stride):
#         for x in range(0, width, stride):
#             end_x = min(x + slice_size[0], width)
#             end_y = min(y + slice_size[1], height)
#             slices.append(tensor[..., y:end_y, x:end_x])
#     return slices

# def merge_tensor_slices(sliced_tensors, original_shape, slice_size=(256, 256), stride=128):
#     """
#     合并切片后的Tensor预测结果。
#     """
#     merged_tensor = torch.zeros(original_shape, device=tensor.device)
#     width, height = original_shape[-2:]
#     num_cols = (width - slice_size[0]) // stride + 1
#     num_rows = (height - slice_size[1]) // stride + 1
    
#     for row in range(num_rows):
#         for col in range(num_cols):
#             start_x = col * stride
#             start_y = row * stride
#             end_x = min(start_x + slice_size[0], width)
#             end_y = min(start_y + slice_size[1], height)
#             merged_tensor[..., start_y:end_y, start_x:end_x] += sliced_tensors[row * num_cols + col]
    
#     # 如果需要平均处理重叠区域，可以在这里添加相应逻辑
#     # merged_tensor /= ...  # 根据重叠次数进行归一化
    
#     return merged_tensor
def test(): 
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    net = Net(model_name=opt.model_name, mode='test').cuda()
    net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    tta=True
    for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
        #img, h,w = downsample_if_needed(img)
        img = Variable(img).cuda()
        pred = net.forward(img)
        #pred=F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False) 
        #img_slices = slice_tensor(img, slice_size=(256, 256), stride=128)
        
        #all_pred_slices = []
        # for slice in img_slices:
        #     # 确保slice在GPU上进行推理
        #     slice = slice.cuda()
        #     pred_slice = net(slice.unsqueeze(0))  # 添加batch维度
        #     all_pred_slices.append(pred_slice.squeeze(0))  # 移除添加的batch维度
        
        # 合并切片预测结果
        #pred = merge_tensor_slices(all_pred_slices, img.shape)
        #pred = pred[:,:,:size[0],:size[1]]
        # if tta:
        #         #x,y,xy flips as TTA
        #         flips = [[-1],[-2],[-2,-1]]
        #         for f in flips:
        #             img = torch.flip(img,f)
                    
        #             y_preds = net.forward(img)
        #             y_preds = torch.flip(y_preds,f)
        #             #y_pred = y_pred[:,:,:size[0],:size[1]]
        #             pred+=y_preds
        # pred=pred/(1+len(flips))
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

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_Sun_Jun__2_16_56_49_2024.txt','w')#'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    #pdb.set_trace()
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        #pdb.set_trace()
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = dataset_name #pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
        
