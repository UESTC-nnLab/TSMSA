import argparse
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os
import torch.nn.functional as F
from mshf_loss import *
# from ASF_models.model  import  Network
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
parser.add_argument("--model_names", default=['proposed_triple'], type=list, 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")              
parser.add_argument("--dataset_names", default=['NUDTMIRSDT'], type=list, 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--dataset_dir", default='/home/pengshuang/Public/NUDT_MIRSDT/', type=str, help="train_dataset_dir")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch sizse")#16
parser.add_argument("--patchSize", type=int, default=512, help="Training patch size")
parser.add_argument("--save", default='./log', type=str, help="Save path of checkpoints")
parser.add_argument("--resume", default='', type=list, help="Resume from exisiting checkpoints (default: None)")#["/home/pengshuang/detect/BasicIRSTD-main/log/PRCV2024_1/ACM_129.pth.tar"]
parser.add_argument("--nEpochs", type=int, default=300, help="Number of epochs")
parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")
parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
parser.add_argument("--seed", type=int, default=42, help="Threshold for test")
parser.add_argument("--train_annotation_path", default='train_NUDTMIRSDT.txt', type=str, help="train_dataset_dir")
parser.add_argument("--val_annotation_path", default='test_NUDTMIRSDT.txt', type=str, help="train_dataset_dir")
# parser.add_argument('--multi-gpus', type=bool, default=False)

global optr
opt = parser.parse_args()

seed_pytorch(opt.seed)
miou = 0
pd = 0
metric = 0
best_metric = 0


def Batch_Augmentation1 (img, mask): 
    
    Random_coefficient = random.randint(0, img.shape[0]-1) 
        
    img_s = []
    mask_s = []
        
    for i in range(img.shape[0]):

        img_s.append(img[i:i+1])
        mask_s.append(mask[i:i+1])

    data_aug = []
    label_aug = []
        
    for i in range(img.shape[0]):
            
        if Random_coefficient > img.shape[0]//2-1:
        
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),3))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),3))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),3))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),3))
                    
        else:
                
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),2))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),2))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),2))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),2))

    data = torch.cat(data_aug, dim=0)
    label = torch.cat(label_aug, dim=0)

    img = F.interpolate(data, size=[512, 512]) 
    mask = F.interpolate(label, size=[512, 512]) 
        
    # data = torch.cat((img,data),0)
    # label = torch.cat((mask,label),0)

    return img, mask

def Batch_Augmentation2 (img, mask): 

    
    Random_coefficient = random.randint(0, img.shape[0]-1) 
        
    img_s = []
    mask_s = []
        
    for i in range(img.shape[0]):

        img_s.append(img[i:i+1])
        mask_s.append(mask[i:i+1])

    data_aug = []
    label_aug = []
        
    for i in range(img.shape[0]):
            
        if Random_coefficient > img.shape[0]//2-1:
        
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),3))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),3))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),3))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),3))
                    
        else:
                
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),2))
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),2))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),2))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),2))

    data = torch.cat(data_aug, dim=0)
    label = torch.cat(label_aug, dim=0)

    data = F.interpolate(data, size=[512, 512]) 
    label = F.interpolate(label, size=[512, 512]) 
        
    img = torch.cat((img,data),0)
    mask = torch.cat((mask,label),0)

    return img, mask


warm_epoch=5
model_path      = ""  #/home/pengshuang/detect/BasicIRSTD-main/log/PRCV2024_dataaugu2/ACM_180.pth.tar
device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
local_rank      = 0
def weights_init(net, init_type='normal', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
from seq_dataset import seqDataset,seqDataset_old
def Batch_Augmentation2 (img, mask): 

    #[4,1,512,512]
    h,w=img.shape[2],img.shape[3]
    Random_coefficient = random.randint(0, img.shape[0]-1) 
    #img: [4,1,509,655]  mask=np.array(mask_s[0].cpu()) mask=np.unique(mask) mask=mask*255 Image.fromarray(mask)
    img_s = []
    mask_s = []  # mask.save('1.png')
        
    for i in range(img.shape[0]): #第几个batch

        img_s.append(img[i:i+1])#[0,1] [1,2]  #[1,1,512,512]
        mask_s.append(mask[i:i+1])

    data_aug = []
    label_aug = []
    # pdb.set_trace()
    for i in range(img.shape[0]):
            
        if Random_coefficient > img.shape[0]//2-1:
        
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),3))#当前batch和后一batch进行拼接 在w上拼接，横着拼
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),3))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),3))#拼接最后一帧和第一帧，在w上拼接，横着拼 就是拼接相邻的
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),3))
                    
        else:
                
            if i < img.shape[0]-1:
                data_aug.append(torch.cat((img_s[i], img_s[i+1]),2))#当前batch和后一batch进行拼接 在H上拼接，竖着拼
                label_aug.append(torch.cat((mask_s[i], mask_s[i+1]),2))
            else:
                data_aug.append(torch.cat((img_s[img.shape[0]-1], img_s[0]),2))
                label_aug.append(torch.cat((mask_s[img.shape[0]-1], mask_s[0]),2))#拼接最后一个batch和第一个batch，在wH拼接，竖着拼
    # pdb.set_trace()
    data = torch.cat(data_aug, dim=0)
    label = torch.cat(label_aug, dim=0)
     
    data = F.interpolate(data, size=[h, w]) 
    label = F.interpolate(label, size=[h, w]) 
        
    img = torch.cat((img,data),0)
    mask = torch.cat((mask,label),0)

    return img, mask
def train():

    # img = Image.open(('/home/public/PRCV2024/' + '/images/' + "02234"+ '.png').replace('//','/')).convert('I')
    # print(img)
    train_annotation_path = opt.train_annotation_path#'train_MWIRSTD.txt'
    # val_annotation_path = 'test_MWIRSTD.txt'
    dataset_dir=opt.dataset_dir#"/home/pengshuang/Public/MWIRSTD/"
    input_shape=[512,512]
    train_set = seqDataset(dataset_dir,train_annotation_path, input_shape, 5, 'train',train_dataset_name=opt.dataset_name)
    
    # train_set = TrainSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.dataset_name, patch_size=opt.patchSize, img_norm_cfg=opt.img_norm_cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
   
    # net=Network().cuda()
    #device = torch.device('cuda')
    net = Net(model_name=opt.model_name, mode='train').cuda()

    # net = Net(model_name=opt.model_name, mode='train').cuda()
    #net = Net(model_name=opt.model_name, mode='train')
    # if opt.multi_gpus:
    #         if torch.cuda.device_count() > 1:
    #             print('use '+str(torch.cuda.device_count())+' gpus')
    #             net = nn.DataParallel(net, device_ids=[0, 1])
    # #net.to(device)

    net.train()
    if model_path != '':
        #------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        #------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        #------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        #------------------------------------------------------#
        model_dict      = net.state_dict()
        #pdb.set_trace()
        pretrained_dict = torch.load(model_path, map_location = device)['state_dict']
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        net.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    epoch_state = 0
    total_loss_list = []
    total_loss_epoch = []
    
    if opt.resume:
        for resume_pth in opt.resume:
            #pdb.set_trace()
            # if opt.dataset_name in resume_pth and opt.model_name in resume_pth:
            print("begin load")
            ckpt = torch.load(resume_pth)
            net.load_state_dict(ckpt['state_dict'])
            epoch_state = ckpt['epoch']
            total_loss_list = ckpt['total_loss']
            for i in range(len(opt.scheduler_settings['step'])):
                opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']
        print("resum load")
    
    ### Default settings  
    pdb.set_trace()              
    if opt.optimizer_name == 'Adam':
        opt.optimizer_settings = {'lr': 5e-4}
        opt.scheduler_name = 'MultiStepLR'
        opt.scheduler_settings = {'epochs':100, 'step': [50, 75], 'gamma': 0.1}  #200, 300
    
    ### Default settings of DNANet                
    if opt.optimizer_name == 'Adagrad':
        opt.optimizer_settings = {'lr': 0.05}
        opt.scheduler_name = 'CosineAnnealingLR'
        opt.scheduler_settings = {'epochs':1500, 'min_lr':1e-5}
        
    opt.nEpochs = opt.scheduler_settings['epochs']
    '''cvpr2024'''
    loss_fun=SLSIoULoss()
    self_down = nn.MaxPool2d(2, 2)
    '''end'''
    optimizer, scheduler = get_optimizer(net, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings, opt.scheduler_settings)
    print("begin training")
    best_metric=0
    for idx_epoch in range(epoch_state, opt.nEpochs):
        for idx_iter, (img, gt_mask) in enumerate(train_loader):
            # pdb.set_trace()
            img, gt_mask = Variable(img).cuda(), Variable(gt_mask[:,-1]).unsqueeze(1).cuda()
            if img.shape[0] == 1:
                continue

            img,gt_mask = Batch_Augmentation2(img, gt_mask)#gt_mask
            #img=img.repeat(1,3,1,1)
            # masks,pred= net.forward(img,tag)
            ''' 原本的loss'''
            pred = net.forward(img)
            loss = net.loss(pred, gt_mask)
            '''end'''
            
            
            '''CVPR的loss'''
            # tag =False
            # if idx_epoch>warm_epoch:
            #     tag = True
            #     #pdb.set_trace()
            #     masks,pred= net.forward(img,tag)
            # else:
            #     tag = False
            #     pred= net.forward(img,tag)
            #     masks=[]
            # loss = 0
 
            # loss = loss + loss_fun(pred, gt_mask, warm_epoch, idx_epoch)
            # for j in range(len(masks)):
            #     if j>0:
            #         gt_mask = self_down(gt_mask)
            #     loss = loss + loss_fun(masks[j], gt_mask, warm_epoch, idx_epoch)
                
            # # loss = loss / (len(gt_mask)+1)
            '''end'''
            total_loss_epoch.append(loss.detach().cpu())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        if (idx_epoch + 1) % 1 == 0:
            total_loss_list.append(float(np.array(total_loss_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' 
                  % (idx_epoch + 1, total_loss_list[-1]))
            total_loss_epoch = []
            
        if (idx_epoch + 1) % 1 == 0: #50
            save_pth = opt.save + '/' + opt.dataset_name +'_'+ opt.model_name+ '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            best_metric=test(save_pth,best_metric)
            
        if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 1 != 0: #50
            save_pth = opt.save + '/' + opt.dataset_name +'_'+ opt.model_name+ '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
            save_checkpoint({
                'epoch': idx_epoch + 1,
                'state_dict': net.state_dict(),
                'total_loss': total_loss_list,
                }, save_pth)
            best_metric=test(save_pth,best_metric)
        # if (idx_epoch + 1) % 5 == 0:
        #     total_loss_list.append(float(np.array(total_loss_epoch).mean()))
        #     print(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,' 
        #           % (idx_epoch + 1, total_loss_list[-1]))
        #     opt.f.write(time.ctime()[4:-5] + ' Epoch---%d, total_loss---%f,\n' 
        #           % (idx_epoch + 1, total_loss_list[-1]))
        #     total_loss_epoch = []
            
        # if (idx_epoch + 1) % 2 == 0:
        #     save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
        #     save_checkpoint({
        #         'epoch': idx_epoch + 1,
        #         'state_dict': net.state_dict(),
        #         'total_loss': total_loss_list,
        #         }, save_pth)
        #     test(save_pth)
            
        # if (idx_epoch + 1) == opt.nEpochs and (idx_epoch + 1) % 50 != 0:
        #     save_pth = opt.save + '/' + opt.dataset_name + '/' + opt.model_name + '_' + str(idx_epoch + 1) + '.pth.tar'
        #     save_checkpoint({
        #         'epoch': idx_epoch + 1,
        #         'state_dict': net.state_dict(),
        #         'total_loss': total_loss_list,
        #         }, save_pth)
        #     test(save_pth)
            
from seq_dataset import seqDatasetval
def test(save_pth,best_metric):
    
    val_annotation_path = opt.val_annotation_path#'test_MWIRSTD.txt'
    dataset_dir=opt.dataset_dir#"/home/pengshuang/Public/MWIRSTD/"
    input_shape=[512,512]
    # test_set = TestSetLoader(opt.dataset_dir, opt.dataset_name, opt.dataset_name, img_norm_cfg=opt.img_norm_cfg)
    # test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    test_set = seqDatasetval(dataset_dir,val_annotation_path, input_shape, 5, 'val',train_dataset_name=opt.dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    # net = Network().cuda()#Net(model_name=opt.model_name, mode='test').cuda()
    net = Net(model_name=opt.model_name, mode='test').cuda()
    # net = torch.nn.DataParallel(net,device_ids=[0,1]).cuda()
    ckpt = torch.load(save_pth)
    net.load_state_dict(ckpt['state_dict'])
    net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size) in enumerate(test_loader):
            # pdb.set_trace()
            img = Variable(img).cuda()
            #img=img.repeat(1,3,1,1)
            pred = net.forward(img)
            pred = pred[:,:,:size[0],:size[1]]
            gt_mask=(gt_mask[:,-1]).unsqueeze(1)#.cuda()
            gt_mask = gt_mask[:,:,:size[0],:size[1]]
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)     
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    miou = results1[1]
    pd = results2[0]
    metric = 0.5 * (miou + pd)
    if metric >best_metric:
        best_metric=metric 
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    print("metic:\t" + str(metric))
    print("best metic:\t" + str(best_metric))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')
    opt.f.write("metic:\t" + str(metric) + '\n')
    opt.f.write("best_metic:\t" + str(best_metric) + '\n')
    return best_metric
def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path

if __name__ == '__main__':
    for dataset_name in opt.dataset_names:
        opt.dataset_name = dataset_name
        for model_name in opt.model_names:
            opt.model_name = model_name
            if not os.path.exists(opt.save):
                os.makedirs(opt.save)
            if not os.path.exists(os.path.dirname(opt.save + '/' + opt.dataset_name + '_' + opt.model_name+'/'+ opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt')):
                os.makedirs(os.path.dirname(opt.save + '/' + opt.dataset_name + '_' + opt.model_name+'/'+ opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt'))
            opt.f = open(opt.save + '/' + opt.dataset_name + '_' + opt.model_name+ '/' + opt.dataset_name + '_' + opt.model_name + '_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
            print(opt.dataset_name + '\t' + opt.model_name)
            train()
            print('\n')
            opt.f.close()
