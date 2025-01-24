import cv2
import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import xml.etree.ElementTree as ET
import time
import torch
from utils import *
# convert to RGB
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 
    

# normalization
def preprocess(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

def rand(a=0, b=1):
        return np.random.rand()*(b-a) + a

def augmentation(images, boxes,h, w, hue=.1, sat=0.7, val=0.4):
    # images [5, w, h, 3], bbox [:,4]
    #------------------------------------------#
    #   翻转图像
    #------------------------------------------#
    filp = rand()<.5
    if filp:
        for i in range(len(images)):
            images[i] = Image.fromarray(images[i].astype('uint8')).convert('RGB').transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        for i in range(len(boxes)):
            boxes[i][[0,2]] = w - boxes[i][[2,0]]

    images      = np.array(images, np.uint8)
    #---------------------------------#
    #   对图像进行色域变换
    #   计算色域变换的参数
    #---------------------------------#
    r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #---------------------------------#
    #   将图像转到HSV上
    #---------------------------------#
    for i in range(len(images)):
        hue, sat, val   = cv2.split(cv2.cvtColor(images[i], cv2.COLOR_RGB2HSV))
        dtype           = images[i].dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        images[i] = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        images[i] = cv2.cvtColor(images[i], cv2.COLOR_HSV2RGB)

    return np.array(images,dtype=np.float32), np.array(boxes,dtype=np.float32)


import pdb
import os
def copy_paste_augmentation_seq(image, mask, target_threshold, scales,pasty_y,pastx_x,max_target_area):
    """
    对输入图像进行copy-paste增强。
    
    参数:
    - image: 输入图像
    - mask: 目标掩码
    - target_threshold: 目标数量阈值，小于该值时进行copy-paste
    - scale_range: 缩放范围

    返回:
    - augmented_image: 增强后的图像
    - augmented_mask: 增强后的掩码
    """
    augmented_image = image.copy()
    augmented_mask = mask.copy()
    h,w=image.shape
    # pdb.set_trace()
    # 找到所有目标的连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    
    if num_labels-1==0:
        return augmented_image, augmented_mask,max_target_area #,previous_x,previous_y
    for num in range(target_threshold-1):  # 复制target_threshold-1次
        scale=scales[num]
        for target_idx in range(1, num_labels):
            ys, xs = np.where(labels == target_idx)
            min_ys=min(ys)
            max_ys=max(ys)
            min_xs=min(xs)
            max_xs=max(xs)
            target_image_patch = image[min_ys:max_ys+1, min_xs:max_xs+1]
            target_mask_patch = mask[min_ys:max_ys+1, min_xs:max_xs+1]
            paste_y=pasty_y[num]+min_ys
            paste_x=pastx_x[num]+min_xs
            # pdb.set_trace()
            #确保粘贴不会超出边界 最大 h-max_target_area  y-max_target_area   最小：0，0
            paste_y=max(min(paste_y,h-2*max_target_area),0)
            paste_x=max(min(paste_x,w-2*max_target_area),0)
            if max(target_mask_patch.shape[0],target_mask_patch.shape[1])>max_target_area:
                max_target_area=max(target_mask_patch.shape[0],target_mask_patch.shape[1])
            ##随机缩放
            target_image_patch = cv2.resize(target_image_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            target_mask_patch = cv2.resize(target_mask_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            patch_ys, patch_xs = np.where(target_mask_patch > 0)
            for dy, dx in zip(patch_ys, patch_xs):
            # pdb.set_trace()
                augmented_image[paste_y + dy, paste_x + dx] = target_image_patch[dy, dx]
                augmented_mask[paste_y + dy, paste_x + dx] = target_mask_patch[dy, dx]

        

    return augmented_image,augmented_mask,max_target_area#,current_x,current_y
class seqDataset(Dataset):
    def __init__(self, dataset_dir,dataset_txt, image_size, num_frame=5 ,type='train',img_norm_cfg=None,train_dataset_name='MWIRSTD'):
        super(seqDataset, self).__init__()
        self.dataset_path = dataset_dir
        self.img_idx = []
        self.anno_idx = []
        self.image_size = image_size
        self.num_frame = num_frame
        self.max_target_area=32
        self.train_dataset_name=train_dataset_name
        # pdb.set_trace()
        if type == 'train':
            self.txt_path = dataset_txt
            self.aug = True
        else:
            self.txt_path = dataset_txt
            self.aug = False
        with open(self.txt_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
                # self.anno_idx.append(np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]]))
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
            # pdb.set_trace()
        else:
            self.img_norm_cfg = img_norm_cfg
        # pdb.set_trace()
        # for i in range(0,1052):
        #     print(i)
        #     images, masks = self.get_data(i)
        # pdb.set_trace()
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        images, masks = self.get_data(index)
       
        #images = np.transpose(images,(3, 0, 1, 2)) #preprocess(

        # if len(box) != 0:
        #     box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
        #     box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        return images, masks
    
    def get_data(self, index):
        # origin=(509,655)
        # pad=(512,672)
        '''random crop time'''
        h_start=None
        w_start=None
        ''''''

        image_data = []
        mask_data=[]
        h, w = self.image_size[0], self.image_size[1]
        file_name = self.img_idx[index]
        # pdb.set_trace()
        # seq_num=file_name.split("-")[0]
        # image_id =file_name.split("-")[1]
        # image_path=os.path.join(self.dataset_path,seq_num)
        
        image_id = int(file_name.split("/")[-1][:-4])
        image_path =file_name.replace(file_name.split("/")[-1], '')
        mask_path=image_path.replace(image_path.split("/")[0], 'masks')
        image_dir=os.path.join(self.dataset_path,image_path)
        mask_dir=os.path.join(self.dataset_path,mask_path)
        target_threshold=4
        scale_range=(0.6, 0.9)
        scales=[]
        pasty_y=[]
        pastx_x=[]
        # txt_output_path = 'img_pad.txt'
        # with open(txt_output_path, 'a') as txt_file:
        for target in range(0,target_threshold-1):
            # paste_y =random.randint(-50,50)#random.randint(0, h/2- max_target_area)
            # paste_x =random.randint(-50,50)#random.randint(0, w/2-max_target_area)
            paste_y =random.randint(int(-h/4+self.max_target_area),int(h/4- self.max_target_area))#random.randint(0, h/2- max_target_area)
            paste_x =random.randint(int(-w/4+self.max_target_area),int(w/4- self.max_target_area))#random.randint(0, w/2-max_target_area)
            scale = random.uniform(scale_range[0], scale_range[1])
            scales.append(scale)
            pasty_y.append(paste_y)
            pastx_x.append(paste_x)
        for id in range(self.num_frame-1,-1,-1): #[0,1,2,3,4]
            try:
                img = Image.open(image_dir +'%d.bmp' % max(image_id - id, 0)).convert('I').resize((512,512), resample=Image.BILINEAR)
            except:
                try:
                   img = Image.open(image_dir +'%d.jpg' % max(image_id - id, 0)).convert('I').resize((512,512), resample=Image.BILINEAR)
                except:
                    # print('img_pth',img_pth)
                    # pdb.set_trace()
                    img = Image.open(image_dir +'%d.png' % max(image_id - id, 0)).convert('I').resize((512,512), resample=Image.BILINEAR)
                
            mask=Image.open(mask_dir +'%d.png' % max(image_id - id, 0)).resize((512,512), resample=Image.NEAREST)
            img,mask,self.max_target_area= copy_paste_augmentation_seq(np.array(img).astype(np.uint8), np.array(mask).astype(np.uint8),target_threshold,scales,pasty_y,pastx_x,self.max_target_area)
            
            # print("填充前",img.size)
            
            


            
            img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
            mask = np.array(mask, dtype=np.float32)  / 255.0
            
            # if img.shape!=origin:
            #     pdb.set_trace()
            #     print("填充前",img.shape)
            #     print(str(image_dir +'%d.jpg' % max(image_id - id, 0)))
            #     txt_file.write("填充前"+'/'+str(img.shape) + '\n')
            #     txt_file.write("填充前"+'/'+str(image_dir +'%d.jpg' % max(image_id - id, 0)) + '\n')
            # # if mask.shape!=origin:
            #     pdb.set_trace()
            #     print("填充前",mask.shape)
            #     print(str(image_dir +'%d.jpg' % max(image_id - id, 0)))
            #     txt_file.write("填充前"+'/'+str(mask.shape) + '\n')
            #     txt_file.write("填充前"+'/'+str(image_dir +'%d.jpg' % max(image_id - id, 0)) + '\n')


            if len(mask.shape) > 2:
                mask = mask[:,:,0]
            ''''random crop time'''
            # if h_start ==None or w_start ==None:
            #     img,mask,h_start,w_start=random_crop_time(img,mask,patch_size=512, pos_prob=0.5)
            # else:
            #     img,mask,h_start,w_start=crop(img,mask,patch_size=512,h_start=h_start,w_start=w_start)

            ''' end'''
            '''pad img'''
            if self.train_dataset_name=='MWIRSTD':
                img = PadImg(img)
                # print("填充后",img.shape)
                # txt_file.write("填充后"+'/'+str(img.shape) + '\n')
                mask = PadImg(mask)
            elif self.train_dataset_name=='IRDST':
                img=pad_image_to_max_dimensions(img, max_height= 768,max_width= 992)
                mask=pad_image_to_max_dimensions(mask, max_height=768,max_width= 992)
            
            else:
                img = PadImg(img)
                # print("填充后",img.shape)
                # txt_file.write("填充后"+'/'+str(img.shape) + '\n')
                mask = PadImg(mask)
            '''end'''
            # if img.shape!=pad:
            #     pdb.set_trace()
            #     print("填充后",img.shape)
            #     print(str(image_dir +'%d.jpg' % max(image_id - id, 0)))

            #     txt_file.write("填充后"+'/'+str(img.shape) + '\n')
            #     txt_file.write("填充后"+'/'+str(image_dir +'%d.jpg' % max(image_id - id, 0)) + '\n')

            # if mask.shape!=pad:
            #     pdb.set_trace()
            #     print("填充后",mask.shape)
            #     print(str(image_dir +'%d.jpg' % max(image_id - id, 0)))
            #     txt_file.write("填充后"+'/'+str(mask.shape) + '\n')
            #     txt_file.write("填充后"+'/'+str(image_dir +'%d.jpg' % max(image_id - id, 0)) + '\n')

            #数据增强
            #img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            # img_patch, mask_patch = self.tranform(img_patch, mask_patch)如何同时旋转，同时裁剪
            img_patch, mask_patch = img, mask
            # img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
            image_data.append(img_patch)
            mask_data.append(mask_patch)


        # pdb.set_trace()    
        # image_data=image_data[::-1]       
        # mask_data=mask_data[::-1] 
        image_data=torch.stack(image_data)
        mask_data=torch.stack(mask_data)
        # image_data = np.array(image_data[::-1]) # 关键帧在后 # [5,w,h,3]
        # mask_data = np.array(mask_data[::-1]) # [:,5]
        if self.aug is True:
            # image_data, label_data[:,:4] = augmentation(image_data,label_data[:,:4],h,w)
            pass
        return image_data, mask_data
def copy_paste_augmentation(image, mask, target_threshold=2, scale_range=(0.6, 0.9)):
    """
    对输入图像进行copy-paste增强。
    
    参数:
    - image: 输入图像
    - mask: 目标掩码
    - target_threshold: 目标数量阈值，小于该值时进行copy-paste
    - scale_range: 缩放范围

    返回:
    - augmented_image: 增强后的图像
    - augmented_mask: 增强后的掩码
    """
    augmented_image = image.copy()
    augmented_mask = mask.copy()
    
    # 找到所有目标的连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels-1 == 0:
        return augmented_image, augmented_mask
    # 如果目标数量小于 target_threshold，则复制某个目标两次
    if num_labels - 1 < target_threshold:
        target_idx = random.choice(range(1, num_labels))
        # pdb.set_trace()
        for _ in range(target_threshold):  # 复制两次
            # 提取目标块
            ys, xs = np.where(labels == target_idx)
            target_image_patch = image[min(ys):max(ys)+1, min(xs):max(xs)+1]
            target_mask_patch = mask[min(ys):max(ys)+1, min(xs):max(xs)+1]
            
            # 随机缩放
            scale = random.uniform(scale_range[0], scale_range[1])
            target_image_patch = cv2.resize(target_image_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            target_mask_patch = cv2.resize(target_mask_patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
            # 随机粘贴位置
            paste_y = random.randint(0, image.shape[0] - target_image_patch.shape[0])
            paste_x = random.randint(0, image.shape[1] - target_image_patch.shape[1])
            
            # 粘贴目标块
            patch_ys, patch_xs = np.where(target_mask_patch > 0)
            for dy, dx in zip(patch_ys, patch_xs):
                # pdb.set_trace()
                augmented_image[paste_y + dy, paste_x + dx] = target_image_patch[dy, dx]
                augmented_mask[paste_y + dy, paste_x + dx] = target_mask_patch[dy, dx]
    
    return augmented_image, augmented_mask
class seqDatasetval(Dataset):
    def __init__(self, dataset_dir,dataset_txt, image_size, num_frame=5 ,type='train',img_norm_cfg=None,train_dataset_name='MWIRSTD'):
        super(seqDatasetval, self).__init__()
        self.dataset_path = dataset_dir
        self.img_idx = []
        self.anno_idx = []
        self.image_size = image_size
        self.num_frame = num_frame
        if type == 'train':
            self.txt_path = dataset_txt
            self.aug = True
        else:
            self.txt_path = dataset_txt
            self.aug = False
        with open(self.txt_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
                # self.anno_idx.append(np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]]))
        if img_norm_cfg == None:
            '''resize到512'''
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir) 
        else:
            self.img_norm_cfg = img_norm_cfg
       
        # for i in range(1053):
        #     images, masks = self.get_data(0)
        # pdb.set_trace()
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        images, masks,[h,w] = self.get_data(index)
       
        #images = np.transpose(images,(3, 0, 1, 2)) #preprocess(

        # if len(box) != 0:
        #     box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
        #     box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        return images, masks,[h,w]
    
    def get_data(self, index):
        image_data = []
        mask_data=[]
        h, w = self.image_size[0], self.image_size[1]
        # print("宽高")
        # print(h,w)
        file_name = self.img_idx[index]
        # seq_num=file_name.split("-")[0]
        # image_id =file_name.split("-")[1]
        # image_path=os.path.join(self.dataset_path,seq_num)
        
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        mask_path=image_path.replace(image_path.split("/")[0], 'masks')
        image_dir=os.path.join(self.dataset_path,image_path)
        mask_dir=os.path.join(self.dataset_path,mask_path)
        for id in range(0, self.num_frame):

            try:
                img = Image.open(image_dir +'%d.bmp' % max(image_id - id, 0)).convert('I').resize((512,512), resample=Image.BILINEAR)
            except:
                try:
                   img = Image.open(image_dir +'%d.jpg' % max(image_id - id, 0)).convert('I').resize((512,512), resample=Image.BILINEAR)
                except:
                    # print('img_pth',img_pth)
                    # pdb.set_trace()
                    img = Image.open(image_dir +'%d.png' % max(image_id - id, 0)).convert('I').resize((512,512), resample=Image.BILINEAR)
     
            mask=Image.open(mask_dir +'%d.png' % max(image_id - id, 0)).resize((512,512), resample=Image.NEAREST)
            img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
            # print("填充前",img.shape)
            mask = np.array(mask, dtype=np.float32)  / 255.0
            if len(mask.shape) > 2:
                mask = mask[:,:,0]
            img = PadImg(img)
            # print("填充后",img.shape)
            mask = PadImg(mask)
            #数据增强
            # img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            # img_patch, mask_patch = self.tranform(img_patch, mask_patch)如何同时旋转，同时裁剪
            img_patch, mask_patch = img, mask
            # img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
            image_data.append(img_patch)
            mask_data.append(mask_patch)


        # pdb.set_trace()    
        image_data=image_data[::-1]       
        mask_data=mask_data[::-1] 
        image_data=torch.stack(image_data)
        mask_data=torch.stack(mask_data)
        # image_data = np.array(image_data[::-1]) # 关键帧在后 # [5,w,h,3]
        # mask_data = np.array(mask_data[::-1]) # [:,5]
        if self.aug is True:
            # image_data, label_data[:,:4] = augmentation(image_data,label_data[:,:4],h,w)
            pass
        return image_data, mask_data,[h,w]

class seqDataset_old(Dataset):
    def __init__(self, dataset_dir,dataset_txt, image_size, num_frame=5 ,type='train',img_norm_cfg=None,train_dataset_name='MWIRSTD'):
        super(seqDataset_old, self).__init__()
        self.dataset_path = dataset_dir
        self.img_idx = []
        self.anno_idx = []
        self.image_size = image_size
        self.num_frame = num_frame
        if type == 'train':
            self.txt_path = dataset_txt
            self.aug = True
        else:
            self.txt_path = dataset_txt
            self.aug = False
        with open(self.txt_path) as f: 
            data_lines = f.readlines()
            self.length = len(data_lines)
            for line in data_lines:
                line = line.strip('\n').split()
                self.img_idx.append(line[0])
                # self.anno_idx.append(np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]]))
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
       
        # for i in range(1053):
        #     images, masks = self.get_data(0)
        # pdb.set_trace()
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        images, masks = self.get_data(index)
       
        #images = np.transpose(images,(3, 0, 1, 2)) #preprocess(

        # if len(box) != 0:
        #     box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
        #     box[:, 0:2] = box[:, 0:2] + ( box[:, 2:4] / 2 )
        return images, masks#,[h,w]
    
    def get_data(self, index):
        image_data = []
        mask_data=[]
        h, w = self.image_size[0], self.image_size[1]
        # print("宽高")
        # print(h,w)
        file_name = self.img_idx[index]
        # seq_num=file_name.split("-")[0]
        # image_id =file_name.split("-")[1]
        # image_path=os.path.join(self.dataset_path,seq_num)
        
        image_id = int(file_name.split("/")[-1][:-4])
        image_path = file_name.replace(file_name.split("/")[-1], '')
        mask_path=image_path.replace(image_path.split("/")[0], 'masks')
        image_dir=os.path.join(self.dataset_path,image_path)
        mask_dir=os.path.join(self.dataset_path,mask_path)
        for id in range(0, self.num_frame):
            try:
                img = Image.open(image_dir +'%d.bmp' % max(image_id - id, 0)).convert('I')
            except:
                img = Image.open(image_dir +'%d.jpg' % max(image_id - id, 0)).convert('I')
            mask=Image.open(mask_dir +'%d.png' % max(image_id - id, 0))
            
            # img,mask = copy_paste_augmentation(np.array(img).astype(np.uint8), np.array(mask).astype(np.uint8))

            img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
            # print("填充前",img.shape)
            mask = np.array(mask, dtype=np.float32)  / 255.0
            if len(mask.shape) > 2:
                mask = mask[:,:,0]
            img = PadImg(img)
            # print("填充后",img.shape)
            mask = PadImg(mask)
            #数据增强
            # img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            # img_patch, mask_patch = self.tranform(img_patch, mask_patch)如何同时旋转，同时裁剪
            img_patch, mask_patch = img, mask
            # img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
            image_data.append(img_patch)
            mask_data.append(mask_patch)


        # pdb.set_trace()    
        image_data=image_data[::-1]       
        mask_data=mask_data[::-1] 
        image_data=torch.stack(image_data)
        mask_data=torch.stack(mask_data)
        # image_data = np.array(image_data[::-1]) # 关键帧在后 # [5,w,h,3]
        # mask_data = np.array(mask_data[::-1]) # [:,5]
        if self.aug is True:
            # image_data, label_data[:,:4] = augmentation(image_data,label_data[:,:4],h,w)
            pass
        return image_data, mask_data#,[h,w]


def dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes

                
            
    
if __name__ == "__main__":
    train_dataset = seqDataset("/home/coco_train.txt", 256, 5, 'train')
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2, collate_fn=dataset_collate)
    t = time.time()
    for index, batch in enumerate(train_dataloader):
        images, targets = batch[0], batch[1]
        print(index)
    print(time.time()-t)