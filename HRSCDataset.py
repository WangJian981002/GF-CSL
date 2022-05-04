import numpy as np
import torch
import json
import cv2
import os
import math
import random
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Circle
import matplotlib.pyplot as plt
import sys
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils.utils import creat_label_heatmap
from utils.smooth_label import gaussian_label
from utils.aug import rotate_image
from DOTA_devkit.DOTA import DOTA

wordname_15 = ['ship']


class HRSCSetv1(Dataset):
    
    #rgb
    mean = np.array([[[0.485, 0.456, 0.406]]],dtype=np.float32)
    std  = np.array([[[0.229, 0.224, 0.225]]],dtype=np.float32)

    def __init__(self, root_dir,img_size=640):
        self.numclasses = len(wordname_15)
        self.class2label = {}
        self.label2class = {}
        for i in range(self.numclasses):
            self.class2label[wordname_15[i]] = i
            self.label2class[i] = wordname_15[i]
        self.imgsize = img_size


        self.DOTA = DOTA(root_dir)
        self.imgids = self.DOTA.getImgIds()

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, idx):
        imgid = self.imgids[idx]
        img = self.DOTA.loadImgs(imgid)[0] #bgr 0~255 np
        ann = self.DOTA.loadAnns(imgId=imgid)

        img,ann = self.flip_aug(img,ann)
        img,ann = self.rot_aug(img,ann)
        img,ann = self.resize(img, ann, self.imgsize)  # 长边resize到imgsize, 再用（imgsize，imgsize）将其装下
        #img = self.gray_aug(img)


        converted_ann = self.convert_poly2cxcyhw(ann) #(N,6) np.float32 [cx,cy,h,w,theta(0~179),class]
        img = self.normalize(img) #rgb (h,w,c) 标准化 np

        return torch.from_numpy(img.transpose(2,0,1)), converted_ann


    def convert_poly2cxcyhw(self,ann):
        #h 代表长边， w代表短边
        converted_ann = np.zeros((0,6),dtype=np.float32) #cx,cy,h,w,theta,class
        for i in range(len(ann)):
            p1,p2,p3,p4 =  ann[i]['poly']
            cx = ((p1[0]+p3[0])/2.0 + (p2[0]+p4[0])/2.0)/2.0
            cy = ((p1[1]+p3[1])/2.0 + (p2[1]+p4[1])/2.0)/2.0
            side1 = self.cal_line_length(p1,p2)
            side2 = self.cal_line_length(p2,p3)
            if side1>side2:
                r1,r2 = p1,p2
                long_side = side1
                short_side = side2
            else:
                r1,r2 = p2,p3
                long_side = side2
                short_side = side1
            if long_side < 4.0 or short_side < 4.0:
                continue

            if r1[1]<r2[1]:
                xx = r1[0] - r2[0]
            else:
                xx = r2[0] - r1[0]
            theta = round((math.acos(xx/long_side)/math.pi)*180)%180 #[0,179]
            cls = self.class2label[ann[i]['name']]

            a = np.zeros((1, 6),dtype=np.float32)
            a[0] = cx,cy,long_side,short_side,theta,cls
            converted_ann = np.append(converted_ann, a, axis=0)
        return converted_ann

    def convert_cxcyhw2poly(self,converted_ann):
        ann = []
        for i in range(len(converted_ann)):
            cx,cy,long_side,short_side,theta,cls = converted_ann[i]
            half_cross_line = math.sqrt(math.pow(long_side, 2) + math.pow(short_side, 2))/2.0 #半对角线长度
            p1 = (cx + half_cross_line * math.cos(theta * math.pi / 180.0 + math.atan(short_side * 1. / long_side)),
                  cy - half_cross_line * math.sin(theta * math.pi / 180.0 + math.atan(short_side * 1. / long_side)))
            p2 = (cx + half_cross_line * math.cos(theta * math.pi / 180.0 - math.atan(short_side * 1. / long_side)),
                  cy - half_cross_line * math.sin(theta * math.pi / 180.0 - math.atan(short_side * 1. / long_side)))
            p3 = (cx - half_cross_line * math.cos(theta * math.pi / 180.0 + math.atan(short_side * 1. / long_side)),
                  cy + half_cross_line * math.sin(theta * math.pi / 180.0 + math.atan(short_side * 1. / long_side)))
            p4 = (cx - half_cross_line * math.cos(theta * math.pi / 180.0 - math.atan(short_side * 1. / long_side)),
                  cy + half_cross_line * math.sin(theta * math.pi / 180.0 - math.atan(short_side * 1. / long_side)))
            ann.append(dict(poly=[p1,p2,p3,p4],name=self.label2class[int(cls)]))

        return ann

    def display(self,img,ann):
        #img (h,w,c) rgb 0~255
        #ann poly type

        plt.imshow(img)
        plt.axis('off')

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []

        for obj in ann:
            c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
            poly = obj['poly']
            polygons.append(Polygon(poly))
            color.append(c)

        p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
        ax.add_collection(p)
        p = PatchCollection(polygons, facecolors='none', edgecolors=color, linewidths=0.5)
        ax.add_collection(p)

        plt.savefig("demo.png", dpi=600)

    def cal_line_length(self,point1, point2):
        return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))














        '''
        image_info = self.coco.loadImgs(self.image_ids[idx])[0]
        path = os.path.join(self.root_dir,'images', self.set_name, image_info['file_name'])
        img=cv2.imread(path)
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # (H,W,C) 0~255 bgr

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.default_resolution
        
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)

        flipped = False
        if np.random.random() < 0.5:
            flipped = True
            img = img[:, ::-1, :]
            c[0] =  width - c[0] - 1
            
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, 
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)

        inp=random_bright(inp)
        inp=random_saturation(inp) #(H,W,C) 0~255 BGR

        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp[:,:,::-1] #(H,W,C) 归一化 RGB
        inp = torch.from_numpy(np.ascontiguousarray(inp.transpose(2, 0, 1))) #(C,H,W) 归一化 RGB tensor


        gt = self.load_annotations(idx,flipped,trans_input,width,input_w,input_h)
        return [inp,gt]
        #inp (C,H,W) 归一化 RGB tensor
        #gt (num_obj,5) [xmin,ymin,w,h,clss] np
        '''

    def flip_aug(self,img,ann,flip_x=0.5,flip_y=0.5):
        H, W, _ = img.shape

        if np.random.rand() < flip_x:
            img = img[:,::-1,:]
            for i in range(len(ann)):
                p1,p2,p3,p4 = ann[i]['poly']
                p1_aug = (W - p1[0], p1[1])
                p2_aug = (W - p2[0], p2[1])
                p3_aug = (W - p3[0], p3[1])
                p4_aug = (W - p4[0], p4[1])
                ann[i]['poly'] = [p1_aug,p2_aug,p3_aug,p4_aug]

        if np.random.rand() < flip_y:
            img = img[::-1,:,:]
            for i in range(len(ann)):
                p1,p2,p3,p4 = ann[i]['poly']
                p1_aug = (p1[0], H - p1[1])
                p2_aug = (p2[0], H - p2[1])
                p3_aug = (p3[0], H - p3[1])
                p4_aug = (p4[0], H - p4[1])
                ann[i]['poly'] = [p1_aug, p2_aug, p3_aug, p4_aug]

        return img,ann

    def resize(self,img,ann,size):
        H, W, _ = img.shape
        long_side = max(H, W)
        ratio = size*1./long_side
        img = cv2.resize(img,(round(W*ratio),round(H*ratio)))
        new_h, new_w, _ = img.shape
        new_img = np.zeros((size,size,3), dtype=np.uint8)
        new_img[:new_h, :new_w, :] = img

        for i in range(len(ann)):
            p1, p2, p3, p4 = ann[i]['poly']
            p1_aug = (p1[0]*ratio , p1[1]*ratio)
            p2_aug = (p2[0]*ratio , p2[1]*ratio)
            p3_aug = (p3[0]*ratio , p3[1]*ratio)
            p4_aug = (p4[0]*ratio , p4[1]*ratio)
            ann[i]['poly'] = [p1_aug, p2_aug, p3_aug, p4_aug]

        return new_img,ann

    def normalize(self,img):
        #img bgr (h,w,c) 0~255
        img = img[:,:,::-1].astype(np.float32)/255.0
        img = (img-self.mean)/self.std

        return img

    def rot_aug(self,img,ann,max_angle=20):

        list_ann = []
        for a in ann:
            p1,p2,p3,p4 = a['poly']
            bbox = [p1[0],p1[1],p2[0],p2[1],p3[0],p3[1],p4[0],p4[1]]
            list_ann.append(([],bbox))
        angle = np.random.randint(-1*max_angle,max_angle+1)

        img, bbox_new_list = rotate_image(img,list_ann,angle=angle)

        for i in range(len(bbox_new_list)):
            bbox = bbox_new_list[i][1]
            p1_new = (bbox[0], bbox[1])
            p2_new = (bbox[2], bbox[3])
            p3_new = (bbox[4], bbox[5])
            p4_new = (bbox[6], bbox[7])
            ann[i]['poly'] = [p1_new, p2_new, p3_new, p4_new]

        return img,ann

    def gray_aug(self,img,p=0.1):
        if np.random.rand() < p:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img

def collater(data):
    imgs=[]
    annos=[]
    for img,anno in data :
        imgs.append(img)
        annos.append(anno)
    imgs = torch.stack(imgs,dim=0) #(N,C,H,W)
    heatmap_t=creat_label_heatmap(imgs,annos,num_classes=1,min_overlap=0.5) #(N,15,H/4,W/4)
    gaussian_smooth=[]
    for anno in annos:
        smooth = np.zeros((len(anno),180),dtype=np.float32)
        for i in range(len(anno)):
            smooth[i,:] = gaussian_label(int(anno[i][4]),180,sig=6).astype(np.float32)
        gaussian_smooth.append(torch.from_numpy(smooth))

    return {'img':imgs,'label':annos,'heatmap_t':heatmap_t,'smooth_label':gaussian_smooth}
    #imgs (N,C,H,W) tensor rgb 标准化
    #annos list list中的每个元素是（num_obj,6）[cx,cy,h,w,theta,class] np
    #heatmap_t (N,1,H/4,W/4) tensor
    #gaussian_smooth list list中每个元素是（num_obj,180） tensor






if __name__ == "__main__":
    from torch.utils.data import DataLoader

    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    trainset = HRSCSetv1('../datasets/HRSC2016/Train')
    print(len(trainset))

    i = 0
    for id in trainset.imgids:
        if id == '100000646':
            print(i)
            break
        else:
            i += 1


    img,convert_ann = trainset[i]
    print(img.size())
    print(convert_ann)
    img = img.numpy().transpose(1,2,0)*std+mean
    img = (img*255.0).astype(np.uint8)

    ann = trainset.convert_cxcyhw2poly(convert_ann)
    trainset.display(img,ann)



    '''
    trainloader = DataLoader(trainset, batch_size=4, shuffle=False, collate_fn=collater)
    it = iter(trainloader)
    data = next(it)
    imgs , labels, heatmaps, smooth_label = data['img'],data['label'],data['heatmap_t'],data['smooth_label']
    print(imgs.size())
    print(heatmaps.size())
    idx=1
    print(labels[idx])
    print(smooth_label[idx].size())
    x=torch.from_numpy(np.arange(len(labels[idx]))).long()
    y=torch.from_numpy(labels[idx][:,4]).long()
    print(smooth_label[idx][x,y])
    print(smooth_label[idx])
    '''




    #img = (imgs[idx].numpy().transpose(1,2,0)*std+mean)*255.0
    #img = img[:,:,::-1].astype(np.uint8)
    #cv2.imwrite('img.png',img)
    #ht = (heatmaps[idx][4].numpy()*255).astype(np.uint8)
    #cv2.imwrite('ht.png', ht)


    
        
    
    
    
    
    
    
    
    
    
    
    