# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 13:08:40 2020

@author: wj
"""

from __future__ import print_function, division
import sys
import os
import random
import math
import argparse
from datetime import datetime
import numpy as np
import cv2
import json
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from nets.resnet_dcn_DFPN_model import ResNet
from utils.utils import decode_per_img
from de import DotaSetv1

parser = argparse.ArgumentParser()


parser.add_argument("--heads", default={'hm': 15,'wh': 2 ,'reg': 2, 'theta':180})
parser.add_argument('--weight_path', default='./checkpoint/dota/centernet_ckpt_140.pth')
parser.add_argument('--score_thr', default=0.15)
parser.add_argument('--img_path', default='P0021.png')

args = parser.parse_args()

def inference_sigle_image():
    print(args.heads)
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    model = ResNet(num_layers=101,heads=args.heads).cuda()

    weight_dict = torch.load(args.weight_path)
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    img = cv2.imread(args.img_path)[:,:,::-1] #(H,W,C) RGB 0~255
    image = ((img.astype(np.float32)/255.0)-mean)/std
    image = image.transpose(2,0,1)#(c,h,w) rgb 标准化
    _, img_h, img_w = image.shape
    input_h , input_w = int(32*np.ceil(img_h/32.0)) , int(32*np.ceil(img_w/32.0))
    input = np.zeros((3, input_h , input_w),dtype=np.float32)
    input[:,:img_h,:img_w] = image
    input = torch.from_numpy(input).unsqueeze(0).cuda()

    with torch.no_grad():
        heatmap, scale, offset, theta = model(input)
        results = decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_h, input_w, args.score_thr)#(total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

    if len(results) == 0:
        print("None of object have been detected")
    else:
        print(results)
        trainset = DotaSetv1('../datasets/trainvalsplit')
        ann = trainset.convert_cxcyhw2poly(results[:,:6])
        trainset.display(img,ann)

def genarate_img_subimage_dict(ori_img_path='../datasets/test/images',sub_img_dir='../datasets/testsplit/images'):
    imgid=[]
    with open('./result/testID.txt','r') as f:
        file = f.readlines()
    for line in file:
        imgid.append(line.strip())
    #imgid ['P0006','P0009',...]

    img_subimage_dict = {}
    for id in imgid:
        img_subimage_dict[id] = []

    subimage_list = os.listdir(sub_img_dir)
    for imgname in subimage_list:
        curid = imgname.split('__')[0]
        img_subimage_dict[curid].append(imgname)

    sum = 0
    for key in img_subimage_dict.keys():
        sum += len(img_subimage_dict[key])
    assert sum == len(subimage_list)

    print("imgid-subimage-dict haved been genarated.")
    return img_subimage_dict

def inference_testset(id_path_dict,out_dir,ori_size=1024,input_size=1120,root_dir='../datasets/testsplit/images'):
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
    label2class = {0:'plane', 1:'baseball-diamond', 2:'bridge', 3:'ground-track-field', 4:'small-vehicle', 5:'large-vehicle',6:'ship', 7:'tennis-court',
                   8:'basketball-court', 9:'storage-tank', 10:'soccer-ball-field', 11:'roundabout', 12:'harbor', 13:'swimming-pool',14:'helicopter'}
    trainset = DotaSetv1('../datasets/trainvalsplit-mix')

    model = ResNet(num_layers=50, heads=args.heads,head_conv=64).cuda()

    print(args.weight_path)
    weight_dict = torch.load(args.weight_path)
    if 'model_state' in weight_dict.keys():
        weight_dict = weight_dict['model_state']
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    num_images = len(os.listdir(root_dir))
    img_i = 0
    j = 0

    for id in id_path_dict.keys():
        results = {}
        for img_name in id_path_dict[id]:
            path = os.path.join(root_dir,img_name)
            img = cv2.imread(path)
            H, W, _ = img.shape
            if H != ori_size or W != ori_size:
                new_img = np.zeros((ori_size,ori_size,3),dtype=np.float32)
                new_img[:H, :W, :] = img
                img = new_img
                j += 1
            img = cv2.resize(img,(input_size,input_size))[:,:,::-1]
            image = ((img.astype(np.float32) / 255.0) - mean) / std
            image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
            input = torch.from_numpy(image).unsqueeze(0).cuda()
            with torch.no_grad():
                heatmap, scale, offset, theta = model(input)
                out = decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size,0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32
                #out = class_specific_decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size, 0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

            img_i+=1
            print('{}/{}'.format(img_i, num_images), end='\r')
            if len(out) == 0: continue

            out[:,:4] = out[:,:4]*ori_size*1. / input_size
            results[img_name.split('.')[0]] = out

        txtpath = os.path.join(out_dir, id + '.txt')
        f = open(txtpath, 'w')
        if len(results) == 0:
            f.close()
            continue

        for subimg_id in results.keys():
            poly_ann = trainset.convert_cxcyhw2poly(results[subimg_id][:,:6])
            for i in range(len(results[subimg_id])):
                cx, cy, h, w, theta, cls, score =  results[subimg_id][i]
                p1,p2,p3,p4 = poly_ann[i]['poly']

                f.write(subimg_id), f.write(" ")
                f.write(label2class[int(cls)]), f.write(" ")
                f.write(str(score)), f.write(" ")
                f.write(str(p1[0])), f.write(" "), f.write(str(p1[1])), f.write(" ")
                f.write(str(p2[0])), f.write(" "), f.write(str(p2[1])), f.write(" ")
                f.write(str(p3[0])), f.write(" "), f.write(str(p3[1])), f.write(" ")
                f.write(str(p4[0])), f.write(" "), f.write(str(p4[1])), f.write('\n')
        f.close()

def inference_testset_FLIP(id_path_dict,out_dir,ori_size=1024,input_size=1120,root_dir='../datasets/testsplit/images'):
    print('*****current test mood is H FLIP test*****')
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
    label2class = {0:'plane', 1:'baseball-diamond', 2:'bridge', 3:'ground-track-field', 4:'small-vehicle', 5:'large-vehicle',6:'ship', 7:'tennis-court',
                   8:'basketball-court', 9:'storage-tank', 10:'soccer-ball-field', 11:'roundabout', 12:'harbor', 13:'swimming-pool',14:'helicopter'}
    trainset = DotaSetv1('../datasets/trainvalsplit-mix')

    model = ResNet(num_layers=101, heads=args.heads,head_conv=128).cuda()

    print(args.weight_path)
    weight_dict = torch.load(args.weight_path)
    if 'model_state' in weight_dict.keys():
        weight_dict = weight_dict['model_state']
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    num_images = len(os.listdir(root_dir))
    img_i = 0
    j = 0

    for id in id_path_dict.keys():
        results = {}
        for img_name in id_path_dict[id]:
            path = os.path.join(root_dir,img_name)
            img = cv2.imread(path)
            H, W, _ = img.shape
            if H != ori_size or W != ori_size:
                new_img = np.zeros((ori_size,ori_size,3),dtype=np.float32)
                new_img[:H, :W, :] = img
                img = new_img
                j += 1
            img = cv2.resize(img,(input_size,input_size))[:,:,::-1]
            image = ((img.astype(np.float32) / 255.0) - mean) / std
            image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
            image = np.ascontiguousarray(image[:,:,::-1])

            input = torch.from_numpy(image).unsqueeze(0).cuda()
            with torch.no_grad():
                heatmap, scale, offset, theta = model(input)
                out = decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size,0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32
                #out = class_specific_decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size, 0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

            img_i+=1
            print('{}/{}'.format(img_i, num_images), end='\r')
            if len(out) == 0: continue

            out[:, 0] = input_size - 1 - out[:, 0]
            out[:, 4] = (180 - out[:, 4]) % 180
            out[:,:4] = out[:,:4]*ori_size*1. / input_size
            results[img_name.split('.')[0]] = out

        txtpath = os.path.join(out_dir, id + '.txt')
        f = open(txtpath, 'w')
        if len(results) == 0:
            f.close()
            continue

        for subimg_id in results.keys():
            poly_ann = trainset.convert_cxcyhw2poly(results[subimg_id][:,:6])
            for i in range(len(results[subimg_id])):
                cx, cy, h, w, theta, cls, score =  results[subimg_id][i]
                p1,p2,p3,p4 = poly_ann[i]['poly']

                f.write(subimg_id), f.write(" ")
                f.write(label2class[int(cls)]), f.write(" ")
                f.write(str(score)), f.write(" ")
                f.write(str(p1[0])), f.write(" "), f.write(str(p1[1])), f.write(" ")
                f.write(str(p2[0])), f.write(" "), f.write(str(p2[1])), f.write(" ")
                f.write(str(p3[0])), f.write(" "), f.write(str(p3[1])), f.write(" ")
                f.write(str(p4[0])), f.write(" "), f.write(str(p4[1])), f.write('\n')
        f.close()

def inference_testset_VFLIP(id_path_dict,out_dir,ori_size=1024,input_size=1120,root_dir='../datasets/testsplit/images'):
    print('*****current test mood is V FLIP test*****')
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)
    label2class = {0:'plane', 1:'baseball-diamond', 2:'bridge', 3:'ground-track-field', 4:'small-vehicle', 5:'large-vehicle',6:'ship', 7:'tennis-court',
                   8:'basketball-court', 9:'storage-tank', 10:'soccer-ball-field', 11:'roundabout', 12:'harbor', 13:'swimming-pool',14:'helicopter'}
    trainset = DotaSetv1('../datasets/trainvalsplit-mix')

    model = ResNet(num_layers=101, heads=args.heads,head_conv=128).cuda()

    print(args.weight_path)
    weight_dict = torch.load(args.weight_path)
    if 'model_state' in weight_dict.keys():
        weight_dict = weight_dict['model_state']
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    num_images = len(os.listdir(root_dir))
    img_i = 0
    j = 0

    for id in id_path_dict.keys():
        results = {}
        for img_name in id_path_dict[id]:
            path = os.path.join(root_dir,img_name)
            img = cv2.imread(path)
            H, W, _ = img.shape
            if H != ori_size or W != ori_size:
                new_img = np.zeros((ori_size,ori_size,3),dtype=np.float32)
                new_img[:H, :W, :] = img
                img = new_img
                j += 1
            img = cv2.resize(img,(input_size,input_size))[:,:,::-1]
            image = ((img.astype(np.float32) / 255.0) - mean) / std
            image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
            image = np.ascontiguousarray(image[:,::-1,:])

            input = torch.from_numpy(image).unsqueeze(0).cuda()
            with torch.no_grad():
                heatmap, scale, offset, theta = model(input)
                out = decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size,0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32
                #out = class_specific_decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_size, input_size, 0.01)  # (total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

            img_i+=1
            print('{}/{}'.format(img_i, num_images), end='\r')
            if len(out) == 0: continue

            out[:, 1] = input_size - 1 - out[:, 1]
            out[:, 4] = (180 - out[:, 4]) % 180
            out[:,:4] = out[:,:4]*ori_size*1. / input_size
            results[img_name.split('.')[0]] = out

        txtpath = os.path.join(out_dir, id + '.txt')
        f = open(txtpath, 'w')
        if len(results) == 0:
            f.close()
            continue

        for subimg_id in results.keys():
            poly_ann = trainset.convert_cxcyhw2poly(results[subimg_id][:,:6])
            for i in range(len(results[subimg_id])):
                cx, cy, h, w, theta, cls, score =  results[subimg_id][i]
                p1,p2,p3,p4 = poly_ann[i]['poly']

                f.write(subimg_id), f.write(" ")
                f.write(label2class[int(cls)]), f.write(" ")
                f.write(str(score)), f.write(" ")
                f.write(str(p1[0])), f.write(" "), f.write(str(p1[1])), f.write(" ")
                f.write(str(p2[0])), f.write(" "), f.write(str(p2[1])), f.write(" ")
                f.write(str(p3[0])), f.write(" "), f.write(str(p3[1])), f.write(" ")
                f.write(str(p4[0])), f.write(" "), f.write(str(p4[1])), f.write('\n')
        f.close()


def divide_into_class_specific(id_path_dict,src_path='./result/test1_ori',dst_path='./result/test1_before_merge'):
    names = locals()
    trainset = DotaSetv1('../datasets/trainvalsplit')
    os.mkdir(dst_path)

    for imgid in id_path_dict.keys():
        src_txt_path = os.path.join(src_path,imgid+'.txt')
        with open(src_txt_path,'r') as f:
            data = f.readlines()
        for n in range(15):
            names['f'+str(n)] = open(os.path.join(dst_path,imgid+"__"+trainset.label2class[n]+".txt"),'w')

        for i in range(len(data)):
            line = data[i].strip().split(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[0])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[2])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[3])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[4])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[5])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[6])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[7])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[8])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[9])
            names['f' + str(trainset.class2label[line[1]])].write(' ')
            names['f' + str(trainset.class2label[line[1]])].write(line[10])
            names['f' + str(trainset.class2label[line[1]])].write('\n')
        for n in range(15):
            names['f' + str(n)].close()

def visulize_from_merged(imgid):
    trainset = DotaSetv1('../datasets/trainvalsplit')

    imgpath = '../datasets/test/images/'+imgid+'.png'
    img =cv2.imread(imgpath)[:,:,::-1]

    results = []
    for i in range(15):
        path = './result/test12/test12_after_merge/'+imgid+'__'+trainset.label2class[i]+'.txt'
        with open(path,'r') as f:
            sigle_class_data = f.readlines()
        if len(sigle_class_data) == 0: continue

        for j in range(len(sigle_class_data)):
            line = sigle_class_data[j].strip().split(' ')
            sigle = dict(poly=[(float(line[2]),float(line[3])),(float(line[4]),float(line[5])),(float(line[6]),float(line[7])),(float(line[8]),float(line[9]))])
            if float(line[1]) > args.score_thr:
                results.append(sigle)

    if len(results) == 0:
        print("none of object have been detected")
    else:
        trainset.display(img,results)

def merged2final(id_path_dict,final_dir='./result/test1_final',merged_dir='./result/test1_after_merge'):
    label2class = {0: 'plane', 1: 'baseball-diamond', 2: 'bridge', 3: 'ground-track-field', 4: 'small-vehicle',
                   5: 'large-vehicle', 6: 'ship', 7: 'tennis-court',
                   8: 'basketball-court', 9: 'storage-tank', 10: 'soccer-ball-field', 11: 'roundabout', 12: 'harbor',
                   13: 'swimming-pool', 14: 'helicopter'}

    for class_id in range(len(label2class)):
        classname = label2class[class_id]
        file = open(os.path.join(final_dir,'Task1_'+classname+'.txt'),'w')

        for img_id in id_path_dict.keys():
            ori_txt_path = merged_dir + "/" + img_id +"__" + classname+'.txt'
            with open(ori_txt_path,'r') as f:
                src = f.readlines()
            if len(src) == 0:continue

            for line_i in range(len(src)):
                line = src[line_i].strip().split(' ')
                file.write(line[0]), file.write(' ')
                file.write(line[1]), file.write(' ')
                file.write(line[2]), file.write(' ')
                file.write(line[3]), file.write(' ')
                file.write(line[4]), file.write(' ')
                file.write(line[5]), file.write(' ')
                file.write(line[6]), file.write(' ')
                file.write(line[7]), file.write(' ')
                file.write(line[8]), file.write(' ')
                file.write(line[9]), file.write('\n')

        file.close()

#for analysis
def visualize_heatmap():
    print(args.heads)
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    model = ResNet(num_layers=50,heads=args.heads).cuda()

    weight_dict = torch.load(args.weight_path)
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    img = cv2.imread(args.img_path)[:,:,::-1] #(H,W,C) RGB 0~255
    image = ((img.astype(np.float32)/255.0)-mean)/std
    image = image.transpose(2,0,1)#(c,h,w) rgb 标准化
    _, img_h, img_w = image.shape
    input_h , input_w = int(32*np.ceil(img_h/32.0)) , int(32*np.ceil(img_w/32.0))
    input = np.zeros((3, input_h , input_w),dtype=np.float32)
    input[:,:img_h,:img_w] = image
    input = torch.from_numpy(input).unsqueeze(0).cuda()

    with torch.no_grad():
        heatmap, scale, offset, theta, seg = model(input)
        results = decode_per_img(heatmap, scale, torch.tanh(offset), theta, input_h, input_w, args.score_thr)#(total_objs,7) [cx,cy,h,w,theta,class,score] np.float32

    print(seg.size())
    print(seg.max())
    print(seg.min())
    segmap = (seg[0,0,:,:].cpu().numpy()*255.0).astype(np.uint8)
    #segmap = cv2.applyColorMap(segmap, cv2.COLORMAP_JET)
    cv2.imwrite('demo.png', segmap)

    #ht = (heatmap[0,6,:,:].cpu().numpy()*255.0).astype(np.uint8)
    #ht_color = cv2.applyColorMap(ht, cv2.COLORMAP_JET)
    #cv2.imwrite('demo.png',ht_color)

def multiscale_test(id_path_dict,in_list,out_dir):
    for id in id_path_dict.keys():
        out_file = open(os.path.join(out_dir, id+'.txt'),'w')
        for result in in_list:
            path = os.path.join("./result", result, result.split('-')[0]+'_ori', id+'.txt')
            with open(path,'r') as f:
                data = f.readlines()
            for line in data:
                out_file.write(line)
        out_file.close()

#for polar
def polar_inference_sigle_with_nms(workdir):
    print(args.heads)
    mean = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
    std = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

    model = ResNet(num_layers=50, heads=args.heads).cuda()

    weight_dict = torch.load(args.weight_path)
    use_weight_dict = {}
    for k, v in weight_dict.items():
        newk = k.replace("module.", "")
        use_weight_dict[newk] = v
    model.load_state_dict(use_weight_dict)
    print("==>finished loading weight")
    model.eval()

    img = cv2.imread(args.img_path)[:, :, ::-1]  # (H,W,C) RGB 0~255
    #img = cv2.imread(args.img_path)[:, :, ::-1]  # (H,W,C) RGB 0~255
    image = ((img.astype(np.float32) / 255.0) - mean) / std
    image = image.transpose(2, 0, 1)  # (c,h,w) rgb 标准化
    _, img_h, img_w = image.shape
    input_h, input_w = int(32 * np.ceil(img_h / 32.0)), int(32 * np.ceil(img_w / 32.0))
    input = np.zeros((3, input_h, input_w), dtype=np.float32)
    input[:, :img_h, :img_w] = image
    input = torch.from_numpy(input).unsqueeze(0).cuda()

    with torch.no_grad():
        heatmap, offset,theta = model(input)
        results = polar_decode_per_img(heatmap, offset,theta, input_h, input_w, args.score_thr)  #(total_objs,10) [class,s,x1,y1,x2,y2,x3,y3,x4,y4] np.float32

    assert len(results) != 0
    if not os.path.isdir(os.path.join(workdir,'before_merge')):
        os.mkdir(os.path.join(workdir,'before_merge'))

    trainset = DotaSetv1('../datasets/test')

    names = locals()
    for n in range(15):
        names['f' + str(n)] = open(os.path.join(workdir,'before_merge', trainset.label2class[n] + ".txt"), 'w')

    for i in range(len(results)):
        cls, s, x1, y1, x2, y2, x3, y3, x4, y4 = results[i]
        names['f' + str(int(cls))].write(args.img_path.split('.')[0]+"__1__0___0")
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(s))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(x1))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(y1))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(x2))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(y2))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(x3))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(y3))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(x4))
        names['f' + str(int(cls))].write(' ')
        names['f' + str(int(cls))].write(str(y4))
        names['f' + str(int(cls))].write('\n')
    for n in range(15):
        names['f' + str(n)].close()

    sys.path.append('..')
    from ResultMerge import mergebypoly
    if not os.path.isdir(os.path.join(workdir, 'after_merge')):
        os.mkdir(os.path.join(workdir,'after_merge'))
    mergebypoly(os.path.join(workdir,'before_merge'), os.path.join(workdir,'after_merge'))

    results = []
    for i in range(15):
        path = workdir + '/after_merge/' + trainset.label2class[i] + '.txt'
        with open(path, 'r') as f:
            sigle_class_data = f.readlines()
        if len(sigle_class_data) == 0: continue

        for j in range(len(sigle_class_data)):
            line = sigle_class_data[j].strip().split(' ')
            sigle = dict(poly=[(float(line[2]), float(line[3])), (float(line[4]), float(line[5])),
                               (float(line[6]), float(line[7])), (float(line[8]), float(line[9]))], name=trainset.label2class[i])
            if float(line[1]) > args.score_thr:
                results.append(sigle)

    trainset.display(img, results)

def eval_dota(out_id,input_size,is_flip=False,is_V=False):
    sys.path.append('..')
    from ResultMerge_multi_process import mergebypoly
    print("***********out id: ",out_id,"***********ori size: 4000, ***********image size: ",input_size)

    os.mkdir(os.path.join('./result', 'test'+str(out_id)))
    os.mkdir(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_ori'))

    id_path_dict = genarate_img_subimage_dict(ori_img_path='../datasets/test/images',sub_img_dir='../datasets/test4000/images')
    if is_flip:
        if is_V:
            inference_testset_VFLIP(id_path_dict,out_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_ori'),ori_size=4000, input_size=input_size, root_dir='../datasets/test4000/images')
        else:
            inference_testset_FLIP(id_path_dict,out_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_ori'),ori_size=4000, input_size=input_size, root_dir='../datasets/test4000/images')
    else:
        inference_testset(id_path_dict, out_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_ori'), ori_size=4000, input_size=input_size,root_dir='../datasets/test4000/images')
    divide_into_class_specific(id_path_dict, src_path=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_ori'),dst_path=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_before_merge'))


    os.mkdir(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_after_merge'))
    mergebypoly(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_before_merge'), os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_after_merge'))

    os.mkdir(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_final'))
    merged2final(id_path_dict, final_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_final'),merged_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_after_merge'))


def eval_dota_2000(out_id,input_size,is_flip=False,is_V=False):
    sys.path.append('..')
    from ResultMerge_multi_process import mergebypoly
    print("***********out id: ",out_id,"***********ori size: 2000, ***********image size: ",input_size)

    os.mkdir(os.path.join('./result', 'test'+str(out_id)))
    os.mkdir(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_ori'))

    id_path_dict = genarate_img_subimage_dict(ori_img_path='../datasets/test/images',sub_img_dir='../datasets/test2000/images')
    if is_flip:
        if is_V:
            inference_testset_VFLIP(id_path_dict,out_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_ori'),ori_size=2000, input_size=input_size, root_dir='../datasets/test2000/images')
        else:
            inference_testset_FLIP(id_path_dict,out_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_ori'),ori_size=2000, input_size=input_size, root_dir='../datasets/test2000/images')
    else:
        inference_testset(id_path_dict, out_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_ori'), ori_size=2000, input_size=input_size,root_dir='../datasets/test2000/images')
    divide_into_class_specific(id_path_dict, src_path=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_ori'),dst_path=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_before_merge'))


    os.mkdir(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_after_merge'))
    mergebypoly(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_before_merge'), os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_after_merge'))

    os.mkdir(os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_final'))
    merged2final(id_path_dict, final_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) + '_final'),merged_dir=os.path.join('./result', 'test' + str(out_id), 'test' + str(out_id) +'_after_merge'))





if __name__ =="__main__":
    
    #inference_sigle_image()
    '''
    id_path_dict=genarate_img_subimage_dict(ori_img_path='../datasets/test/images',sub_img_dir='../datasets/test4000/images')
    #inference_testset(id_path_dict,out_dir='./result/test47/test47_ori',ori_size=4000,input_size=5024,root_dir='../datasets/test4000/images')
    #multiscale_test(id_path_dict, in_list=['test9-5024','test10-4512','test11-4000','test12-3520','test13-3008','test14-2528','test15-2016'], out_dir='./result/test16/test16_ori')
    multiscale_test(id_path_dict, in_list=['test73-MS_HF','test75-6000','test76-5500','test77-5024','test78-4512','test79-4000','test80-3520','test81-3008','test82-2528','test83-2016'], out_dir='./result/test84/test84_ori')
    divide_into_class_specific(id_path_dict, src_path='./result/test84/test84_ori', dst_path='./result/test84/test84_before_merge')

    sys.path.append('..')
    from ResultMerge_multi_process import mergebypoly

    os.mkdir('./result/test84/test84_after_merge')
    mergebypoly('./result/test84/test84_before_merge','./result/test84/test84_after_merge')
    os.mkdir('./result/test84/test84_final')
    merged2final(id_path_dict, final_dir='./result/test84/test84_final', merged_dir='./result/test84/test84_after_merge')
    

    #visulize_from_merged('P0055')
    #visualize_heatmap()

    #polar_inference_sigle_with_nms('./c')

    #eval_dota_2000(out_id=18, input_size=4000)
    #eval_dota_2000(out_id=19, input_size=3008)
    #eval_dota_2000(out_id=38, input_size=2752)
    #eval_dota_2000(out_id=40, input_size=2016)
    #eval_dota(out_id=20, input_size=5024)
    #eval_dota(out_id=36, input_size=4512)
    #eval_dota(out_id=21, input_size=4000)
    #eval_dota(out_id=22, input_size=3520)
    #eval_dota(out_id=23, input_size=3008)
    #eval_dota(out_id=24, input_size=2528)
    #eval_dota(out_id=25, input_size=2016)

    #eval_dota_2000(out_id=26, input_size=4000, is_flip=True)
    #eval_dota_2000(out_id=27, input_size=3008, is_flip=True)
    #eval_dota_2000(out_id=39, input_size=2752, is_flip=True)
    #eval_dota(out_id=28, input_size=5024, is_flip=True)
    #eval_dota(out_id=37, input_size=4512, is_flip=True)
    #eval_dota(out_id=29, input_size=4000, is_flip=True)
    #eval_dota(out_id=30, input_size=3520, is_flip=True)
    #eval_dota(out_id=31, input_size=3008, is_flip=True)
    #eval_dota(out_id=32, input_size=2528, is_flip=True)
    #eval_dota(out_id=33, input_size=2016, is_flip=True)
    
    






    

    #eval_dota_2000(out_id=55, input_size=3008)
    #eval_dota_2000(out_id=56, input_size=2752)
    #eval_dota(out_id=57, input_size=5024)
    #eval_dota(out_id=58, input_size=4512)
    #eval_dota(out_id=59, input_size=4000)
    #eval_dota(out_id=60, input_size=3520)
    #eval_dota(out_id=61, input_size=3008)
    #eval_dota(out_id=62, input_size=2528)
    #eval_dota(out_id=63, input_size=2016)


    #eval_dota_2000(out_id=64, input_size=3008, is_flip=True)
    #eval_dota_2000(out_id=65, input_size=2752, is_flip=True)
    #eval_dota(out_id=66, input_size=5024, is_flip=True)
    #eval_dota(out_id=67, input_size=4512, is_flip=True)
    #eval_dota(out_id=68, input_size=4000, is_flip=True)
    #eval_dota(out_id=69, input_size=3520, is_flip=True)
    #eval_dota(out_id=70, input_size=3008, is_flip=True)
    #eval_dota(out_id=71, input_size=2528, is_flip=True)
    #eval_dota(out_id=72, input_size=2016, is_flip=True)

    #eval_dota_2000(out_id=75, input_size=3008, is_flip=True,is_V=True)
    #eval_dota_2000(out_id=76, input_size=2752, is_flip=True,is_V=True)
    #eval_dota(out_id=77, input_size=5024, is_flip=True,is_V=True)
    eval_dota(out_id=78, input_size=4512, is_flip=True,is_V=True)
    eval_dota(out_id=79, input_size=4000, is_flip=True,is_V=True)
    #eval_dota(out_id=80, input_size=3520, is_flip=True,is_V=True)
    #eval_dota(out_id=81, input_size=3008, is_flip=True,is_V=True)
    #eval_dota(out_id=82, input_size=2528, is_flip=True,is_V=True)
    #eval_dota(out_id=83, input_size=2016, is_flip=True,is_V=True)
    '''
    eval_dota(out_id=85, input_size=4000)
    
    

    

















    
    
    
    
    
    
    
    
    
    
    
    
    
    
