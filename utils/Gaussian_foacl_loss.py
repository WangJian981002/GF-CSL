# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:20:12 2020

@author: wj
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class losses(nn.Module):


    use_smooth = True

    def forward(self, heatmap,scale,offset,theta,heatmap_t,label,smoothlabel):
        '''
        heatmap: (N,15,H/4,W/4) cuda  0~1
        scale: (N,2,H/4,W/4) cuda \\if use gaussian  (N,4,H/4,W/4)
        offset: (N,2,H/4,W/4) cuda \\if use gaussian  (N,4,H/4,W/4) 未经tanh激活
        theta: (N,180,H/4,W/4) cuda 0~1
        heatmap_t: (N,15,H/4,W/4) tensor cuda float32
        label: list list中的每个元素是（num_obj,6）[cx,cy,h,w,theta,class] np
        smoothlabel:list list中的每个元素是（num_obj,180）tensor cpu

        '''
        alpha=2.0
        beita=4.0
        eps=0.00001

        label_new=[]
        for i in range(len(label)):
            if len(label[i]) == 0: continue
            l=torch.zeros((label[i].shape[0],7))
            l[:,1:7]=torch.from_numpy(label[i])
            l[:,0]=i
            label_new.append(l)
        if len(label_new) == 0:
            return 0*heatmap.sum(), 0*scale.sum(), 0*offset.sum(), 0*theta.sum()
        label_new = torch.cat(label_new, 0)  # (s_num,7) torch.float32 cpu [idx,cx,cy,h,w,theta,class]

        N = max(1, label_new.size(0))

        idx = label_new[:, 0].long()
        cx = (label_new[:, 1]/4.0).long()
        cy = (label_new[:, 2]/4.0).long()
        clss = label_new[:, 6].long()

        '''compute center point loss'''

        pos = heatmap[idx, clss, cy, cx]
        pos_loss = -torch.pow(1 - pos, alpha) * torch.log(pos + eps)  # 正样本损失 （s_sum）

        neg_loss = -torch.pow(1 - heatmap_t, beita) * torch.pow(heatmap, alpha) * torch.log(1 - heatmap + eps)  # 负样本损失 (N,1,H/4,W/4)

        center_loss = (pos_loss.sum() + neg_loss.sum()) / N

        '''compute scale&offset loss'''
        scale_ph = torch.clamp(scale[idx, 0, cy, cx], max=math.log(5000 / 4.0))
        scale_pw = torch.clamp(scale[idx, 1, cy, cx], max=math.log(5000 / 4.0))
        #scale_ph = scale[idx, 0, cy, cx]
        #scale_pw = scale[idx, 1, cy, cx]
        offset_ph = torch.tanh(offset[idx, 0, cy, cx])
        offset_pw = torch.tanh(offset[idx, 1, cy, cx])
        # all in cuda

        scale_th = torch.log(label_new[:, 3] / 4.0).cuda()
        scale_tw = torch.log(label_new[:, 4] / 4.0).cuda()
        offset_th = (label_new[:, 2]/4.0 - (cy.float() + 0.5)).cuda()
        offset_tw = (label_new[:, 1]/4.0 - (cx.float() + 0.5)).cuda()

        # L1 loss
        if self.use_smooth == False :
            diff_s_h = torch.abs(scale_th - scale_ph)
            diff_s_w = torch.abs(scale_tw - scale_pw)
            diff_o_h = torch.abs(offset_th - offset_ph)
            diff_o_w = torch.abs(offset_tw - offset_pw)

            scale_loss = (diff_s_h.sum() + diff_s_w.sum()) / N
            offset_loss = (diff_o_h.sum() + diff_o_w.sum()) / N
        # smooth L1 loss
        else :

            diff_s_h = torch.abs(scale_th - scale_ph)
            diff_s_w = torch.abs(scale_tw - scale_pw)
            diff_o_h = torch.abs(offset_th - offset_ph)
            diff_o_w = torch.abs(offset_tw - offset_pw)

            scale_loss = (torch.where(torch.le(diff_s_h, 1.0 / 9.0),
                                      0.5 * 9.0 * torch.pow(diff_s_h, 2),
                                      diff_s_h - 0.5 / 9.0).sum() +
                          torch.where(torch.le(diff_s_w, 1.0 / 9.0),
                                      0.5 * 9.0 * torch.pow(diff_s_w, 2),
                                      diff_s_w - 0.5 / 9.0).sum()
                          ) / N

            offset_loss = (torch.where(torch.le(diff_o_h, 1.0 / 9.0),
                                       0.5 * 9.0 * torch.pow(diff_o_h, 2),
                                       diff_o_h - 0.5 / 9.0).sum() +
                           torch.where(torch.le(diff_o_w, 1.0 / 9.0),
                                       0.5 * 9.0 * torch.pow(diff_o_w, 2),
                                       diff_o_w - 0.5 / 9.0).sum()
                           ) / N

        '''compute theta loss'''
        refer_theta = theta[idx,:,cy,cx]
        smoothlabel = torch.cat(smoothlabel,dim=0).cuda()

        t_theta = label_new[:, 5].long()
        pos_angle = refer_theta[torch.arange(len(t_theta)), t_theta]
        pos_angle_loss = -torch.pow(1 - pos_angle, alpha) * torch.log(pos_angle + eps)
        neg_angle_loss = -torch.pow(1 - smoothlabel, beita) * torch.pow(refer_theta, alpha) * torch.log(1 - refer_theta + eps)
        theta_loss = (pos_angle_loss.sum() + neg_angle_loss.sum()) / N



        return center_loss, scale_loss, offset_loss, theta_loss








