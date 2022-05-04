import torch.nn as nn
import torch
import math
import sys
sys.path.append('./nets')
from utils.aware_Gaussian_focal_loss import losses
from resnet_dcn_DFPN import get_pose_net
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class ResNet(nn.Module):

    def __init__(self, num_layers,heads):
        super(ResNet, self).__init__()

        self.backbone=get_pose_net(num_layers=num_layers,heads=heads)
        self.losses=losses()

        self.sig = nn.Sigmoid()

    def forward(self, input):

        if self.training:
            x=input['img'] #cuda
            label=input['label']
            heatmap_t=input['heatmap_t'] #cuda
            smooth_label= input['smooth_label']
        else :
            x = input

        out=self.backbone(x)[0]
        heatmap=self.sig(out['hm'])#(N,15,H/4,W/4)
        scale=out['wh']#(N,2,H/4,W/4)
        offset=out['reg']#(N,2,H/4,W/4)
        theta=self.sig(out['theta'])#(N,180,H/4,W/4)

        if self.training:
            return self.losses(heatmap,scale,offset,theta,heatmap_t,label,smooth_label)
        else:
            return [heatmap,scale,offset,theta]


