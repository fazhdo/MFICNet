import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init)
from mmcv.cnn.bricks import DropPath
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

from mmdet3d.models.backbones import PointNet2SASSG
from .resnet import ResNet
from .seresnet import SEResNet
from mmseg.models.decode_heads import PSPHead2

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PSPUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)


class SENet(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channel, in_channel // 4, bias=False)
        self.fc2 = nn.Linear(in_channel // 4, in_channel, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.Sigmoid()
    
    def forward(self, x):
        # x = x.transpose(2, 1) # 4 1792 12288
        b, _, c = x.size()
        Fsq_out = self.gap(x.transpose(2, 1)).view(b, c) # 4 1792 1
        out = self.act2(self.fc2(self.act1(self.fc1(Fsq_out))))[:, None, :]  # 4 1 1792
        out = out.expand_as(x) #  4 12288 1792
        out = x * out # 对应元素相乘
        return out + x
    
'''class ReWeight(nn.Module):
     def __init__(self):
        super(ReWeight, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(128, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(128, 256, 1)
        self.conv2_rgb2 = torch.nn.Conv1d(256, 128, 1)
        self.conv2_cld2 = torch.nn.Conv1d(256, 128, 1)
        self.conv2_rgbd = torch.nn.Conv1d(256, 256, 1)
        self.conv2_rgbd2 = torch.nn.Conv1d(256, 128, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.act = nn.Sigmoid()
        
    def forward(self, rgb_f, cld_f, Ch):
        rgb = F.relu(self.conv2_rgb(rgb_f))
        rgb = F.relu(self.conv2_rgb2(rgb))
        cld = F.relu(self.conv2_cld(cld_f))
        cld = F.relu(self.conv2_cld2(cld))
        rgbd = torch.cat((rgb, cld), dim=1)
        Feat = F.relu(self.conv2_rgbd(rgbd))
        Feat = F.relu(self.conv2_rgbd2(Feat))
        Mask_pre = self.gap(Feat.transpose(2, 1))
        Mask = self.act(Mask_pre.transpose(2, 1))
        Mask = Mask.repeat(1, Ch, 1)

        return rgb_f * Mask, cld_f * (1 - Mask) '''
class ReWeight(nn.Module):
    def __init__(self):
        super(ReWeight, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(128, 128, 1)
        self.conv2_cld = torch.nn.Conv1d(128, 128, 1)
        self.conv2_rgbd = torch.nn.Conv1d(256, 128, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.act = nn.Sigmoid()
        
    def forward(self, rgb_f, cld_f, Ch):
        rgb = F.relu(self.conv2_rgb(rgb_f))
        cld = F.relu(self.conv2_cld(cld_f))
        rgbd = torch.cat((rgb, cld), dim=1)
        Feat = F.relu(self.conv2_rgbd(rgbd))
        Mask_pre = self.gap(Feat.transpose(2, 1))
        Mask = self.act(Mask_pre.transpose(2, 1))
        Mask = Mask.repeat(1, Ch, 1)

        return rgb_f * Mask, cld_f * (1 - Mask)

class DenseFusion(nn.Module):
    def __init__(self):
        super(DenseFusion, self).__init__()
        self.conv2_rgb = torch.nn.Conv1d(128, 256, 1)
        self.conv2_cld = torch.nn.Conv1d(128, 256, 1)

        self.conv3 = torch.nn.Conv1d(256, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(7500)
        # self.attn = SENet(1792)
        
        self.rew = ReWeight()

    def forward(self, rgb_emb, cld_emb): # ([4, 64, 12288] ([4, 256, 12288]
        bs, Ch, n_pts = cld_emb.size()

        rgb_emb, cld_emb = self.rew(rgb_emb, cld_emb, Ch)

        feat_1 = torch.cat((rgb_emb, cld_emb), dim=1)
        # feat_1 = feat_1.transpose(2, 1)
        # import pdb;pdb.set_trace()
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))

        feat_2 = torch.cat((rgb, cld), dim=1)

        rgbd = F.relu(self.conv3(feat_1))
        rgbd = F.relu(self.conv4(rgbd))

        ap_x = self.ap1(rgbd)
    
        ap_x = ap_x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        
        feat_num = torch.cat([feat_1, feat_2, ap_x], 1) # 256 + 512 + 1024 = 1792
        feat_num = feat_num.transpose(2, 1) # ([4, 12288, 1792])
        # feat_num = self.attn(feat_num)
        ''' bs, Ch, n_pts = cld_emb.size()
        # rgb_emb, cld_emb = self.rew(rgb_emb, cld_emb, Ch)
        rgb = F.relu(self.conv2_rgb(rgb_emb))
        cld = F.relu(self.conv2_cld(cld_emb))
        feat_num = torch.cat((rgb, cld), dim=1)
        feat_num = feat_num.transpose(2, 1) '''
        
        ''' rgb = F.relu(self.conv2_rgb(rgb_emb))
        feat_num = rgb.transpose(2, 1) '''
        return feat_num

@BACKBONES.register_module()
class SEResNet_PointNet2(nn.Module):
    def __init__(self, depth, dataset='7S', se_ratio=16, Pn_in_channels=3, Pn_num_points=(2048, 1024, 512, 256), 
                 Pn_radius=(0.2, 0.4, 0.8, 1.2), Pn_num_samples=(64, 32, 16, 16), 
                 Pn_sa_channels=((64, 64, 128), (128, 128, 256), (128, 128, 256),(128, 128, 256)), 
                 Pn_fp_channels=((256, 256), (256, 256), (256, 256), (256, 256)), psp_param=None, **kwargs):
        
        super(SEResNet_PointNet2, self).__init__()

        # Get resnet backbone
        self.cnnModle = SEResNet(depth, **kwargs)

        # Get psp head
        self.psphead = PSPHead2(**psp_param)

        # Get pointnet++ backbone
        self.pointnet2 = PointNet2SASSG(Pn_in_channels, Pn_num_points, Pn_radius, Pn_num_samples, Pn_sa_channels, Pn_fp_channels)

        # Feature fusion function
        self.densefusion = DenseFusion()

        # Up sample
        self.up_1 = PSPUpsample(512, 256) # size up 2 times
        self.up_2 = PSPUpsample(256, 128) # size up 2 times
        self.up_3 = PSPUpsample(128, 128) # size up 2 times

        # Drop out
        self.drop_1 = nn.Dropout2d(p=0.3)
        self.drop_2 = nn.Dropout2d(p=0.15)

        # Final layer
    

    def forward(self, x):
        cu_x = [item.to("cuda", non_blocking=True) for item in x]
        rgb, cld, choose = cu_x
        # rgb [4, 3, 480, 640]
        # cld [4, 12288, 3]
        # choose [4, 1, 12288]
        
        rgb_f = self.cnnModle(rgb) # [4, 512, 60, 80]
        rgb_f = self.psphead(rgb_f)
        # rgb_f = self.drop_1(rgb_f)
        rgb_f = self.up_1(rgb_f)
        # rgb_f = self.drop_2(rgb_f)
        rgb_f = self.up_2(rgb_f)
        rgb_f = self.up_3(rgb_f) # [4, 64, 480, 640]
        bs, di, _, _ = rgb_f.size()

        rgb_emb = rgb_f.view(bs, di, -1) # [4, 64, 480*640]
        choose = choose.repeat(1, di, 1) # [4, 1, 12288] -> [4, 64, 12288]
        rgb_emb = torch.gather(rgb_emb, 2, choose).contiguous() # [4, 64, 12288]
        
        dpt_f = self.pointnet2(cld) # [4, 256, 12288]
        # import pdb;pdb.set_trace()
        rgbd_f = self.densefusion(rgb_emb, dpt_f['fp_features'][4]) # [4, 320, 12288]
        
        return rgbd_f