# Copyright (c) OpenMMLab. All rights reserved.
import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import HEADS


@HEADS.register_module()
class RGBDHead(BaseModule):
    def __init__(self,
                in_channel=1792,
                norm_cfg=dict(type='BN'),
                init_cfg=None):
        super(RGBDHead, self).__init__(init_cfg=init_cfg)

        self.in_channel = in_channel
        self.conv_reg1 = nn.Sequential(
            nn.Linear(self.in_channel, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )

        self.conv_reg2 =nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.conv_reg3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self.coord_conv = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        self.coord_reg = torch.nn.Linear(64, 3)

        self.uncer_conv = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        self.uncer_reg = torch.nn.Linear(64, 1)

        # self.att = TransformerBlock(512, 8, 2, True, 'WithBias')
        # self.compensator = Feature_compensator(512, 16)

    def forward(self, feat, **kwargs):

        # global_feat = self.att(feat)
        # feat = self.compensator(global_feat, feat)
        feat = self.conv_reg3(self.conv_reg2(self.conv_reg1(feat)))
        coord = self.coord_reg(self.coord_conv(feat))
        uncer = self.uncer_reg(self.uncer_conv(feat))
        uncer = torch.sigmoid(uncer)
        return coord, uncer



# top-k attention---->maintained by xt
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv_3 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.qkv_dwconv_5 = nn.Conv2d(dim * 3, dim * 3, kernel_size=5, stride=1, padding=2, groups=dim * 3, bias=bias)
        self.qkv_dwconv_7 = nn.Conv2d(dim * 3, dim * 3, kernel_size=7, stride=1, padding=3, groups=dim * 3, bias=bias)

        self.qkv_pconv =  nn.Conv2d(dim * 9, dim * 3, kernel_size=1, stride=1, padding=0, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attns = nn.ParameterList([torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True) for _ in range(3)])

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = torch.cat([self.qkv_dwconv_3(self.qkv(x)), self.qkv_dwconv_5(self.qkv(x)), self.qkv_dwconv_7(self.qkv(x))], dim=1)
        qkv = self.qkv_pconv(qkv)

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        _, _, C, _ = q.shape

        attn_masks = []
        for top_k in [int(C/2), int(C*2/3), int(C*3/4)]:
            mask = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
            index = torch.topk(attn, k=top_k, dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            attn_mask = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))
            attn_mask = attn_mask.softmax(dim=-1)
            attn_masked_v = attn_mask @ v
            attn_masks.append(attn_masked_v)
        out = sum([attn_mask * attn for (attn_mask, attn) in zip(attn_masks, self.attns)])


        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class Feature_compensator(nn.Module):
    
    def __init__(self, channels=64, r=4):
        super(Feature_compensator, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        # import matplotlib.pyplot as plt
        # src_g = plt.imshow(wei[0].mean(dim=0)[5:-5, 5:-5].detach().cpu().numpy(), cmap="jet")
        # plt.colorbar(src_g)
        # plt.show()
        # src_l = plt.imshow((1 - wei[0].mean(dim=0))[5:-5, 5:-5].detach().cpu().numpy(), cmap="jet")
        # plt.colorbar(src_l)
        # plt.show()
        # exit()

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv_3 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv_5 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=5, stride=1, padding=2, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x3_1, x3_2 = self.dwconv_3(x).chunk(2, dim=1)
        x3 = F.gelu(x3_1) * x3_2

        x5_1, x5_2 = self.dwconv_5(x).chunk(2, dim=1)
        x5 = F.gelu(x5_1) * x5_2

        x = self.project_out(torch.cat([x3, x5], dim=1))
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):

        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##  Sparse Transformer Block (STB) 
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        import matplotlib.pyplot as plt
        x = x + self.attn(self.norm1(x))
        # import matplotlib.pyplot as plt
        # plt.imshow(x[0].mean(dim=0)[2:-2, 2:-2].detach().cpu().numpy(), cmap="jet")
        # plt.show()
        x = x + self.ffn(self.norm2(x))

        return x
