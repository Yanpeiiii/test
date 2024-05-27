import math
import numpy as np
import torch
import torch.nn as nn
import copy

from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.modules.utils import _pair


class PreNorm(nn.Module):
    def __init__(self, channel_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(channel_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return self.fn(x, **kwargs)


class DepthWise(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, groups=dim_in, stride=stride,
                      bias=bias),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=(1, 1), bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class AttentionDW(nn.Module):
    def __init__(self, dim_in, dim_out, stride_kv, stride_q, kernel_size=3, padding_kv=1,
                 padding_q=1, qkv_bias=False, attn_drop=0., proj_drop=0., num_heads=4):
        super(AttentionDW, self).__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.scale = dim_out ** -0.5
        self.num_heads = num_heads

        self.conv_proj_q = self._build_projection(dim_in, dim_out, kernel_size, padding=padding_q, stride=stride_q)
        self.conv_proj_k = self._build_projection(dim_in, dim_out, kernel_size, padding=padding_kv, stride=stride_kv)
        self.conv_proj_v = self._build_projection(dim_in, dim_out, kernel_size, padding=padding_kv, stride=stride_kv)

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.conv_proj = nn.Conv2d(dim_out, dim_out, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attend = nn.Softmax(dim=-1)

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride,):
        proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in),
            nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False),
        )

        return proj

    def forward(self, x):
        b, n, h, w = x.shape
        head = self.num_heads
        q = self.conv_proj_q(x)
        k = self.conv_proj_k(x)
        v = self.conv_proj_v(x)

        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b (h w) c')
        v = rearrange(v, 'b c h w -> b (h w) c')

        q = rearrange(q, 'b t (h d) -> b h t d', h=head)
        k = rearrange(k, 'b t (h d) -> b h t d', h=head)
        v = rearrange(v, 'b t (h d) -> b h t d', h=head)

        attn_score = torch.einsum('b h l k, b h t k -> b h l t', q, k) * self.scale
        attn = self.attend(attn_score)
        attn = self.attn_drop(attn)
        out = torch.einsum('b h l t, b h t v -> b h l v', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', h=head, x=h, y=w)

        out = self.conv_proj(out)
        out = self.proj_drop(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SpatialTransformer(nn.Module):
    def __init__(self, depth, dim_in, dim_out, stride_kv, stride_q, padding_q, padding_kv, attn_drop=0., proj_drop=0.,
                 num_heads=8, mlp_mult=4, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim_in, AttentionDW(dim_in=dim_in, dim_out=dim_out, stride_kv=stride_kv, stride_q=stride_q,
                                            kernel_size=3, padding_q=padding_q, padding_kv=padding_kv, qkv_bias=False,
                                            attn_drop=attn_drop, proj_drop=proj_drop, num_heads=num_heads)),
                PreNorm(dim_out, FeedForward(dim_out, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
