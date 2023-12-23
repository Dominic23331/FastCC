import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_


class Linear_BN(nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    @torch.no_grad()
    def fuse(self):
        l, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[:, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.flatten(0, 1)).reshape_as(x)


class SFT(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 key_dim,
                 num_heads=8,
                 attn_ratio=2,
                 stride=2,
                 resolution=14,
                 resolution_=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio

        self.resolution_2 = resolution_
        h = self.dh + nh_kd
        self.kv = Linear_BN(in_dim, h)

        activation = torch.nn.Hardswish

        self.q = torch.nn.Sequential(
            Linear_BN(in_dim, nh_kd))
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim))

        self.stride = stride
        self.resolution = resolution

        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, resolution_, resolution))
        trunc_normal_(self.attention_biases)

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
                               1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale + self.attention_biases
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x
