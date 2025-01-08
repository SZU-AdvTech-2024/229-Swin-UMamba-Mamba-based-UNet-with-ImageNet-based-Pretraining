import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

import logging

logger = logging.getLogger(__name__)

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels=768, atrous_rates=(6, 12, 18), out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)  #(B,C,H,W) dim=1
        return self.project(res)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    Reference: http://arxiv.org/abs/2401.10166
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchExpand(nn.Module):
    """
    Reference: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x) # B, H, W, 2C
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4) #B, 2H, 2W, C/2
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4) #B, 2H, 2W, C/2

        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Reference:
        - GitHub: https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/swin_transformer_unet_skip_expand_decoder_sys.py
        - Paper: https://arxiv.org/pdf/2105.05537.pdf
    """

    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # H, W = self.input_resolution
        x = x.permute(0, 2, 3, 1)  # B, C, H, W ==> B, H, W, C
        x = self.expand(x) #B, H, W, 16C
        B, H, W, C = x.shape
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))     #B, 4H, 4W, C/4
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        return x  # .permute(0, 3, 1, 2)


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape  #h,w,c -> h,w,c -> h/2,w/2,4c -> h/2,w/2,2c

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,  # input shape
            d_state=16,  # 隐状态shape
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,  # parameters range
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # delta步长的秩
        # initialize in_proj  输入映射
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        # data reflection
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        # weight parameter
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj
        # dt_proj ========
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # A D===========
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # 输入x经过x_proj映射得到数据依赖的三个参数B , C , Δ ，其中Δ \DeltaΔ 得到的维度是dt_rank，还需要进行一个(dt_rank, d_inner)的线性映射
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization   观测矩阵A初始化
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # （B,L,C)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSMEncoder(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, depths=[2, 2, 9, 2],
                 dims=[96, 192, 384, 768], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_ret = []
        x_ret.append(x)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 3, 1, 2))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)

        return x_ret  # ([1, 3, 256, 256]), ([1, 96, 64, 64]),([1, 192, 32, 32]),([1, 384, 16, 16]),([1, 768, 8, 8])

class LKPE(nn.Module):
    def __init__(self, dim: int, dim_scale: int = 2, norm_layer: Type[nn.Module] = nn.LayerNorm):
        super(LKPE, self).__init__()
        self.dim = dim
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=True)
        )
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = x.view(B, H, W, C)
        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x)
        x = x.reshape(B, H * 2, W * 2, C // 4)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return x

class FLKPE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dim_scale: int = 4,
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super(FLKPE, self).__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Sequential(
            nn.Conv2d(dim, dim * 16, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim * 16, dim * 16, kernel_size=3, padding=1, groups=dim * 16, bias=True)
        )

        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)
        self.out = nn.Conv2d(self.output_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)

        x = rearrange(x, pattern="b c h w -> b h w c")
        B, H, W, C = x.shape

        x = rearrange(x, pattern="b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)
        x = x.reshape(B, H * self.dim_scale, W * self.dim_scale, self.output_dim)

        x = rearrange(x, pattern="b h w c -> b c h w")
        return self.out(x)

class UNetResDecoder(nn.Module):
    def __init__(
            self,
            num_classes: int,
            deep_supervision,
            features_per_stage: Union[Tuple[int, ...], List[int]] = None,
            drop_path_rate: float = 0.2,
            d_state: int = 16,
    ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()

        encoder_output_channels = features_per_stage      #[96,192,384,768]
        self.deep_supervision = deep_supervision
        # self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder_output_channels)  #== 4

        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
        depths = [2, 2, 2, 2]

        # we start with the bottleneck and work out way up
        stages = []
        expand_layers = []
        seg_layers = []
        concat_back_dim = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder_output_channels[-s]
            input_features_skip = encoder_output_channels[-(s + 1)]
            expand_layers.append(PatchExpand(
                input_resolution=None,
                dim=input_features_below,
                dim_scale=2,
                norm_layer=nn.LayerNorm,
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(VSSLayer(
                dim=input_features_skip,
                depth=2,
                attn_drop=0.,
                drop_path=dpr[sum(depths[:s - 1]):sum(depths[:s])],
                d_state=math.ceil(2 * input_features_skip / 6) if d_state is None else d_state,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
            ))
            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))
            concat_back_dim.append(nn.Linear(2 * input_features_skip, input_features_skip))

        # for final prediction
        expand_layers.append(FinalPatchExpand_X4(
            input_resolution=None,
            dim=encoder_output_channels[0],
            dim_scale=4,
            norm_layer=nn.LayerNorm,
        ))
        stages.append(nn.Identity())
        seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
                x = self.concat_back_dim[s](x)
            x = self.stages[s](x).permute(0, 3, 1, 2)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

class DMC_fusion(nn.Module):
    def __init__(self, in_channels, up_kwargs=None):
        super(DMC_fusion, self).__init__()

        self.up_kwargs = up_kwargs

        self.conv4_2 = nn.Sequential(
            nn.Conv2d(in_channels[3], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[2]),
            nn.ReLU(inplace=True))
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels[2], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[1]),
            nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels[1], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True))
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels[0], in_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels[0]),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3, x4):
        x4_1 = x4
        x4_2 = F.interpolate(self.conv4_2(x4_1), scale_factor=2, **self.up_kwargs)
        x3_1 = x4_2 * (self.conv3_1(x3))
        x3_2 = F.interpolate(self.conv3_2(x3_1), scale_factor=2, **self.up_kwargs)
        x2_1 = x3_2 * (self.conv2_1(x2))
        x2_2 = F.interpolate(self.conv2_2(x2_1), scale_factor=2, **self.up_kwargs)
        x1_1 = x2_2 * (self.conv1_1(x1))

        return x1_1, x2_1, x3_1, x4_1

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class LPA(nn.Module):
    def __init__(self, in_channel):
        super(LPA, self).__init__()
        self.ca = ChannelAttention(in_channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x0, x1 = x.chunk(2, dim=2)
        x0 = x0.chunk(2, dim=3)
        x1 = x1.chunk(2, dim=3)
        x0 = [self.ca(x0[-2]) * x0[-2], self.ca(x0[-1]) * x0[-1]]
        x0 = [self.sa(x0[-2]) * x0[-2], self.sa(x0[-1]) * x0[-1]]

        x1 = [self.ca(x1[-2]) * x1[-2], self.ca(x1[-1]) * x1[-1]]
        x1 = [self.sa(x1[-2]) * x1[-2], self.sa(x1[-1]) * x1[-1]]

        x0 = torch.cat(x0, dim=3)
        x1 = torch.cat(x1, dim=3)
        x3 = torch.cat((x0, x1), dim=2)

        x4 = self.ca(x) * x
        x4 = self.sa(x4) * x4
        x = x3 + x4
        return x

class DMC1(nn.Module):
    def __init__(self, in_channels, width=384, up_kwargs=None):
        super(DMC1, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(2 * width, width, 1, padding=0, bias=False),  # 调整通道数1*1卷积
            nn.BatchNorm2d(width))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        # feat = torch.concat(tensors=(feats[0],feats[1]),dim=1)
        # feat = self.conv_out(feat)
        feat = feats[0]+feats[1]
        return feat


class DMC2(nn.Module):
    def __init__(self, in_channels, width=192, up_kwargs=None):
        super(DMC2, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(3 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        # feat = feats[0] * feats[1] * feats[2]
        # feat = torch.concat(tensors=(feats[0], feats[1],feats[2]), dim=1)
        feat = feats[0]+feats[1]+feats[2]
        # feat = self.conv_out(feat)
        return feat


class DMC3(nn.Module):
    def __init__(self, in_channels, width=96, up_kwargs=None):
        super(DMC3, self).__init__()
        self.up_kwargs = up_kwargs

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels[-4], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.conv_out = nn.Sequential(
            nn.Conv2d(4 * width, width, 1, padding=0, bias=False),
            nn.BatchNorm2d(width))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3]), self.conv2(inputs[-4])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), **self.up_kwargs)
        feats[-3] = F.interpolate(feats[-3], (h, w), **self.up_kwargs)
        feats[-4] = F.interpolate(feats[-4], (h, w), **self.up_kwargs)
        # feat = feats[0] * feats[1] * feats[2] * feats[3]
        feat = feats[0]+feats[1]+feats[2]+feats[3]
        # feat = self.conv_out(feat)
        return feat

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

# class UNetResDecoder(nn.Module):
#     def __init__(
#             self,
#             num_classes: int,
#             deep_supervision,
#             features_per_stage: Union[Tuple[int, ...], List[int]] = None,
#             drop_path_rate: float = 0.2,
#             d_state: int = 16,
#     ):
#         """
#         This class needs the skips of the encoder as input in its forward.
#
#         the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
#         are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
#         features and the lowest skip as inputs
#         the decoder has two (three) parts in each stage:
#         1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
#         2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
#         3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
#         :param encoder:
#         :param num_classes:
#         :param n_conv_per_stage:
#         :param deep_supervision:
#         """
#         super().__init__()
#
#         encoder_output_channels = features_per_stage           #[96,192,384,768]
#         self.deep_supervision = deep_supervision
#         # self.encoder = encoder
#         self.num_classes = num_classes
#         n_stages_encoder = len(encoder_output_channels)
#
#         dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder - 1) * 2)]
#         depths = [2, 2, 2, 2]
#
#         # we start with the bottleneck and work out way up
#         stages = []
#         expand_layers = []
#         seg_layers = []
#         concat_back_dim = []
#         for s in range(1, n_stages_encoder):
#             input_features_below = encoder_output_channels[-s]
#             input_features_skip = encoder_output_channels[-(s + 1)]
#             expand_layers.append(LKPE(
#                 dim=input_features_below,
#                 dim_scale=2,
#                 norm_layer=nn.LayerNorm,
#             ))
#             # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
#             stages.append(VSSLayer(
#                 dim=input_features_skip,
#                 depth=2,
#                 attn_drop=0.,
#                 drop_path=dpr[sum(depths[:s - 1]):sum(depths[:s])],
#                 d_state=math.ceil(2 * input_features_skip / 6) if d_state is None else d_state,
#                 norm_layer=nn.LayerNorm,
#                 downsample=None,
#                 use_checkpoint=False,
#             ))
#             # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
#             # then a model trained with deep_supervision=True could not easily be loaded at inference time where
#             # deep supervision is not needed. It's just a convenience thing
#             seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))
#             concat_back_dim.append(nn.Linear(2 * input_features_skip, input_features_skip))
#
#         # for final prediction
#         expand_layers.append(FLKPE(
#             dim=encoder_output_channels[0],
#             dim_scale=4,
#             norm_layer=nn.LayerNorm,
#         ))
#         stages.append(nn.Identity())
#         seg_layers.append(nn.Conv2d(input_features_skip, num_classes, 1, 1, 0, bias=True))
#
#         self.stages = nn.ModuleList(stages)
#         self.expand_layers = nn.ModuleList(expand_layers)
#         self.seg_layers = nn.ModuleList(seg_layers)
#         self.concat_back_dim = nn.ModuleList(concat_back_dim)
#
#     def forward(self, skips):
#         """
#         we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
#         :param skips:
#         :return:
#         """
#         lres_input = skips[-1]
#         seg_outputs = []
#         for s in range(len(self.stages)):
#             x = self.expand_layers[s](lres_input)
#             if s < (len(self.stages) - 1):
#                 x = torch.cat((x, skips[-(s + 2)].permute(0, 2, 3, 1)), -1)
#                 x = self.concat_back_dim[s](x)
#             x = self.stages[s](x).permute(0, 3, 1, 2)
#             if self.deep_supervision:
#                 seg_outputs.append(self.seg_layers[s](x))
#             elif s == (len(self.stages) - 1):
#                 seg_outputs.append(self.seg_layers[-1](x))
#             lres_input = x
#
#         # invert seg outputs so that the largest segmentation prediction is returned first
#         seg_outputs = seg_outputs[::-1]  #(1,1,256,256) (1,1,64,64)(1,1,32,32) (1,1,16,16)
#
#         if not self.deep_supervision:
#             r = seg_outputs[0]
#         else:
#             r = seg_outputs
#         return r

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
class SwinPA(nn.Module):
    def __init__(self, out_planes=1, encoder='vmamba_T',use_checkpoint=False):
        super(SwinPA, self).__init__()
        self.encoder = encoder
        if self.encoder == 'vmamba_T':
            mutil_channel = [96 ,192, 384, 768]
            self.vssm_encoder = VSSMEncoder(dim=96,drop_path_rate=0.2,use_checkpoint=use_checkpoint)
        elif self.encoder == 'vmamba_S':
            mutil_channel = [128, 256, 512, 1024]
            self.vssm_encoder = VSSMEncoder()
        elif self.encoder == 'vmamba_B':
            mutil_channel = [64, 128, 320, 512]
            self.vssm_encoder = VSSMEncoder() # [64, 128, 320, 512]
            # path = './pretrained_pth/pvt_v2_b2.pth'
            # save_model = torch.load(path)
            # model_dict = self.backbone.state_dict()
            # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            # model_dict.update(state_dict)
            # self.backbone.load_state_dict(model_dict)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dmc1 = DMC1([mutil_channel[2], mutil_channel[3]], width=mutil_channel[2], up_kwargs=up_kwargs)
        self.dmc2 = DMC2([mutil_channel[1], mutil_channel[2], mutil_channel[3]], width=mutil_channel[1],
                         up_kwargs=up_kwargs)
        self.dmc3 = DMC3([mutil_channel[0], mutil_channel[1], mutil_channel[2], mutil_channel[3]],
                         width=mutil_channel[0], up_kwargs=up_kwargs)
        self.DMC_fusion = DMC_fusion(mutil_channel, up_kwargs=up_kwargs)
        self.lpa1 = LPA(in_channel=mutil_channel[0])
        self.lpa2 = LPA(in_channel=mutil_channel[1])
        self.lpa3 = LPA(in_channel=mutil_channel[2])
        self.lpa4 = LPA(in_channel=mutil_channel[3])

        self.decoder = UNetResDecoder(num_classes=1,deep_supervision=False,features_per_stage=[96,192,384,768])

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.vssm_encoder(x)

        x1 = self.dmc3(x1, x2, x3, x4)  #乘法融合
        x2 = self.dmc2(x2, x3, x4)
        x3 = self.dmc1(x3, x4)
        x4 = x4
        x1, x2, x3, x4 = self.DMC_fusion(x1, x2, x3, x4)  # 整理为一个模块

        y1 = self.lpa1(x1)
        y2 = self.lpa2(x2)
        y3 = self.lpa3(x3)
        y4 = self.lpa4(x4)
        skip = [x0,y1,y2,y3,y4]
        out = self.decoder(skip)
        return out
