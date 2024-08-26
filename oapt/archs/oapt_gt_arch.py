# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------
import sys
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY
import torch.nn.functional as F
import copy
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SFB(nn.Module):
    def __init__(self, dim=180, out = 180, kernel=3):
        super().__init__()
        self.res_branch1 = nn.Sequential(
                        nn.Conv2d(dim, dim, kernel, 1, kernel//2),
                        nn.LeakyReLU(),
                        nn.Conv2d(dim, dim, kernel, 1, kernel//2)
                        )
        self.res_branch2_head = nn.Sequential(
                        nn.Conv2d(dim, dim, kernel, 1, kernel//2),
                        nn.LeakyReLU()
                        )
        self.res_branch2_body = nn.Sequential(
                        nn.Conv2d(2*dim, 2*dim, kernel, 1, kernel//2),
                        nn.LeakyReLU()
                        )
        self.res_branch2_tail = nn.Conv2d(dim, dim, 1, 1)
        self.conv = nn.Conv2d(2*dim, out, 1, 1)
    def forward(self,x):
        _, _, H, W = x.shape
        x1 = x + self.res_branch1(x)
        x2 = self.res_branch2_head(x)
        y = torch.fft.rfft2(x2)
        y_imag = y.imag
        y_real = y.real
        y = torch.cat([y_real, y_imag], dim=1)
        y = self.res_branch2_body(y)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W))
        x2 = y + x2
        x2 = self.res_branch2_tail(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x

def check_image_size_(x, pad_size):
    b, h, w, c = x.shape
    mod_pad_h = (pad_size - h % pad_size) % pad_size
    mod_pad_w = (pad_size - w % pad_size) % pad_size
    x = x.permute(0,3,1,2)
    try:
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "reflect")
    except BaseException:
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), "constant")
    x = x.permute(0,2,3,1)
    return x
        
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
        h, after check patchsize
        w,
    """
    x = check_image_size_(x, window_size)
    B, H, W, C = x.shape
    x = x.contiguous().view(B, H // window_size, window_size, W // window_size, window_size, C) #
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, H, W

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.contiguous().view(B, H // window_size, W // window_size, window_size, window_size, -1) #
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def degrade_partition(x, offset, CatMode=True):
    """
    Args:
        x: (B, H, W, C)
        offset list (int): double jpeg compression non-align stride
        CatMode (bool): cat the same degraded patch together
    Returns:
        output: degraded parts list , H, W
    """
    B, _, _, C = x.shape
    # 每个8x8块内划分，[0:8-x1,0:8-x2], [0:8-x1, 8-x2:8], [8-x1:8, 0:8-x2], [8-x1:8,8-x2:8] x1,x2~[0,4]
    x_8x8, H, W = window_partition(x, 8) # (H//8* W//8 *B, 8, 8, C)
    x_degade1 = x_8x8[:, 0:8-offset[0], 0:8-offset[1], :] # like windows style
    if offset[1]== 0:
        x_degade2 = None
    else:
        x_degade2 = x_8x8[:, 0:8-offset[0], 8-offset[1]:8, :]
    if offset[0]== 0:
        x_degade3 = None
    else:
        x_degade3 = x_8x8[:, 8-offset[0]:8, 0:8-offset[1], :]
        
    if offset[0]== 0 or offset[1]== 0:
        x_degade4 = None
    else:
        x_degade4 = x_8x8[:, 8-offset[0]:8, 8-offset[1]:8, :]
    
    del x_8x8
    if CatMode:
        # cat the same degrade parts   .contiguous().view()  .reshape()
        if x_degade1 is not None:
            x_degade1 = x_degade1.contiguous().view(B, (8-offset[0])*(H//8), (8-offset[1])*(W//8), C)
        if x_degade2 is not None:
            x_degade2 = x_degade2.contiguous().view(B, (8-offset[0])*(H//8), offset[1]*(W//8), C)
        if x_degade3 is not None:
            x_degade3 = x_degade3.contiguous().view(B, offset[0]*(H//8), (8-offset[1])*(W//8), C)
        if x_degade4 is not None:
            x_degade4 = x_degade4.contiguous().view(B, offset[0]*(H//8), offset[1]*(W//8), C)

    return [x_degade1, x_degade2, x_degade3, x_degade4], H, W

def degrade_reverse(parts, H, W, offset, CatMode=True):
    """
    Args:
        parts: 4 regions of degraded parts (B, hhh, www, C) hhh可能是拼凑之后的，也可能是拼凑之前的；
        window_size(int): window partition size, 0 for no more window partition
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    part1, part2, part3, part4 = parts[0], parts[1], parts[2], parts[3]
    del parts
    if part1 is not None:
        B, _, _, C = part1.shape
    if part2 is not None:
        B, _, _, C = part2.shape
    if part3 is not None:
        B, _, _, C = part3.shape
    if part4 is not None:
        B, _, _, C = part4.shape
    # catmode
    if CatMode:
        # the same parts are cat together
        if part1 is not None:
            part1 = part1.contiguous().view(B*H//8*W//8, 8-offset[0], 8-offset[1], C) #B*W//8*H//8, (8-offset), (8-offset), C
        if part2 is not None:
            part2 = part2.contiguous().view(B*H//8*W//8, 8-offset[0], offset[1], C)
        if part3 is not None:
            part3 = part3.contiguous().view(B*H//8*W//8, offset[0], 8-offset[1], C)
        if part4 is not None:
            part4 = part4.contiguous().view(B*H//8*W//8, offset[0], offset[1], C)
    # reverse
    x_8x8 = torch.zeros((B*H//8*W//8, 8, 8, C), dtype=torch.float32, device=part1.device)
    if part1 is not None:
        x_8x8[:, 0:8-offset[0], 0:8-offset[1], :] = part1
    if part2 is not None:
        x_8x8[:, 0:8-offset[0], 8-offset[1]:8, :] = part2
    if part3 is not None:
        x_8x8[:, 8-offset[0]:8, 0:8-offset[1], :] = part3
    if part4 is not None:
        x_8x8[:, 8-offset[0]:8, 8-offset[1]:8, :] = part4
    del part1, part2, part3, part4
    x = window_reverse(x_8x8, 8, H, W)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0: 
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows, _, _ = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def parts_attn(self,x1):
        if x1 is not None:
            B, H1, W1, C = x1.shape
            # cyclic shift
            if self.shift_size > 0:
                x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x1, h1, w1 = window_partition(x1, self.window_size) 
            # calculate mask
            if self.shift_size>0:
                mask1 = self.calculate_mask([h1, w1]).to(x1.device)
            else:
                mask1 = None
            # SW-MSA for degraded region
            # self-attention
            x1 = self.attn(x1.contiguous().view(-1, self.window_size * self.window_size, C), mask=mask1) #
            del mask1
            # merge windows
            x1 = x1.contiguous().view(-1, self.window_size, self.window_size, C) # 
            x1 = window_reverse(x1, self.window_size, h1, w1)
            # reverse cyclic shift
            if self.shift_size > 0 :
                x1 = torch.roll(x1[:,:H1,:W1,:], shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x1 = x1[:,:H1,:W1,:]
        return x1


    def forward(self, x, x_size, offset):
        # print(f"shift:{self.shift_size}")
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
    
        if isinstance(offset, int):
            offset_h, offset_w = offset, offset
        else:
            offset = offset[0]
            offset_h, offset_w = int(torch.round(offset[0]/1.)), int(torch.round(offset[1]/1.))
        
        offset = [offset_h, offset_w]

        if offset_h+offset_w > 0:
            # partition degraded region into 4 parts 有可能存在None的区域，例如offset=[0,2]
            [x1, x2, x3, x4], h, w = degrade_partition(x, offset, CatMode=True)
            # catmode : window attention
            x1 = self.parts_attn(x1)
            x2 = self.parts_attn(x2)
            x3 = self.parts_attn(x3)
            x4 = self.parts_attn(x4)
            # merge degraded parts
            shifted_x = degrade_reverse([x1,x2,x3,x4],h, w, offset, CatMode=True)
            x = shifted_x[:,:H,:W,:]
        else:
            # cyclic shift
            if self.shift_size > 0:
                # print('shift window attention')
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x

            # partition windows
            shifted_x, h1, w1 = window_partition(shifted_x, self.window_size)
            # W-MSA 
            if self.input_resolution == x_size:
                shifted_x = self.attn(shifted_x.contiguous().view(-1, self.window_size * self.window_size, C), mask=self.attn_mask) 
            else:
                if(self.shift_size>0):
                    mask1 = self.calculate_mask([h1,w1]).to(x.device)               
                else:
                    mask1 = None
                shifted_x = self.attn(shifted_x.contiguous().view(-1, self.window_size * self.window_size, C), mask=mask1)
            # merge windows
            shifted_x = shifted_x.contiguous().view(-1, self.window_size, self.window_size, C) #
            shifted_x = window_reverse(shifted_x, self.window_size, h1, w1)

            # reverse cyclic shift
            if self.shift_size > 0:
                x = torch.roll(shifted_x[:,:H,:W,:], shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x[:,:H,:W,:]

        del shifted_x

        x = x.contiguous().view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, DenseFirstPositions, mode,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.DenseFirstPositions = DenseFirstPositions
        self.mode = mode

        # build blocks
        if self.mode == 'interval':
            self.interval = 1 # 忽略shift，offset 和 dense 交替出现
            self.blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    shift_size=0 if ((i // 2) % 2 == 0) else window_size // 2, 
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer)
                for i in range(depth)])
        else:
            self.interval = 2 #offset、offsetshift 在一起，dense、denseShift在一起，二者交替出现
            self.blocks = nn.ModuleList([
                SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer)
                for i in range(depth)])
        


        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, offset):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                if self.DenseFirstPositions:
                    if (i//self.interval) % 2 == 0: #先local window 再 degarded partition window
                        x = blk(x, x_size, 0) 
                    else:
                        # print('here print out the feature...')
                        # x_numpy = copy.deepcopy(x.view(1,240,240,180)).cpu().detach().numpy()
                        # np.save('before_pc.npy',x_numpy)
                        # print(x_numpy.shape)
                        x = blk(x, x_size, offset)
                        # x_numpy = copy.deepcopy(x.view(1,240,240,180)).cpu().detach().numpy()
                        # np.save('after_pc.npy',x_numpy)
                        # print(x_numpy.shape)
                        # sys.exit(0)
                else:
                    if (i//self.interval) % 2 == 0: #先degarded partition window 再 local window
                        x = blk(x, x_size, offset) 
                    else:
                        x = blk(x, x_size, 0)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


# class BasicLayer(nn.Module):
#     """ A basic Swin Transformer layer for one stage.

#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#         use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
#     """

#     def __init__(self, dim, input_resolution, depth, num_heads, window_size, DenseFirstPositions, mode,
#                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
#                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.use_checkpoint = use_checkpoint
#         self.DenseFirstPositions = DenseFirstPositions
#         self.mode = mode

#         # build blocks
#         if self.mode == 'interval':
#             self.interval = 1 # 忽略shift，offset 和 dense 交替出现
#             self.blocks = nn.ModuleList([
#                 SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
#                                     num_heads=num_heads, window_size=window_size,
#                                     shift_size=0 if ((i // 2) % 2 == 0) else window_size // 2, 
#                                     mlp_ratio=mlp_ratio,
#                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                     drop=drop, attn_drop=attn_drop,
#                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                     norm_layer=norm_layer)
#                 for i in range(depth)])
#         else:
#             self.interval = 2 #offset、offsetshift 在一起，dense、denseShift在一起，二者交替出现
#             self.blocks = nn.ModuleList([
#                 SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
#                                     num_heads=num_heads, window_size=window_size,
#                                     shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                     mlp_ratio=mlp_ratio,
#                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                     drop=drop, attn_drop=attn_drop,
#                                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                     norm_layer=norm_layer)
#                 for i in range(depth)])
        


#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None

#     def forward(self, x, x_size, offset):
#         for i, blk in enumerate(self.blocks):
#             if self.use_checkpoint:
#                 x = checkpoint.checkpoint(blk, x)
#             else:
#                 if self.DenseFirstPositions:
#                     if (i//self.interval) % 2 == 0: #串联，先local window 再 degarded partition window
#                         x = blk(x, x_size, 0) 
#                     else:
#                         x = blk(x, x_size, offset)
#                 else:
#                     if (i//self.interval) % 2 == 0: #串联，先degarded partition window 再 local window
#                         x = blk(x, x_size, offset) 
#                     else:
#                         x = blk(x, x_size, 0)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops

class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, DenseFirstPositions, mode,
                mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                        input_resolution=input_resolution,
                                        depth=depth,
                                        num_heads=num_heads,
                                        window_size=window_size,
                                        DenseFirstPositions=DenseFirstPositions,
                                        mode=mode,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop, attn_drop=attn_drop,
                                        drop_path=drop_path,
                                        norm_layer=norm_layer,
                                        downsample=downsample,
                                        use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                    nn.Conv2d(dim // 4, dim, 3, 1, 1))
        elif resi_connection == 'sfb':
            self.conv = SFB(dim = dim, out = dim, kernel = 3)

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size, offset):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, offset), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops

@ARCH_REGISTRY.register()
class OAPT_gt(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                window_size=7, DenseFirstPositions=True,mode='interval',
                mlp_ratio=4., qkv_bias=True, qk_scale=None,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                **kwargs):
        super(OAPT_gt, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0], patches_resolution[1]),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        DenseFirstPositions=DenseFirstPositions,
                        mode=mode,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint,
                        img_size=img_size,
                        patch_size=patch_size,
                        resi_connection=resi_connection
                        )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        elif resi_connection == 'sfb':
            self.conv_after_body = SFB(dim = embed_dim, out = embed_dim, kernel = 3)

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                    nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
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

    def forward_features(self, x, offset):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        # i = 0
        for layer in self.layers:
            x = layer(x, x_size, offset)
        #     if i==1:
        #         sys.exit(0)
        #     i += 1
        # sys.exit(0)
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, offset):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, offset)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, offset)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, offset)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # for image denoising and JPEG compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, offset)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        
        return x

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
  
    height = 224
    width = 224
    model = OAPT_gt(
        upscale=1,
        in_chans=1,
        img_size=224,
        window_size=7,
        DenseFirstPositions=True,
        mode='non-interval',
        img_range=255.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='',
        resi_connection='1conv')
    # print(model)
    # print(height, width, model.flops() / 1e9)
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model,(1, height, width),as_strings=True,print_per_layer_stat=False)
    print('MACs:  ' + macs)
    print('Params: ' + params)

    # import cv2
    # import numpy as np
    # I = cv2.imread("/home/moqiao/workplace/dataset/Classic5/barbara.bmp")
    # sh = I.shape
    # cv2.imwrite("I.png",I)
    # I = torch.tensor(np.expand_dims(I,0))
    # offset = [0,4]
    # Is,H,W = degrade_partition(I,offset,CatMode=True)
    # if Is[0] is not None:
    #     Is1 = np.array(Is[0][0])
    #     cv2.imwrite("s1.png",Is1)
    # if Is[1] is not None:
    #     Is2 = np.array(Is[1][0])
    #     cv2.imwrite("s2.png",Is2)
    # if Is[2] is not None:
    #     Is3 = np.array(Is[2][0])
    #     cv2.imwrite("s3.png",Is3)
    # if Is[3] is not None:
    #     Is4 = np.array(Is[3][0])
    #     cv2.imwrite("s4.png",Is4)
    
    
    # Ii = np.array(degrade_reverse(Is, H, W, offset, CatMode=True)[0][:sh[0],:sh[1],:])
    # cv2.imwrite("Ii.png",Ii)


