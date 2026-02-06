import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Mlp
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from inspect import isfunction
import fourier_enoder
from rope import SegmentedRoPEExpCached

def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def modulate(x, shift, scale):
    #return x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    return x * (1 + scale) + shift

def _softplus_inv(target):
    # 求 softplus(x)=target 的 x（即 softplus^{-1}(target)）
    # softplus(x) = ln(1+exp(x)) -> invert: x = ln(exp(target)-1)
    return math.log(math.exp(target) - 1.0 + 1e-12)

def get_cond(rx,ry,sx,sy):
    deltaX = (rx - sx)/2
    deltaY = (ry - sy)/2
    midX   = (rx + sx) / 2
    midY   = (ry + sy) / 2
    offset = torch.sqrt(deltaX**2 + deltaY**2)
    azimuth_rad = torch.arctan2(deltaY, deltaX)
    azimuth_deg = torch.from_numpy((np.degrees(azimuth_rad.cpu().numpy()) + 360.0) % 360.0)
    return deltaX, deltaY, midX, midY, offset,azimuth_deg


def normalize(arr, amin=None, amax=None):
    # linear map to [-1,1]; if amin/amax None, use arr min/max
    if amin is None: amin = torch.min(arr)
    if amax is None: amax = torch.max(arr)
    d = amax - amin if (amax - amin) != 0 else 1.0
    arr=(arr - amin) / d 
    return arr ,amin, amax


class AdaTimeModulation(nn.Module):
    def __init__(self, hidden_dim, time_dim, eps=1e-6):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 6)
        )
        nn.init.zeros_(self.time_proj[-1].weight)
        nn.init.zeros_(self.time_proj[-1].bias)
        self.rms = nn.functional.normalize      # L2 归一化代替 mean/std

    def forward(self, x, t):
        shift_msa, scale_msa, gate_msa,\
        shift_mlp, scale_mlp, gate_mlp = self.time_proj(t).chunk(6, dim=-1)

        def _mod(inp, shift, scale, gate):
            n = self.rms(inp, dim=(2, 3), eps=1e-6)      # HW 两维一起归一化
            return inp + gate[:, :, None, None] * (n * (1 + scale[:, :, None, None]) + shift[:, :, None, None])

        x = _mod(x, shift_msa, scale_msa, gate_msa)
        x = _mod(x, shift_mlp, scale_mlp, gate_mlp)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model: 词向量的维度
        max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        # 创建一个 max_len x d_model 的矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        # 偶数位置：sin，奇数位置：cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引维度

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]，为了广播匹配batch维度
        self.register_buffer('pe', pe)  # 不作为参数更新

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        返回：添加了位置编码后的张量
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.device)


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=16, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels: int):

        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.act = MYact()  
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)
        
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        # 输出维度(batch_size, time_channels)
        return emb

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        QWEN3 
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.shift = nn.Parameter(torch.zeros(hidden_size))
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

class MYact(nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.gelu=nn.GELU()
        self.silu=nn.SiLU()
    def forward(self, x):
        return self.silu(x)

class Emb(nn.Module):
    def __init__(self, d_model: int, minLog: float, maxLog: float):
        super().__init__()
        assert d_model % 4 == 0, "d_model 必须能被 4 整除，当前为 {}".format(d_model)

        # 二维坐标用的频率向量：长度 = d_model // 4
        f_pair = torch.exp(torch.linspace(minLog, maxLog, d_model // 4))
        # 标量用的频率向量：长度 = d_model // 2
        f_scalar = torch.exp(torch.linspace(minLog, maxLog, d_model // 2)

        )
        self.register_buffer("f_pair", f_pair, persistent=False)
        self.register_buffer("f_scalar", f_scalar, persistent=False)

        self.linear = nn.Conv2d(
            d_model, d_model,
            kernel_size=1, stride=1, padding=0, bias=True
        )

        # 可选的缩放参数，先保留（下面我给出怎么用）
        self.alpha = nn.Parameter(torch.tensor(_softplus_inv(1.0)), requires_grad=True)
        self.beta  = nn.Parameter(torch.tensor(_softplus_inv(1.0)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入有两种合法模式：

        1) 标量模式（scalar mode）
           x: [B, H, W] 或 [B, 1, H, W]
           表示每个空间位置一个标量（例如时间、logτ、某个属性）

        2) 二维坐标模式（pair mode）
           x: [B, H, W, 2]
           表示每个空间位置一个二维向量（例如 (offset, azimuth)）

        输出：
           out: [B, d_model, H, W]
        """
        # ---------- 二维坐标模式：x[...,2] ----------
        if x.dim() == 4 and x.shape[-1] == 2:
            # x: [B, H, W, 2]
            B, H, W, _ = x.shape
            x1 = x[..., 0]        # [B, H, W]
            x2 = x[..., 1]        # [B, H, W]
            x1 = x1.unsqueeze(1)  # [B, 1, H, W]
            x2 = x2.unsqueeze(1)  # [B, 1, H, W]
            #频率 reshape 成 [1, F, 1, 1] 方便 broadcast
            f = self.f_pair.view(1, -1, 1, 1)  # F = d_model//4
            # Fourier features
            x1 = x1 * f * (2 * math.pi)       # [B, F, H, W]
            x2 = x2 * f * (2 * math.pi)
            x1_enc = torch.cat([torch.sin(x1), torch.cos(x1)], dim=1)  # [B, 2F, H, W]
            x2_enc = torch.cat([torch.sin(x2), torch.cos(x2)], dim=1)  # [B, 2F, H, W]
            # 2F + 2F = 4F = d_model
            x_enc = torch.cat([x1_enc, x2_enc], dim=1)                 # [B, d_model, H, W]
        # ---------- 标量模式：x 是一个场 ----------
        else:
            # 允许 x 是 [B, H, W] 或 [B, 1, H, W]
            if x.dim() == 3:
                # [B, H, W] -> [B, 1, H, W]
                x = x.unsqueeze(1)
            elif x.dim() == 4 and x.shape[1] == 1:
                # 已经是 [B,1,H,W] 直接用
                pass
            else:
                raise ValueError(
                    f"标量模式下，x 期望形状为 [B,H,W] 或 [B,1,H,W]，"
                    f"当前为 {x.shape}"
                )
            B, C, H, W = x.shape
            # f_scalar 长度 = d_model//2
            f = self.f_scalar.view(1, -1, 1, 1)  # [1, F, 1, 1]
            x = x * f * (2 * math.pi)          # [B, F, H, W]
            x_enc = torch.cat([torch.sin(x), torch.cos(x)], dim=1)  # [B, 2F, H, W] = [B,d_model,H,W]
        out = self.linear(x_enc)  # [B, d_model, H, W]
        # 如果你想用 alpha / beta 做一个可学习的残差缩放，可以这样打开：
        # a = F.softplus(self.alpha)
        # b = F.softplus(self.beta)
        # out = a * out + b * x_enc     # out 里混合了原始 Fourier feature 和线性变换
        return out
##Encoder
##conv1d for time
class Resblock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        # 确保groups数量合适
        #print(self.n_groups,in_channels)
        # 第一层
        self.norm1 = GroupNorm(
            num_channels=in_channels,
        )
        self.act1 = MYact()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1,3),
            padding=(0,1),
        )
        
        # 第二层
        self.norm2 = GroupNorm(
            num_channels=out_channels,
        )
        self.act2 = MYact()
        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1,3),
            padding=(0,1),
        )
        
        # 残差连接
        if in_channels != out_channels:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels,kernel_size=1,padding=0)
        else:
            self.shortcut = torch.nn.Identity()
            
        # 时间编码
        self.adaLN=AdaTimeModulation(time_dim=time_channels,hidden_dim=out_channels)
        self.time_emb = torch.nn.Linear(time_channels, out_channels)
        self.time_act = MYact()
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # 主路径
        #print(x.shape)
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # 时间调制
        h = h + self.time_emb(self.time_act(t))[:, :, None, None]
        #h=self.adaLN(h,t)
        
        # 第二层处理
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        # 残差连接
        return h + self.shortcut(x)

class Downsample(nn.Module):
    """
    时间维度下采样
    """
    def __init__(self, n_channels, i,stride:int=2):
        super().__init__()
        # 只对时间维度操作
        self.conv_0 = nn.Conv2d(n_channels, n_channels, (1, stride+1), stride=(1, stride), padding=(0, stride//2),)
        self.conv_1 = nn.Conv2d(n_channels, n_channels, (1, stride+1), stride=(1, stride), padding=(0, stride//2),)
        self.conv_2 = nn.Conv2d(n_channels, n_channels, (1, stride+1), stride=(1, stride), padding=(0, stride//2),)
        self.i = i
        self.conv_list = torch.nn.ModuleList([self.conv_0,self.conv_1,self.conv_2])

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv_list[self.i](x)

class WidthUpsample1D(nn.Module):
    def __init__(self, in_channels,out_channels,width_scale):
        super().__init__()
        self.width_scale = width_scale
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * width_scale,  
            kernel_size=(5,3),
            padding=(2,1),
            stride=1,
        )
    def forward(self, x):
        # x: (B, C, H, W)
        #print(x.shape)
        x = self.conv(x)  # (B, C × r, H, W)
        B, C_r, H, W = x.shape
        C = C_r // self.width_scale
        x = x.view(B, C, self.width_scale, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # (B, C, H, W, r)
        x = x.reshape(B, C, H, W * self.width_scale)  # (B, C, H, W × r)
        return x

class WidthUpsample_Block(nn.Module):
    def __init__(self, n_channels, stride=2):
        super().__init__()
        self.upsample = WidthUpsample1D(
            in_channels=n_channels,
            out_channels=n_channels,
            width_scale=stride,
        )
    def forward(self, x):
        # (B, C, H, W) -> (B, C, H, W * stride)
        x_up = self.upsample(x)
       #residual = x_up
        #out = self.conv1(x_up)
        #out = self.act(out)
        #out = self.conv2(out)
        return x_up #+ residual

class Upsample(nn.Module):
    """
    时间维度上采样
    """
    def __init__(self, n_channels, i,stride=2):
        super().__init__()
        ##Note: 这里stride=2,即每次上采样2倍
        self.conv_0 = WidthUpsample_Block(n_channels, stride)
        #self.conv_0 = nn.ConvTranspose2d(n_channels, n_channels, (1, stride*2), stride=(1, stride), padding=(0, stride//2))
        self.conv_1 = WidthUpsample_Block(n_channels, stride)
        #self.conv_1 = nn.ConvTranspose2d(n_channels, n_channels, (1, stride*2), stride=(1, stride), padding=(0, stride//2))
        self.conv_2 = WidthUpsample_Block(n_channels, stride)
        #self.conv_2 = nn.ConvTranspose2d(n_channels, n_channels, (1, stride*2), stride=(1, stride), padding=(0, stride//2))
        self.i = i
        #self.conv_list = torch.nn.ModuleList([self.conv_0, self.conv_1, self.conv_2])
        self.conv_list = torch.nn.ModuleList([self.conv_0,self.conv_1,self.conv_2])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv_list[self.i](x)


class RelativePositionBias(nn.Module):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size  # 窗口大小 (D_win, T_win)，用于构建相对位置索引
        self.num_heads = num_heads
        self.rel_pos_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        self.register_buffer("relative_position_index", self.get_position_index())

    def get_position_index(self):
        coords_d = torch.arange(self.window_size[0])
        coords_t = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_d, coords_t], indexing="ij"))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1]  - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def forward(self):
        return self.rel_pos_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        ).permute(2, 0, 1)


class WindowAttention2D(nn.Module):
    #window_size=(15,8), num_heads=8, shift_size=(5,4),
    ##window_size=(9,8), num_heads=8, shift_size=(3,4),
    #window_size=(17,8), num_heads=8, shift_size=(8,4)
    #def __init__(self, dim, window_size=(16,8), shift_size=(8,4), num_heads=8, attn_drop=0.1, proj_drop=0.1)
    def __init__(
        self, 
        dim, 
        window_size=(32,1248//8), 
        shift_size=(16,0), 
        num_heads=8, 
        attn_drop=0.1, 
        proj_drop=0.1,
        qk_norm=False,
        *,
        use_rope: bool = True,
        rope_n_pos: int = 4,
        rope_min_log: float = -12,
        rope_max_log: float = 0,
        rope_mapper: str = "linear",
        rope_hidden: int = 128,
        rope_p_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_bias = RelativePositionBias(window_size, num_heads)
        self.last_score_map = None
        
        # RoPE 相关参数（与 TraceAxisAttention2D 保持一致）
        self.use_rope = bool(use_rope)
        self.rope_dim = None
        self.rope = None
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        if self.use_rope:
            _rope_dim = self.head_dim 
            _rope_dim = (_rope_dim // 2) * 2  
            if _rope_dim < 2:
                self.use_rope = False
            else:
                self.rope_dim = _rope_dim
                self.rope = SegmentedRoPEExpCached(
                    D=self.rope_dim * self.num_heads,
                    N=self.num_heads,
                    n_pos=int(rope_n_pos),
                    min_log=float(rope_min_log),
                    max_log=float(rope_max_log),
                    mapper=str(rope_mapper),
                    hidden=int(rope_hidden),
                    p_scale=float(rope_p_scale),
                )
    
    @staticmethod
    def _default_trace_pos(B, H, device):
        """
        默认 trace 位置: 归一化到 [0,1] 的索引，shape = [B, H, 1]
        用 float32 生成，后续在 RoPE 内部按 out_dtype 再转换
        """
        if H <= 1:
            pos_1d = torch.zeros((1,), device=device, dtype=torch.float32)
        else:
            pos_1d = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=torch.float32)
        return pos_1d.view(1, H, 1).expand(B, H, 1)

    ##2d attention mask
    def create_attn_mask(self, B, D, T, win_h, win_w, shift_h, shift_w, device):
        # 无 shift -> 不需要 mask
        if shift_h == 0 and shift_w == 0:
            return None

        # 要求 shift < window（Swin 的约束），若不满足则取模
        if not (0 < shift_h < win_h):
            shift_h = shift_h % win_h
        if not (0 < shift_w < win_w):
            shift_w = shift_w % win_w

        num_win_h = D // win_h
        num_win_w = T // win_w
        assert D % win_h == 0 and T % win_w == 0, "D/T must be divisible by window size"

        # img_mask: [D, T], 每个像素标记其所属 region id（3x3 区域分配）
        img_mask = torch.zeros((D, T), device=device, dtype=torch.long)

        cnt = 0
        # 和 Swin 一致的三段切分（左/中/右或上/中/下）
        h_slices = (slice(0, -win_h), slice(-win_h, -shift_h), slice(-shift_h, None))
        w_slices = (slice(0, -win_w), slice(-win_w, -shift_w), slice(-shift_w, None))

        for h in h_slices:
            for w in w_slices:
                img_mask[h, w] = cnt
                cnt += 1

        # 分窗并得到窗口 id 序列
        mask_windows = img_mask.view(num_win_h, win_h, num_win_w, win_w).permute(0, 2, 1, 3).reshape(-1, win_h * win_w)
        # mask_windows: [num_windows, N]
        # attn_mask_bool: [num_windows, N, N] True 表示不同 region -> 需屏蔽
        attn_mask_bool = (mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2))  # [num_windows, N, N]
        # 扩展到 heads dim 前的 shape: [num_windows, 1, N, N]
        attn_mask = attn_mask_bool.unsqueeze(1).to(torch.bool)  # bool
        # 扩展到 batch：B * num_windows
        attn_mask = attn_mask.repeat(B, 1, 1, 1)  # [B * num_windows, 1, N, N]
        return attn_mask
    
    def create_trace_keep_mask(self, B, D, T,win_h, win_w, shift_h, num_win_h, num_win_w, device):
        """
        1d trace attention mask
        仅仅计算地震道之间的相关性
        返回 bool mask：True=允许注意力，False=屏蔽
        形状对齐 trace-axis 注意力：最终会扩展到 [B*num_windows*win_w, win_h, win_h]
        """
        if shift_h == 0:
            return None

        # 1D Swin-style 三段切分，只在 D 维
        if not (0 < shift_h < win_h):
            shift_h = shift_h % win_h

        img_mask = torch.zeros((D,), device=device, dtype=torch.long)
        cnt = 0
        h_slices = (slice(0, -win_h), slice(-win_h, -shift_h), slice(-shift_h, None))
        for h in h_slices:
            img_mask[h] = cnt
            cnt += 1

        # 每个 D-window 的 region id: [num_win_h, win_h]
        mask_windows_d = img_mask.view(num_win_h, win_h)

        # 扩展到所有 T-window： [num_win_h*num_win_w, win_h]
        mask_windows = mask_windows_d[:, None, :].expand(num_win_h, num_win_w, win_h).reshape(num_win_h * num_win_w, win_h)

        # keep mask: 同 region 才允许注意力 -> True allowed
        keep = (mask_windows.unsqueeze(1) == mask_windows.unsqueeze(2))  # [num_win_h*num_win_w, win_h, win_h]

        # 扩到 batch： [B*num_windows, win_h, win_h]
        keep = keep.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_win_h * num_win_w, win_h, win_h)

        # 你的注意力是对每个窗口的每个 time 列单独算，所以再 repeat_interleave(win_w)
        keep = keep.repeat_interleave(win_w, dim=0)  # [B*num_windows*win_w, win_h, win_h]
        return keep

    def forward(self, x, pos=None):
        """
        x: [B, D, T, C]
        pos(可选): [B, D, rope_n_pos] 或 [B, D, 4]，对应每个 D 位置的位置编码
        returns: same shape
        """
        B, D, T, C = x.shape
        win_h, win_w = self.window_size
        shift_h, shift_w = self.shift_size
        shift_w = 0 
        # basic checks
        assert D % win_h == 0 and T % win_w == 0, f"D/T must be divisible by window size: got D={D},T={T},win_h={win_h},win_w={win_w}"
        if shift_h >= win_h or shift_w >= win_w:
            shift_h = shift_h % win_h
            shift_w = shift_w % win_w

        num_win_h = D // win_h
        num_win_w = T // win_w
        num_windows = B * num_win_h * num_win_w
        N = win_h * win_w
        
        x_shifted = x
        pos_shifted = pos
        if shift_h > 0 or shift_w > 0:
            x_shifted = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(1, 2))
            if pos is not None:
                pos_shifted = torch.roll(pos, shifts=(-shift_h,), dims=(1,))

        x_windows = x_shifted.view(B, num_win_h, win_h, num_win_w, win_w, C)
        x_windows = x_windows.permute(0, 1, 3, 2, 4, 5).reshape(B * num_win_h * num_win_w, win_h, win_w, C)  # [B*num_windows, win_h, win_w, C]

        pos_windows = None
        if pos_shifted is not None:
            P = pos_shifted.shape[-1]
            pos_windows = pos_shifted.reshape(B, num_win_h, win_h, P)
            pos_windows = pos_windows[:, :, None, :, :].expand(B, num_win_h, num_win_w, win_h, P)
            pos_windows = pos_windows.reshape(B * num_win_h * num_win_w, win_h, P)

        x_windows_reshaped = x_windows.permute(0, 2, 1, 3).contiguous()  # [B*num_windows, win_w, win_h, C]
        x_windows_reshaped = x_windows_reshaped.view(B * num_win_h * num_win_w * win_w, win_h, C)  # [B*num_windows*win_w, win_h, C]
        
        qkv = self.qkv(x_windows_reshaped)  # [B*num_windows*win_w, win_h, 3*C]
        qkv = qkv.view(B * num_win_h * num_win_w * win_w, win_h, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*num_windows*win_w, num_heads, win_h, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B*num_windows*win_w, num_heads, win_h, head_dim]
        
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
    
        if self.use_rope and self.rope is not None and self.rope_dim is not None:
            num_windows = B * num_win_h * num_win_w
            if pos_windows is not None:
                pos_in = pos_windows  # [B*num_windows, win_h, rope_n_pos]
            else:
                pos_in = self._default_trace_pos(num_windows, win_h, device=x.device)  # [B*num_windows, win_h, 1]
            
            # 计算 cos/sin，shape: [B*num_windows, heads, win_h, half]
            self.rope.precompute_cos_sin(pos_in, out_dtype=q.dtype, device=q.device)
            cos = self.rope.cos  # [B*num_windows, heads, win_h, half]
            sin = self.rope.sin  # [B*num_windows, heads, win_h, half]

            rope_dim = self.rope_dim
            half = rope_dim // 2

            q = q.contiguous().view(B * num_win_h * num_win_w, win_w, self.num_heads, win_h, self.head_dim)
            k = k.contiguous().view(B * num_win_h * num_win_w, win_w, self.num_heads, win_h, self.head_dim)

            q_tail = q[..., rope_dim:]
            k_tail = k[..., rope_dim:]

            q_rot = q[..., :rope_dim].contiguous().view(B * num_win_h * num_win_w, win_w, self.num_heads, win_h, half, 2)
            k_rot = k[..., :rope_dim].contiguous().view(B * num_win_h * num_win_w, win_w, self.num_heads, win_h, half, 2)

            q_even, q_odd = q_rot[..., 0], q_rot[..., 1]  # [B*num_windows, win_w, heads, win_h, half]
            k_even, k_odd = k_rot[..., 0], k_rot[..., 1]

            cos_ = cos.unsqueeze(1)  # [B*num_windows, 1, heads, win_h, half]
            sin_ = sin.unsqueeze(1)  # [B*num_windows, 1, heads, win_h, half]

            q_even2 = q_even * cos_ - q_odd * sin_
            q_odd2 = q_even * sin_ + q_odd * cos_
            k_even2 = k_even * cos_ - k_odd * sin_
            k_odd2 = k_even * sin_ + k_odd * cos_

            q_rot2 = torch.stack([q_even2, q_odd2], dim=-1).view(B * num_win_h * num_win_w, win_w, self.num_heads, win_h, rope_dim)
            k_rot2 = torch.stack([k_even2, k_odd2], dim=-1).view(B * num_win_h * num_win_w, win_w, self.num_heads, win_h, rope_dim)

            q = torch.cat([q_rot2, q_tail], dim=-1).view(B * num_win_h * num_win_w * win_w, self.num_heads, win_h, self.head_dim).contiguous()
            k = torch.cat([k_rot2, k_tail], dim=-1).view(B * num_win_h * num_win_w * win_w, self.num_heads, win_h, self.head_dim).contiguous()
       
        attn_mask = None
        if shift_h > 0:
            keep_mask = self.create_trace_keep_mask(
                B=B, D=D, T=T,
                win_h=win_h, win_w=win_w,
                shift_h=shift_h,
                num_win_h=num_win_h, num_win_w=num_win_w,
                device=x.device
            )  # keep_mask: [B*, L, L], True=允许
            attn_mask = (~keep_mask).unsqueeze(1).to(torch.bool).contiguous()  # [B*,1,L,L], True=屏蔽

        if hasattr(F, "scaled_dot_product_attention") and torch.__version__ >= "2.0.0":
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,  
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*,heads,L,L]
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn_output = attn @ v
        
        # [B*num_windows*win_w, num_heads, win_h, head_dim] -> [B*num_windows*win_w, win_h, C]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B*num_windows*win_w, win_h, num_heads, head_dim]
        attn_output = attn_output.view(B * num_win_h * num_win_w * win_w, win_h, C)  # [B*num_windows*win_w, win_h, C]
        
        out = self.proj(attn_output)  # [B*num_windows*win_w, win_h, C]
        out = self.proj_drop(out)
        
        out = out.view(B * num_win_h * num_win_w, win_w, win_h, C)  # [B*num_windows, win_w, win_h, C]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B*num_windows, win_h, win_w, C]
        out = out.view(B, num_win_h, num_win_w, win_h, win_w, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, D, T, C)

        if shift_h > 0 or shift_w > 0:
            out = torch.roll(out, shifts=(shift_h, shift_w), dims=(1, 2))

        return out


# ========== 新增：Trace-axis global attention ==========
class TraceAxisAttention2D(nn.Module):
    """
    Trace-axis global attention: 对每个时间位置 w，在 H 维度上做全局多头自注意力。

    位置编码:
    - 默认开启 RoPE（Rotary Positional Embedding），使用 `rope.py::SegmentedRoPEExpCached`
    - RoPE 作用在 q/k 上，沿 trace 维（H）旋转
    - 若 forward 不传 pos，则使用归一化的 trace 索引作为 pos（[0,1]）
    
    输入: x: [B, H, W, C]
    输出: [B, H, W, C]
    
    机制: 对每个 w，将 H 条道作为序列，做全局 MHSA。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.1,
        qk_norm=True,
        *,
        use_rope: bool = True,
        rope_n_pos: int = 4,
        rope_min_log: float = -12,
        rope_max_log: float = 0,
        rope_mapper: str = "linear",
        rope_hidden: int = 128,
        rope_p_scale: dict = None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_rope = bool(use_rope)
        self.rope_dim = None
        self.rope = None
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
           self.q_norm = None
           self.k_norm = None
        if self.use_rope:
            _rope_dim = self.head_dim 
            _rope_dim = (_rope_dim // 2) * 2  
            if _rope_dim < 2:
                self.use_rope = False
            else:
                self.rope_dim = _rope_dim
                self.rope = SegmentedRoPEExpCached(
                    D=self.rope_dim * self.num_heads,
                    N=self.num_heads,
                    n_pos=int(rope_n_pos),
                    min_log=float(rope_min_log),
                    max_log=float(rope_max_log),
                    mapper=str(rope_mapper),
                    hidden=int(rope_hidden),
                    p_scale=rope_p_scale,
                )

    @staticmethod
    def _default_trace_pos(B, H, device):
        """
        默认 trace 位置: 归一化到 [0,1] 的索引，shape = [B, H, 1]
        用 float32 生成，后续在 RoPE 内部按 out_dtype 再转换
        """
        if H <= 1:
            pos_1d = torch.zeros((1,), device=device, dtype=torch.float32)
        else:
            pos_1d = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=torch.float32)
        return pos_1d.view(1, H, 1).expand(B, H, 1)

    def forward(self, x, pos=None):
        """
        x: [B, H, W, C]
        返回: [B, H, W, C]

        pos(可选): [B, H] 或 [B, H, rope_n_pos]，对应每条道的“位置/坐标”特征
        """
        B, H, W, C = x.shape
        
        # 对每个时间位置 w，在 H 维度上做注意力
        # reshape: [B, H, W, C] -> [B*W, H, C]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()  # [B, W, H, C]
        x_reshaped = x_reshaped.view(B * W, H, C)  # [B*W, H, C]
        qkv = self.qkv(x_reshaped)  # [B*W, H, 3*C]
        qkv = qkv.view(B * W, H, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()  # [3, B*W, num_heads, H, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B*W, num_heads, H, head_dim]
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.use_rope and self.rope is not None and self.rope_dim is not None:
            if pos is None:
                pos_in = self._default_trace_pos(B, H, device=x.device)  # [B,H,1]
            else:
                if pos.dim() == 2:
                    pos_in = pos.unsqueeze(-1)
                else:
                    pos_in = pos
                if pos_in.shape[0] != B or pos_in.shape[1] != H:
                    raise RuntimeError(
                        f"TraceAxisAttention2D: pos shape must be [B,H] or [B,H,n_pos], "
                        f"got {tuple(pos.shape)} while x is {tuple(x.shape)}"
                    )
            self.rope.precompute_cos_sin(pos_in, out_dtype=q.dtype, device=q.device)
            cos = self.rope.cos
            sin = self.rope.sin

            rope_dim = self.rope_dim
            half = rope_dim // 2

            # reshape 成 [B, W, heads, H, head_dim]，避免把 W 展开到 batch 导致 cos/sin 复制
            q = q.contiguous().view(B, W, self.num_heads, H, self.head_dim)
            k = k.contiguous().view(B, W, self.num_heads, H, self.head_dim)

            q_tail = q[..., rope_dim:]
            k_tail = k[..., rope_dim:]

            q_rot = q[..., :rope_dim].contiguous().view(B, W, self.num_heads, H, half, 2)
            k_rot = k[..., :rope_dim].contiguous().view(B, W, self.num_heads, H, half, 2)

            q_even, q_odd = q_rot[..., 0], q_rot[..., 1]  # [B,W,heads,H,half]
            k_even, k_odd = k_rot[..., 0], k_rot[..., 1]

            cos_ = cos.unsqueeze(1)  # [B,1,heads,H,half]
            sin_ = sin.unsqueeze(1)  # [B,1,heads,H,half]

            q_even2 = q_even * cos_ - q_odd * sin_
            q_odd2 = q_even * sin_ + q_odd * cos_
            k_even2 = k_even * cos_ - k_odd * sin_
            k_odd2 = k_even * sin_ + k_odd * cos_

            q_rot2 = torch.stack([q_even2, q_odd2], dim=-1).view(B, W, self.num_heads, H, rope_dim)
            k_rot2 = torch.stack([k_even2, k_odd2], dim=-1).view(B, W, self.num_heads, H, rope_dim)

            q = torch.cat([q_rot2, q_tail], dim=-1).view(B * W, self.num_heads, H, self.head_dim).contiguous()
            k = torch.cat([k_rot2, k_tail], dim=-1).view(B * W, self.num_heads, H, self.head_dim).contiguous()
        
        # Scaled dot-product attention
        if hasattr(F, 'scaled_dot_product_attention') and torch.__version__ >= '2.0.0':
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False
            )  # [B*W, num_heads, H, head_dim]
        else:
            # 回退到手写实现
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*W, num_heads, H, H]
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn_output = attn @ v  # [B*W, num_heads, H, head_dim]
        
        # 恢复形状: [B*W, num_heads, H, head_dim] -> [B*W, H, C]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B*W, H, num_heads, head_dim]
        attn_output = attn_output.view(B * W, H, C)  # [B*W, H, C]
        
        # 输出投影
        out = self.proj(attn_output)  # [B*W, H, C]
        out = self.proj_drop(out)
        
        # 恢复原始形状: [B*W, H, C] -> [B, H, W, C]
        out = out.view(B, W, H, C)  # [B, W, H, C]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, H, W, C]
        
        return out

class TraceAxisAttention2D_gate(nn.Module):
    """
    Trace-axis global attention: 对每个时间位置 w，在 H 维度上做全局多头自注意力。

    位置编码:
    - 默认开启 RoPE（Rotary Positional Embedding），使用 `rope.py::SegmentedRoPEExpCached`
    - RoPE 作用在 q/k 上，沿 trace 维（H）旋转
    - 若 forward 不传 pos，则使用归一化的 trace 索引作为 pos（[0,1]）
    
    输入: x: [B, H, W, C]
    输出: [B, H, W, C]
    
    机制: 对每个 w，将 H 条道作为序列，做全局 MHSA。
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.1,
        qk_norm=True,
        *,
        ##gate cfg
        headwise_attn_output_gate:bool = False,
        elementwise_attn_output_gate:bool = True,
        ##rope cfg
        use_rope: bool = True,
        rope_n_pos: int = 4,
        rope_min_log: float = -12,
        rope_max_log: float = 0,
        rope_mapper: str = "linear",
        rope_hidden: int = 128,
        rope_p_scale: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        #self.qkv = nn.Linear(dim, dim * 3, bias=False)
        ##gate init
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        assert headwise_attn_output_gate or elementwise_attn_output_gate, "at least one of headwise_attn_output_gate or elementwise_attn_output_gate must be True"
        if headwise_attn_output_gate:
            self.q = nn.Linear(dim, dim+self.num_heads, bias=False)
        elif elementwise_attn_output_gate:
            self.q = nn.Linear(dim, dim * 2, bias=False)
        else:
            self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_rope = bool(use_rope)
        self.rope_dim = None
        self.rope = None
        
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
           self.q_norm = None
           self.k_norm = None
        if self.use_rope:
            _rope_dim = self.head_dim 
            _rope_dim = (_rope_dim // 2) * 2  
            if _rope_dim < 2:
                self.use_rope = False
            else:
                self.rope_dim = _rope_dim
                self.rope = SegmentedRoPEExpCached(
                    D=self.rope_dim * self.num_heads,
                    N=self.num_heads,
                    n_pos=int(rope_n_pos),
                    min_log=float(rope_min_log),
                    max_log=float(rope_max_log),
                    mapper=str(rope_mapper),
                    hidden=int(rope_hidden),
                    p_scale=float(rope_p_scale),
                )

    @staticmethod
    def _default_trace_pos(B, H, device):
        """
        默认 trace 位置: 归一化到 [0,1] 的索引，shape = [B, H, 1]
        用 float32 生成，后续在 RoPE 内部按 out_dtype 再转换
        """
        if H <= 1:
            pos_1d = torch.zeros((1,), device=device, dtype=torch.float32)
        else:
            pos_1d = torch.linspace(0.0, 1.0, steps=H, device=device, dtype=torch.float32)
        return pos_1d.view(1, H, 1).expand(B, H, 1)

    def forward(self, x, pos=None,return_weights=False):
        """
        x: [B, H, W, C]
        返回: [B, H, W, C]

        pos(可选): [B, H] 或 [B, H, rope_n_pos]，对应每条道的“位置/坐标”特征
        """
        B, H, W, C = x.shape
        
        # 对每个时间位置 w，在 H 维度上做注意力
        # reshape: [B, H, W, C] -> [B*W, H, C]
        x_reshaped = x.permute(0, 2, 1, 3).contiguous()  # [B, W, H, C]
        x_reshaped = x_reshaped.view(B * W, H, C)  # [B*W, H, C]
        q = self.q(x_reshaped)
        k = self.k(x_reshaped)
        k = k.view(B * W, H,self.num_heads, self.head_dim).transpose(1, 2)  # -> [BW, heads, H, head_dim]
        v = self.v(x_reshaped)
        v = v.view(B * W, H,self.num_heads, self.head_dim).transpose(1, 2)  # -> [BW, heads, H, head_dim]
        if self.headwise_attn_output_gate:
            q, gate_score = torch.split(q, [self.head_dim * self.num_heads, self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B * W, H, self.num_heads, 1)
            q = q.reshape(B * W, H, self.num_heads,self.head_dim).transpose(1, 2)
        elif self.elementwise_attn_output_gate:
            q, gate_score = torch.split(q, [self.head_dim * self.num_heads, self.head_dim * self.num_heads], dim=-1)
            gate_score = gate_score.reshape(B * W, H, self.num_heads, self.head_dim)
            q = q.reshape(B * W, H, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = q.reshape(B * W, H, self.num_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        if self.use_rope and self.rope is not None and self.rope_dim is not None:
            if pos is None:
                pos_in = self._default_trace_pos(B, H, device=x.device)  # [B,H,1]
            else:
                if pos.dim() == 2:
                    pos_in = pos.unsqueeze(-1)
                else:
                    pos_in = pos
                if pos_in.shape[0] != B or pos_in.shape[1] != H:
                    raise RuntimeError(
                        f"TraceAxisAttention2D: pos shape must be [B,H] or [B,H,n_pos], "
                        f"got {tuple(pos.shape)} while x is {tuple(x.shape)}"
                    )
            self.rope.precompute_cos_sin(pos_in, out_dtype=q.dtype, device=q.device)
            cos = self.rope.cos
            sin = self.rope.sin

            rope_dim = self.rope_dim
            half = rope_dim // 2

            # reshape 成 [B, W, heads, H, head_dim]，避免把 W 展开到 batch 导致 cos/sin 复制
            q = q.contiguous().view(B, W, self.num_heads, H, self.head_dim)
            k = k.contiguous().view(B, W, self.num_heads, H, self.head_dim)

            q_tail = q[..., rope_dim:]
            k_tail = k[..., rope_dim:]

            q_rot = q[..., :rope_dim].contiguous().view(B, W, self.num_heads, H, half, 2)
            k_rot = k[..., :rope_dim].contiguous().view(B, W, self.num_heads, H, half, 2)

            q_even, q_odd = q_rot[..., 0], q_rot[..., 1]  # [B,W,heads,H,half]
            k_even, k_odd = k_rot[..., 0], k_rot[..., 1]

            cos_ = cos.unsqueeze(1)  # [B,1,heads,H,half]
            sin_ = sin.unsqueeze(1)  # [B,1,heads,H,half]

            q_even2 = q_even * cos_ - q_odd * sin_
            q_odd2 = q_even * sin_ + q_odd * cos_
            k_even2 = k_even * cos_ - k_odd * sin_
            k_odd2 = k_even * sin_ + k_odd * cos_

            q_rot2 = torch.stack([q_even2, q_odd2], dim=-1).view(B, W, self.num_heads, H, rope_dim)
            k_rot2 = torch.stack([k_even2, k_odd2], dim=-1).view(B, W, self.num_heads, H, rope_dim)

            q = torch.cat([q_rot2, q_tail], dim=-1).view(B * W, self.num_heads, H, self.head_dim).contiguous()
            k = torch.cat([k_rot2, k_tail], dim=-1).view(B * W, self.num_heads, H, self.head_dim).contiguous()
        
        # Scaled dot-product attention
        if hasattr(F, 'scaled_dot_product_attention') and torch.__version__ >= '2.0.0':
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False 
            )  # [B*W, num_heads, H, head_dim]
        else:
            # 回退到手写实现
            attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*W, num_heads, H, H]
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            attn_output = attn @ v  # [B*W, num_heads, H, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B*W, H, num_heads, head_dim]
        if self.headwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)
        elif self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)
        else:
            attn_output = attn_output
        attn_output = attn_output.view(B * W, H, C)  # [B*W, H, C]
        out = self.proj(attn_output)  # [B*W, H, C]
        out = self.proj_drop(out)
        out = out.view(B, W, H, C)  # [B, W, H, C]
        out = out.permute(0, 2, 1, 3).contiguous()  # [B, H, W, C]
        return out

class DiTBlock_windows(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads,mlp_ratio=4.0,):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention2D(dim=hidden_size,num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        #self.time_emb=nn.Linear(time_dim,hidden_size)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    def forward(self, x,c,rope_pos=None):
        #print(c.shape)
        #print(x.shape)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        #print(shift_msa.shape,scale_msa.shape,gate_msa.shape,shift_mlp.shape,scale_mlp.shape,gate_mlp.shape)
        x1=modulate(self.norm1(x), shift_msa.unsqueeze(-2),scale_msa.unsqueeze(-2))
        x = x + gate_msa.unsqueeze(-2)* self.attn(x1,pos=rope_pos)
        x = x + gate_mlp.unsqueeze(-2) * self.mlp(modulate(self.norm2(x), shift_mlp.unsqueeze(-2), scale_mlp.unsqueeze(-2)))
        return x


# ========== 新增：使用 Trace-axis attention 的 DiTBlock ==========
class DiTBlockTrace_noRoPE(nn.Module):
    """
    DiT block with Trace-axis global attention instead of WindowAttention2D.
    保持 adaLN-Zero conditioning 逻辑不变。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = TraceAxisAttention2D(dim=hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x, c):
        """
        x: [B, H, W, C]
        c: [B, H, C] (fourier_emb)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x1 = modulate(self.norm1(x), shift_msa.unsqueeze(-2), scale_msa.unsqueeze(-2))
        x = x + gate_msa.unsqueeze(-2) * self.attn(x1)
        x = x + gate_mlp.unsqueeze(-2) * self.mlp(modulate(self.norm2(x), shift_mlp.unsqueeze(-2), scale_mlp.unsqueeze(-2)))
        return x

class DiTBlockTrace(nn.Module):
    """
    DiT block with Trace-axis global attention instead of WindowAttention2D.
    保持 adaLN-Zero conditioning 逻辑不变。
    支持通过 rope_pos 参数传递位置信息给 RoPE。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,rope_p_scale=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = TraceAxisAttention2D(dim=hidden_size, num_heads=num_heads,rope_p_scale=rope_p_scale)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x, c, rope_pos=None):
        """
        x: [B, H, W, C]
        c: [B, H, C] (fourier_emb)
        rope_pos: 可选，[B, H] 或 [B, H, n_pos]，用于 RoPE 的位置编码
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x1 = modulate(self.norm1(x), shift_msa.unsqueeze(-2), scale_msa.unsqueeze(-2))
        x = x + gate_msa.unsqueeze(-2) * self.attn(x1, pos=rope_pos)
        x = x + gate_mlp.unsqueeze(-2) * self.mlp(modulate(self.norm2(x), shift_mlp.unsqueeze(-2), scale_mlp.unsqueeze(-2)))
        return x

class DiTBlockTrace_gate(nn.Module):
    """
    DiT block with Trace-axis global attention instead of WindowAttention2D.
    保持 adaLN-Zero conditioning 逻辑不变。
    支持通过 rope_pos 参数传递位置信息给 RoPE。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = TraceAxisAttention2D_gate(dim=hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.1)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x, c, rope_pos=None):
        """
        x: [B, H, W, C]
        c: [B, H, C] (fourier_emb)
        rope_pos: 可选，[B, H] 或 [B, H, n_pos]，用于 RoPE 的位置编码
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x1 = modulate(self.norm1(x), shift_msa.unsqueeze(-2), scale_msa.unsqueeze(-2))
        x = x + gate_msa.unsqueeze(-2) * self.attn(x1, pos=rope_pos)
        x = x + gate_mlp.unsqueeze(-2) * self.mlp(modulate(self.norm2(x), shift_mlp.unsqueeze(-2), scale_mlp.unsqueeze(-2)))
        return x

class DiTBlockNoCond(nn.Module):
    """
    与 DiTBlock 结构保持一致，但不使用任何条件
    从而使中间 attention/MLP 模块完全不受位置/条件控制。
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WindowAttention2D(dim=hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c=None):
        # 忽略 c
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

'''class SeisDiT(torch.nn.Module):
    ##adaLN-zero
    def __init__(
        self,
        image_channels,
        n_channels=64,
        channel=[1,2,2,4],
        d_model=512,
        nhead=4,
        dropout=0.1,
        num_layers=12,
        output_channels=1,
        res_blocks=1,
        strides=[2,2,2,1],
        f_dict=None,
        pe_type='transformer',      
        #label_dim=5
    ):
        super(SeisDiT, self).__init__()
        # alpha = (2*num_layers)**0.25
        # beta =  (8*num_layers)**(-0.25)
        self.image_channels = image_channels
        self.n_channels= n_channels
        self.channel = channel
        n_res=len(channel)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.num_layers = num_layers

        self.tokenizer = torch.nn.Conv2d(
            1, n_channels//2, kernel_size=(1, 1), padding=(0, 0),bias=True
        )
        self.tokenizer_cond = torch.nn.Conv2d(
            1, n_channels//2, kernel_size=(1, 1), padding=(0, 0),bias=True
        )
        self.mask_adapter_n=torch.nn.Conv2d(
            n_channels, n_channels, kernel_size=(1, 1), padding=(0, 0),bias=True
        )
        self.mask_adapter_d = torch.nn.Conv2d(
            d_model, d_model, kernel_size=(1, 1), padding=(0, 0),bias=True
        )

        self.time_emb = TimeEmbedding(d_model)
        self.fourier_encoder=fourier_enoder.Seismic5DEncoder(coord_dim=4,max_freq=128,out_dim=d_model,num_bands=32,pe_type = pe_type)
        last_channel = n_channels*channel[-1]*channel[-2]*channel[-3]

        self.to_attn = torch.nn.Conv2d(last_channel, d_model, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=True)
        self.to_unet = torch.nn.Conv2d(d_model, last_channel, kernel_size=(1,1), stride=(1,1), padding=(0,0), bias=True)
        attenL =[]
        for i in range(num_layers):
            attenL.append(DiTBlockTrace(hidden_size=d_model,num_heads=nhead))
            # attenL.append(Attention_Block(d_model=d_model))
        self.attenL = torch.nn.ModuleList(attenL)
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        down = []  
        out_channels = in_channels = n_channels
        for i in range(n_res):
            out_channels = in_channels * channel[i]
            for _ in range(res_blocks): 
                # print(out_channels,in_channels)
                down.append(
                    Resblock(in_channels, out_channels, d_model,)
                )
                in_channels = out_channels
            if i < n_res - 1:  
                down.append(Downsample(in_channels,i,strides[i]))

        self.down = torch.nn.ModuleList(down)
        up = []
        in_channels = out_channels  
        for i in reversed(range(n_res)):
            out_channels = in_channels
            for _ in range(res_blocks):
                up.append(
                    Resblock(in_channels+out_channels, out_channels,d_model,)
                )
            out_channels = in_channels // channel[i]
            up.append(
                Resblock(in_channels+out_channels, out_channels,d_model,)
            )  
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels, i - 1,strides[i-1]))
        self.up = torch.nn.ModuleList(up)
        self.ac=MYact()
        self.norm = torch.nn.GroupNorm(16,in_channels,eps=1e-5)
        self.final = torch.nn.Conv2d(
            in_channels, output_channels, kernel_size=(1, 5), padding=(0, 2),stride=(1,1),bias=True
        )
        self.A=torch.nn.Conv2d(in_channels,output_channels,kernel_size=(1,5),stride=(1,1),padding=(0,2),bias=True)
        nn.init.zeros_(self.A.weight)
        nn.init.zeros_(self.A.bias)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,condL=None,log_tau=None,time_axis=None,training=False):
        B,_,_,T = x.shape  
        x_in,x_cond = x[:,0:1],x[:,1:2] 
        mask = torch.all(x_cond == 0, dim=-1, keepdim=True).to(x_cond.dtype)  # (B,1,H,1)
        mask = mask.expand(-1, -1, -1, 1)
        STD=x_cond.std(dim=(2,3),keepdim=True)+1e-2
        a = (x_cond.std(-1,keepdim=True)+STD*0.05)*(1-mask)+mask*STD
        x_cond/=a
        x_in/=a
        x_cond = self.tokenizer_cond(x_cond)
        x_in=self.tokenizer(x_in)
        x=torch.cat([x_in,x_cond],dim=1)
        x= (1-mask)*x+mask*self.mask_adapter_n(x)

        t = self.time_emb(t)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)      
        x=self.to_attn(x)
        x = (1-mask)*x+mask*self.mask_adapter_d(x) 
        B,D,H,W=x.shape
        fourier_emb = None
        if condL is not None:
            rx, ry, sx, sy = condL
            x_mean = rx.mean(dim=-1, keepdim=True)
            y_mean = ry.mean(dim=-1, keepdim=True)
            sx = sx - x_mean
            sy = sy - y_mean
            rx = rx - x_mean
            ry = ry - y_mean
            pos_emb=torch.stack([rx,ry,sx,sy], dim=-1)
            fourier_emb=self.fourier_encoder(pos_emb)
        if fourier_emb is None:
            dummy_pos_emb = torch.zeros(B, H, 4, device=x.device, dtype=x.dtype)
            fourier_emb = self.fourier_encoder(dummy_pos_emb)
        fourier_emb =fourier_emb+t.unsqueeze(1)
        x=x.permute(0,2,3,1)                        
        #x=x.permute(0,2,3,1)
        for atten in self.attenL:
            x= atten(x,fourier_emb)
        shift, scale = self.adaLN_modulation(fourier_emb).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(-2), scale.unsqueeze(-2))
        x = x.permute(0,3,1,2).contiguous()
        x = self.to_unet(x)#+h0
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)    
        A = torch.exp(self.A(x))*mask*STD +(1-mask)*a     
        x=self.final(self.ac(self.norm(x)))
        #print("x:",x.shape)
        #print("A:",A.shape)
        x=x*A
        return x'''

class SeisDiT(torch.nn.Module):
    ##adaLN-zero
    def __init__(
        self,
        image_channels,
        n_channels=64,
        channel=[1,2,2,2],
        d_model=512,
        nhead=6,
        dropout=0.1,
        num_layers=12,
        output_channels=1,
        res_blocks=2,
        strides=[2,2,2,1],
        f_dict=None,
        pe_type='transformer',   
        rope_p_scale=1.0,
        #label_dim=5
    ):
        super(SeisDiT, self).__init__()
        # alpha = (2*num_layers)**0.25
        # beta =  (8*num_layers)**(-0.25)
        self.image_channels = image_channels
        self.n_channels= n_channels
        self.channel = channel
        n_res=len(channel)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.num_layers = num_layers

        self.tokenizer = torch.nn.Conv2d(
            image_channels, n_channels, kernel_size=(1, 3), padding=(0, 1),bias=True
        )
        self.mask_adapter_n=torch.nn.Conv2d(
            n_channels, n_channels, kernel_size=(1, 3), padding=(0, 1),bias=True
        )
        self.mask_adapter_d = torch.nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1),bias=True
        )
        self.time_emb = TimeEmbedding(d_model)
        #nn.init.zeros_(self.time_axis_mlp[-1].weight)
        #nn.init.zeros_(self.time_axis_mlp[-1].bias)
        self.fourier_encoder=fourier_enoder.Seismic5DEncoder(coord_dim=4,max_freq=128,out_dim=d_model,num_bands=32,pe_type = pe_type)
        last_channel = n_channels*channel[-1]*channel[-2]*channel[-3]

        self.to_attn = torch.nn.Conv2d(last_channel, d_model, kernel_size=(1,3), stride=(1,1), padding=(0,1), bias=True)
        self.to_unet = torch.nn.Conv2d(d_model, last_channel, kernel_size=(1,3), stride=(1,1), padding=(0,1), bias=True)
        attenL =[]
        # embTL=[]
        # ========== 修改：使用 DiTBlockTrace 替代 DiTBlock ==========
        for i in range(num_layers):
            attenL.append(DiTBlockTrace(hidden_size=d_model,num_heads=nhead,rope_p_scale=rope_p_scale))
            # attenL.append(Attention_Block(d_model=d_model))
        self.attenL = torch.nn.ModuleList(attenL)
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        down = []  
        out_channels = in_channels = n_channels
        for i in range(n_res):
            out_channels = in_channels * channel[i]
            for _ in range(res_blocks): 
                # print(out_channels,in_channels)
                down.append(
                    Resblock(in_channels, out_channels, d_model,)
                )
                in_channels = out_channels
            if i < n_res - 1:  
                down.append(Downsample(in_channels,i,strides[i]))

        self.down = torch.nn.ModuleList(down)
        up = []
        in_channels = out_channels  
        for i in reversed(range(n_res)):
            out_channels = in_channels
            for _ in range(res_blocks):
                up.append(
                    Resblock(in_channels+out_channels, out_channels,d_model,)
                )
            out_channels = in_channels // channel[i]
            up.append(
                Resblock(in_channels+out_channels, out_channels,d_model,)
            )  
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels, i - 1,strides[i-1]))
        self.up = torch.nn.ModuleList(up)
        self.ac=MYact()
        self.norm = torch.nn.GroupNorm(16,in_channels,eps=1e-5)
        self.final = torch.nn.Conv2d(
            in_channels, output_channels, kernel_size=(1, 5), padding=(0, 2)
        )
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,condL=None,log_tau=None,time_axis=None,training=False):
        B,_,_,T = x.shape   
        x_in,x_cond = x[:,0:1],x[:,1:2] 
        mask = torch.all(x_cond == 0, dim=-1, keepdim=True).to(x_cond.dtype)  
        mask = mask.expand(-1, -1, -1, 1)
        x=self.tokenizer(x)
        x= (1-mask)*x+mask*self.mask_adapter_n(x)
        t = self.time_emb(t)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x=self.to_attn(x)
        x = (1-mask)*x+mask*self.mask_adapter_d(x) 
        B,D,H,W=x.shape
        fourier_emb = None
        if condL is not None:
            rx, ry, sx, sy = condL
            x_mean = rx.mean(dim=-1, keepdim=True)
            y_mean = ry.mean(dim=-1, keepdim=True)
            sx = sx - x_mean
            sy = sy - y_mean
            rx = rx - x_mean
            ry = ry - y_mean
            pos_emb=torch.stack([rx,ry,sx,sy], dim=-1)
            fourier_emb=self.fourier_encoder(pos_emb)
        if fourier_emb is None:
            dummy_pos_emb = torch.zeros(B, H, 4, device=x.device, dtype=x.dtype)
            fourier_emb = self.fourier_encoder(dummy_pos_emb)
        #fourier_emb =fourier_emb+t.unsqueeze(1)
        fourier_emb =t.unsqueeze(1)
        x=x.permute(0,2,3,1)                        
        #x=x.permute(0,2,3,1)
        for atten in self.attenL:
            x= atten(x,fourier_emb)
        shift, scale = self.adaLN_modulation(fourier_emb).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift.unsqueeze(-2), scale.unsqueeze(-2))
        x = x.permute(0,3,1,2).contiguous()
        x = self.to_unet(x)#+h0
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)           
        x=self.final(self.ac(self.norm(x)))
        return x   


class SeisDiTRope(torch.nn.Module):
    """
    基于 SeisDiT 的网络，使用 RoPE 位置编码。
    与 SeisDiT 的区别：
    - 使用修改后的 DiTBlockTrace，支持传递 rope_pos 参数
    - 在 forward 中从 condL 提取位置信息（rx/ry）作为 RoPE 的位置输入
    """
    def __init__(
        self,
        image_channels,
        n_channels=64,
        channel=[1,2,2,2],
        d_model=512,
        nhead=8,
        dropout=0.1,
        num_layers=12,
        output_channels=1,
        res_blocks=2,
        strides=[2,2,2,1],
        f_dict=None,
        pe_type='transformer',
        rope_p_scale=1.0,
    ):
        super(SeisDiTRope, self).__init__()
        self.image_channels = image_channels
        self.n_channels = n_channels
        self.channel = channel
        n_res = len(channel)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.num_layers = num_layers
        #self.fourier_encoder=fourier_enoder.Seismic5DEncoder(coord_dim=4,max_freq=128,out_dim=d_model,num_bands=32,pe_type = pe_type)
        self.tokenizer = torch.nn.Conv2d(
            image_channels//2, n_channels, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.tokenizer_c = torch.nn.Conv2d(image_channels//2, n_channels, (1,3), padding=(0,1), bias=True)
        self.fuse = torch.nn.Conv2d(2*n_channels, n_channels, kernel_size=(1,1), padding=(0,0), bias=True)
        self.mask_adapter_n = torch.nn.Conv2d(
            n_channels, n_channels, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.mask_adapter_d = torch.nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.time_emb = TimeEmbedding(d_model)
        last_channel = n_channels * channel[-1] * channel[-2] * channel[-3]

        self.to_attn = torch.nn.Conv2d(
            last_channel, d_model, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.to_unet = torch.nn.Conv2d(
            d_model, last_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.Geomlp = nn.Sequential(
            nn.Linear(4, d_model*2),
            nn.SiLU(),
            nn.Linear(d_model*2, d_model),
        )
        nn.init.zeros_(self.Geomlp[-1].weight)
        nn.init.zeros_(self.Geomlp[-1].bias)
        self.geo_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.geo_gate.weight)
        nn.init.zeros_(self.geo_gate.bias)
        attenL = []
        for i in range(num_layers):
            attenL.append(DiTBlockTrace(hidden_size=d_model, num_heads=nhead,rope_p_scale=rope_p_scale))
        self.attenL = torch.nn.ModuleList(attenL)
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_res):
            out_channels = in_channels * channel[i]
            for _ in range(res_blocks):
                down.append(Resblock(in_channels, out_channels, d_model))
                in_channels = out_channels
            if i < n_res - 1:
                down.append(Downsample(in_channels, i, strides[i]))

        self.down = torch.nn.ModuleList(down)
        up = []
        in_channels = out_channels
        for i in reversed(range(n_res)):
            out_channels = in_channels
            for _ in range(res_blocks):
                up.append(Resblock(in_channels + out_channels, out_channels, d_model))
            out_channels = in_channels // channel[i]
            up.append(Resblock(in_channels + out_channels, out_channels, d_model))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels, i - 1, strides[i - 1]))
        self.up = torch.nn.ModuleList(up)
        self.ac = MYact()
        self.norm = torch.nn.GroupNorm(8, in_channels, eps=1e-5)
        self.final = torch.nn.Conv2d(
            in_channels, output_channels, kernel_size=(1, 5), padding=(0, 2)
        )
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condL=None, log_tau=None, time_axis=None, training=False):
        B, _, _, T = x.shape
        x_in, x_cond = x[:, 0:1], x[:, 1:2]
        mask = torch.all(x_cond == 0, dim=-1, keepdim=True).to(x_cond.dtype)
        mask = mask.expand(-1, -1, -1, 1)
        x_in = self.tokenizer(x_in)
        x_cond = self.tokenizer_c(x_cond)
        x = torch.cat([x_in, x_cond], dim=1)
        x = self.fuse(x)
        x = x + (1 - mask) * self.mask_adapter_n(x)

        t = self.time_emb(t)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.to_attn(x)
        x = x + (1 - mask) * self.mask_adapter_d(x)
        
        fourier_emb = None
        if condL is not None:
            rx, ry, sx, sy = condL
            x_mean = sx.mean(dim=-1, keepdim=True)
            y_mean = sy.mean(dim=-1, keepdim=True)
            sx = sx - x_mean
            sy = sy - y_mean
            rx = rx - x_mean
            ry = ry - y_mean
            pos_emb = torch.stack([rx, ry, sx, sy], dim=-1) # (B, H, 4)
            fourier_emb=self.Geomlp(pos_emb)
            
        if fourier_emb is None:
            dummy_pos_emb = torch.zeros(B, H, 4, device=x.device, dtype=x.dtype)
            fourier_emb = self.Geomlp(dummy_pos_emb)
            
        fourier_emb =self.geo_gate(fourier_emb)*0+t.unsqueeze(1)
        #fourier_emb =self.geo_gate(fourier_emb)+t.unsqueeze(1)
        
        x = x.permute(0, 2, 3, 1)

        for atten in self.attenL:
            x = atten(x, fourier_emb, rope_pos=pos_emb)
        
        shift, scale = self.adaLN_modulation(fourier_emb).chunk(2, dim=-1)
        
        x = modulate(self.norm_final(x), shift.unsqueeze(-2), scale.unsqueeze(-2))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.to_unet(x)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        x = self.final(self.ac(self.norm(x)))
        return x

class SeisDiTRopeV2(torch.nn.Module):
    """
    基于 SeisDiT 的网络，使用 RoPE 位置编码。
    与 SeisDiT 的区别：
    - 使用修改后的 DiTBlockTrace，支持传递 rope_pos 参数
    - 在 forward 中从 condL 提取位置信息（rx/ry）作为 RoPE 的位置输入
    """
    def __init__(
        self,
        image_channels,
        n_channels=64,
        channel=[1,2,2,2],
        d_model=512,
        nhead=8,
        dropout=0.1,
        num_layers=12,
        output_channels=1,
        res_blocks=2,
        strides=[2,2,2,1],
        f_dict=None,
        pe_type='transformer',
        rope_p_scale=1.0,
    ):
        super(SeisDiTRopeV2, self).__init__()
        self.image_channels = image_channels
        self.n_channels = n_channels
        self.channel = channel
        n_res = len(channel)
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.num_layers = num_layers
        #self.fourier_encoder=fourier_enoder.Seismic5DEncoder(coord_dim=4,max_freq=128,out_dim=d_model,num_bands=32,pe_type = pe_type)
        self.tokenizer = torch.nn.Conv2d(
            image_channels//2, n_channels, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.tokenizer_c = torch.nn.Conv2d(image_channels//2, n_channels, (1,3), padding=(0,1), bias=True)
        self.fuse = torch.nn.Conv2d(2*n_channels, n_channels, kernel_size=(1,1), padding=(0,0), bias=True)
        self.mask_adapter_n = torch.nn.Conv2d(
            n_channels, n_channels, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.mask_adapter_d = torch.nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.time_emb = TimeEmbedding(d_model)
        last_channel = n_channels * channel[-1] * channel[-2] * channel[-3]

        self.to_attn = torch.nn.Conv2d(
            last_channel, d_model, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.to_unet = torch.nn.Conv2d(
            d_model, last_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.Geomlp = nn.Sequential(
            nn.Linear(2, d_model*2),
            nn.SiLU(),
            nn.Linear(d_model*2, d_model),
        )
        nn.init.zeros_(self.Geomlp[-1].weight)
        nn.init.zeros_(self.Geomlp[-1].bias)
        self.geo_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.geo_gate.weight)
        nn.init.zeros_(self.geo_gate.bias)
        attenL = []
        for i in range(num_layers):
            attenL.append(DiTBlockTrace(hidden_size=d_model, num_heads=nhead,rope_p_scale=rope_p_scale))
        self.attenL = torch.nn.ModuleList(attenL)
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
        down = []
        out_channels = in_channels = n_channels
        for i in range(n_res):
            out_channels = in_channels * channel[i]
            for _ in range(res_blocks):
                down.append(Resblock(in_channels, out_channels, d_model))
                in_channels = out_channels
            if i < n_res - 1:
                down.append(Downsample(in_channels, i, strides[i]))

        self.down = torch.nn.ModuleList(down)
        up = []
        in_channels = out_channels
        for i in reversed(range(n_res)):
            out_channels = in_channels
            for _ in range(res_blocks):
                up.append(Resblock(in_channels + out_channels, out_channels, d_model))
            out_channels = in_channels // channel[i]
            up.append(Resblock(in_channels + out_channels, out_channels, d_model))
            in_channels = out_channels
            if i > 0:
                up.append(Upsample(in_channels, i - 1, strides[i - 1]))
        self.up = torch.nn.ModuleList(up)
        self.ac = MYact()
        self.norm = torch.nn.GroupNorm(8, in_channels, eps=1e-5)
        self.final = torch.nn.Conv2d(
            in_channels, output_channels, kernel_size=(1, 5), padding=(0, 2)
        )
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condL=None, log_tau=None, time_axis=None, training=False):
        B, _, _, T = x.shape
        x_in, x_cond = x[:, 0:1], x[:, 1:2]
        mask = torch.all(x_cond == 0, dim=-1, keepdim=True).to(x_cond.dtype)
        mask = mask.expand(-1, -1, -1, 1)
        x_in = self.tokenizer(x_in)
        x_cond = self.tokenizer_c(x_cond)
        x = torch.cat([x_in, x_cond], dim=1)
        x = self.fuse(x)
        x = x + (1 - mask) * self.mask_adapter_n(x)

        t = self.time_emb(t)
        h = [x]
        for m in self.down:
            x = m(x, t)
            h.append(x)
        x = self.to_attn(x)
        x = x + (1 - mask) * self.mask_adapter_d(x)
        
        fourier_emb = None
        if condL is not None:
            rx, ry, sx, sy = condL
            x_mean = sx.mean(dim=-1, keepdim=True)
            y_mean = sy.mean(dim=-1, keepdim=True)
            sx = sx - x_mean
            sy = sy - y_mean
            rx = rx - x_mean
            ry = ry - y_mean
            pos_emb = torch.stack([rx, ry, sx, sy], dim=-1) # (B, H, 4)
            fourier_emb=self.Geomlp(pos_emb[:,:,:2])
            
        if fourier_emb is None:
            dummy_pos_emb = torch.zeros(B, H, 4, device=x.device, dtype=x.dtype)
            fourier_emb = self.Geomlp(dummy_pos_emb)
            
        fourier_emb =self.geo_gate(fourier_emb)+t.unsqueeze(1)
        #fourier_emb =self.geo_gate(fourier_emb)+t.unsqueeze(1)
        
        x = x.permute(0, 2, 3, 1)

        for atten in self.attenL:
            x = atten(x, fourier_emb, rope_pos=pos_emb[:,:,:])
        
        shift, scale = self.adaLN_modulation(fourier_emb).chunk(2, dim=-1)
        
        x = modulate(self.norm_final(x), shift.unsqueeze(-2), scale.unsqueeze(-2))
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.to_unet(x)
        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x, t)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        x = self.final(self.ac(self.norm(x)))
        return x

# ========== 测试代码 ==========
if __name__ == "__main__":
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # ===== 1. 构造模型实例 =====
    model = SeisDiTRope(
        image_channels=2,
        n_channels=64,
        channel=[1,2,4,8],
        d_model=512,
        nhead=8,
        dropout=0.1,
        num_layers=4,  # 测试时使用较小的层数
        output_channels=1,
        res_blocks=2,
        strides=[2,2,2,1],
        pe_type="transformer",
        rope_p_scale=1.0,
    ).to(device)
    
    model.eval()
    
    # ===== 2. 构造测试数据 =====
    B = 2          # batch size
    C = 2          # input channels
    H = 32         # trace 方向长度（深度）
    W = 128        # time 方向长度
    
    # 输入地震数据
    x = torch.randn(B, C, H, W, device=device)
    
    # 扩散时间步
    t = torch.randint(0, 1000, (B,), device=device).float()
    
    # 几何信息（条件）
    rx = torch.randn(B, H, device=device)
    ry = torch.randn(B, H, device=device)
    sx = torch.randn(B, H, device=device)
    sy = torch.randn(B, H, device=device)
    condL = (rx, ry, sx, sy)
    
    # ===== 3. 前向测试 =====
    print("开始前向传播测试...")
    with torch.no_grad():
        y = model(x, t, condL=condL)
    
    # ===== 4. 验证输出 =====
    print(f"输入形状: {x.shape}")   # [B, C, H, W]
    print(f"输出形状: {y.shape}")   # 期望 [B, 1, H, W]
    
    # 断言检查
    assert y.shape == (B, 1, H, W), \
        f"输出形状不匹配！期望 {(B, 1, H, W)}，实际 {y.shape}"
    assert not torch.isnan(y).any(), "输出包含 NaN 值！"
    assert not torch.isinf(y).any(), "输出包含 Inf 值！"
    
    print("✓ 测试通过！")
    print(f"  - 输入形状: {x.shape}")
    print(f"  - 输出形状: {y.shape}")
    print(f"  - 模型使用 Trace-axis global attention")

