import math
import torch
import torch.nn as nn

class SegmentedRoPEExpCached(nn.Module):
    def __init__(self, D, N, n_pos, min_log=0.0, max_log=4.0, mapper="linear", hidden=128, p_scale=None, use_time=False):
        super().__init__()
        assert D % N == 0
        d_seg = D // N
        assert d_seg % 2 == 0

        self.D, self.N, self.n_pos = D, N, n_pos
        self.d_seg = d_seg
        self.half = d_seg // 2
        self.p_scale = p_scale

        self.use_time = use_time
        if self.use_time:
            if mapper == "linear":
                self.map = nn.Linear(n_pos-1, N, bias=True)
            elif mapper == "equal":
                self.map = nn.Identity()
            else:
                self.map = nn.Sequential(
                    nn.Linear(n_pos-1, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, N),
                )
            self.time_map = nn.Sequential(
                nn.Linear(1, hidden//2),
                nn.SiLU(),
                nn.Linear(hidden//2, N),
            )
        else:
            if mapper == "linear":
                self.map = nn.Linear(n_pos, N, bias=True)
            elif mapper == "equal":
                self.map = nn.Identity()
            else:
                self.map = nn.Sequential(
                    nn.Linear(n_pos, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, N),
                )
        # 传统 Transformer RoPE 频率设定: theta_i = 10000^(-2i/d_seg)
        # 其中 i 从 0 到 half-1，d_seg 是每个头的维度
        # 标准公式: inv_freq[i] = 10000^(-2i/d_seg)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.half, dtype=torch.float32) * 2.0 / self.d_seg))
        self.register_buffer("freq", inv_freq, persistent=False)

    def precompute_cos_sin(self, pos: torch.Tensor, out_dtype: torch.dtype, device: torch.device):
        """
        pos: (B,t,n) -> cos/sin: (B,t,N,half)
        """
        pos_f = pos.to(torch.float32) # 关键：float32
        B, t, n = pos.shape
        assert n == self.n_pos #[rx,ry,sx,sy]
        scale = pos_f.new_tensor([
        self.p_scale["rx"],
        self.p_scale["ry"],
        self.p_scale["sx"],
        self.p_scale["sy"],
        ]).view(1, 1, -1)
        pos_f = pos_f * scale
        #if self.p_scale is not None:
        #    pos[:,:,0] = pos[:,:,0]*self.p_scale["rx"]
        #    pos[:,:,1] = pos[:,:,1]*self.p_scale["ry"]
        #    pos[:,:,2] = pos[:,:,2]*self.p_scale["sx"]
        #    pos[:,:,3] = pos[:,:,3]*self.p_scale["sy"]
        
        p = self.map(pos_f)  # (B,t,N)
        phase = (
            p.to(dtype=torch.float32).unsqueeze(-1)
            * self.freq.to(device=device).to(torch.float32).view(1, 1, 1, -1)
            * (2.0 * math.pi)
        )  # (B,t,N,half)
        #print(f"phase shape: {phase.shape}")
        self.cos = torch.cos(phase).to(dtype=out_dtype).permute(0,2, 1, 3)
        self.sin = torch.sin(phase).to(dtype=out_dtype).permute(0,2, 1, 3)
        #return cos, sin
        
    def precompute_cos_sin_time(self, space_pos: torch.Tensor, time_pos: torch.Tensor, out_dtype: torch.dtype, device: torch.device):
        """
        pos: (B,t,n) -> cos/sin: (B,t,N,half)
        """
        pos_f = space_pos.to(torch.float32) # 关键：float32
        B, t, n = space_pos.shape
        assert n == self.n_pos-1 #[rx,ry,sx,sy]
        scale = pos_f.new_tensor([
        self.p_scale["rx"],
        self.p_scale["ry"],
        self.p_scale["sx"],
        self.p_scale["sy"],
        ]).view(1, 1, -1)
        pos_f = pos_f * scale
        time_pos_f = time_pos.to(torch.float32)
        time_p = self.time_map(time_pos_f)
        #if self.p_scale is not None:
        #    pos[:,:,0] = pos[:,:,0]*self.p_scale["rx"]
        #    pos[:,:,1] = pos[:,:,1]*self.p_scale["ry"]
        #    pos[:,:,2] = pos[:,:,2]*self.p_scale["sx"]
        #    pos[:,:,3] = pos[:,:,3]*self.p_scale["sy"]
        p = self.map(pos_f)  # (B,t,N)
        p = p + time_p
        phase = (
            p.to(dtype=torch.float32).unsqueeze(-1)
            * self.freq.to(device=device).to(torch.float32).view(1, 1, 1, -1)
            * (2.0 * math.pi)
        )  # (B,t,N,half)
        #print(f"phase shape: {phase.shape}")
        self.cos = torch.cos(phase).to(dtype=out_dtype).permute(0,2, 1, 3)
        self.sin = torch.sin(phase).to(dtype=out_dtype).permute(0,2, 1, 3)
        #return cos, sin

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        x:   (B, head_num, t, head_dim)
        cos/sin: (B, N, t, half) where N=num_heads, half=head_dim//2
        注意: cos/sin 在 precompute_cos_sin 中被 permute 成 (B, N, t, half)
        """
        B, head_num, t, head_dim = x.shape
        # cos/sin shape: (B, N, t, half) where N=num_heads
        # 确保维度匹配
        if cos.shape[0] != B or cos.shape[1] != head_num or cos.shape[2] != t:
            raise RuntimeError(
                f"Dimension mismatch: x shape {x.shape}, cos shape {cos.shape}. "
                f"Expected cos shape: (B={B}, N={head_num}, t={t}, half={head_dim//2})"
            )
        if cos.shape[3] != head_dim // 2:
            raise RuntimeError(
                f"Half dimension mismatch: cos.shape[3]={cos.shape[3]}, head_dim//2={head_dim//2}"
            )
        
        # 将 x reshape 为 (B, head_num, t, head_dim//2, 2) 以便应用旋转
        x_reshaped = x.view(B, head_num, t, head_dim // 2, 2)
        x_even = x_reshaped[..., 0]  # (B, head_num, t, head_dim//2)
        x_odd  = x_reshaped[..., 1]  # (B, head_num, t, head_dim//2)
        
        # cos/sin 已经是 (B, head_num, t, half) 形状，可以直接广播
        # 执行旋转: (B, head_num, t, head_dim//2)
        y_even = x_even * cos - x_odd * sin
        y_odd  = x_even * sin + x_odd * cos

        # 重新组合: stack 然后 reshape
        y = torch.stack([y_even, y_odd], dim=-1)  # (B, head_num, t, head_dim//2, 2)
        return y.view(B, head_num, t, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos, sin = self.cos, self.sin
        return self.apply_rotary(x, cos, sin)