from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modules import MoeMLP, Mlp
from seisdit_trace_axis import (
    MYact,
    Downsample,
    Resblock,
    TimeEmbedding,
    TraceAxisAttention2D,
    Upsample,
    modulate,
)


@dataclass
class MoEConfig:
    num_experts: int
    capacity: int
    n_shared_experts: int = 0
    interleave: bool = False
    init_MoeMLP: bool = False
    capacity_schedule: object | None = None


def _get_config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


#################################################################################
#                                DiffMoE Layer                                  #
#################################################################################


class SparseMoEBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, experts, hidden_dim, num_experts, n_shared_experts=0, capacity=2, mlp_ratio=4.0):
        super().__init__()
        self.gate_weight = nn.Parameter(torch.empty((num_experts, hidden_dim)))
        nn.init.normal_(self.gate_weight, std=0.006)
        self.experts = nn.ModuleList(experts)
        self.capacity = capacity
        self.num_experts = num_experts

        self.n_shared_experts = n_shared_experts

        self.capacity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_experts, bias=True),
        )

        if self.n_shared_experts > 0:
            mlp_hidden_dim = int(hidden_dim * mlp_ratio * 2)
            approx_gelu = lambda: nn.GELU(approximate="tanh")
            self.shared_experts = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

        ema_decay = 0.95
        expert_threshold = torch.tensor([0.0] * num_experts)
        self.register_buffer("expert_threshold", expert_threshold)
        ema_decay = torch.tensor([ema_decay])
        self.register_buffer("ema_decay", ema_decay)

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        return self.forward_eval(x)

    def update_threshold(self, capacity_pred):
        if self.training:
            capacity_pred = F.sigmoid(capacity_pred)
            S = capacity_pred.size(0)
            topk = int((S / self.num_experts) * self.capacity)
            topk = max(1, min(S, topk))
            threshold = self.expert_threshold
            ema_decay = self.ema_decay

            for i in range(self.num_experts):
                capacity_pred_scores, _ = torch.topk(capacity_pred[:, i], k=topk, dim=-1, sorted=True)
                quantile = capacity_pred_scores[-1].detach()
                threshold[i] = threshold[i] * ema_decay + (1 - ema_decay) * quantile

            dist.all_reduce(threshold, op=dist.ReduceOp.SUM)
            threshold /= dist.get_world_size()
            self.expert_threshold = threshold

    def forward_train(self, x):
        B, s, D = x.shape
        identity = x

        x = x.view(-1, D)
        S = x.shape[0]

        capacity_pred = self.capacity_predictor(x.detach())
        k = int((S / self.num_experts) * self.capacity)
        k = max(1, min(S, k))

        logits = F.linear(x, self.gate_weight, None)
        scores = logits.softmax(dim=-1).permute(1, 0)

        gating, index = torch.topk(scores, k=k, dim=-1, sorted=False)

        mask = torch.zeros((self.num_experts, S), dtype=x.dtype, device=x.device)
        mask.scatter_(1, index, 1.0)

        gating_expanded = gating.unsqueeze(-1)

        expert_inputs = x[index]
        expert_outputs = torch.stack(
            [expert(expert_inputs[i]) for i, expert in enumerate(self.experts)]
        )
        gated_outputs = gating_expanded * expert_outputs
        y = torch.zeros((S * self.num_experts, D), dtype=x.dtype, device=x.device)
        offset = torch.arange(0, self.num_experts).unsqueeze(1).to(device=x.device) * S
        index = (index + offset.long()).view(-1)

        gated_outputs_flat = gated_outputs.view(-1, D)

        y = torch.scatter(
            y,
            0,
            index.unsqueeze(1).expand(-1, D),
            gated_outputs_flat,
        )

        y = y.view(self.num_experts, S, D).sum(dim=0, keepdim=False)

        self.update_threshold(capacity_pred)

        x_out = y.view(B, s, D)

        ones = mask.permute(1, 0).view(B, s, self.num_experts)

        capacity_pred = capacity_pred.view(B, s, self.num_experts)

        if self.n_shared_experts > 0:
            x_out = x_out + self.shared_experts(identity)

        return x_out, ones, capacity_pred

    def forward_eval(self, x):
        B, s, D = x.shape
        identity = x

        x = x.view(-1, D)
        S = x.shape[0]

        capacity_pred = self.capacity_predictor(x.detach())
        capacity_pred = F.sigmoid(capacity_pred)
        threshold = self.expert_threshold

        logits = F.linear(x, self.gate_weight, None)
        scores = logits.softmax(dim=-1).permute(-1, -2)

        y = torch.zeros_like(x, dtype=x.dtype)
        processed_tokens = 0

        for i, expert in enumerate(self.experts):
            k_fixed = torch.where(capacity_pred[:, i] > threshold[i], 1, 0).sum()
            k_fixed = min(S, int(k_fixed.item()))
            processed_tokens += k_fixed
            if k_fixed == 0:
                continue
            gating, index = torch.topk(scores[i], k=k_fixed, dim=-1, sorted=False)
            y[index, :] += gating.unsqueeze(-1) * expert(x[index, :])

        _ = processed_tokens / S / self.num_experts
        x_out = y.view(B, s, D)

        if self.n_shared_experts > 0:
            x_out = x_out + self.shared_experts(identity)
        return x_out, None, None


#################################################################################
#                            Trace-axis MoE DiT Blocks                           #
#################################################################################


class DiTBlockTraceMoE(nn.Module):
    """
    Trace-axis DiT block with optional DiffMoE MLP.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_swiglu=False, MoE_config=None, use_moe=False):
        super().__init__()
        self.use_moe = use_moe
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = TraceAxisAttention2D(dim=hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        if use_moe:
            if use_swiglu is False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = SparseMoEBlock(
                    experts=[
                        Mlp(
                            in_features=hidden_size,
                            hidden_features=mlp_hidden_dim,
                            act_layer=approx_gelu,
                            drop=0,
                        )
                        for _ in range(MoE_config.num_experts)
                    ],
                    hidden_dim=hidden_size,
                    num_experts=MoE_config.num_experts,
                    capacity=MoE_config.capacity,
                    n_shared_experts=MoE_config.n_shared_experts,
                    mlp_ratio=mlp_ratio,
                )
            else:
                self.mlp = SparseMoEBlock(
                    experts=[
                        MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim)
                        for _ in range(MoE_config.num_experts)
                    ],
                    hidden_dim=hidden_size,
                    num_experts=MoE_config.num_experts,
                    capacity=MoE_config.capacity,
                    n_shared_experts=MoE_config.n_shared_experts,
                )
        else:
            if use_swiglu is False:
                approx_gelu = lambda: nn.GELU(approximate="tanh")
                self.mlp = Mlp(
                    in_features=hidden_size,
                    hidden_features=mlp_hidden_dim,
                    act_layer=approx_gelu,
                    drop=0.1,
                )
            else:
                self.mlp = MoeMLP(hidden_size=hidden_size, intermediate_size=mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, rope_pos=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x1 = modulate(self.norm1(x), shift_msa.unsqueeze(-2), scale_msa.unsqueeze(-2))
        x = x + gate_msa.unsqueeze(-2) * self.attn(x1, pos=rope_pos)
        if self.use_moe:
            x_mlp, ones, pred_c = self.mlp(modulate(self.norm2(x), shift_mlp.unsqueeze(-2), scale_mlp.unsqueeze(-2)))
            x = x + gate_mlp.unsqueeze(-2) * x_mlp
            return x, ones, pred_c
        x = x + gate_mlp.unsqueeze(-2) * self.mlp(
            modulate(self.norm2(x), shift_mlp.unsqueeze(-2), scale_mlp.unsqueeze(-2))
        )
        return x, None, None


class SeisDiTRopeMoE(torch.nn.Module):
    """
    SeisDiT (Trace-axis) with DiffMoE-style MLP layers.
    """

    def __init__(
        self,
        image_channels,
        n_channels=64,
        channel=(1, 2, 2, 2),
        d_model=512,
        nhead=8,
        dropout=0.1,
        num_layers=12,
        output_channels=1,
        res_blocks=2,
        strides=(2, 2, 2, 1),
        f_dict=None,
        pe_type="transformer",
        rope_p_scale=1.0,
        use_swiglu=False,
        MoE_config=None,
        CapacityPred_loss_weight=0.01,
    ):
        super().__init__()
        _ = f_dict, pe_type, dropout
        if MoE_config is None:
            raise ValueError("MoE_config is required for SeisDiTRopeMoE.")

        self.image_channels = image_channels
        self.n_channels = n_channels
        self.channel = list(channel)
        n_res = len(self.channel)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.MoE_config = MoE_config
        self.CapacityPred_loss_weight = CapacityPred_loss_weight

        use_moe_flag = [True] * num_layers
        if _get_config_value(self.MoE_config, "interleave", False):
            use_moe_flag = [i % 2 == 1 for i in range(num_layers)]

        self.tokenizer = torch.nn.Conv2d(
            image_channels // 2, n_channels, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.tokenizer_c = torch.nn.Conv2d(
            image_channels // 2, n_channels, (1, 3), padding=(0, 1), bias=True
        )
        self.fuse = torch.nn.Conv2d(2 * n_channels, n_channels, kernel_size=(1, 1), padding=(0, 0), bias=True)
        self.mask_adapter_n = torch.nn.Conv2d(
            n_channels, n_channels, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.mask_adapter_d = torch.nn.Conv2d(
            d_model, d_model, kernel_size=(1, 3), padding=(0, 1), bias=True
        )
        self.time_emb = TimeEmbedding(d_model)
        last_channel = n_channels * self.channel[-1] * self.channel[-2] * self.channel[-3]

        self.to_attn = torch.nn.Conv2d(
            last_channel, d_model, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.to_unet = torch.nn.Conv2d(
            d_model, last_channel, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=True
        )
        self.Geomlp = nn.Sequential(
            nn.Linear(4, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        nn.init.zeros_(self.Geomlp[-1].weight)
        nn.init.zeros_(self.Geomlp[-1].bias)
        self.geo_gate = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.geo_gate.weight)
        nn.init.zeros_(self.geo_gate.bias)

        attenL = []
        for i in range(num_layers):
            attenL.append(
                DiTBlockTraceMoE(
                    hidden_size=d_model,
                    num_heads=nhead,
                    mlp_ratio=4.0,
                    use_swiglu=use_swiglu,
                    MoE_config=MoE_config,
                    use_moe=use_moe_flag[i],
                )
            )
        self.attenL = torch.nn.ModuleList(attenL)
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        down = []
        out_channels = in_channels = n_channels
        for i in range(n_res):
            out_channels = in_channels * self.channel[i]
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
            out_channels = in_channels // self.channel[i]
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

        self.capacity_schedule = _get_config_value(self.MoE_config, "capacity_schedule", None)
        if self.capacity_schedule:
            self.training_iters = -1

        self.init_MoeMLP = _get_config_value(self.MoE_config, "init_MoeMLP", False)
        self._init_moe_weights()

    def _init_moe_weights(self):
        def init_MoeMLP(module, std=0.006):
            nn.init.normal_(module.gate_proj.weight, std=std)
            nn.init.normal_(module.up_proj.weight, std=std)
            nn.init.normal_(module.down_proj.weight, std=std)

        if self.init_MoeMLP:
            for block in self.attenL:
                if block.use_moe:
                    for expert in block.mlp.experts:
                        init_MoeMLP(expert)

    def forward(self, x: torch.Tensor, t: torch.Tensor, condL=None, log_tau=None, time_axis=None, training=False):
        _ = log_tau, time_axis, training
        B, _, _, _ = x.shape
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
        _, _, H, _ = x.shape

        fourier_emb = None
        pos_emb = None
        if condL is not None:
            rx, ry, sx, sy = condL
            x_mean = sx.mean(dim=-1, keepdim=True)
            y_mean = sy.mean(dim=-1, keepdim=True)
            sx = sx - x_mean
            sy = sy - y_mean
            rx = rx - x_mean
            ry = ry - y_mean
            pos_emb = torch.stack([rx, ry, sx, sy], dim=-1)
            fourier_emb = self.Geomlp(pos_emb)

        if fourier_emb is None:
            dummy_pos_emb = torch.zeros(B, H, 4, device=x.device, dtype=x.dtype)
            fourier_emb = self.Geomlp(dummy_pos_emb)
            pos_emb = dummy_pos_emb

        fourier_emb = self.geo_gate(fourier_emb) * 0 + t.unsqueeze(1)

        x = x.permute(0, 2, 3, 1)

        if self.training and self.capacity_schedule:
            num_experts = _get_config_value(self.MoE_config, "num_experts", None)
            capacity = _get_config_value(self.MoE_config, "capacity", None)
            schedule = self.capacity_schedule
            stage_I_iters = _get_config_value(schedule, "capacity_schedule_stage_I_iters", 0)
            stage_II_iters = _get_config_value(schedule, "capacity_schedule_stage_II_iters", 0)

            if self.training_iters <= stage_I_iters:
                capacity = num_experts
            elif self.training_iters <= stage_II_iters:
                capacity = capacity + (num_experts - capacity) * (
                    stage_II_iters - self.training_iters
                ) / (stage_II_iters - stage_I_iters)

            for block in self.attenL:
                if block.use_moe:
                    block.mlp.capacity = capacity

        ones_list = []
        pred_c_list = []
        layer_idx_list = []
        for layer_idx, atten in enumerate(self.attenL):
            x, ones, pred_c = atten(x, fourier_emb, rope_pos=pos_emb)
            if ones is not None:
                ones_list.append(ones)
                pred_c_list.append(pred_c)
                layer_idx_list.append(layer_idx)

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

        return x, "Capacity_Pred", layer_idx_list, ones_list, pred_c_list, self.CapacityPred_loss_weight
