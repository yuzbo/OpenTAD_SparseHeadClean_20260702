# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.registry import MODELS
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_init
from mmaction.utils import ConfigType, OptConfigType
from mmaction.models.backbones.vit_mae import get_sinusoid_encoding

from .time_aligned_rasterizer import TimeAlignedRasterizer


class Adapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
        tara_cfg: Optional[dict] = None,
        multiscale_cfg: Optional[dict] = None,
        time_film_cfg: Optional[dict] = None,
        gap_residual_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)
        tara_cfg = {} if tara_cfg is None else dict(tara_cfg)
        tara_mode = tara_cfg.pop("mode", "none")
        multiscale_cfg = {} if multiscale_cfg is None else dict(multiscale_cfg)
        time_film_cfg = {} if time_film_cfg is None else dict(time_film_cfg)
        gap_residual_cfg = {} if gap_residual_cfg is None else dict(gap_residual_cfg)

        # temporal depth-wise convolution
        self.temporal_size = temporal_size
        self.dwconv = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=hidden_dims,
        )
        self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)
        self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
        self.dwconv.bias.data.zero_()
        self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
        self.conv.bias.data.zero_()
        self.multiscale_convs = None
        self.multiscale_scale = None
        self.multiscale_scale_max = None
        multiscale_dilations = multiscale_cfg.pop("dilations", None)
        if multiscale_dilations:
            rng_state = torch.get_rng_state()
            try:
                self.multiscale_convs = nn.ModuleList()
                self.multiscale_scale = nn.Parameter(torch.tensor(float(multiscale_cfg.pop("init_scale", 0.0))))
                self.multiscale_scale_max = multiscale_cfg.pop("scale_max", None)
                if self.multiscale_scale_max is not None:
                    self.multiscale_scale_max = float(self.multiscale_scale_max)
                for branch_dilation in multiscale_dilations:
                    branch_dilation = int(branch_dilation)
                    branch = nn.Sequential(
                        nn.Conv1d(
                            hidden_dims,
                            hidden_dims,
                            kernel_size=kernel_size,
                            stride=1,
                            padding=(kernel_size // 2) * branch_dilation,
                            dilation=branch_dilation,
                            groups=hidden_dims,
                        ),
                        nn.Conv1d(hidden_dims, hidden_dims, 1),
                    )
                    branch[0].weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
                    branch[0].bias.data.zero_()
                    branch[1].weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
                    branch[1].bias.data.zero_()
                    self.multiscale_convs.append(branch)
            finally:
                torch.set_rng_state(rng_state)

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0
        self.time_film = None
        self.time_film_max_delta = None
        self.time_film_mode = None
        if time_film_cfg:
            self.time_film_mode = time_film_cfg.pop("mode", "scale_shift")
            if self.time_film_mode not in {"scale_shift", "shift"}:
                raise ValueError(f"Unsupported adapter time FiLM mode: {self.time_film_mode}")
            input_dim = int(time_film_cfg.pop("input_dim", 5))
            film_hidden_dim = int(time_film_cfg.pop("hidden_dim", hidden_dims))
            self.time_film_max_delta = time_film_cfg.pop("max_delta", 0.20)
            if self.time_film_max_delta is not None:
                self.time_film_max_delta = float(self.time_film_max_delta)
            if time_film_cfg:
                raise ValueError(f"Unknown adapter time FiLM options: {sorted(time_film_cfg.keys())}")
            output_dim = hidden_dims * 2 if self.time_film_mode == "scale_shift" else hidden_dims
            self.time_film = nn.Sequential(
                nn.Linear(input_dim, film_hidden_dim),
                nn.GELU(),
                nn.Linear(film_hidden_dim, output_dim),
            )
            trunc_normal_init(self.time_film[0], std=0.02, bias=0)
            constant_init(self.time_film[-1], 0)

        self.gap_residual = None
        self.gap_residual_max_delta = None
        self.gap_residual_mode = None
        if gap_residual_cfg:
            self.gap_residual_mode = gap_residual_cfg.pop("mode", "diff_residual")
            if self.gap_residual_mode != "diff_residual":
                raise ValueError(f"Unsupported adapter gap residual mode: {self.gap_residual_mode}")
            input_dim = int(gap_residual_cfg.pop("input_dim", 5))
            gap_hidden_dim = int(gap_residual_cfg.pop("hidden_dim", hidden_dims))
            self.gap_residual_max_delta = gap_residual_cfg.pop("max_delta", 0.20)
            if self.gap_residual_max_delta is not None:
                self.gap_residual_max_delta = float(self.gap_residual_max_delta)
            if gap_residual_cfg:
                raise ValueError(f"Unknown adapter gap residual options: {sorted(gap_residual_cfg.keys())}")
            self.gap_residual = nn.Sequential(
                nn.Linear(input_dim, gap_hidden_dim),
                nn.GELU(),
                nn.Linear(gap_hidden_dim, hidden_dims * 2),
            )
            trunc_normal_init(self.gap_residual[0], std=0.02, bias=0)
            constant_init(self.gap_residual[-1], 0)

        if tara_mode in {"none", None}:
            self.time_rasterizer = None
        else:
            self.time_rasterizer = TimeAlignedRasterizer(channels=hidden_dims, mode=tara_mode, **tara_cfg)

    def _reshape_time_features(self, time_embed: Optional[Tensor], token_count: int, outer_batches: int, dtype, device):
        if time_embed is None:
            raise ValueError("Adapter time metadata is required, but time_embed is None.")
        if time_embed.dim() != 3:
            raise ValueError(f"time_embed must have shape [B, T_token, D], got {tuple(time_embed.shape)}")
        if time_embed.shape[1] != token_count:
            raise ValueError(
                f"time_embed token count ({time_embed.shape[1]}) does not match adapter chunk token count ({token_count})."
            )
        if time_embed.shape[0] * token_count != outer_batches * self.temporal_size:
            raise ValueError(
                "time_embed cannot be regrouped to adapter temporal axis: "
                f"time_embed={tuple(time_embed.shape)}, outer_batches={outer_batches}, temporal_size={self.temporal_size}."
            )
        return time_embed.to(device=device, dtype=dtype).reshape(outer_batches, self.temporal_size, -1)

    def _temporal_conv(self, attn: Tensor, h: int, w: int) -> Tensor:
        return self._temporal_conv_with(attn, h, w, self.dwconv, self.conv)

    def _temporal_conv_with(self, attn: Tensor, h: int, w: int, dwconv: nn.Module, conv: nn.Module) -> Tensor:
        group_count, _, _, _, channels = attn.shape
        attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)  # [b*h*w,c,t]
        attn = dwconv(attn)
        attn = conv(attn)
        return attn.unflatten(0, (group_count, h, w)).permute(0, 4, 1, 2, 3)

    def _multiscale_scale_value(self, dtype, device):
        scale = self.multiscale_scale.to(device=device, dtype=dtype)
        if self.multiscale_scale_max is not None:
            scale = torch.tanh(scale) * self.multiscale_scale_max
        return scale

    def _apply_multiscale_temporal_conv(self, attn: Tensor, h: int, w: int) -> Tensor:
        source_attn = self._temporal_conv(attn, h, w)
        if self.multiscale_convs is None:
            return source_attn

        branch_sum = None
        for branch in self.multiscale_convs:
            branch_attn = self._temporal_conv_with(attn, h, w, branch[0], branch[1])
            branch_sum = branch_attn if branch_sum is None else branch_sum + branch_attn
        branch_attn = branch_sum / len(self.multiscale_convs)
        scale = self._multiscale_scale_value(dtype=source_attn.dtype, device=source_attn.device)
        return source_attn + scale * branch_attn

    def _apply_time_film(self, attn: Tensor, time_embed: Optional[Tensor], token_count: int, h: int, w: int) -> Tensor:
        if self.time_film is None:
            return attn

        time_features = self._reshape_time_features(
            time_embed=time_embed,
            token_count=token_count,
            outer_batches=attn.shape[0],
            dtype=attn.dtype,
            device=attn.device,
        )
        film = self.time_film(time_features)
        if self.time_film_max_delta is not None:
            film = torch.tanh(film) * self.time_film_max_delta
        if time_features.shape[-1] > 1:
            film = film * time_features[..., 1:2].clamp(0.0, 1.0)
        film = film[:, :, None, None, :].expand(-1, -1, h, w, -1)
        if self.time_film_mode == "scale_shift":
            scale, shift = film.chunk(2, dim=-1)
            return attn * (1.0 + scale) + shift
        return attn + film

    def _apply_gap_residual(self, attn: Tensor, time_embed: Optional[Tensor], token_count: int, h: int, w: int) -> Tensor:
        if self.gap_residual is None:
            return attn

        time_features = self._reshape_time_features(
            time_embed=time_embed,
            token_count=token_count,
            outer_batches=attn.shape[0],
            dtype=attn.dtype,
            device=attn.device,
        )
        gate = self.gap_residual(time_features)
        if self.gap_residual_max_delta is not None:
            gate = torch.tanh(gate) * self.gap_residual_max_delta

        valid = None
        if time_features.shape[-1] > 1:
            valid = time_features[..., 1:2].clamp(0.0, 1.0)
            gate = gate * valid

        left_gate, right_gate = gate.chunk(2, dim=-1)
        left = torch.roll(attn, shifts=1, dims=1)
        right = torch.roll(attn, shifts=-1, dims=1)
        if valid is None:
            left_valid = left_gate.new_ones(left_gate.shape[:-1] + (1,))
            right_valid = right_gate.new_ones(right_gate.shape[:-1] + (1,))
        else:
            left_valid = valid * torch.roll(valid, shifts=1, dims=1)
            right_valid = valid * torch.roll(valid, shifts=-1, dims=1)
        left_valid[:, :1] = 0.0
        right_valid[:, -1:] = 0.0

        left_delta = left_gate[:, :, None, None, :] * left_valid[:, :, None, None, :] * (left - attn)
        right_delta = right_gate[:, :, None, None, :] * right_valid[:, :, None, None, :] * (right - attn)
        return attn + left_delta + right_delta

    def forward(self, x: Tensor, h: int, w: int, time_embed: Optional[Tensor] = None) -> Tensor:
        inputs = x

        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)

        # temporal depth-wise convolution
        B, N, C = x.shape  # 48, 8*10*10, 384
        token_count = N // (h * w)
        attn = x.reshape(-1, self.temporal_size, h, w, C)  # [b,t,h,w,c]
        attn = self._apply_time_film(attn, time_embed, token_count, h, w)
        attn = self._apply_gap_residual(attn, time_embed, token_count, h, w)

        if self.time_rasterizer is not None:
            source_attn = None
            if self.time_rasterizer.use_branch_residual:
                source_attn = self._apply_multiscale_temporal_conv(attn, h, w)
            time_features = self._reshape_time_features(
                time_embed=time_embed,
                token_count=token_count,
                outer_batches=attn.shape[0],
                dtype=x.dtype,
                device=x.device,
            )
            grouped = attn.permute(0, 2, 3, 4, 1).reshape(attn.shape[0], h * w, C, self.temporal_size)
            grouped, uniform_center, uniform_valid, source_center, source_valid = self.time_rasterizer.source_to_uniform(
                grouped,
                time_features,
            )
            attn = grouped.reshape(attn.shape[0], h, w, C, self.temporal_size).permute(0, 4, 1, 2, 3)
            attn = self._apply_multiscale_temporal_conv(attn, h, w)
            grouped = attn.permute(0, 2, 3, 4, 1).reshape(attn.shape[0], h * w, C, self.temporal_size)
            grouped = self.time_rasterizer.uniform_to_source(
                grouped,
                uniform_center,
                uniform_valid,
                source_center,
                source_valid,
            )
            tara_attn = grouped.reshape(attn.shape[0], h, w, C, self.temporal_size).permute(0, 4, 1, 2, 3)
            if source_attn is not None:
                attn = self.time_rasterizer.mix_with_source(source_attn, tara_attn)
            else:
                attn = tara_attn
        else:
            attn = self._apply_multiscale_temporal_conv(attn, h, w)

        attn = attn.reshape(B, N, C)
        x = x + attn

        x = self.up_proj(x)
        return x * self.gamma + inputs


class PlainAdapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x * self.gamma + inputs


class Attention(BaseModule):
    """Multi-head Self-attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        init_cfg: OptConfigType = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads

        self.scale = qk_scale or head_embed_dims**-0.5

        if qkv_bias:
            self._init_qv_bias()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def _init_qv_bias(self) -> None:
        self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the attention block, same size as inputs.
        """
        B, N, C = x.shape

        if hasattr(self, "q_bias"):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # standard self-attention
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        # fast attention
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(BaseModule):
    """The basic block in the Vision Transformer.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        act_cfg (dict or ConfigDict): Config for activation layer in FFN.
            Defaults to `dict(type='GELU')`.
        norm_cfg (dict or ConfigDict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: ConfigType = dict(type="GELU"),
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        init_cfg: OptConfigType = None,
        with_cp: bool = False,
        use_adapter: bool = False,
        adapter_mlp_ratio: float = 0.25,
        temporal_size: int = 384,
        adapter_tara_cfg: Optional[dict] = None,
        adapter_multiscale_cfg: Optional[dict] = None,
        adapter_time_film_cfg: Optional[dict] = None,
        adapter_gap_residual_cfg: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.use_adapter = use_adapter

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
        )

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False,
        )

        if self.use_adapter:
            self.adapter = Adapter(
                embed_dims=embed_dims,
                kernel_size=3,
                dilation=1,
                temporal_size=temporal_size,
                mlp_ratio=adapter_mlp_ratio,
                tara_cfg=adapter_tara_cfg,
                multiscale_cfg=adapter_multiscale_cfg,
                time_film_cfg=adapter_time_film_cfg,
                gap_residual_cfg=adapter_gap_residual_cfg,
            )

    def forward(self, x: Tensor, h, w, time_embed: Optional[Tensor] = None) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the transformer block, same size as inputs.
        """

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

            if self.use_adapter:
                x = self.adapter(x, h, w, time_embed=time_embed)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x, use_reentrant=False)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class VisionTransformerAdapter(BaseModule):
    """Vision Transformer with support for patch or hybrid CNN input stage. An
    impl of `VideoMAE: Masked Autoencoders are Data-Efficient Learners for
    Self-Supervised Video Pre-Training <https://arxiv.org/pdf/2203.12602.pdf>`_

    We add the checkpointing and frozen stage to the original VisionTransformer.

    Args:
        img_size (int or tuple): Size of input image.
            Defaults to 224.
        patch_size (int): Spatial size of one patch. Defaults to 16.
        in_channels (int): The number of channels of he input.
            Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        depth (int): number of blocks in the transformer.
            Defaults to 12.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 12.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        norm_cfg (dict or Configdict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        num_frames (int): Number of frames in the video. Defaults to 16.
        tubelet_size (int): Temporal size of one patch. Defaults to 2.
        use_mean_pooling (bool): If True, take the mean pooling over all
            positions. Defaults to True.
        pretrained (str, optional): Name of pretrained model. Default: None.
        return_feat_map (bool): If True, return the feature in the shape of
            `[B, C, T, H, W]`. Defaults to False.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: int = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        num_frames: int = 16,  # frames per attention
        tubelet_size: int = 2,
        use_mean_pooling: int = True,
        pretrained: Optional[str] = None,
        return_feat_map: bool = False,
        with_cp: bool = False,
        adapter_mlp_ratio: float = 0.25,
        total_frames: int = 768,
        adapter_index: list = [3, 5, 7, 11],
        use_irregular_time_embed: bool = False,
        add_irregular_time_embed: bool = True,
        time_embed_dim: int = 5,
        time_embed_hidden: int = 128,
        time_embed_scale: float = 0.25,
        time_embed_scale_max: Optional[float] = None,
        time_embed_warmup_epochs: int = 0,
        adapter_tara_cfg: Optional[dict] = None,
        adapter_multiscale_cfg: Optional[dict] = None,
        adapter_time_film_cfg: Optional[dict] = None,
        adapter_gap_residual_cfg: Optional[dict] = None,
        debug_time_grid: bool = False,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        **kwargs,
    ) -> None:
        if pretrained:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp

        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.use_irregular_time_embed = use_irregular_time_embed
        self.add_irregular_time_embed = add_irregular_time_embed
        self.debug_time_grid = debug_time_grid
        self.time_embed_scale_max = None if time_embed_scale_max is None else float(time_embed_scale_max)
        self.time_embed_warmup_epochs = int(time_embed_warmup_epochs)
        self.current_train_epoch = 0

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
        )

        grid_size = img_size // patch_size
        num_patches = grid_size**2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)

        # sine-cosine positional embeddings
        pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
        self.register_buffer("pos_embed", pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = ModuleList(
            [
                Block(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    init_cfg=init_cfg,
                    use_adapter=i in adapter_index,
                    adapter_mlp_ratio=adapter_mlp_ratio,
                    temporal_size=total_frames // tubelet_size,
                    adapter_tara_cfg=adapter_tara_cfg,
                    adapter_multiscale_cfg=adapter_multiscale_cfg,
                    adapter_time_film_cfg=adapter_time_film_cfg,
                    adapter_gap_residual_cfg=adapter_gap_residual_cfg,
                )
                for i in range(depth)
            ]
        )

        if use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.fc_norm = None

        self.return_feat_map = return_feat_map

        if self.use_irregular_time_embed and self.add_irregular_time_embed:
            self.time_embed_mlp = nn.Sequential(
                nn.Linear(time_embed_dim, time_embed_hidden),
                nn.GELU(),
                nn.Linear(time_embed_hidden, embed_dims),
            )
            self.time_embed_scale = nn.Parameter(torch.tensor(float(time_embed_scale)))
        else:
            self.time_embed_mlp = None
            self.time_embed_scale = None

        # count the number of parameters in the backbone
        num_vit_param = sum(p.numel() for name, p in self.named_parameters() if "adapter" not in name)
        num_adapter_param = sum(p.numel() for name, p in self.named_parameters() if "adapter" in name)
        ratio = num_adapter_param / num_vit_param * 100
        print("ViT's param: {}, Adapter's params: {}, ratio: {:2.1f}%".format(num_vit_param, num_adapter_param, ratio))

    def set_train_epoch(self, curr_epoch):
        self.current_train_epoch = int(curr_epoch)

    def _time_embed_scale_value(self, dtype, device):
        scale = self.time_embed_scale.to(device=device, dtype=dtype)
        if self.time_embed_scale_max is not None:
            scale = torch.tanh(scale) * self.time_embed_scale_max
        if self.training and self.time_embed_warmup_epochs > 0 and self.current_train_epoch < self.time_embed_warmup_epochs:
            scale = scale * 0.0
        return scale

    def _build_irregular_time_embedding(self, time_embed: Optional[Tensor], token_count: int, spatial_size: int, dtype, device):
        if self.time_embed_mlp is None or time_embed is None:
            return None
        if time_embed.dim() != 3:
            raise ValueError(f"time_embed must have shape [B, T_token, D], got {tuple(time_embed.shape)}")
        if time_embed.shape[1] != token_count:
            raise ValueError(
                f"time_embed token count ({time_embed.shape[1]}) does not match backbone token count ({token_count})."
            )

        time_token_embed = self.time_embed_mlp(time_embed.to(device=device, dtype=dtype))
        time_token_embed = time_token_embed.repeat_interleave(spatial_size, dim=1)
        return self._time_embed_scale_value(dtype=dtype, device=device) * time_token_embed

    def forward(self, x: Tensor, time_embed: Optional[Tensor] = None) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
        Returns:
            Tensor: The feature of the input
                samples extracted by the backbone.
        """
        self._freeze_layers()

        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]
        time_token_count = x.shape[1] // (h * w)
        if (h, w) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size, self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(h, w), mode="bicubic", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        raw_time_embed = time_embed if self.use_irregular_time_embed else None
        irregular_time_embed = self._build_irregular_time_embedding(
            time_embed=raw_time_embed,
            token_count=time_token_count,
            spatial_size=h * w,
            dtype=x.dtype,
            device=x.device,
        )

        x = x + pos_embed
        if irregular_time_embed is not None:
            x = x + irregular_time_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, h, w, time_embed=raw_time_embed)

        x = self.norm(x)

        if self.return_feat_map:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            x = x.permute(0, 4, 1, 2, 3)
            return x

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))

        return x[:, 0]

    def _freeze_layers(self):
        """Prevent all the parameters not in the adapters"""

        # freeze patch_embed
        self.patch_embed.eval()
        for m in self.patch_embed.modules():
            for param in m.parameters():
                param.requires_grad = False

        # freeze blocks except the adapter's parameters
        for block in self.blocks:
            for m, n in block.named_children():
                if "adapter" not in m and m != "drop_path":
                    n.eval()
                    for param in n.parameters():
                        param.requires_grad = False
