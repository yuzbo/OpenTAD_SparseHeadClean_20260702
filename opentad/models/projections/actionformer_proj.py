import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule, TransformerBlock
from ..builder import PROJECTIONS
from ..utils import normalize_temporal_grid_input, downsample_temporal_grid


@PROJECTIONS.register_module()
class Conv1DTransformerProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        attn_cfg=None,  # n_head n_mha_win_size attn_pdrop
        path_pdrop=0.0,  # dropout rate for drop path
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0,  # drop out the input feature
    ):
        super().__init__()
        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.proj_pdrop = conv_cfg["proj_pdrop"]
        self.scale_factor = 2  # as default
        self.n_mha_win_size = attn_cfg["n_mha_win_size"]
        self.n_head = attn_cfg["n_head"]
        self.attn_pdrop = 0.0  # as default
        self.path_pdrop = path_pdrop
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

        if isinstance(self.n_mha_win_size, int):
            self.mha_win_size = [self.n_mha_win_size] * (1 + arch[-1])
        else:
            assert len(self.n_mha_win_size) == (1 + arch[-1])
            self.mha_win_size = self.n_mha_win_size

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    out_channels,
                    self.n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=self.attn_pdrop,
                    proj_pdrop=self.proj_pdrop,
                    path_pdrop=self.path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


@PROJECTIONS.register_module()
class GridAwareConv1DTransformerProj(Conv1DTransformerProj):
    def forward(self, x, mask, temporal_grid=None):
        temporal_grid = normalize_temporal_grid_input(temporal_grid, mask)

        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        out_feats = (x,)
        out_masks = (mask,)
        out_grids = (temporal_grid,)

        for idx in range(len(self.branch)):
            temporal_grid = downsample_temporal_grid(temporal_grid)
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)
            out_grids += (temporal_grid,)

        return out_feats, out_masks, out_grids


@PROJECTIONS.register_module()
class DensePassthroughConv1DTransformerProj(Conv1DTransformerProj):
    """Dense Conv1DTransformerProj with temporal-grid passthrough.

    This wrapper keeps the dense feature path identical to Conv1DTransformerProj
    and only forwards/downsamples the temporal grid so an irregular-aware head
    can still decode on the native timeline.
    """

    def forward(self, x, mask, temporal_grid=None):
        temporal_grid = normalize_temporal_grid_input(temporal_grid, mask)
        out_feats, out_masks = super().forward(x, mask)

        if temporal_grid is None:
            out_grids = tuple(None for _ in out_feats)
        else:
            out_grids = (temporal_grid,)
            current_grid = temporal_grid
            for _ in range(1, len(out_feats)):
                current_grid = downsample_temporal_grid(current_grid)
                out_grids += (current_grid,)

        return out_feats, out_masks, out_grids


def get_sinusoid_encoding(n_position, d_hid):
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)
