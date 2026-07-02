import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS, build_backbone
from ..bricks import AffineDropPath, Scale
from ..utils.post_processing import load_predictions, save_predictions
from .actionformer import ActionFormer


class CompletionResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        groups = 32 if channels % 32 == 0 else 1
        self.norm = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, valid_mask):
        residual = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = (x + residual) * valid_mask.unsqueeze(1).to(x.dtype)
        return x


class MinimalTemporalCompletion(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=512,
        num_blocks=4,
        kernel_size=5,
        dilation_growth=2,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels + 1, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList()
        dilation = 1
        for _ in range(num_blocks):
            self.blocks.append(
                CompletionResidualBlock(
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            dilation *= max(int(dilation_growth), 1)
        groups = 32 if hidden_channels % 32 == 0 else 1
        self.output_norm = nn.GroupNorm(groups, hidden_channels)
        self.output_act = nn.GELU()
        self.output_proj = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, observed_feat, observed_mask, valid_mask):
        observed_mask_f = observed_mask.unsqueeze(1).to(observed_feat.dtype)
        valid_mask_f = valid_mask.unsqueeze(1).to(observed_feat.dtype)
        x = torch.cat([observed_feat, observed_mask_f], dim=1)
        x = self.input_proj(x) * valid_mask_f
        for block in self.blocks:
            x = block(x, valid_mask)
        x = self.output_proj(self.output_act(self.output_norm(x))) * valid_mask_f

        missing_mask_f = (valid_mask & ~observed_mask).unsqueeze(1).to(observed_feat.dtype)
        completed = observed_feat * observed_mask_f + x * missing_mask_f
        return completed * valid_mask_f


@DETECTORS.register_module()
class SparseCompletionActionFormer(ActionFormer):
    def __init__(
        self,
        projection,
        rpn_head,
        neck=None,
        backbone=None,
        completion=None,
        oracle_backbone=None,
    ):
        super().__init__(projection=projection, rpn_head=rpn_head, neck=neck, backbone=backbone)

        completion_cfg = {} if completion is None else dict(completion)
        completion_type = completion_cfg.pop("type", "MinimalTemporalCompletion")
        self.feature_loss_weight = float(completion_cfg.pop("feature_loss_weight", 0.0))
        self.completion_type = completion_type
        self.oracle_fill_mode = completion_cfg.pop("fill_mode", "holes")

        if completion_type == "oracle":
            if self.oracle_fill_mode not in {"holes", "full"}:
                raise ValueError(f"Unsupported oracle fill_mode: {self.oracle_fill_mode}")
            self.completion = None
        elif completion_type in {"nearest", "linear", "identity", "scatter", "none"}:
            self.completion = None
        elif completion_type == "MinimalTemporalCompletion":
            self.completion = MinimalTemporalCompletion(in_channels=self.projection.in_channels, **completion_cfg)
        else:
            raise ValueError(f"Unsupported completion type: {completion_type}")

        if oracle_backbone is not None:
            self.oracle_backbone = build_backbone(oracle_backbone)

    @property
    def with_oracle_backbone(self):
        return hasattr(self, "oracle_backbone") and self.oracle_backbone is not None

    def _extract_sparse_features(self, inputs, metas):
        if self.with_backbone:
            return self.backbone(inputs, metas=metas)
        return inputs

    def _extract_oracle_features(self, oracle_inputs):
        if oracle_inputs is None:
            return None
        if not self.with_oracle_backbone:
            raise ValueError("oracle_inputs were provided, but oracle_backbone is not configured.")
        with torch.no_grad():
            oracle_feat = self.oracle_backbone(oracle_inputs)
        if oracle_feat.shape[-1] != self.max_seq_len:
            oracle_feat = F.interpolate(oracle_feat, size=self.max_seq_len, mode="linear", align_corners=False)
        return oracle_feat.detach()

    def _scatter_sparse_to_dense(self, sparse_feat, sparse_mask, metas):
        batch_size, channels, _ = sparse_feat.shape
        dense_len = self.max_seq_len
        dense_feat = sparse_feat.new_zeros(batch_size, channels, dense_len)
        observed_mask = torch.zeros(batch_size, dense_len, device=sparse_feat.device, dtype=torch.bool)
        valid_mask = torch.zeros(batch_size, dense_len, device=sparse_feat.device, dtype=torch.bool)

        for sample_idx, (mask, meta) in enumerate(zip(sparse_mask, metas)):
            selected_positions = meta.get("irregular_selected_positions", None)
            valid_len = meta.get("irregular_selected_valid_len", None)
            if selected_positions is None or valid_len is None:
                raise ValueError(
                    "SparseCompletionActionFormer requires irregular_selected_positions and irregular_selected_valid_len in metas."
                )

            positions = torch.as_tensor(selected_positions, device=sparse_feat.device, dtype=torch.float32).flatten()
            sparse_valid = min(int(mask.sum().item()), int(positions.numel()))
            dense_valid = int(round(float(valid_len)))
            dense_valid = max(min(dense_valid, dense_len), 1)
            valid_mask[sample_idx, :dense_valid] = True

            if sparse_valid <= 0:
                continue

            positions = positions[:sparse_valid].round().to(torch.long).clamp_(0, dense_valid - 1)
            dense_feat[sample_idx, :, positions] = sparse_feat[sample_idx, :, :sparse_valid]
            observed_mask[sample_idx, positions] = True

        dense_feat = dense_feat * valid_mask.unsqueeze(1).to(dense_feat.dtype)
        return dense_feat, observed_mask, valid_mask

    def _complete_dense_features(self, observed_feat, observed_mask, valid_mask, oracle_feat=None):
        if self.completion_type == "oracle":
            if oracle_feat is None:
                raise ValueError("Oracle completion requires oracle_inputs / oracle_backbone.")
            valid_mask_f = valid_mask.unsqueeze(1).to(observed_feat.dtype)
            if self.oracle_fill_mode == "full":
                return oracle_feat * valid_mask_f
            observed_mask_f = observed_mask.unsqueeze(1).to(observed_feat.dtype)
            missing_mask_f = (valid_mask & ~observed_mask).unsqueeze(1).to(observed_feat.dtype)
            oracle_feat = oracle_feat * valid_mask_f
            return observed_feat * observed_mask_f + oracle_feat * missing_mask_f

        if self.completion_type in {"identity", "scatter", "none"}:
            return observed_feat * valid_mask.unsqueeze(1).to(observed_feat.dtype)

        if self.completion_type in {"nearest", "linear"}:
            return self._fill_dense_features(observed_feat, observed_mask, valid_mask, mode=self.completion_type)

        return self.completion(observed_feat, observed_mask, valid_mask)

    def _fill_dense_features(self, observed_feat, observed_mask, valid_mask, mode="nearest"):
        batch_size, channels, dense_len = observed_feat.shape
        completed = observed_feat.new_zeros(batch_size, channels, dense_len)

        for sample_idx in range(batch_size):
            sample_valid = valid_mask[sample_idx]
            dense_valid = int(sample_valid.sum().item())
            if dense_valid <= 0:
                continue

            query_idx = torch.arange(dense_valid, device=observed_feat.device, dtype=torch.long)
            obs_idx = torch.nonzero(observed_mask[sample_idx, :dense_valid], as_tuple=False).flatten()
            if obs_idx.numel() == 0:
                continue

            sample_feat = observed_feat[sample_idx, :, :dense_valid]
            if obs_idx.numel() == 1:
                fill_feat = sample_feat[:, obs_idx[0]].unsqueeze(-1).expand(-1, dense_valid)
                completed[sample_idx, :, :dense_valid] = fill_feat
                continue

            insert = torch.searchsorted(obs_idx, query_idx, right=False)
            left_ptr = (insert - 1).clamp(min=0)
            right_ptr = insert.clamp(max=obs_idx.numel() - 1)
            left_pos = obs_idx[left_ptr]
            right_pos = obs_idx[right_ptr]
            left_feat = sample_feat[:, left_pos]
            right_feat = sample_feat[:, right_pos]

            if mode == "nearest":
                choose_right = (query_idx - left_pos).abs() >= (right_pos - query_idx).abs()
                chosen_pos = torch.where(choose_right, right_pos, left_pos)
                completed[sample_idx, :, :dense_valid] = sample_feat[:, chosen_pos]
                continue

            same_pos = left_pos == right_pos
            denom = (right_pos - left_pos).to(observed_feat.dtype).clamp_min(1.0)
            alpha = (query_idx - left_pos).to(observed_feat.dtype) / denom
            alpha = torch.where(same_pos, torch.zeros_like(alpha), alpha).clamp_(0.0, 1.0)
            completed[sample_idx, :, :dense_valid] = left_feat * (1.0 - alpha.unsqueeze(0)) + right_feat * alpha.unsqueeze(0)

        return completed * valid_mask.unsqueeze(1).to(observed_feat.dtype)

    def _feature_reconstruction_loss(self, dense_feat, oracle_feat, observed_mask, valid_mask):
        if self.feature_loss_weight <= 0.0 or oracle_feat is None:
            return None
        missing_mask = (valid_mask & ~observed_mask).unsqueeze(1)
        if not missing_mask.any():
            return None
        pred = dense_feat[missing_mask.expand_as(dense_feat)]
        target = oracle_feat[missing_mask.expand_as(oracle_feat)]
        if pred.numel() == 0:
            return None
        return F.smooth_l1_loss(pred, target, reduction="mean") * self.feature_loss_weight

    def _run_dense_detector(self, dense_feat, dense_mask, gt_segments=None, gt_labels=None, **kwargs):
        x, masks = self.pad_data(dense_feat, dense_mask)
        if self.with_projection:
            x, masks = self.projection(x, masks)
        if self.with_neck:
            x, masks = self.neck(x, masks)

        if gt_segments is None or gt_labels is None:
            return self.rpn_head.forward_test(x, masks, **kwargs)
        return self.rpn_head.forward_train(x, masks, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, oracle_inputs=None, **kwargs):
        sparse_feat = self._extract_sparse_features(inputs, metas)
        observed_feat, observed_mask, dense_valid_mask = self._scatter_sparse_to_dense(sparse_feat, masks, metas)
        oracle_feat = self._extract_oracle_features(oracle_inputs)
        dense_feat = self._complete_dense_features(observed_feat, observed_mask, dense_valid_mask, oracle_feat=oracle_feat)

        losses = self._run_dense_detector(dense_feat, dense_valid_mask, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)
        feature_loss = self._feature_reconstruction_loss(dense_feat, oracle_feat, observed_mask, dense_valid_mask)
        if feature_loss is not None:
            losses["completion_loss"] = feature_loss
        losses["cost"] = sum(value for value in losses.values())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, oracle_inputs=None, **kwargs):
        sparse_feat = self._extract_sparse_features(inputs, metas)
        observed_feat, observed_mask, dense_valid_mask = self._scatter_sparse_to_dense(sparse_feat, masks, metas)
        oracle_feat = self._extract_oracle_features(oracle_inputs)
        dense_feat = self._complete_dense_features(observed_feat, observed_mask, dense_valid_mask, oracle_feat=oracle_feat)
        return self._run_dense_detector(dense_feat, dense_valid_mask, gt_segments=None, gt_labels=None, **kwargs)

    def forward_detection(self, inputs, masks, metas, infer_cfg, post_cfg, **kwargs):
        # Oracle completion needs eval-time kwargs such as oracle_inputs, so the
        # generic BaseDetector.forward_detection path is not sufficient here.
        if infer_cfg.load_from_raw_predictions:
            predictions = load_predictions(metas, infer_cfg)
        else:
            predictions = self.forward_test(inputs, masks, metas, infer_cfg, **kwargs)
            if infer_cfg.save_raw_prediction:
                save_predictions(predictions, metas, infer_cfg.folder)

        results = self.post_processing(predictions, metas, post_cfg, **kwargs)
        return results

    def get_optim_groups(self, cfg):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        def _skip_prefix(name):
            return name.startswith("backbone") or name.startswith("oracle_backbone")

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if _skip_prefix(fpn):
                    continue

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    no_decay.add(fpn)

        param_dict = {
            pn: p for pn, p in self.named_parameters() if (not _skip_prefix(pn)) and p.requires_grad
        }
        decay = {pn for pn in decay if pn in param_dict}
        no_decay = {pn for pn in no_decay if pn in param_dict}

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay sets!"
        )

        return [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": cfg["weight_decay"],
                "lr": cfg["lr"],
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
                "lr": cfg["lr"],
            },
        ]
