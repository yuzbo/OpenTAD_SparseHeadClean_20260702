import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule, Scale


@HEADS.register_module()
class AnchorFreeHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        filter_similar_gt=True,
        use_regress_range=True,
        assigner=None,
        assignment_debug=None,
        cls_residual_cfg=None,
        reg_residual_cfg=None,
        cls_geometry_cfg=None,
        reg_geometry_cfg=None,
        quality_head_cfg=None,
        sample_reliability_cfg=None,
    ):
        super(AnchorFreeHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt
        self.use_regress_range = use_regress_range
        self.assignment_debug = assignment_debug or {}
        self.assignment_debug_enabled = bool(self.assignment_debug.get("enabled", False))
        self.cls_residual_cfg = None if cls_residual_cfg is None else dict(cls_residual_cfg)
        self.reg_residual_cfg = None if reg_residual_cfg is None else dict(reg_residual_cfg)
        self.cls_geometry_cfg = None if cls_geometry_cfg is None else dict(cls_geometry_cfg)
        self.reg_geometry_cfg = None if reg_geometry_cfg is None else dict(reg_geometry_cfg)
        self.quality_head_cfg = {} if quality_head_cfg is None else dict(quality_head_cfg)
        self.sample_reliability_cfg = {} if sample_reliability_cfg is None else dict(sample_reliability_cfg)
        self.cls_geometry_enabled = bool(self.cls_geometry_cfg and self.cls_geometry_cfg.get("enabled", True))
        self.cls_geometry_input_dim = int((self.cls_geometry_cfg or {}).get("input_dim", 5))
        self.cls_geometry_max_delta = (self.cls_geometry_cfg or {}).get("max_delta", 0.20)
        if self.cls_geometry_max_delta is not None:
            self.cls_geometry_max_delta = float(self.cls_geometry_max_delta)
        self.reg_geometry_enabled = bool(self.reg_geometry_cfg and self.reg_geometry_cfg.get("enabled", True))
        self.reg_geometry_input_dim = int((self.reg_geometry_cfg or {}).get("input_dim", 5))
        self.reg_geometry_max_delta = (self.reg_geometry_cfg or {}).get("max_delta", 0.20)
        if self.reg_geometry_max_delta is not None:
            self.reg_geometry_max_delta = float(self.reg_geometry_max_delta)
        self.quality_head_enabled = bool(self.quality_head_cfg.get("enabled", False))
        self.quality_loss_weight = float(self.quality_head_cfg.get("loss_weight", 0.0))
        self.quality_score_alpha = float(self.quality_head_cfg.get("score_alpha", 0.0))
        self.quality_target_mode = self.quality_head_cfg.get("target_mode", "assigned_iou")
        self.quality_positive_weight = float(self.quality_head_cfg.get("positive_weight", 1.0))
        self.quality_negative_weight = float(self.quality_head_cfg.get("negative_weight", 1.0))
        self.quality_loss_normalizer = self.quality_head_cfg.get("loss_normalizer", "valid")
        self.quality_keep_loss_graph_when_weight_zero = bool(
            self.quality_head_cfg.get("keep_loss_graph_when_weight_zero", False)
        )
        self.sample_reliability_enabled = bool(self.sample_reliability_cfg.get("enabled", False))
        self.sample_reliability_mode = self.sample_reliability_cfg.get("mode", "support")
        self.sample_reliability_apply_to = self.sample_reliability_cfg.get("apply_to", "all")
        self.sample_reliability_min_weight = float(self.sample_reliability_cfg.get("min_weight", 0.60))
        self.sample_reliability_gamma = float(self.sample_reliability_cfg.get("gamma", 1.0))
        valid_sample_reliability_keys = {"enabled", "mode", "apply_to", "min_weight", "gamma"}
        unknown_sample_reliability_keys = set(self.sample_reliability_cfg) - valid_sample_reliability_keys
        if unknown_sample_reliability_keys:
            raise ValueError(f"Unknown sample reliability options: {sorted(unknown_sample_reliability_keys)}")
        if self.sample_reliability_mode not in {"support"}:
            raise ValueError(f"Unsupported sample reliability mode: {self.sample_reliability_mode}")
        if self.sample_reliability_apply_to not in {"all", "positive"}:
            raise ValueError(f"Unsupported sample reliability apply_to: {self.sample_reliability_apply_to}")
        if not 0.0 <= self.sample_reliability_min_weight <= 1.0:
            raise ValueError("sample_reliability min_weight must be in [0, 1]")
        if self.sample_reliability_gamma <= 0:
            raise ValueError("sample_reliability gamma must be positive")
        valid_quality_target_modes = {"assigned_iou", "max_iou", "positive_max_iou"}
        if self.quality_target_mode not in valid_quality_target_modes:
            raise ValueError(f"Unsupported quality target mode: {self.quality_target_mode}")
        valid_quality_normalizers = {"valid", "weighted", "positive"}
        if self.quality_loss_normalizer not in valid_quality_normalizers:
            raise ValueError(f"Unsupported quality loss normalizer: {self.quality_loss_normalizer}")
        if self.quality_loss_normalizer == "positive" and self.quality_negative_weight > 0:
            raise ValueError("quality loss_normalizer='positive' requires negative_weight=0.0")

        self.loss_weight = loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(loss_normalizer))  # save in the state_dict

        # point generator
        self.prior_generator = build_prior_generator(prior_generator)

        self._init_layers()

        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)
        self.assigner = build_loss(assigner) if assigner is not None else None
        self._train_epoch = None
        self._last_assigner_stats = []
        self._reset_assignment_diag()

    def set_train_epoch(self, curr_epoch):
        self._train_epoch = int(curr_epoch)
        self._reset_assignment_diag()

    def _reset_assignment_diag(self):
        num_levels = len(self.prior_generator.strides) if hasattr(self, "prior_generator") else 0
        self._assignment_diag = {
            "epoch": self._train_epoch,
            "iters": 0,
            "samples": 0,
            "gt": 0,
            "valid_points": 0,
            "pos_points": 0,
            "weighted_pos": 0.0,
            "valid_weight_sum": 0.0,
            "valid_weight_count": 0,
            "valid_weight_lt1": 0,
            "valid_weight_eq0": 0,
            "pos_weight_sum": 0.0,
            "pos_weight_count": 0,
            "pos_weight_lt1": 0,
            "per_level_valid": [0 for _ in range(num_levels)],
            "per_level_pos": [0 for _ in range(num_levels)],
            "per_level_pos_weight": [0.0 for _ in range(num_levels)],
            "per_level_reg_loss_sum": [0.0 for _ in range(num_levels)],
            "per_level_reg_iou_sum": [0.0 for _ in range(num_levels)],
            "per_level_reg_target_len_sum": [0.0 for _ in range(num_levels)],
            "per_level_reg_count": [0 for _ in range(num_levels)],
            "reg_loss_sum": 0.0,
            "reg_loss_count": 0,
            "reg_iou_sum": 0.0,
            "reg_iou_count": 0,
            "reg_target_len_sum": 0.0,
            "reg_target_len_count": 0,
            "reg_target_len_min": None,
            "reg_target_len_max": None,
            "reg_pred_len_sum": 0.0,
            "reg_pred_len_count": 0,
            "reg_len_bin_edges": [16.0, 32.0, 64.0, 128.0, 256.0],
            "reg_len_bin_count": [0 for _ in range(6)],
            "reg_len_bin_loss_sum": [0.0 for _ in range(6)],
            "reg_len_bin_iou_sum": [0.0 for _ in range(6)],
            "raw_candidate_count_sum": 0,
            "raw_candidate_count_count": 0,
            "raw_candidate_count_min": None,
            "raw_candidate_count_max": None,
            "candidate_count_sum": 0,
            "candidate_count_count": 0,
            "candidate_count_min": None,
            "candidate_count_max": None,
            "dynamic_k_sum": 0,
            "dynamic_k_count": 0,
            "dynamic_k_min": None,
            "dynamic_k_max": None,
            "matched_count_sum": 0,
            "matched_count_count": 0,
            "matched_count_min": None,
            "matched_count_max": None,
            "candidate_point_count": 0,
            "confuse_point_count": 0,
            "matched_point_count": 0,
        }

    def _update_minmax(self, min_key, max_key, value):
        if self._assignment_diag[min_key] is None or value < self._assignment_diag[min_key]:
            self._assignment_diag[min_key] = value
        if self._assignment_diag[max_key] is None or value > self._assignment_diag[max_key]:
            self._assignment_diag[max_key] = value

    def _assignment_level_ids(self, points, device):
        level_ids = []
        for level, point in enumerate(points):
            level_ids.append(torch.full((point.shape[0],), level, dtype=torch.long, device=device))
        return torch.cat(level_ids, dim=0)

    @torch.no_grad()
    def _update_assignment_diag(self, points, valid_mask, gt_cls, pos_mask, target_weights, gt_segments):
        if not self.assignment_debug_enabled:
            return

        diag = self._assignment_diag
        diag["iters"] += 1
        diag["samples"] += int(gt_cls.shape[0])
        diag["gt"] += sum(int(segment.shape[0]) for segment in gt_segments)
        diag["valid_points"] += int(valid_mask.sum().item())
        diag["pos_points"] += int(pos_mask.sum().item())

        if target_weights is None:
            weights = torch.ones_like(valid_mask, dtype=torch.float32)
        else:
            weights = target_weights.to(dtype=torch.float32)

        valid_weights = weights[valid_mask]
        pos_weights = weights[pos_mask]
        diag["valid_weight_sum"] += float(valid_weights.sum().item()) if valid_weights.numel() > 0 else 0.0
        diag["valid_weight_count"] += int(valid_weights.numel())
        diag["valid_weight_lt1"] += int((valid_weights < 0.999).sum().item()) if valid_weights.numel() > 0 else 0
        diag["valid_weight_eq0"] += int((valid_weights <= 0.0).sum().item()) if valid_weights.numel() > 0 else 0
        diag["pos_weight_sum"] += float(pos_weights.sum().item()) if pos_weights.numel() > 0 else 0.0
        diag["pos_weight_count"] += int(pos_weights.numel())
        diag["pos_weight_lt1"] += int((pos_weights < 0.999).sum().item()) if pos_weights.numel() > 0 else 0
        diag["weighted_pos"] += float((pos_mask.float() * weights).sum().item())

        level_ids = self._assignment_level_ids(points, valid_mask.device)
        for level in range(len(diag["per_level_valid"])):
            level_mask = level_ids == level
            valid_level = valid_mask[:, level_mask]
            pos_level = pos_mask[:, level_mask]
            weight_level = weights[:, level_mask]
            diag["per_level_valid"][level] += int(valid_level.sum().item())
            diag["per_level_pos"][level] += int(pos_level.sum().item())
            diag["per_level_pos_weight"][level] += float((pos_level.float() * weight_level).sum().item())

        for stats in self._last_assigner_stats:
            for key, sum_key, count_key, min_key, max_key in (
                (
                    "raw_candidate_counts",
                    "raw_candidate_count_sum",
                    "raw_candidate_count_count",
                    "raw_candidate_count_min",
                    "raw_candidate_count_max",
                ),
                ("candidate_counts", "candidate_count_sum", "candidate_count_count", "candidate_count_min", "candidate_count_max"),
                ("dynamic_ks", "dynamic_k_sum", "dynamic_k_count", "dynamic_k_min", "dynamic_k_max"),
                ("matched_counts", "matched_count_sum", "matched_count_count", "matched_count_min", "matched_count_max"),
            ):
                for value in stats.get(key, []):
                    diag[sum_key] += int(value)
                    diag[count_key] += 1
                    self._update_minmax(min_key, max_key, int(value))
            diag["candidate_point_count"] += int(stats.get("candidate_point_count", 0))
            diag["confuse_point_count"] += int(stats.get("confuse_point_count", 0))
            diag["matched_point_count"] += int(stats.get("matched_point_count", 0))

    @torch.no_grad()
    def _update_regression_diag(self, points, pos_mask, pred_segments, target_segments, reg_loss_values):
        if not self.assignment_debug_enabled or pred_segments.numel() == 0:
            return

        diag = self._assignment_diag
        losses = reg_loss_values.detach().to(dtype=torch.float32)
        preds = pred_segments.detach().to(dtype=torch.float32)
        targets = target_segments.detach().to(dtype=torch.float32)

        target_len = (targets[:, 1] - targets[:, 0]).clamp(min=0.0)
        pred_len = (preds[:, 1] - preds[:, 0]).clamp(min=0.0)
        inter = (torch.minimum(preds[:, 1], targets[:, 1]) - torch.maximum(preds[:, 0], targets[:, 0])).clamp(min=0.0)
        union = (pred_len + target_len - inter).clamp(min=1e-6)
        ious = inter / union

        finite = torch.isfinite(losses) & torch.isfinite(ious) & torch.isfinite(target_len) & torch.isfinite(pred_len)
        if not finite.any():
            return

        losses = losses[finite]
        ious = ious[finite]
        target_len = target_len[finite]
        pred_len = pred_len[finite]

        diag["reg_loss_sum"] += float(losses.sum().item())
        diag["reg_loss_count"] += int(losses.numel())
        diag["reg_iou_sum"] += float(ious.sum().item())
        diag["reg_iou_count"] += int(ious.numel())
        diag["reg_target_len_sum"] += float(target_len.sum().item())
        diag["reg_target_len_count"] += int(target_len.numel())
        diag["reg_pred_len_sum"] += float(pred_len.sum().item())
        diag["reg_pred_len_count"] += int(pred_len.numel())
        self._update_minmax("reg_target_len_min", "reg_target_len_max", float(target_len.min().item()))
        self._update_minmax("reg_target_len_min", "reg_target_len_max", float(target_len.max().item()))

        level_ids = self._assignment_level_ids(points, pos_mask.device)
        pos_levels = level_ids[None, :].expand_as(pos_mask)[pos_mask][finite]
        for level in range(len(diag["per_level_reg_count"])):
            level_pos = pos_levels == level
            if not level_pos.any():
                continue
            diag["per_level_reg_loss_sum"][level] += float(losses[level_pos].sum().item())
            diag["per_level_reg_iou_sum"][level] += float(ious[level_pos].sum().item())
            diag["per_level_reg_target_len_sum"][level] += float(target_len[level_pos].sum().item())
            diag["per_level_reg_count"][level] += int(level_pos.sum().item())

        edges = diag["reg_len_bin_edges"]
        bin_ids = torch.bucketize(target_len, target_len.new_tensor(edges), right=False)
        for bin_idx in range(len(diag["reg_len_bin_count"])):
            bin_pos = bin_ids == bin_idx
            if not bin_pos.any():
                continue
            diag["reg_len_bin_count"][bin_idx] += int(bin_pos.sum().item())
            diag["reg_len_bin_loss_sum"][bin_idx] += float(losses[bin_pos].sum().item())
            diag["reg_len_bin_iou_sum"][bin_idx] += float(ious[bin_pos].sum().item())

    def collect_debug_state(self):
        if not self.assignment_debug_enabled:
            return {}

        diag = self._assignment_diag

        def safe_avg(sum_key, count_key):
            count = diag[count_key]
            return float(diag[sum_key] / count) if count else 0.0

        def safe_list_avg(sum_key, count_key):
            return [
                round(float(total / count), 4) if count else 0.0
                for total, count in zip(diag[sum_key], diag[count_key])
            ]

        def safe_bin_avg(sum_key):
            return [
                round(float(total / count), 4) if count else 0.0
                for total, count in zip(diag[sum_key], diag["reg_len_bin_count"])
            ]

        pos_count = max(diag["pos_points"], 1)
        valid_count = max(diag["valid_points"], 1)
        return {
            "assign_epoch": diag["epoch"],
            "assign_iters": diag["iters"],
            "assign_samples": diag["samples"],
            "assign_gt": diag["gt"],
            "assign_valid_points": diag["valid_points"],
            "assign_pos_points": diag["pos_points"],
            "assign_pos_per_sample": float(diag["pos_points"] / max(diag["samples"], 1)),
            "assign_pos_per_gt": float(diag["pos_points"] / max(diag["gt"], 1)),
            "assign_weighted_pos": round(diag["weighted_pos"], 4),
            "assign_weighted_pos_per_gt": float(diag["weighted_pos"] / max(diag["gt"], 1)),
            "assign_valid_weight_mean": safe_avg("valid_weight_sum", "valid_weight_count"),
            "assign_valid_weight_lt1_frac": float(diag["valid_weight_lt1"] / valid_count),
            "assign_valid_weight_eq0_frac": float(diag["valid_weight_eq0"] / valid_count),
            "assign_pos_weight_mean": float(diag["pos_weight_sum"] / pos_count),
            "assign_pos_weight_lt1_frac": float(diag["pos_weight_lt1"] / pos_count),
            "assign_per_level_valid": diag["per_level_valid"],
            "assign_per_level_pos": diag["per_level_pos"],
            "assign_per_level_pos_weight": [round(value, 4) for value in diag["per_level_pos_weight"]],
            "assign_dynamic_k_mean": safe_avg("dynamic_k_sum", "dynamic_k_count"),
            "assign_dynamic_k_min": diag["dynamic_k_min"],
            "assign_dynamic_k_max": diag["dynamic_k_max"],
            "assign_raw_candidate_count_mean": safe_avg("raw_candidate_count_sum", "raw_candidate_count_count"),
            "assign_raw_candidate_count_min": diag["raw_candidate_count_min"],
            "assign_raw_candidate_count_max": diag["raw_candidate_count_max"],
            "assign_candidate_count_mean": safe_avg("candidate_count_sum", "candidate_count_count"),
            "assign_candidate_count_min": diag["candidate_count_min"],
            "assign_candidate_count_max": diag["candidate_count_max"],
            "assign_matched_count_mean": safe_avg("matched_count_sum", "matched_count_count"),
            "assign_matched_count_min": diag["matched_count_min"],
            "assign_matched_count_max": diag["matched_count_max"],
            "assign_candidate_point_count": diag["candidate_point_count"],
            "assign_confuse_point_count": diag["confuse_point_count"],
            "assign_matched_point_count": diag["matched_point_count"],
            "assign_reg_loss_mean": safe_avg("reg_loss_sum", "reg_loss_count"),
            "assign_reg_iou_mean": safe_avg("reg_iou_sum", "reg_iou_count"),
            "assign_reg_target_len_mean": safe_avg("reg_target_len_sum", "reg_target_len_count"),
            "assign_reg_target_len_min": diag["reg_target_len_min"],
            "assign_reg_target_len_max": diag["reg_target_len_max"],
            "assign_reg_pred_len_mean": safe_avg("reg_pred_len_sum", "reg_pred_len_count"),
            "assign_reg_per_level_loss": safe_list_avg("per_level_reg_loss_sum", "per_level_reg_count"),
            "assign_reg_per_level_iou": safe_list_avg("per_level_reg_iou_sum", "per_level_reg_count"),
            "assign_reg_per_level_target_len": safe_list_avg("per_level_reg_target_len_sum", "per_level_reg_count"),
            "assign_reg_len_bin_edges": diag["reg_len_bin_edges"],
            "assign_reg_len_bin_count": diag["reg_len_bin_count"],
            "assign_reg_len_bin_loss": safe_bin_avg("reg_len_bin_loss_sum"),
            "assign_reg_len_bin_iou": safe_bin_avg("reg_len_bin_iou_sum"),
        }

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_heads()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.cls_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.reg_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_heads(self):
        """Initialize predictor layers of the head."""
        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])
        self.cls_residual = None
        self.cls_residual_scale = None
        self.reg_residual = None
        self.reg_residual_scale = None
        self.cls_geometry_embed = None
        self.cls_geometry_residual = None
        self.reg_geometry_embed = None
        self.reg_geometry_residual = None
        self.quality_head = None
        cls_residual_cfg = self.cls_residual_cfg
        if cls_residual_cfg is not None:
            kernel_size = int(cls_residual_cfg.get("kernel_size", 3))
            padding = kernel_size // 2
            if bool(cls_residual_cfg.get("depthwise", True)):
                self.cls_residual = nn.Sequential(
                    nn.Conv1d(
                        self.feat_channels,
                        self.feat_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=self.feat_channels,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=1),
                )
            else:
                hidden_channels = int(cls_residual_cfg.get("hidden_channels", self.feat_channels))
                self.cls_residual = nn.Sequential(
                    nn.Conv1d(self.feat_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channels, self.num_classes, kernel_size=1),
                )
            self.cls_residual_scale = nn.Parameter(torch.tensor(float(cls_residual_cfg.get("init_scale", 0.0))))
        reg_residual_cfg = self.reg_residual_cfg
        if reg_residual_cfg is not None:
            kernel_size = int(reg_residual_cfg.get("kernel_size", 3))
            padding = kernel_size // 2
            if bool(reg_residual_cfg.get("depthwise", True)):
                self.reg_residual = nn.Sequential(
                    nn.Conv1d(
                        self.feat_channels,
                        self.feat_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        groups=self.feat_channels,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(self.feat_channels, 2, kernel_size=1),
                )
            else:
                hidden_channels = int(reg_residual_cfg.get("hidden_channels", self.feat_channels))
                self.reg_residual = nn.Sequential(
                    nn.Conv1d(self.feat_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channels, 2, kernel_size=1),
            )
            self.reg_residual_scale = nn.Parameter(torch.tensor(float(reg_residual_cfg.get("init_scale", 0.0))))
        cls_geometry_cfg = self.cls_geometry_cfg
        if self.cls_geometry_enabled:
            hidden_channels = int(cls_geometry_cfg.get("hidden_channels", max(self.feat_channels // 4, 16)))
            kernel_size = int(cls_geometry_cfg.get("kernel_size", 3))
            padding = kernel_size // 2
            self.cls_geometry_embed = nn.Sequential(
                nn.Conv1d(self.cls_geometry_input_dim, hidden_channels, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.cls_geometry_residual = nn.Sequential(
                nn.Conv1d(self.feat_channels + hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channels, self.num_classes, kernel_size=1),
            )
            nn.init.constant_(self.cls_geometry_residual[-1].weight, 0.0)
            nn.init.constant_(self.cls_geometry_residual[-1].bias, 0.0)
        reg_geometry_cfg = self.reg_geometry_cfg
        if self.reg_geometry_enabled:
            hidden_channels = int(reg_geometry_cfg.get("hidden_channels", max(self.feat_channels // 4, 16)))
            kernel_size = int(reg_geometry_cfg.get("kernel_size", 3))
            padding = kernel_size // 2
            self.reg_geometry_embed = nn.Sequential(
                nn.Conv1d(self.reg_geometry_input_dim, hidden_channels, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            self.reg_geometry_residual = nn.Sequential(
                nn.Conv1d(self.feat_channels + hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_channels, 2, kernel_size=1),
            )
            nn.init.constant_(self.reg_geometry_residual[-1].weight, 0.0)
            nn.init.constant_(self.reg_geometry_residual[-1].bias, 0.0)
        if self.quality_head_enabled:
            kernel_size = int(self.quality_head_cfg.get("kernel_size", 3))
            self.quality_head = nn.Conv1d(self.feat_channels, 1, kernel_size=kernel_size, padding=kernel_size // 2)
            nn.init.constant_(self.quality_head.weight, float(self.quality_head_cfg.get("weight_init", 0.0)))
            nn.init.constant_(self.quality_head.bias, float(self.quality_head_cfg.get("bias_init", 0.0)))

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

    def _apply_cls_residual(self, cls_feat, cls_logits):
        if self.cls_residual is None:
            return cls_logits
        cls_logits = cls_logits + self.cls_residual_scale.to(dtype=cls_logits.dtype) * self.cls_residual(cls_feat)
        return cls_logits

    def _apply_reg_residual(self, reg_feat, reg_raw):
        if self.reg_residual is None:
            return reg_raw
        reg_raw = reg_raw + self.reg_residual_scale.to(dtype=reg_raw.dtype) * self.reg_residual(reg_feat)
        return reg_raw

    def _sample_geometry_features(self, meta, base_len, device, dtype, input_dim):
        positions = meta.get("irregular_selected_positions", None) if meta is not None else None
        valid_len = meta.get("irregular_selected_valid_len", None) if meta is not None else None
        if positions is None or valid_len is None:
            return torch.zeros(input_dim, base_len, device=device, dtype=dtype)

        positions = torch.as_tensor(positions, device=device, dtype=torch.float32).flatten()
        true_count = int(positions.numel())
        if true_count <= 0:
            return torch.zeros(input_dim, base_len, device=device, dtype=dtype)

        valid_len = max(float(valid_len), 1.0)
        true_positions = positions[: min(true_count, base_len)]
        real_count = int(true_positions.numel())
        expected_gap = max(valid_len / max(true_count, 1), 1e-6)

        left_gap = torch.full((real_count,), expected_gap, device=device, dtype=torch.float32)
        right_gap = torch.full((real_count,), expected_gap, device=device, dtype=torch.float32)
        if real_count > 1:
            diffs = (true_positions[1:] - true_positions[:-1]).clamp(min=1e-6)
            left_gap[1:] = diffs
            right_gap[:-1] = diffs
        local_gap = 0.5 * (left_gap + right_gap)
        support = (expected_gap / local_gap.clamp(min=1e-6)).clamp(max=1.0)

        features = [
            (true_positions / valid_len).clamp(min=0.0, max=1.0),
            torch.ones(real_count, device=device, dtype=torch.float32),
            torch.log((left_gap / expected_gap).clamp(min=1e-6)),
            torch.log((right_gap / expected_gap).clamp(min=1e-6)),
            support,
        ]
        sample = torch.stack(features[:input_dim], dim=0)
        if input_dim > sample.shape[0]:
            pad = torch.zeros(input_dim - sample.shape[0], real_count, device=device, dtype=torch.float32)
            sample = torch.cat([sample, pad], dim=0)

        if real_count < base_len:
            pad = torch.zeros(input_dim, base_len - real_count, device=device, dtype=torch.float32)
            sample = torch.cat([sample, pad], dim=1)
        elif real_count > base_len:
            sample = sample[:, :base_len]
        return sample.to(dtype=dtype)

    def _build_geometry_levels(self, metas, mask_list, dtype, device, input_dim, enabled):
        if not enabled or metas is None or len(metas) == 0:
            return None
        batch_size = mask_list[0].shape[0]
        if len(metas) != batch_size:
            return None

        base_len = int(mask_list[0].shape[-1])
        base_features = torch.stack(
            [
                self._sample_geometry_features(
                    meta,
                    base_len,
                    device=device,
                    dtype=dtype,
                    input_dim=input_dim,
                )
                for meta in metas
            ],
            dim=0,
        )
        levels = []
        for mask in mask_list:
            target_len = int(mask.shape[-1])
            if target_len == base_len:
                level_features = base_features
            else:
                level_features = F.interpolate(base_features, size=target_len, mode="linear", align_corners=False)
            level_features = level_features * mask[:, None, :].to(dtype=dtype)
            levels.append(level_features)
        return levels

    def _build_cls_geometry_levels(self, metas, mask_list, dtype, device):
        return self._build_geometry_levels(
            metas=metas,
            mask_list=mask_list,
            dtype=dtype,
            device=device,
            input_dim=self.cls_geometry_input_dim,
            enabled=self.cls_geometry_enabled,
        )

    def _build_reg_geometry_levels(self, metas, mask_list, dtype, device):
        return self._build_geometry_levels(
            metas=metas,
            mask_list=mask_list,
            dtype=dtype,
            device=device,
            input_dim=self.reg_geometry_input_dim,
            enabled=self.reg_geometry_enabled,
        )

    def _sample_reliability_vector(self, meta, base_len, device, dtype):
        positions = meta.get("irregular_selected_positions", None) if meta is not None else None
        valid_len = meta.get("irregular_selected_valid_len", None) if meta is not None else None
        if positions is None or valid_len is None:
            return torch.ones(base_len, device=device, dtype=dtype)

        positions = torch.as_tensor(positions, device=device, dtype=torch.float32).flatten()
        true_count = int(positions.numel())
        if true_count <= 0:
            return torch.ones(base_len, device=device, dtype=dtype)

        valid_len_tensor = torch.as_tensor(valid_len, device=device, dtype=torch.float32).flatten()
        valid_len_value = float(valid_len_tensor[0].item()) if valid_len_tensor.numel() > 0 else 1.0
        valid_len_value = max(valid_len_value, 1.0)
        true_positions = positions[: min(true_count, base_len)]
        real_count = int(true_positions.numel())
        expected_gap = max(valid_len_value / max(true_count, 1), 1e-6)

        left_gap = torch.full((real_count,), expected_gap, device=device, dtype=torch.float32)
        right_gap = torch.full((real_count,), expected_gap, device=device, dtype=torch.float32)
        if real_count > 1:
            diffs = (true_positions[1:] - true_positions[:-1]).clamp(min=1e-6)
            left_gap[1:] = diffs
            right_gap[:-1] = diffs
        local_gap = 0.5 * (left_gap + right_gap)
        support = (expected_gap / local_gap.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
        reliability = support.pow(self.sample_reliability_gamma).clamp(min=self.sample_reliability_min_weight, max=1.0)

        if real_count < base_len:
            pad = torch.ones(base_len - real_count, device=device, dtype=torch.float32)
            reliability = torch.cat([reliability, pad], dim=0)
        elif real_count > base_len:
            reliability = reliability[:base_len]
        return reliability.to(dtype=dtype)

    def _build_sample_reliability_weights(self, metas, mask_list, dtype, device):
        if not self.sample_reliability_enabled or metas is None or len(metas) == 0:
            return None
        batch_size = mask_list[0].shape[0]
        if len(metas) != batch_size:
            return None

        base_len = int(mask_list[0].shape[-1])
        base_weights = torch.stack(
            [
                self._sample_reliability_vector(
                    meta,
                    base_len,
                    device=device,
                    dtype=dtype,
                )
                for meta in metas
            ],
            dim=0,
        )[:, None, :]

        levels = []
        for mask in mask_list:
            target_len = int(mask.shape[-1])
            if target_len == base_len:
                level_weights = base_weights
            else:
                level_weights = F.interpolate(base_weights, size=target_len, mode="linear", align_corners=False)
            level_weights = level_weights[:, 0, :].clamp(min=self.sample_reliability_min_weight, max=1.0)
            level_weights = torch.where(
                mask,
                level_weights,
                torch.ones_like(level_weights),
            )
            levels.append(level_weights)
        return torch.cat(levels, dim=1).to(dtype=dtype)

    def _apply_sample_reliability_weights(self, target_weights, reliability_weights, valid_mask, pos_mask):
        if reliability_weights is None:
            return target_weights
        if target_weights is None:
            target_weights = torch.ones_like(valid_mask, dtype=reliability_weights.dtype)
        else:
            target_weights = target_weights.to(dtype=reliability_weights.dtype)

        reliability_weights = reliability_weights.to(device=valid_mask.device, dtype=target_weights.dtype)
        if reliability_weights.shape != valid_mask.shape:
            raise ValueError(
                "sample reliability weights shape mismatch: "
                f"weights={tuple(reliability_weights.shape)} valid_mask={tuple(valid_mask.shape)}"
            )
        if target_weights.shape != valid_mask.shape:
            raise ValueError(
                "sample reliability target weights shape mismatch: "
                f"target_weights={tuple(target_weights.shape)} valid_mask={tuple(valid_mask.shape)}"
            )
        reliability_weights = torch.where(valid_mask, reliability_weights, torch.ones_like(reliability_weights))
        if self.sample_reliability_apply_to == "positive":
            return torch.where(pos_mask, target_weights * reliability_weights, target_weights)
        return target_weights * reliability_weights

    def _apply_cls_geometry_residual(self, cls_feat, cls_logits, geometry):
        if self.cls_geometry_residual is None or geometry is None:
            return cls_logits
        geometry = geometry.to(device=cls_feat.device, dtype=cls_feat.dtype)
        geometry_embed = self.cls_geometry_embed(geometry)
        delta = self.cls_geometry_residual(torch.cat([cls_feat, geometry_embed], dim=1))
        if self.cls_geometry_max_delta is not None:
            delta = torch.tanh(delta) * self.cls_geometry_max_delta
        return cls_logits + delta

    def _apply_reg_geometry_residual(self, reg_feat, reg_raw, geometry):
        if self.reg_geometry_residual is None or geometry is None:
            return reg_raw
        geometry = geometry.to(device=reg_feat.device, dtype=reg_feat.dtype)
        geometry_embed = self.reg_geometry_embed(geometry)
        delta = self.reg_geometry_residual(torch.cat([reg_feat, geometry_embed], dim=1))
        if self.reg_geometry_max_delta is not None:
            delta = torch.tanh(delta) * self.reg_geometry_max_delta
        return reg_raw + delta

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, metas=None, **kwargs):
        cls_pred = []
        reg_pred = []
        quality_pred = []
        cls_geometry = self._build_cls_geometry_levels(
            metas=metas,
            mask_list=mask_list,
            dtype=feat_list[0].dtype,
            device=feat_list[0].device,
        )
        reg_geometry = self._build_reg_geometry_levels(
            metas=metas,
            mask_list=mask_list,
            dtype=feat_list[0].dtype,
            device=feat_list[0].device,
        )

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_raw = self._apply_cls_residual(cls_feat, self.cls_head(cls_feat))
            geometry = None if cls_geometry is None else cls_geometry[l]
            cls_raw = self._apply_cls_geometry_residual(cls_feat, cls_raw, geometry)
            cls_pred.append(cls_raw)
            reg_raw = self._apply_reg_residual(reg_feat, self.reg_head(reg_feat))
            geometry = None if reg_geometry is None else reg_geometry[l]
            reg_raw = self._apply_reg_geometry_residual(reg_feat, reg_raw, geometry)
            reg_pred.append(F.relu(self.scale[l](reg_raw)))
            if self.quality_head_enabled:
                quality_pred.append(self.quality_head(reg_feat.detach()))

        points = self.prior_generator(feat_list)

        quality_pred = quality_pred if self.quality_head_enabled else None
        losses = self.losses(
            cls_pred,
            reg_pred,
            mask_list,
            points,
            gt_segments,
            gt_labels,
            quality_pred=quality_pred,
            metas=metas,
        )
        return losses

    def forward_test(self, feat_list, mask_list, metas=None, **kwargs):
        cls_pred = []
        reg_pred = []
        quality_pred = []
        cls_geometry = self._build_cls_geometry_levels(
            metas=metas,
            mask_list=mask_list,
            dtype=feat_list[0].dtype,
            device=feat_list[0].device,
        )
        reg_geometry = self._build_reg_geometry_levels(
            metas=metas,
            mask_list=mask_list,
            dtype=feat_list[0].dtype,
            device=feat_list[0].device,
        )

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_raw = self._apply_cls_residual(cls_feat, self.cls_head(cls_feat))
            geometry = None if cls_geometry is None else cls_geometry[l]
            cls_raw = self._apply_cls_geometry_residual(cls_feat, cls_raw, geometry)
            cls_pred.append(cls_raw)
            reg_raw = self._apply_reg_residual(reg_feat, self.reg_head(reg_feat))
            geometry = None if reg_geometry is None else reg_geometry[l]
            reg_raw = self._apply_reg_geometry_residual(reg_feat, reg_raw, geometry)
            reg_pred.append(F.relu(self.scale[l](reg_raw)))
            if self.quality_head_enabled:
                quality_pred.append(self.quality_head(reg_feat.detach()))

        points = self.prior_generator(feat_list)

        # get refined proposals and scores
        quality_pred = quality_pred if self.quality_head_enabled else None
        proposals, scores = self.get_valid_proposals_scores(
            points, reg_pred, cls_pred, mask_list, quality_pred=quality_pred
        )  # list [T,2]
        return proposals, scores

    def get_refined_proposals(self, points, reg_pred):
        points = torch.cat(points, dim=0)  # [T,4]
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)  # [B,T,2]

        start = points[:, 0][None] - reg_pred[:, :, 0] * points[:, 3][None]
        end = points[:, 0][None] + reg_pred[:, :, 1] * points[:, 3][None]
        proposals = torch.stack((start, end), dim=-1)  # [B,T,2]
        return proposals

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, mask_list, quality_pred=None):
        # apply regression to get refined proposals
        proposals = self.get_refined_proposals(points, reg_pred)  # [B,T,2]
        # proposal scores
        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()  # [B,T,num_classes]
        if quality_pred is None or self.quality_score_alpha <= 0:
            quality_scores = [None] * scores.shape[0]
        else:
            quality_scores = torch.cat(quality_pred, dim=-1).permute(0, 2, 1).sigmoid()  # [B,T,1]

        # mask out invalid, and return a list with batch size
        masks = torch.cat(mask_list, dim=1)  # [B,T]
        new_proposals, new_scores = [], []
        for proposal, score, mask, quality_score in zip(proposals, scores, masks, quality_scores):
            if quality_score is not None:
                quality_score = quality_score.clamp(min=1e-6, max=1.0)
                score = score * quality_score.pow(self.quality_score_alpha)
            new_proposals.append(proposal[mask])  # [T,2]
            new_scores.append(score[mask])  # [T,num_classes]
        return new_proposals, new_scores

    @staticmethod
    def _segment_iou_1d(pred_segments, target_segments, eps=1e-6):
        pred_len = (pred_segments[:, 1] - pred_segments[:, 0]).clamp(min=0.0)
        target_len = (target_segments[:, 1] - target_segments[:, 0]).clamp(min=0.0)
        inter = (
            torch.minimum(pred_segments[:, 1], target_segments[:, 1])
            - torch.maximum(pred_segments[:, 0], target_segments[:, 0])
        ).clamp(min=0.0)
        union = (pred_len + target_len - inter).clamp(min=eps)
        return inter / union

    @staticmethod
    def _pairwise_segment_iou_1d(pred_segments, target_segments, eps=1e-6):
        pred_len = (pred_segments[:, 1] - pred_segments[:, 0]).clamp(min=0.0)
        target_len = (target_segments[:, 1] - target_segments[:, 0]).clamp(min=0.0)
        inter = (
            torch.minimum(pred_segments[:, None, 1], target_segments[None, :, 1])
            - torch.maximum(pred_segments[:, None, 0], target_segments[None, :, 0])
        ).clamp(min=0.0)
        union = (pred_len[:, None] + target_len[None, :] - inter).clamp(min=eps)
        return inter / union

    @torch.no_grad()
    def _max_iou_quality_target(self, all_pred_segments, valid_mask, gt_segments, dtype=None):
        target_dtype = all_pred_segments.dtype if dtype is None else dtype
        quality_target = torch.zeros_like(valid_mask, dtype=target_dtype)
        for batch_idx, gt_segment in enumerate(gt_segments):
            valid = valid_mask[batch_idx]
            if not valid.any() or gt_segment.numel() == 0:
                continue
            pred = all_pred_segments[batch_idx, valid].detach()
            target = gt_segment.detach().to(device=pred.device, dtype=pred.dtype)
            quality_target[batch_idx, valid] = self._pairwise_segment_iou_1d(pred, target).max(dim=1).values.to(
                dtype=target_dtype
            )
        return quality_target

    def _quality_loss(
        self,
        quality_pred,
        valid_mask,
        pos_mask,
        pred_segments,
        target_segments,
        all_pred_segments=None,
        gt_segments=None,
    ):
        quality_pred = torch.cat(quality_pred, dim=-1).squeeze(1)
        quality_pred = quality_pred.float()
        quality_logits = quality_pred[valid_mask]
        if quality_logits.numel() == 0:
            return quality_pred.sum() * 0

        if self.quality_target_mode == "assigned_iou":
            quality_target = torch.zeros_like(valid_mask, dtype=quality_pred.dtype)
            if pred_segments.numel() > 0:
                quality_target[pos_mask] = self._segment_iou_1d(pred_segments.detach(), target_segments.detach())
        elif self.quality_target_mode in ("max_iou", "positive_max_iou"):
            if all_pred_segments is None or gt_segments is None:
                raise ValueError("max_iou quality target requires all_pred_segments and gt_segments")
            max_iou_target = self._max_iou_quality_target(
                all_pred_segments.detach(),
                valid_mask,
                gt_segments,
                dtype=quality_pred.dtype,
            )
            if self.quality_target_mode == "positive_max_iou":
                quality_target = torch.zeros_like(valid_mask, dtype=quality_pred.dtype)
                quality_target[pos_mask] = max_iou_target[pos_mask]
            else:
                quality_target = max_iou_target
        else:
            raise ValueError(f"Unsupported quality target mode: {self.quality_target_mode}")

        quality_weight = torch.zeros_like(valid_mask, dtype=quality_pred.dtype)
        quality_weight[valid_mask] = self.quality_negative_weight
        if pos_mask.any():
            quality_weight[pos_mask] = self.quality_positive_weight

        quality_target = quality_target[valid_mask]
        quality_weight = quality_weight[valid_mask]
        quality_loss = F.binary_cross_entropy_with_logits(quality_logits, quality_target, reduction="none")
        quality_loss = (quality_loss * quality_weight).sum()

        if self.quality_loss_normalizer == "weighted":
            denom = quality_weight.sum().clamp(min=1.0)
        elif self.quality_loss_normalizer == "positive":
            denom = (pos_mask.to(dtype=quality_pred.dtype) * self.quality_positive_weight).sum().clamp(min=1.0)
        elif self.quality_loss_normalizer == "valid":
            denom = valid_mask.sum().clamp(min=1).to(dtype=quality_loss.dtype)
        else:
            raise ValueError(f"Unsupported quality loss normalizer: {self.quality_loss_normalizer}")
        quality_loss /= denom.to(dtype=quality_loss.dtype)
        return quality_loss

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels, quality_pred=None, metas=None):
        if self.assigner is None:
            gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)
            target_weights = None
            self._last_assigner_stats = []
        else:
            gt_cls, gt_reg, target_weights = self.prepare_targets_with_assigner(
                points, mask_list, cls_pred, reg_pred, gt_segments, gt_labels
            )

        # positive mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        if target_weights is not None:
            target_weights = torch.stack(target_weights)
        reliability_weights = self._build_sample_reliability_weights(
            metas=metas,
            mask_list=mask_list,
            dtype=torch.float32,
            device=valid_mask.device,
        )
        target_weights = self._apply_sample_reliability_weights(
            target_weights=target_weights,
            reliability_weights=reliability_weights,
            valid_mask=valid_mask,
            pos_mask=pos_mask,
        )
        if target_weights is not None:
            num_pos = (pos_mask.float() * target_weights).sum().item()
        else:
            num_pos = pos_mask.sum().item()

        self._update_assignment_diag(points, valid_mask, gt_cls, pos_mask, target_weights, gt_segments)

        # maintain an EMA of foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1. classification loss
        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]

        # optional label smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        if target_weights is None:
            cls_loss = self.cls_loss(cls_pred, gt_target, reduction="sum")
        else:
            valid_cls_weights = target_weights.to(dtype=torch.float32)
            valid_cls_weights = valid_cls_weights[valid_mask]
            cls_loss = self.cls_loss(cls_pred, gt_target, reduction="none")
            cls_loss = (cls_loss * valid_cls_weights[:, None]).sum()
        cls_loss /= loss_normalizer

        # 2. regression using IoU/GIoU/DIOU loss (defined on positive samples)
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)  # [B,2,T]
        all_pred_segments = self.get_refined_proposals(points, reg_pred)
        all_target_segments = self.get_refined_proposals(points, gt_reg)
        pred_segments = all_pred_segments[pos_mask]
        target_segments = all_target_segments[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            # giou loss defined on positive samples
            reg_loss_values = self.reg_loss(pred_segments, target_segments, reduction="none")
            self._update_regression_diag(points, pos_mask, pred_segments, target_segments, reg_loss_values)
            if target_weights is None:
                reg_loss = reg_loss_values.sum()
            else:
                pos_weights = target_weights[pos_mask]
                reg_loss = (reg_loss_values * pos_weights).sum()
            reg_loss /= loss_normalizer

        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        losses = {"cls_loss": cls_loss, "reg_loss": reg_loss * loss_weight}
        if self.quality_head_enabled and quality_pred is not None:
            if self.quality_loss_weight > 0:
                quality_loss = self._quality_loss(
                    quality_pred,
                    valid_mask,
                    pos_mask,
                    pred_segments,
                    target_segments,
                    all_pred_segments=all_pred_segments,
                    gt_segments=gt_segments,
                )
                losses["quality_loss"] = quality_loss * self.quality_loss_weight
            elif self.quality_loss_weight <= 0 and self.quality_keep_loss_graph_when_weight_zero:
                quality_zero_loss = sum(pred.float().sum() for pred in quality_pred) * 0
                losses["quality_loss"] = quality_zero_loss
        return losses

    @torch.no_grad()
    def prepare_targets_with_assigner(self, points, mask_list, cls_preds, reg_preds, gt_segments, gt_labels):
        cls_preds = torch.cat([x.permute(0, 2, 1) for x in cls_preds], dim=1)
        reg_preds = torch.cat([x.permute(0, 2, 1) for x in reg_preds], dim=1)
        masks = torch.cat(mask_list, dim=1)
        concat_points = torch.cat(points, dim=0)
        assign_points = concat_points
        if not self.use_regress_range:
            assign_points = concat_points.clone()
            assign_points[:, 1] = 0.0
            assign_points[:, 2] = float("inf")
        num_pts = concat_points.shape[0]
        point_inds = torch.arange(num_pts, device=concat_points.device)
        gt_cls, gt_reg, weights = [], [], []
        batch_assigner_stats = []

        for mask, cls_pred, reg_pred, gt_segment, gt_label in zip(
            masks, cls_preds, reg_preds, gt_segments, gt_labels
        ):
            num_gts = gt_segment.shape[0]

            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                weights.append(concat_points.new_ones((num_pts,), dtype=torch.float32))
                continue

            assign_matrix, min_inds, weight = self.assigner.assign(
                cls_pred, assign_points, reg_pred, gt_segment, gt_label, mask
            )
            if hasattr(self.assigner, "get_last_stats"):
                batch_assigner_stats.append(self.assigner.get_last_stats())

            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = concat_points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - concat_points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = assign_matrix.to(reg_targets.dtype) @ gt_label_one_hot
            cls_targets.clamp_(min=0.0, max=1.0)

            reg_targets = reg_targets[point_inds, min_inds]
            reg_targets /= concat_points[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
            weights.append(weight)

        self._last_assigner_stats = batch_assigner_stats
        return gt_cls, gt_reg, weights

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        concat_points = torch.cat(points, dim=0)
        num_pts = concat_points.shape[0]
        gt_cls, gt_reg = [], []

        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            num_gts = gt_segment.shape[0]

            # corner case where current sample does not have actions
            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                continue

            # compute the lengths of all segments -> F T x N
            lens = gt_segment[:, 1] - gt_segment[:, 0]
            lens = lens[None, :].repeat(num_pts, 1)

            # compute the distance of every point to each segment boundary
            # auto broadcasting for all reg target-> F T x N x2
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = concat_points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - concat_points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                # center of all segments F T x N
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                # center sampling based on stride radius
                # compute the new boundaries:
                # concat_points[:, 3] stores the stride
                t_mins = center_pts - concat_points[:, 3, None] * self.center_sample_radius
                t_maxs = center_pts + concat_points[:, 3, None] * self.center_sample_radius
                # prevent t_mins / maxs from over-running the action boundary
                # left: torch.maximum(t_mins, gt_segs[:, :, 0])
                # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
                # F T x N (distance to the new boundary)
                cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
                # F T x N x 2
                center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
                # F T x N
                inside_gt_seg_mask = center_seg.min(-1)[0] > 0
            else:
                # inside an gt action
                inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

            # limit the regression range for each location
            max_regress_distance = reg_targets.max(-1)[0]
            # F T x N
            if self.use_regress_range:
                inside_regress_range = torch.logical_and(
                    (max_regress_distance >= concat_points[:, 1, None]),
                    (max_regress_distance <= concat_points[:, 2, None]),
                )
            else:
                inside_regress_range = torch.ones_like(inside_gt_seg_mask)

            # if there are still more than one actions for one moment
            # pick the one with the shortest duration (easiest to regress)
            lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
            lens.masked_fill_(inside_regress_range == 0, float("inf"))
            # F T x N -> F T
            min_len, min_len_inds = lens.min(dim=1)

            # corner case: multiple actions with very similar durations (e.g., THUMOS14)
            if self.filter_similar_gt:
                min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float("inf")))
            else:
                min_len_mask = lens < float("inf")
            min_len_mask = min_len_mask.to(reg_targets.dtype)

            # cls_targets: F T x C; reg_targets F T x 2
            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = min_len_mask @ gt_label_one_hot
            # to prevent multiple GT actions with the same label and boundaries
            cls_targets.clamp_(min=0.0, max=1.0)
            # OK to use min_len_inds
            reg_targets = reg_targets[range(num_pts), min_len_inds]
            # normalization based on stride
            reg_targets /= concat_points[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
        return gt_cls, gt_reg
