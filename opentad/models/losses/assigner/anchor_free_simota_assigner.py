# modify from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/bbox/assigners/sim_ota_assigner.py

import torch
from torch.nn import functional as F
from ...builder import LOSSES
from ..focal_loss import sigmoid_focal_loss

INF = float("inf")


@LOSSES.register_module()
class AnchorFreeSimOTAAssigner(object):
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
    """

    def __init__(
        self,
        center_radius=2.5,
        iou_weight=3.0,
        cls_weight=1.0,
        keep_percent=0.65,
        confuse_weight=0.1,
        topk=None,
        min_k=1,
        dynamic_k=None,
        dynamic_k_mode="candidate_count",
        min_candidate_iou=0.0,
        filter_shortest_gt=True,
    ):
        if dynamic_k is not None:
            dynamic_k = dict(dynamic_k)
            dynamic_k.pop("type", None)
            keep_percent = dynamic_k.pop("keep_percent", keep_percent)
            dynamic_k_mode = dynamic_k.pop("mode", dynamic_k_mode)
            min_k = dynamic_k.pop("min_k", min_k)
            min_candidate_iou = dynamic_k.pop("min_candidate_iou", min_candidate_iou)
            if dynamic_k:
                raise ValueError(f"Unsupported dynamic_k options: {sorted(dynamic_k)}")

        if dynamic_k_mode not in {"candidate_count", "iou_sum"}:
            raise ValueError(f"Unsupported dynamic_k_mode: {dynamic_k_mode}")
        if float(min_candidate_iou) < 0.0:
            raise ValueError(f"min_candidate_iou must be non-negative: {min_candidate_iou}")

        self.center_radius = center_radius
        self.iou_weight = iou_weight
        self.cls_weight = cls_weight
        self.keep_percent = keep_percent
        self.confuse_weight = confuse_weight
        self.topk = topk
        self.min_k = min_k
        self.dynamic_k_mode = dynamic_k_mode
        self.min_candidate_iou = min_candidate_iou
        self.filter_shortest_gt = filter_shortest_gt
        self._last_stats = {}

    def get_last_stats(self):
        return dict(self._last_stats)

    def assign(
        self,
        out_scores,
        points,
        out_offsets,
        gt_bboxes,
        gt_labels,
        valid_masks,
        gt_bboxes_ignore=None,
        eps=1e-7,
    ):
        assign_result = self._assign(out_scores, points, out_offsets, gt_bboxes, gt_labels, valid_masks, eps)
        return assign_result

    def bbox_overlaps(self, points, out_offsets, gt_bboxes, eps: float = 1e-8):
        num_bboxes = len(out_offsets)
        num_gt = len(gt_bboxes)

        # gt_segs from (Mx2) to (NxMx2)
        gt_segs = gt_bboxes[None].expand(num_bboxes, num_gt, 2)
        left = points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - points[:, 0, None]
        gt_offset = torch.stack((left, right), dim=-1)
        gt_offset /= points[:, 3, None].unsqueeze(-1)

        # out_offsets from (Nx2) to (NxMx2)
        out_offsets = out_offsets.unsqueeze(1).expand(num_bboxes, num_gt, 2).clone()

        # set invalid positions = 0
        invalid_position = (gt_offset < 0).sum(-1) > 0
        gt_offset[invalid_position] = 0.0
        out_offsets[invalid_position] = 0.0

        num_valid = len(out_offsets)
        iou_cost, pairwise_ious = custom_ctr_diou_loss_1d(out_offsets.reshape(-1, 2), gt_offset.reshape(-1, 2))
        iou_cost = iou_cost.reshape(num_valid, num_gt)
        iou_cost[invalid_position] = INF
        pairwise_ious = pairwise_ious.reshape(num_valid, num_gt)
        pairwise_ious[invalid_position] = 0
        return iou_cost, pairwise_ious

    @torch.no_grad()
    def _assign(self, out_scores, points, out_offsets, gt_bboxes, gt_labels, valid_mask, eps=1e-7):
        num_gt = gt_bboxes.size(0)

        valid_out_ofsets = out_offsets[valid_mask]
        valid_out_scores = out_scores[valid_mask]
        valid_points = points[valid_mask]
        num_valid = valid_out_ofsets.size(0)

        pre_assign_mask = self.within_center(valid_points, gt_bboxes)

        iou_cost, pairwise_ious = self.bbox_overlaps(valid_points, valid_out_ofsets, gt_bboxes)

        gt_onehot_label = F.one_hot(gt_labels.long(), out_scores.shape[-1])
        gt_onehot_label = gt_onehot_label.float().unsqueeze(0).repeat(num_valid, 1, 1)

        valid_out_scores = valid_out_scores.unsqueeze(1).repeat(1, num_gt, 1)
        cls_cost = sigmoid_focal_loss(valid_out_scores, gt_onehot_label).sum(-1)
        cost_matrix = cls_cost * self.cls_weight + iou_cost * self.iou_weight
        valid_cost_matrix_inds = cost_matrix < INF
        valid_cost_matrix_inds = torch.logical_and(pre_assign_mask, valid_cost_matrix_inds)

        matching_matrix, min_inds, weight = self.dynamic_k_matching(
            cost_matrix, num_gt, valid_mask, valid_cost_matrix_inds, pairwise_ious
        )

        # convert to original mask
        original_matrix = torch.zeros((len(valid_mask), num_gt)).to(valid_mask.device)
        original_matrix[valid_mask] = matching_matrix

        original_min_inds = torch.zeros((len(valid_mask),), dtype=torch.long).to(min_inds.device)

        original_min_inds[valid_mask] = min_inds

        original_weight = torch.zeros((len(valid_mask),), dtype=torch.float32).to(weight.device)

        original_weight[valid_mask] = weight

        return original_matrix, original_min_inds, original_weight

    def dynamic_k_matching(self, cost, num_gt, valid_mask, valid_cost_matrix_inds, pairwise_ious):
        positive_pos = valid_cost_matrix_inds
        raw_candidate_counts = positive_pos.sum(0)
        if float(self.min_candidate_iou) > 0.0:
            iou_gate = pairwise_ious >= float(self.min_candidate_iou)
            positive_pos = torch.logical_and(positive_pos, iou_gate)
        matching_matrix = cost.new_zeros(positive_pos.shape)
        pre_assign_weight = cost.new_ones((len(cost),))

        pre_assign_weight[positive_pos.sum(1) > 0] = self.confuse_weight

        candidate_counts = positive_pos.sum(0)
        if self.dynamic_k_mode == "iou_sum":
            dynamic_ks = candidate_counts.new_zeros(candidate_counts.shape)
            for gt_idx in range(num_gt):
                candidate_count = int(candidate_counts[gt_idx].item())
                if candidate_count <= 0:
                    continue
                topk = candidate_count if self.topk is None else min(candidate_count, int(self.topk))
                valid_ious = pairwise_ious[:, gt_idx].masked_fill(~positive_pos[:, gt_idx], 0.0)
                dynamic_ks[gt_idx] = (
                    torch.topk(valid_ious, k=topk, largest=True).values.sum().long().clamp(min=int(self.min_k))
                )
        elif self.dynamic_k_mode == "candidate_count":
            dynamic_ks = candidate_counts
            if self.topk is not None:
                dynamic_ks = dynamic_ks.clamp(max=int(self.topk))
            dynamic_ks = (dynamic_ks.float() * self.keep_percent).long()
            dynamic_ks = torch.where(candidate_counts > 0, dynamic_ks.clamp(min=int(self.min_k)), dynamic_ks)
        else:
            raise ValueError(f"Unsupported dynamic_k_mode: {self.dynamic_k_mode}")
        dynamic_ks = torch.minimum(dynamic_ks, candidate_counts)

        for gt_idx in range(num_gt):
            k = int(dynamic_ks[gt_idx].item())
            if k <= 0:
                continue
            masked_cost = cost[:, gt_idx].masked_fill(~positive_pos[:, gt_idx], INF)
            _, pos_idx = torch.topk(masked_cost, k=k, largest=False)
            matching_matrix[pos_idx, gt_idx] = 1.0
            pre_assign_weight[pos_idx] = 1.0

        # Resolve multi-match: each point assigned to at most one GT
        multi_match = matching_matrix.sum(1) > 1
        if multi_match.any():
            multi_rows = multi_match.nonzero(as_tuple=True)[0]
            selected_cost = cost[multi_rows].masked_fill(matching_matrix[multi_rows] == 0, INF)
            best_gt = selected_cost.argmin(dim=1)
            matching_matrix[multi_rows] = 0
            matching_matrix[multi_rows, best_gt] = 1.0

        masked_cost = cost.masked_fill(matching_matrix == 0, INF)
        _, min_inds = masked_cost.min(1)

        matched_counts = matching_matrix.sum(0).to(torch.long)
        self._last_stats = {
            "dynamic_k_mode": self.dynamic_k_mode,
            "min_candidate_iou": float(self.min_candidate_iou),
            "raw_candidate_counts": [int(x) for x in raw_candidate_counts.detach().cpu().tolist()],
            "candidate_counts": [int(x) for x in candidate_counts.detach().cpu().tolist()],
            "dynamic_ks": [int(x) for x in dynamic_ks.detach().cpu().tolist()],
            "matched_counts": [int(x) for x in matched_counts.detach().cpu().tolist()],
            "candidate_point_count": int((positive_pos.sum(1) > 0).sum().item()),
            "confuse_point_count": int((pre_assign_weight < 1.0).sum().item()),
            "matched_point_count": int((matching_matrix.sum(1) > 0).sum().item()),
        }

        return matching_matrix, min_inds, pre_assign_weight

    @torch.no_grad()
    def within_center(self, concat_points, gt_segment):
        # concat_points : F T x 4 (t, regression range, stride)
        # gt_segment : N (#Events) x 2
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # compute the lengths of all segments -> F T x N
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)

        if num_gts == 0:
            return lens * 0.0

        # compute the distance of every point to each segment boundary
        # auto broadcasting for all reg target-> F T x N x2
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        reg_targets = torch.stack((left, right), dim=-1)

        # center of all segments F T x N
        center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
        # center sampling based on stride radius
        # compute the new boundaries:
        # concat_points[:, 3] stores the stride
        t_mins = center_pts - concat_points[:, 3, None] * self.center_radius
        t_maxs = center_pts + concat_points[:, 3, None] * self.center_radius
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

        # limit the regression range for each location
        max_regress_distance = reg_targets.max(-1)[0]
        # F T x N
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]), (max_regress_distance <= concat_points[:, 2, None])
        )

        # lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        # lens.masked_fill_(inside_regress_range == 0, float('inf'))
        # pre_assign_mask = lens < INF

        lens.masked_fill_(inside_gt_seg_mask == 0, INF)
        lens.masked_fill_(inside_regress_range == 0, INF)
        # F T x N -> F T
        min_len, min_len_inds = lens.min(dim=1)

        if self.filter_shortest_gt:
            # corner case: multiple actions with very similar durations (e.g., THUMOS14)
            pre_assign_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < INF))
        else:
            pre_assign_mask = lens < INF

        return pre_assign_mask.to(reg_targets.dtype)


def custom_ctr_diou_loss_1d(input_offsets: torch.Tensor, target_offsets: torch.Tensor, eps: float = 1e-8):
    input_offsets = input_offsets.float()
    target_offsets = target_offsets.float()

    lp, rp = input_offsets[:, 0], input_offsets[:, 1]
    lg, rg = target_offsets[:, 0], target_offsets[:, 1]

    # intersection key points
    lkis = torch.min(lp, lg)
    rkis = torch.min(rp, rg)

    # iou
    intsctk = rkis + lkis
    unionk = (lp + rp) + (lg + rg) - intsctk
    iouk = intsctk / unionk.clamp(min=eps)

    # smallest enclosing box
    lc = torch.max(lp, lg)
    rc = torch.max(rp, rg)
    len_c = lc + rc

    # offset between centers
    rho = 0.5 * (rp - lp - rg + lg)

    # diou
    loss = 1.0 - iouk + torch.square(rho / len_c.clamp(min=eps))

    return loss, iouk
