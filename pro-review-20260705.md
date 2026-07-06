---
date: 2026-07-05
source: Pro review fallback response
status: recorded
scope: Sparse-head route implementation and experiment-attribution review
boundary: Pro reported that the GitHub repository URL returned 404, so this review was based on the fallback bundle rather than a live repository checkout.
---

# Pro Review 2026-07-05

## Access Boundary

According to a document from 2026-07-05: **无法访问仓库**。你给的 GitHub URL 打开返回 404，因此我没有完成 live GitHub 仓库的逐行审查；下面结论基于我能读取到的同日 fallback 审查包 `opentad_sparsehead_pro_review_bundle.md`、其中的 diff/config/test/root-cause 记录，以及 GitHub 404 结果。这个边界很重要：**我可以给出代码与实验路线判断，但不能把它说成“已审查私有仓库当前 HEAD”。**

## 0. 总体 Verdict

**当前 sparse-head 路线不是“完全正确”。更准确的判断是：实现部分有可用组件，但当前实验归因不成立，且存在高概率的监督/尺度/坐标语义错误。**

最强证据是：训练没有崩，HeadV3 fixed 完整训练到 `Average-mAP=40.20`，没有 Traceback/OOM/non-finite loss；`nogeometry=39.32`，`reggate=38.32`；而 `bridge_hard_linear_openrange` 提升到 `42.44`，但仍远低于 dense reference `51.59`，高 IoU 也仍弱。assignment audit 显示 current bridge hard 的 positive 几乎只在低层 `[83, 22, 0, 0, 0, 0]`，openrange 后变成 `[184, 151, 104, 56, 27, 18]`。这说明主问题不是训练崩溃，而是 **assignment / regression range / scale / GT axis / regression encoding 的监督合同不一致**。

## 1. 明确 bug / 逻辑错误检查

### 1.1 IrregularPointGeneratorV2 的 point scale 语义

当前 `IrregularPointGeneratorV2` 使用：

```python
point_scale = cell_left + cell_right
```

并在 `range_mode="hard"` 下用：

```python
reg_min = reg_range[0] * point_scale
reg_max = reg_range[1] * point_scale
```

同时 point tensor 保留 `[center, reg_min, reg_max, cell_left, cell_right]`，测试也明确确认：`cell_left=2, cell_right=2, regression_range=(1,2)` 会得到 `reg_min=4, reg_max=8`。

这个实现**机械上自洽**，但有严重语义风险：如果 config 里的 `regression_range=[(0,4),(4,8),(8,16),...]` 是从 dense ActionFormer 继承来的“绝对时长/level duration bin”语义，那么再乘 `cell_left+cell_right` 就会把范围放大，导致 assignment 与 dense control 不等价。bundle 也明确记录 `temporal_grid["level_scale"]` 是 `0.5 * (cell_left + cell_right)`，而 V2 使用的是 full span，这两个定义不能混用。

**结论：**
`cell_left + cell_right` 作为 symmetric-linear regression denominator 可以作为一个设计选择，但把 dense-style regression range 也乘以它，是当前最可疑的逻辑错误之一。它更像是 **config/语义 bug**，而不是简单 Python bug。

### 1.2 regression range 是否错误乘以 cell_left + cell_right，压没高层 positive？

是，高概率成立。

证据非常直接：current bridge hard 的 per-level positive 是 `[83, 22, 0, 0, 0, 0]`，高层完全没有 positive；openrange 把所有层的 range 改为 `(0,10000)` 后，positive 恢复为 `[184, 151, 104, 56, 27, 18]`，并且 mAP 从 HeadV3 fixed `40.20` 提升到 `42.44`。这强烈说明原始 range gate 正在错误地过滤掉大量高层候选。

但注意：`corrected/absolute raw` 的 audit 是 `[3, 7, 18, 26, 7, 0]`，不是 openrange 那种全层大量恢复。这说明问题不是“只要把 range 打开就行”，而是需要同时审计 **absolute range、center radius、GT length bucket、point scale denominator、decode scale**。

### 1.3 openrange 为什么提升到 42.44？它证明什么，不能证明什么？

`openrange` 的配置把所有 level 的 `regression_range` 设成 `(0,10000)`，仍然是 `range_mode="hard"`。

它证明三件事：

第一，训练流程没有崩；HeadV3/Bridge 的低分不是 OOM、NaN 或预测数异常造成的。第二，原始 regression range / scale 过滤确实在压制 positive，尤其是高层。第三，给高层恢复监督能带来 mAP 改善，`42.44 > 40.20`。

但它**不能证明**当前 head 已经 dense-like，也不能作为最终修复。因为 openrange 实质上移除了 ActionFormer 的重要 scale prior，让所有 GT 都可能跨层产生 positives；这会造成监督扩散、level specialization 变差，并且很可能解释为什么它虽然提升到 42.44，却仍远低于 dense 51.59，且 `mAP@0.7=15.20` 仍很弱。

**结论：**
openrange 是高价值诊断，不是最终方案。

### 1.4 `range_mode="absolute"` 是否正确？是否应优先于 openrange？

从当前代码和测试看，`absolute` 的机械实现是正确的：它不再乘 `cell_left + cell_right`，而是直接把 `reg_min/reg_max` 填成 config 中的绝对数值；测试确认 `(8,16)` 在不同 cell size 下都保持 `8/16`。

所以如果 config 的 `(0,4),(4,8),(8,16)...` 语义本来就是 dense ActionFormer 的 absolute duration bin，那么 **absrange 应该优先于 openrange**。不过当前 `corrected/absolute raw positives=[3,7,18,26,7,0]` 仍然不够解释全部问题，所以 absrange 不能单独宣布修复；它必须和 center radius scale、GT axis、regression denominator 一起 audit。

### 1.5 bridge hard linear 是否真的 dense-like？

**不是完全 dense-like，只是“dense-like 的尝试”。**

它确实配置为 `assignment_mode="hard"`、`regression_mode="symmetric_linear"`；相关测试也覆盖了 hard/soft/log/openrange/absrange configs。

但它和 dense ActionFormer 至少仍有这些差异：

1. center sampling 半径使用了 irregular `point_scale`，代码片段显示类似 `radius = point_scale * center_sample_radius`。如果 `point_scale=cell_left+cell_right`，半径语义就不同于 dense stride / level stride。
2. regression range 在 current hard 下按 full cell span 缩放，而 dense control 的 range 语义通常是 level/duration bin，不应被不规则 cell span 再次放大。
3. regression decode/encode 使用 left/right cell scale 或 full-span denominator，而 dense 是更接近 `reg * stride` 的线性距离。bundle 明确把 dense linear 与 V2/V3 `log1p/expm1` 或 scale-based encode/decode 列为未隔离变量。
4. dense control 使用 selected-axis GT，而 HeadV3/Bridge 使用 native-axis GT；这使 dense 51.59 与 sparse 40.20 不是纯 head-only 对照。
5. 还需要确认 hard assignment 是否严格复刻 dense 的 shortest-GT conflict resolution、loss normalizer、valid mask、label target、center fallback 行为。现有证据不足以证明完全一致。

**结论：**
bridge hard linear 可以作为 isolation scaffold，但不能声称“已经 dense-like”。当前首跑更差并不反驳 dense-like 思路，而是说明实现仍混入了 range/scale/axis 不一致。

### 1.6 HeadV2/V3 soft assignment 是否造成过多弱正样本、跨 GT 冲突、监督扩散？

高概率是。

bundle 的 root-cause notes 明确记录：V2/V3 使用 per-GT top-k soft assignment、soft classification target、soft regression weight、asymmetric log regression；早期诊断出现 `440` 到 `1016` positive points，`pos_mass/count≈0.71`，并且存在很多 weak cross-GT positives。这和 dense ActionFormer “较少、清晰、每点 shortest-GT 冲突解决”的监督风格差异很大。

`reggate` 把 regression range gate 加回 soft path 以后只有 `38.32`，比 fixed `40.20` 更差。这反驳了“只缺 regression range gate”这个窄假设，但不反驳“soft assignment 本身扩散监督”这个主假设。

### 1.7 dense control 与 HeadV3/bridge 的 GT 坐标轴差异是否污染结论？

是，污染强归因结论。

证据显示 HeadV3 和 bridge 的 `LoadFrames` 里 `remap_gt_to_selected_axis=False`，而 dense_control 和 dense_head_grid 是 `True`。测试也专门断言了这个差异。

因此 `dense 51.59 -> HeadV3 40.20` 不能被写成“只换 head 导致 -11.39 mAP”。更严谨说法是：**selected-axis dense ActionFormer reference 与 native-axis sparse-head route 之间存在约 11 mAP 差距**。这可以作为路线目标差距，但不能作为单变量 head 归因。

### 1.8 GridAware vs DensePassthrough cross-over 是否只是 route sanity？

是。它只能作为 route sanity，不是 projection/neck 强归因。

cross-over configs 覆盖了 `dense_head_gridaware`、`bridge_hard_linear_densepass`、`headv3_densepass`，但这些组合同时改变 projection/neck/head/point generator/GT axis/temporal-grid path。bundle 也明确说 cross-over 应被视为 route sanity，而不是 projection/neck attribution。

## 2. 当前核心问题排序

我会这样排序：

**第 1 位：regression range / scale / center-radius mismatch。**
证据最强。current bridge hard 高层 positive 被压没；openrange 恢复所有层 positive 并提升 mAP；IrregularPointGeneratorV2 的 full-span scale 与 dense-style range 语义冲突。

**第 2 位：assignment diffusion / soft target 扩散。**
HeadV2/V3 soft assignment 与 dense ActionFormer 的 hard shortest-GT assignment 差异过大，positive 过多且弱；`reggate=38.32` 说明只加 range gate 不够。

**第 3 位：GT axis mismatch。**
它未必是 mAP 下降的唯一根因，但它直接破坏 dense 51.59 vs HeadV3/bridge 的强归因。必须做 matched-axis 诊断，否则论文/表格结论不成立。

**第 4 位：regression encoding mismatch。**
dense 是 linear distance 风格；V2/V3/bridge 的 log 或 scale-based encoding 可能让边界校准过保守，尤其伤 @0.6/@0.7。需要 hard_linear vs hard_log 在相同 range/axis/radius 下隔离。

**第 5 位：fixed center radius 不适配 irregular grid。**
它其实是第 1 位的一部分。当前若用 full span 做半径，center sampling 会偏离 dense stride 语义。应该把 `radius_base` 单独可配，至少比较 `half_cell_span`、`min_side`、`dense_level_scale`。

**第 6 位：boundary aux / geometry / predictor kernel / modulation。**
`nogeometry=39.32` 低于 fixed `40.20`，说明 geometry 不是一阶原因；boundary aux 从旧 42.41 到当前 40.20 可能贡献约 2 mAP，但解释不了 dense gap；predictor kernel 这类因素应延后。

## 3. 下一步最小实验计划：先 assignment audit，不要直接长训

### 3.1 必跑的同 batch audit

在同一批 train batch、同一 seed、同一 dataloader sample ids 下比较：

1. `bridge_hard_linear_n16r4` 当前 hard range。
2. `bridge_hard_linear_openrange_n16r4`。
3. `bridge_hard_linear_absrange_n16r4`。
4. 新增 `bridge_hard_linear_absrange_radiuslevel_n16r4`：absolute range + corrected center-radius base。
5. 可选 dense selected-axis control 与 dense native-axis diagnostic，用来拆 GT axis 影响。

至少输出这些字段：

```text
sample_id
video_name / window_start
num_gt
gt_length_bucket
gt_label
gt_covered_any
gt_covered_level_bitmap
gt_num_assigned_points
per_level_pos_count
per_level_candidate_count_before_range
per_level_candidate_count_after_range
range_fail_count_by_level
center_fail_count_by_level
inside_gt_count_by_level
assigned_gt_idx_hist
multi_gt_conflict_count_before_shortest
shortest_gt_resolved_count
reg_target_left_p50/p90/max
reg_target_right_p50/p90/max
encoded_reg_p50/p90/max
decode_reconstruction_max_error
radius_base_p50/p90 by level
valid_mask_true_count by level
```

GT length buckets 建议：

```text
[0,4), [4,8), [8,16), [16,32), [32,64), [64,128), [128,inf)
```

### 3.2 什么时候才值得启动长训？

只有满足这些 audit gates 才值得长训：

```text
G1: 高层 positive 不再系统性为 0，除非同 batch 确实没有长 GT。
G2: GT coverage 接近 dense matched-axis control；不能只看 32/33，要看每个 length bucket。
G3: per-level positive 分布与 GT length bucket 合理对应，不能像 current hard 那样 [83,22,0,0,0,0]。
G4: positive 数不能像 openrange 那样无约束膨胀；若超过 dense control 2-3x，需要解释。
G5: encoded -> decoded target reconstruction error 接近 0；训练和 inference decode 必须同一尺度。
G6: native-axis 与 selected-axis 的差异被单独记录，不允许再把 51.59 vs 40.20 作为 head-only 结论。
```

### 3.3 长训优先级

如果 audit 通过，长训优先级如下：

1. **P0：`bridge_hard_linear_absrange_radiuslevel`**
   这是最小修复版：hard assignment + absolute range + corrected radius + symmetric linear。它最能回答“dense-like sparse/native head 是否能恢复”。

2. **P1：`bridge_hard_log_absrange_radiuslevel`**
   只改 regression encoding，隔离 log1p/expm1 是否伤 high-IoU。

3. **P2：`bridge_soft_topk1_binary_linear_absrange_radiuslevel`**
   在 range/radius 修正后再测 soft-vs-hard，避免把 range bug 误判成 soft bug。

4. **P3：matched-axis diagnostic**
   dense native-axis 或 bridge selected-axis，只用于 attribution，不用于 deployable sparse claim。

5. **P4：geometry / boundary aux / predictor kernel**
   等监督合同修复后再做。

## 4. 关键实现建议与代码片段

### 4.1 不要再让 dense-style range 隐式乘 full cell span

文件：`opentad/models/dense_heads/prior_generator/irregular_point_generator.py`

建议保留已有 `absolute`，但把 dense-like config 全部切到 `range_mode="absolute"`。同时给 `hard` 改名或加注释，避免误用。

```python
# opentad/models/dense_heads/prior_generator/irregular_point_generator.py

class IrregularPointGeneratorV2(IrregularPointGenerator):
    def __call__(self, feat_list, temporal_grid_list):
        pts_list = []
        for feat, temporal_grid, reg_range in zip(
            feat_list, temporal_grid_list, self.regression_range
        ):
            center = temporal_grid["center"]
            scale_left = temporal_grid["cell_left"].to(center.dtype).clamp_min(1e-6)
            scale_right = temporal_grid["cell_right"].to(center.dtype).clamp_min(1e-6)
            full_cell_span = (scale_left + scale_right).clamp_min(1e-6)

            reg_range = torch.as_tensor(reg_range, dtype=center.dtype, device=center.device)

            if self.range_mode == "absolute":
                # Dense ActionFormer-style duration bins: do NOT scale by local cell span.
                reg_min = torch.full_like(center, float(reg_range[0]))
                reg_max = torch.full_like(center, float(reg_range[1]))
            elif self.range_mode == "hard":
                # Scaled-by-cell-span mode. Do not use for dense-equivalent bins.
                reg_min = reg_range[0] * full_cell_span
                reg_max = reg_range[1] * full_cell_span
            elif self.range_mode == "overlap_band":
                center_scale = 0.5 * (reg_range[0] + reg_range[1]) * full_cell_span
                half_width = 0.5 * (reg_range[1] - reg_range[0]) * full_cell_span
                overlap_pad = self.overlap_factor * full_cell_span
                reg_min = (center_scale - half_width - overlap_pad).clamp_min(0.0)
                reg_max = center_scale + half_width + overlap_pad
            else:
                raise ValueError(f"Unsupported range_mode: {self.range_mode}")

            # Keep existing 5-field layout.
            points = torch.stack(
                [center, reg_min, reg_max, scale_left, scale_right],
                dim=-1,
            )
            pts_list.append(points)
        return pts_list
```

新增 config：

```python
# configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4.py

_base_ = ["./input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py"]

model = dict(
    rpn_head=dict(
        assignment_mode="hard",
        regression_mode="symmetric_linear",
        center_radius_scale="half_cell_span",
        reg_denom_mode="half_cell_span",
        prior_generator=dict(
            range_mode="absolute",
            regression_range=[
                (0, 4),
                (4, 8),
                (8, 16),
                (16, 32),
                (32, 64),
                (64, 10000),
            ],
        ),
    )
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4"
```

### 4.2 把 center-radius scale 和 regression denominator 拆开

文件：`opentad/models/dense_heads/irregular_actionformer_bridge_head.py`

当前 hard path 片段显示 center radius 使用 `point_scale`，这很可能就是 full cell span。建议新增两个显式选项：

```python
# opentad/models/dense_heads/irregular_actionformer_bridge_head.py

def _scale_base(self, left_scale, right_scale, point_scale, mode: str):
    if mode == "full_cell_span":
        return point_scale.clamp_min(self.reg_denom_floor)
    if mode == "half_cell_span":
        return (0.5 * point_scale).clamp_min(self.reg_denom_floor)
    if mode == "min_side":
        return torch.minimum(left_scale, right_scale).clamp_min(self.reg_denom_floor)
    if mode == "left_right_mean":
        return (0.5 * (left_scale + right_scale)).clamp_min(self.reg_denom_floor)
    raise ValueError(f"Unsupported scale mode: {mode}")
```

在 `__init__` 里：

```python
self.center_radius_scale = center_radius_scale
self.reg_denom_mode = reg_denom_mode
```

在 hard assignment center sampling 里：

```python
radius_base = self._scale_base(
    left_scale=left_scale,
    right_scale=right_scale,
    point_scale=point_scale,
    mode=self.center_radius_scale,
)
radius = radius_base[:, None] * self.center_sample_radius
```

在线性 regression encode/decode 里：

```python
denom = self._scale_base(
    left_scale=left_scale,
    right_scale=right_scale,
    point_scale=point_scale,
    mode=self.reg_denom_mode,
).clamp_min(self.reg_denom_floor)
```

关键原则：**训练 encode 和 inference decode 必须使用同一个 denom，decode 不能用 GT length。**

### 4.3 dense-like hard assignment 应该显式实现 shortest-GT conflict

文件：`opentad/models/dense_heads/irregular_actionformer_bridge_head.py`

建议 hard path 保持 GT 只在 target assignment 中使用；inference decode 不使用 GT。

```python
@torch.no_grad()
def _prepare_targets_hard_dense_like(self, points, gt_segments, gt_labels):
    point_list = self._points_per_sample(points, len(gt_segments))

    all_gt_cls = []
    all_gt_reg = []
    all_reg_weight = []
    all_debug = []

    for point, gt_segment, gt_label in zip(point_list, gt_segments, gt_labels):
        num_pts = point.shape[0]
        center_t, reg_min, reg_max, left_scale, right_scale, point_scale = self._point_fields(point)

        gt_cls = point.new_zeros((num_pts, self.num_classes))
        gt_reg = point.new_zeros((num_pts, 2))
        reg_weight = point.new_zeros((num_pts,))
        assigned_gt = torch.full((num_pts,), -1, device=point.device, dtype=torch.long)

        if gt_segment.numel() == 0:
            all_gt_cls.append(gt_cls)
            all_gt_reg.append(gt_reg)
            all_reg_weight.append(reg_weight)
            all_debug.append(dict(assigned_gt=assigned_gt))
            continue

        gt_segment = gt_segment.to(device=point.device, dtype=point.dtype)
        gt_label = gt_label.to(device=point.device).long()

        gt_start = gt_segment[:, 0]
        gt_end = gt_segment[:, 1]
        gt_len = (gt_end - gt_start).clamp_min(self.reg_denom_floor)

        left = center_t[:, None] - gt_start[None, :]
        right = gt_end[None, :] - center_t[:, None]
        reg_targets = torch.stack([left, right], dim=-1)

        inside_gt = reg_targets.min(dim=-1).values > 0

        if self.center_sample == "radius":
            gt_center = 0.5 * (gt_start + gt_end)
            radius_base = self._scale_base(
                left_scale, right_scale, point_scale, self.center_radius_scale
            )
            radius = radius_base[:, None] * self.center_sample_radius
            t_min = torch.maximum(gt_center[None, :] - radius, gt_start[None, :])
            t_max = torch.minimum(gt_center[None, :] + radius, gt_end[None, :])
            center_mask = (center_t[:, None] > t_min) & (center_t[:, None] < t_max)
        else:
            center_mask = inside_gt

        max_regress_distance = reg_targets.max(dim=-1).values
        range_mask = (
            (max_regress_distance >= reg_min[:, None])
            & (max_regress_distance <= reg_max[:, None])
        )

        candidate = inside_gt & center_mask & range_mask

        # Dense-style conflict resolution: choose the shortest GT among candidates.
        gt_len_matrix = gt_len[None, :].expand(num_pts, -1).clone()
        gt_len_matrix = gt_len_matrix.masked_fill(~candidate, float("inf"))
        min_len, min_idx = gt_len_matrix.min(dim=1)
        pos = torch.isfinite(min_len)

        if pos.any():
            assigned_gt[pos] = min_idx[pos]
            gt_cls[pos] = F.one_hot(
                gt_label[min_idx[pos]], self.num_classes
            ).to(gt_cls.dtype)

            raw_left = left[torch.arange(num_pts, device=point.device), min_idx]
            raw_right = right[torch.arange(num_pts, device=point.device), min_idx]
            gt_reg[pos] = self._encode_regression_targets(
                raw_left[pos],
                raw_right[pos],
                left_scale[pos],
                right_scale[pos],
                point_scale[pos],
            )
            reg_weight[pos] = 1.0

        all_gt_cls.append(gt_cls)
        all_gt_reg.append(gt_reg)
        all_reg_weight.append(reg_weight)
        all_debug.append(
            dict(
                assigned_gt=assigned_gt,
                candidate_count=candidate.sum(dim=1),
                range_fail=(inside_gt & center_mask & ~range_mask).sum(dim=1),
            )
        )

    return all_gt_cls, all_gt_reg, all_reg_weight, all_debug
```

注意：如果原始 dense `AnchorFreeHead` 没有 “missing_gt fallback”，dense-like bridge 也不应悄悄保留 fallback。否则 audit 会高估 GT coverage。

### 4.4 decode 不允许使用 GT length

建议把 decode 写成纯 point-field 函数：

```python
def _decode_segments(self, points, reg_pred):
    center_t, _, _, left_scale, right_scale, point_scale = self._point_fields(points)

    if self.regression_mode == "symmetric_linear":
        denom = self._scale_base(
            left_scale, right_scale, point_scale, self.reg_denom_mode
        )
        distances = F.relu(reg_pred) * denom[:, None]
        left_dist = distances[:, 0]
        right_dist = distances[:, 1]

    elif self.regression_mode == "asymmetric_log1p":
        pred = F.relu(reg_pred)
        left_dist = torch.expm1(pred[:, 0]) * left_scale
        right_dist = torch.expm1(pred[:, 1]) * right_scale

    else:
        raise ValueError(f"Unsupported regression_mode: {self.regression_mode}")

    return torch.stack(
        [center_t - left_dist, center_t + right_dist],
        dim=-1,
    )
```

这满足你的要求：**inference decode 不用 GT length；训练和推理尺度一致。**

### 4.5 新增 assignment audit 工具

建议新增：

```text
tools/audit_sparse_head_assignment.py
```

核心输出：

```python
def length_bucket(length):
    edges = [0, 4, 8, 16, 32, 64, 128, float("inf")]
    for i in range(len(edges) - 1):
        if edges[i] <= length < edges[i + 1]:
            return f"[{edges[i]},{edges[i+1]})"
    return "unknown"
```

伪代码：

```python
@torch.no_grad()
def summarize_targets(points, masks, gt_segments, gt_labels, target_debug, level_lengths):
    rows = []
    level_offsets = []
    start = 0
    for li, n in enumerate(level_lengths):
        level_offsets.append((li, start, start + n))
        start += n

    for b, dbg in enumerate(target_debug):
        assigned = dbg["assigned_gt"]  # [N], -1 for negative
        for level, lo, hi in level_offsets:
            assigned_l = assigned[lo:hi]
            pos = assigned_l >= 0

            rows.append({
                "batch_idx": b,
                "level": level,
                "pos_count": int(pos.sum().item()),
                "valid_count": int(masks[level][b].sum().item()) if isinstance(masks, list) else None,
            })

        for gi, seg in enumerate(gt_segments[b]):
            covered = bool((assigned == gi).any().item())
            dur = float((seg[1] - seg[0]).item())
            rows.append({
                "batch_idx": b,
                "gt_idx": gi,
                "gt_length": dur,
                "gt_length_bucket": length_bucket(dur),
                "gt_covered": covered,
                "gt_assigned_points": int((assigned == gi).sum().item()),
            })

    return rows
```

命令形态：

```bash
python tools/audit_sparse_head_assignment.py \
  --configs \
    configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py \
    configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py \
    configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py \
    configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_radiuslevel_n16r4.py \
  --split train \
  --num-batches 8 \
  --seed 20260705 \
  --out logs/sparse_head_assignment_audit_20260705
```

## 5. 哪些结果支持或推翻当前判断？

### 支持“range/scale 是主因”的结果

```text
absrange + corrected radius 后：
- 高层 positive 不再为 0；
- GT length bucket coverage 接近 dense matched-axis control；
- per-level positives 不像 openrange 那样无约束膨胀；
- 长训 mAP 明显超过 42.44，尤其 @0.6/@0.7 改善。
```

### 推翻“range/scale 是主因”的结果

```text
同 batch audit 中 current hard、absrange、corrected radius 的 per-level positives 和 GT coverage 几乎相同；
或者 absrange+radius 修正后仍然高层无 positive，但 batch 中确实存在长 GT。
```

那时应转向检查 temporal grid construction、valid mask、多尺度下采样、native-axis segment units 是否错位。

### 支持“soft assignment 扩散是主因”的结果

```text
在相同 absrange/radius/axis 下：
hard_linear 明显优于 soft_topk1_binary_linear 和 HeadV3 soft；
soft 的 positive count / pos_mass / cross-GT conflict 明显更高。
```

### 支持“regression encoding 是 high-IoU 主因”的结果

```text
在相同 hard assignment + absrange + radius 下：
hard_linear 的 @0.6/@0.7 明显高于 hard_log。
```

### 支持“GT axis 污染归因”的结果

```text
dense native-axis control 明显低于 selected-axis dense 51.59；
或 bridge selected-axis diagnostic 明显接近 dense selected-axis。
```

这不一定说明 native-axis 是错的；它说明 dense 51.59 不能作为 head-only attribution。

## 6. 需要补充的完整 diff bundle

因为 live GitHub 仓库不可访问，要完成真正 Pro 级逐行审查，你需要提供一个最小 bundle：

```text
git status --short
git branch --show-current
git rev-parse HEAD
git remote -v
git diff --stat
git diff
```

以及以下文件：

```text
opentad/models/dense_heads/anchor_free_head.py
opentad/models/dense_heads/irregular_actionformer_head_v2.py
opentad/models/dense_heads/irregular_actionformer_head_v3.py
opentad/models/dense_heads/irregular_actionformer_bridge_head.py
opentad/models/dense_heads/prior_generator/irregular_point_generator.py
opentad/models/detectors/irregular_actionformer.py
opentad/models/utils/temporal_grid.py
所有 GridAware / DensePassthrough projection/neck 模块
```

configs：

```text
input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py
input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py
input_random_fixed_50pct_adapter_irregular_bridge_hard_log_n16r4.py
input_random_fixed_50pct_adapter_irregular_bridge_soft_topk1_binary_linear_n16r4.py
input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py
input_random_fixed_50pct_adapter_densehead_gridaware_n16r4.py
如果已有：absrange / densepass / nogeometry / reggate configs
```

记录与工具：

```text
tests/test_adapter_native_dense_headv2_contracts.py
root-cause-notes.md
assignment audit 脚本与 JSON/CSV 输出
完整训练命令、seed、resolved config dump
HeadV3 fixed / openrange / dense control 的 stdout、stderr、metrics json
```

## 7. 最终清晰结论

**当前实现是否完全正确？**
不是。当前路线是“部分可运行、部分可诊断”，但不能宣称 sparse/irregular head 已正确复刻 dense ActionFormer，也不能用当前结果支持强 mAP 结论。

**最可能的 1-3 个根因：**

1. **regression range / point scale / center radius 语义错配**：dense-style range 被 full cell span 缩放，高层 positive 被压没；openrange 恢复 positive 并提升 mAP 是强证据。
2. **HeadV2/V3 soft assignment 监督扩散**：过多弱正样本、跨 GT 冲突、soft cls/reg weight 与 dense hard assignment 不一致。
3. **GT axis mismatch + regression encoding mismatch**：selected-axis dense 51.59 与 native-axis sparse 40.20 不是纯 head-only 对照；linear vs log/scale decode 仍可能是 high-IoU 弱的关键。

**最小修复路径：**

```text
先做同 batch assignment audit
-> 修 absolute range
-> 修 center radius scale / regression denominator
-> hard dense-like assignment 复刻 shortest-GT conflict
-> 验证 encode/decode 训练推理一致且 inference 不用 GT
-> audit 通过后只长训 P0: bridge_hard_linear_absrange_radiuslevel
```

**可以支持路线继续的结果：**

```text
P0 audit 恢复合理 per-level positives 和 GT length-bucket coverage；
P0 长训超过 openrange 42.44，且 @0.6/@0.7 明显改善；
matched-axis diagnostic 解释 dense 51.59 的坐标轴差异；
hard_linear corrected 明显优于 soft/log 变体。
```

**会推翻当前判断或要求换方向的结果：**

```text
absrange + radius 修正后 assignment 仍与 dense matched-axis 大幅不一致；
decode reconstruction 失败或需要 GT length 才能恢复；
P0 长训不超过 openrange，且 @0.6/@0.7 无改善；
dense native-axis control 也崩到相近水平，说明问题可能在 native-axis route / temporal-grid / detector integration，而不只是 sparse head。
```

当前最合理的决策是：**HOLD_LONG_TRAIN；先实现 assignment audit 与 absrange/radius/dense-like hard assignment 修复。**
