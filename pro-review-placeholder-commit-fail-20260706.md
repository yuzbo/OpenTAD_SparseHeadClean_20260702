## A. 总判决

**总判决：FAIL。**
我不能把 `<COMMIT_URL>` 当作可审查的固定 commit；它不是 hash/URL。因此本次只能审查 `codex/sparse-head-clean-20260702` 分支在 2026-07-06 可见的 HEAD。该分支确实存在并在 2026-07-06 更新，但“固定指定 commit 已核验”这一项不成立。([GitHub][1])

| Gate                        | 判决                                                                                   |
| --------------------------- | ------------------------------------------------------------------------------------ |
| 是否允许 remote sync            | **WARN / 允许**，仅限同步、静态检查、Stage 0-2 诊断                                                 |
| 是否允许 Linux preflight        | **允许**，但当前预期会失败，不能作为 green gate                                                      |
| 是否允许 same-batch audit       | **不允许作为官方等价结论**；当前工具还没有真正实现 official-vs-current diff                                 |
| 是否允许 full training / 长训     | **不允许**                                                                              |
| 是否允许 dense-equivalent claim | **不允许**                                                                              |
| 当前 40/42 vs 65 崩溃归因         | 更像 **supervision / scale / axis / postprocess contract 偏移**，不能归因于 sparse sampling 本身 |

官方 ActionFormer 的关键合同很明确：PointGenerator 输出 `[point, reg_min, reg_max, stride]`，点中心是 `t * stride + 0.5 * stride`；assignment 用 stride 作为 center sampling 半径基准；regression range 使用 PointGenerator 中的绝对范围；regression target 编码除以 stride；decode 再乘 stride。官方 THUMOS 配置中 strides 为 `[1,2,4,8,16,32]`，regression ranges 为 `[(0,4),(4,8),(8,16),(16,32),(32,64),(64,10000)]`，`center_sample_radius=1.5`。([GitHub][2])

当前分支的若干“修复声明”与代码不一致：`allow_center_fallback_inside_gt` 未检出；`IrregularPointGeneratorV2` 未检出 `dense_compat_mode="official_actionformer"`；`convert_to_seconds` 工具函数没有 `source_axis` 参数；单类别 post-processing 分支仍未执行 pre-NMS threshold/topk；`tools/check_fail_closed_config.py` 访问为 404；`tools/audit_sparse_head_assignment.py` 未检出 `official_vs_current_assignment_diff` 及声明的 diff 字段。([GitHub][3])

---

## B. 逐文件逐行问题

### 1. `opentad/models/dense_heads/irregular_actionformer_bridge_head.py`

**P0 — hidden fallback 没有 fail-closed。**
函数 `_build_candidate_mask` 中，当某个 GT 没有 candidate point 时，代码计算 `missing_gt = ~candidate_mask.any(dim=0)` 后，直接执行 `candidate_mask[:, missing_gt] = inside_gt_seg[:, missing_gt]`。这就是“center sampling 缺失时退回 inside-GT”的 hidden fallback。它没有看到你声明的 `allow_center_fallback_inside_gt=False` 默认关闭开关。([GitHub][3])

**为什么影响 mAP / 可信度：**
官方 ActionFormer 是严格 center sampling + regression range + shortest-GT conflict resolution；如果当前 route 在缺 center 点时扩大成 inside-GT，会改变 positive set、assigned class、regression target 分布。即便 hard route 当前不走 `_build_candidate_mask`，这个 fallback 仍是代码库内隐藏路径，不能声称 fail-closed 或 official-equivalent。

**与官方差异：**
官方 assignment 不做“缺 center 点时放宽到 inside GT”的补偿；它用 center sampling mask、regression range mask、shortest GT 冲突解决，再除以 stride 编码。([GitHub][4])

**修复代码：**

```python
# __init__
def __init__(..., allow_center_fallback_inside_gt: bool = False, ...):
    ...
    self.allow_center_fallback_inside_gt = bool(allow_center_fallback_inside_gt)

# _build_candidate_mask
missing_gt = ~candidate_mask.any(dim=0)
if missing_gt.any():
    if self.allow_center_fallback_inside_gt:
        candidate_mask[:, missing_gt] = inside_gt_seg[:, missing_gt]
    else:
        if hasattr(self, "_latest_debug_state"):
            self._latest_debug_state["missing_center_gt_count"] = int(missing_gt.sum().item())
        # keep fail-closed: no fallback positives
return candidate_mask
```

---

**P1 — scale contract 部分修复成立，但只修了 full-cell legacy。**
`center_radius_scale`、`reg_denom_mode`、`allow_legacy_full_cell_span` 已经存在；代码对 `full_cell_span` legacy 模式做了显式 opt-in 检查。这个部分是 PASS。([GitHub][3])

但 dense-equivalent 仍要求三件事同时成立：

```text
center_radius_scale = "point_radius"
reg_denom_mode      = "left_right_mean"  # 在 official dense-compatible point 中等于 stride
point fields        = [center, reg_min, reg_max, stride, stride, stride, stride]
```

当前 legacy hard/openrange/absrange config 仍显式使用 `full_cell_span`，不能作为 official-compatible route。

---

### 2. `opentad/models/dense_heads/prior_generator/irregular_point_generator.py`

**P0 — 声明的 `dense_compat_mode="official_actionformer"` 不存在。**
`IrregularPointGeneratorV2` 的构造函数包含 `range_mode`、`decode_scale_mode`、`radius_scale_mode`，默认分别是 `"hard"`、`"cell"`、`"geometric_mean"`；未看到 `dense_compat_mode`。([GitHub][5])

**为什么影响 mAP / 可信度：**
如果没有一个 fail-closed 的 dense compatibility mode，配置可以悄悄落入 cell-span/range-scale/几何半径语义。这样 same-batch assignment 可能与官方 dense ActionFormer 不等价，尤其会改变：

```text
center sampling 半径
regression range gate
regression encode/decode denominator
per-level positive count
```

**与官方差异：**
官方 PointGenerator 的 stride 同时承担：point stride、center sampling radius base、regression target denominator、decode multiplier。([GitHub][6])

**修复代码：**

```python
class IrregularPointGeneratorV2:
    def __init__(
        self,
        strides,
        regression_range,
        use_temporal_grid=True,
        range_mode="hard",
        decode_scale_mode="cell",
        radius_scale_mode="geometric_mean",
        dense_compat_mode=None,
        **kwargs,
    ):
        if dense_compat_mode is not None:
            if dense_compat_mode != "official_actionformer":
                raise ValueError(f"Unsupported dense_compat_mode={dense_compat_mode}")

            # Fail-closed: official-compatible mode owns these fields.
            if range_mode not in {"hard", "absolute"}:
                raise ValueError("official_actionformer requires range_mode='absolute'")
            if decode_scale_mode not in {"cell", "level_stride"}:
                raise ValueError("official_actionformer requires decode_scale_mode='level_stride'")
            if radius_scale_mode not in {"geometric_mean", "level_stride"}:
                raise ValueError("official_actionformer requires radius_scale_mode='level_stride'")

            range_mode = "absolute"
            decode_scale_mode = "level_stride"
            radius_scale_mode = "level_stride"

        self.dense_compat_mode = dense_compat_mode
        self.range_mode = range_mode
        self.decode_scale_mode = decode_scale_mode
        self.radius_scale_mode = radius_scale_mode
        ...
```

---

### 3. `opentad/models/detectors/irregular_actionformer.py`

**P1 — detector 内部有 source-axis shim，但不是工具级显式合同。**
`IrregularActionFormer` 中有 `_segments_to_axis(..., source_axis, target_axis)` 和 `_segments_to_seconds(..., source_axis)`，其中 `_segments_to_seconds` 通过临时修改 `irregular_native_axis` 传给 `convert_to_seconds`。这只是 detector 局部 shim，不是 `convert_to_seconds(source_axis=...)` 的正式 API。([GitHub][7])

**为什么影响 mAP / 可信度：**
selected-axis、native-axis、seconds-axis 混用是 TAD 高 IoU 崩溃的典型源头。若工具函数仍靠 meta flag 推断，二次 selected→native 转换或漏转换很难被 scanner 抓住。

**修复方向：**
把 `source_axis` 下沉到 `convert_to_seconds` 工具函数，并要求所有调用者显式传入 `"selected"` 或 `"native"`。禁止隐式读取 `irregular_native_axis` 作为唯一判断。

---

**P1 — 单类别 post-processing 仍未执行 pre-NMS threshold/topk。**
当前代码读取了 `pre_nms_thresh`、`pre_nms_topk`，但在 `num_classes == 1` 分支中直接 squeeze score、生成 label，没有 threshold/topk；多类别分支才执行 filter/topk。([GitHub][7])

**为什么影响 mAP / 可信度：**
THUMOS 是 20 类，所以这未必解释当前 40/42；但它直接推翻“单类别 post-processing 现在也执行 pre-NMS filter”的修复声明，也会污染单类 sanity / synthetic route 的等价性。

**修复代码：**

```python
if num_classes == 1:
    scores_1d = scores.squeeze(-1)
    keep = scores_1d > pre_nms_thresh
    segments = segments[keep]
    scores_1d = scores_1d[keep]

    if pre_nms_topk > 0 and scores_1d.numel() > pre_nms_topk:
        sorted_scores, order = scores_1d.sort(descending=True)
        order = order[:pre_nms_topk]
        scores = sorted_scores[:pre_nms_topk]
        segments = segments[order]
    else:
        scores = scores_1d

    labels = scores.new_zeros(scores.shape, dtype=torch.long)
else:
    # existing multi-class branch
    ...
```

---

### 4. `opentad/models/utils/post_processing/utils.py`

**P0 — `convert_to_seconds` 没有 `source_axis` 参数。**
函数签名仍是 `convert_to_seconds(segments, meta)`；它根据 `irregular_selected_positions`、`valid_len`、`irregular_native_axis` 等 meta 推断是否做 selected→dense/native 转换。([GitHub][8])

**为什么影响 mAP / 可信度：**
post-processing / NMS / eval 的坐标轴如果隐式推断，会让 selected/native 二次转换极难定位。40/42 这类崩溃不一定来自这里，但这里足以阻断 dense-equivalent claim。

**修复代码：**

```python
def convert_to_seconds(segments, meta, source_axis: str):
    if source_axis not in {"selected", "native"}:
        raise ValueError(f"source_axis must be 'selected' or 'native', got {source_axis}")

    out = segments
    if source_axis == "selected":
        selected_positions = meta.get("irregular_selected_positions", None)
        valid_len = meta.get("valid_len", None)
        if selected_positions is not None and valid_len is not None:
            out = selected_axis_to_dense_axis(
                out,
                selected_positions.to(out.device),
                int(valid_len),
            )

    snippet_stride = float(meta["snippet_stride"])
    fps = float(meta["fps"])
    offset_frames = float(meta.get("window_start", 0))
    return out * snippet_stride / fps + offset_frames / fps
```

---

### 5. `tools/check_fail_closed_config.py`

**P0 — 文件不存在 / 404。**
目标路径打开为 404，因此“新增 whole-config fail-closed scanner”声明不成立。

**为什么影响 mAP / 可信度：**
现在不能系统扫描 diagnostic GT、teacher/cache、raw prediction、prediction shortcut、oracle、silent fallback 等配置污染。性能低不代表没有 shortcut；只是说明当前结论不能被信任。

**最小修复代码：**

```python
#!/usr/bin/env python
import argparse
import re
import sys
from pathlib import Path
from mmengine.config import Config

FORBIDDEN = re.compile(
    r"(diagnostic_gt|teacher|cache|raw[_-]?pred|prediction[_-]?shortcut|"
    r"oracle|use_gt|uses_gt|ground_truth|eval_shortcut)",
    re.IGNORECASE,
)

ALLOWLIST_KEYS = {
    "ann_file",          # annotation 文件名可能自然包含 gt 字样时需显式审查
    "gt_path_doc_only",  # 仅示例，实际按项目最小化
}

def walk(obj, prefix=""):
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if k not in ALLOWLIST_KEYS and FORBIDDEN.search(str(k)):
                hits.append((path, "<key>"))
            hits.extend(walk(v, path))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits.extend(walk(v, f"{prefix}[{i}]"))
    else:
        text = str(obj)
        if FORBIDDEN.search(text):
            hits.append((prefix, text))
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    failed = False
    for cfg_path in args.configs:
        cfg = Config.fromfile(cfg_path)
        hits = walk(cfg.to_dict())
        if hits:
            failed = True
            print(f"[FAIL] {cfg_path}")
            for path, value in hits:
                print(f"  {path}: {value}")
        else:
            print(f"[PASS] {cfg_path}")

    if failed and args.strict:
        sys.exit(2)

if __name__ == "__main__":
    main()
```

---

### 6. `tools/audit_sparse_head_assignment.py`

**P0 — 不是 official-vs-current same-batch audit。**
当前 audit 工具能构建 dataloader、跑当前 model/head/prior generator，并输出 current route 的 per-level positive count、GT coverage、encoded/decoded diagnostics；但未看到 `official_vs_current_assignment_diff`、`positive_mask_diff_count`、`assigned_class_diff_count`、`encoded_target_max_abs_diff`、`decoded_target_iou` 等声明字段。([GitHub][9])

**为什么影响 mAP / 可信度：**
如果没有 same-batch official target builder，就只能说“当前 route 自洽”，不能说“与官方 dense ActionFormer assignment 等价”。

**修复核心：**

```python
def official_dense_targets(points, gt_segments, gt_labels, center_sample_radius):
    # points: [N, 4] = center, reg_min, reg_max, stride
    centers = points[:, 0]
    strides = points[:, 3]
    reg_min = points[:, 1]
    reg_max = points[:, 2]

    gt_l = gt_segments[:, 0]
    gt_r = gt_segments[:, 1]
    gt_len = gt_r - gt_l

    left = centers[:, None] - gt_l[None, :]
    right = gt_r[None, :] - centers[:, None]
    reg_targets = torch.stack([left, right], dim=-1)

    inside = reg_targets.min(dim=-1).values > 0
    max_reg = reg_targets.max(dim=-1).values
    in_range = (max_reg >= reg_min[:, None]) & (max_reg <= reg_max[:, None])

    gt_centers = 0.5 * (gt_l + gt_r)
    radius = strides[:, None] * center_sample_radius
    cb_l = torch.maximum(gt_l[None, :], gt_centers[None, :] - radius)
    cb_r = torch.minimum(gt_r[None, :], gt_centers[None, :] + radius)
    in_center = (centers[:, None] >= cb_l) & (centers[:, None] <= cb_r)

    valid = inside & in_range & in_center
    masked_len = gt_len[None, :].repeat(points.shape[0], 1)
    masked_len[~valid] = float("inf")
    min_len, assign = masked_len.min(dim=1)

    pos = torch.isfinite(min_len)
    cls = gt_labels.new_full((points.shape[0],), -1)
    cls[pos] = gt_labels[assign[pos]]

    reg = reg_targets[torch.arange(points.shape[0]), assign]
    reg = reg / strides[:, None].clamp(min=1e-6)
    reg[~pos] = 0
    return cls, reg, pos, assign
```

```python
def official_vs_current_assignment_diff(official, current):
    o_cls, o_reg, o_pos, _ = official
    c_cls, c_reg, c_pos, _ = current

    return {
        "positive_mask_diff_count": int((o_pos != c_pos).sum().item()),
        "assigned_class_diff_count": int(((o_cls != c_cls) & (o_pos | c_pos)).sum().item()),
        "encoded_target_max_abs_diff": float((o_reg[o_pos & c_pos] - c_reg[o_pos & c_pos]).abs().max().item())
            if (o_pos & c_pos).any() else 0.0,
    }
```

---

### 7. `tools/verify_bridge_dense_equivalence.py`

**P1 — synthetic verifier 有价值，但不能证明真实 dataloader 等价。**
该工具文档写明是 synthetic sanity verifier；它构造 official/bridge points，测试 stride1、multi-level range gate、generated V2 levelstride 等 synthetic case。([GitHub][10])

**为什么影响结论：**
它没有覆盖真实 THUMOS dataloader、padding mask、batched variable length、selected/native axis、post-processing seconds、NMS、missing-center fallback 负例。它只能作为 Stage 0 单元测试，不能替代 Stage 1 same-batch audit。

**修复方向：**
增加 dataloader verifier：

```python
def verify_real_batch_points_and_targets(batch, official_head, bridge_head):
    official_points = official_head.prior_generator(batch["feat_list"])
    bridge_points = bridge_head.prior_generator(batch["feat_list"], batch["temporal_grid"])

    # point center / stride parity
    for op, bp in zip(official_points, bridge_points):
        assert torch.allclose(op[:, 0], bp[:, 0], atol=1e-6)
        assert torch.allclose(op[:, 1], bp[:, 1], atol=1e-6)
        assert torch.allclose(op[:, 2], bp[:, 2], atol=1e-6)
        assert torch.allclose(op[:, 3], bp[:, 3], atol=1e-6)  # stride == decode/radius scale
```

---

### 8. `tests/test_adapter_native_dense_headv2_contracts.py`

**P1 — 测试覆盖了 scale contract，但没有覆盖你声明的新修复。**
测试中能看到对 config 中 `range_mode="absolute"`、`decode_scale_mode="level_stride"`、`radius_scale_mode="level_stride"` 的检查，也有 `allow_legacy_full_cell_span` 的默认/legacy 测试，以及 synthetic uniform-grid target match。([GitHub][11])

但未检出以下声明项：`official_actionformer`、`convert_to_seconds`、single-class postprocess、`allow_center_fallback_inside_gt`。([GitHub][11])

**修复方向：**
新增至少四个测试：

```text
test_bridge_missing_center_fallback_disabled_by_default
test_point_generator_v2_official_actionformer_mode_forces_absolute_levelstride
test_convert_to_seconds_requires_source_axis_and_no_double_conversion
test_single_class_postprocess_applies_pre_nms_threshold_and_topk
```

---

### 9. configs：各 route 实际验证了什么

#### `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py`

**P0 if treated as official-compatible.**
该配置使用 hard assignment + symmetric linear，但仍设置 `center_radius_scale="full_cell_span"`、`reg_denom_mode="full_cell_span"`、`allow_legacy_full_cell_span=True`，prior `range_mode="hard"`。它是 legacy cell-span route，不是官方 dense-compatible route。([GitHub][12])

它验证的是：**cell-span scale + hard local range**。
它不能验证 official ActionFormer 等价。

---

#### `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py`

**P0 if treated as official-compatible.**
该配置把 regression ranges 改成全开放 `(0,10000)` / `range_mode="open"`，但 bridge 仍是 `full_cell_span` legacy。([GitHub][13])

它验证的是：**去掉 regression range gate 后，legacy scale 是否仍崩**。
若只从 40 到 42，说明 regression range 不是唯一根因，但不证明 scale 正确。

---

#### `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py`

**P0 if treated as official-compatible.**
它只把 prior 改为 `range_mode="absolute"`，bridge 仍使用 `full_cell_span` / legacy。([GitHub][14])

它验证的是：**absolute range alone**。
它仍不能证明 official dense 等价。

---

#### `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py`

**P1 — corrected candidate，但不是 exact official dense-compatible。**
它改成 `center_radius_scale="point_radius"`、`reg_denom_mode="left_right_mean"`、`allow_legacy_full_cell_span=False`，prior 使用 `range_mode="absolute"`、`decode_scale_mode="level_stride"`、`radius_scale_mode="level_stride"`；但 regression ranges 是 expanded：`[(0,8),(2,16),(4,32),(8,64),(16,128),(32,10000)]`，不是官方 `[(0,4),(4,8),...]`。([GitHub][15])

它验证的是：**official-like scale + expanded range**。
它是好 ablation，但不能叫 exact dense-equivalent。

---

#### `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4.py`

**P1 — scale 更接近，但 range gate 不同。**
它使用 `point_radius`、`left_right_mean`、`level_stride` decode/radius，但 prior `range_mode="level_stride"` 会把 range 语义变成 stride-scaled，而不是官方 absolute regression range。([GitHub][16])

它验证的是：**level-stride range scaling**。
它不是官方 dense-compatible route。

---

#### `input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py`

**P1 — uniform sparse/native-axis corrected bridge candidate。**
dataset 使用 `uniform_fixed_subsample`、`keep_ratio=0.5`、`remap_gt_to_selected_axis=False`，即 native-axis sparse route。([GitHub][17])

它验证的是：**等间隔 50% native-axis irregular bridge 是否能恢复**。
它不能直接与 selected-axis dense baseline 做 apples-to-apples，除非 Stage 1/2 证明 assignment 与秒轴完全一致。

---

#### `input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py`

**WARN — route 存在，但没有完成结论。**
该配置文档说明是 Stage-2 official dense selected-axis sanity；它继承 dense control，GT/proposals 在 selected axis，pipeline 中 `remap_gt_to_selected_axis=True`。([GitHub][18])

它应该作为“本仓库是否能复现 50% uniform ≈65 Avg-mAP”的必要 gate。
但它只是配置存在；未运行前不能声称 dense baseline 已经被本仓库官方等价复现。

---

### 10. `root-cause-notes.md`

**P2 — notes 的判断方向合理，但不能替代代码 gate。**
文档记录了 fixed 约 40.20、openrange 约 42.44、等间隔 50% baseline 约 65，并明确把 route-level collapse 指向 bridge/supervision contract；也列出 missing-center fallback、`convert_to_seconds source_axis`、scanner、audit 等 blocker。([GitHub][19])

这份 notes 与本次审查结论一致，但它是仓库自述，不是独立证据。长训和论文结论仍必须由 Stage 0-5 gate 决定。

---

## C. 当前实现是否完成上次建议

| 建议项                                             |                 判决 | 说明                                                                                                                              |
| ----------------------------------------------- | -----------------: | ------------------------------------------------------------------------------------------------------------------------------- |
| Bridge fallback fail-closed                     | **FAIL / partial** | `allow_legacy_full_cell_span` scale contract 已 fail-closed；但 `allow_center_fallback_inside_gt` 未见，soft fallback 仍 unconditional |
| PointGeneratorV2 official dense-compatible mode | **FAIL / partial** | 可以手动配出 `absolute + level_stride + level_stride`，但没有 `dense_compat_mode="official_actionformer"`                                 |
| explicit `source_axis` seconds conversion       | **FAIL / partial** | detector 有 shim；工具函数 `convert_to_seconds` 仍无 `source_axis`                                                                      |
| single-class pre-NMS filter                     |           **FAIL** | 单类分支未执行 threshold/topk                                                                                                          |
| whole-config fail-closed scanner                |           **FAIL** | 文件 404                                                                                                                          |
| official-vs-current same-batch audit diff       |           **FAIL** | 当前 audit 是 current diagnostics，不是官方 same-batch diff                                                                             |
| batched/masked/full dataloader verifier         |    **WARN / FAIL** | synthetic verifier 有，但真实 dataloader/mask/axis 未覆盖                                                                               |
| official dense selected-axis sanity route       |           **WARN** | config 存在，未证明跑通、未证明接近 65                                                                                                        |
| remote Linux preflight readiness                |           **WARN** | 可以跑，但当前不能作为 green gate；预期暴露失败项                                                                                                  |

---

## D. 性能崩溃根因排序 Top 10

### 1. Assignment / supervision contract 偏离官方 dense

**最可能。**
当前 hard/openrange/absrange legacy route 使用 full-cell span；soft route 有 hidden fallback；corrected route 又改变 regression ranges。官方 dense 的 positive set、class assignment、target encoding 都由 stride/range/center sampling 共同决定。任何一个偏移都可能让 65 掉到 40/42。

**最小验证：**
Stage 1 same-batch audit：要求

```text
positive_mask_diff_count == 0
assigned_class_diff_count == 0
encoded_target_max_abs_diff <= 1e-6
decoded_target_max_abs_diff <= 1e-6
decoded_target_iou_min >= 0.999999
official_per_level_positive_count == current_per_level_positive_count
gt_coverage_diff == 0
```

---

### 2. Regression range 语义错误

`hard/local_cell_span`、`open`、`absolute`、`expanded absolute`、`levelstride` 分别改变不同变量。openrange 从 40 到 42 不能说明 range 正确，只说明“range gate 不是唯一原因”。

**最小验证：**
同一 batch 输出 per-level positive count 与 GT coverage，对比 official range。再加 exact official range config：

```python
regression_range=[(0,4),(4,8),(8,16),(16,32),(32,64),(64,10000)]
range_mode="absolute"
decode_scale_mode="level_stride"
radius_scale_mode="level_stride"
```

---

### 3. Regression encode/decode denominator 错误

官方 reg target 除以 stride；legacy route 用 full cell span。稀疏 50% 时 cell span 与 stride 不等价，边界会系统性缩放错。

**最小验证：**
在 same-batch audit 中比较 official decoded segment 与 current decoded segment：

```text
decoded_target_max_abs_diff
decoded_target_iou
left/right error histogram
```

---

### 4. selected-axis vs native-axis GT 坐标混杂

selected-axis dense sanity 与 native-axis irregular bridge 是不同实验。若 GT remap、proposal decode、seconds conversion 任一处混轴，高 IoU 会严重掉。

**最小验证：**
构造 3 段 toy GT：

```text
selected segment -> native segment -> seconds
native segment -> seconds
```

要求 roundtrip 误差为 0，且 NMS 前后的 segment axis 被显式标记。

---

### 5. post-processing 秒轴转换 / NMS mismatch

`convert_to_seconds` 仍隐式推断 source axis；detector shim 不能覆盖所有调用路径。NMS 若在错误轴执行，proposal 排序和高 IoU 都会崩。

**最小验证：**

```text
save raw decoded proposals before axis conversion
save proposals after native conversion
save proposals after seconds conversion
compare against official dense selected-axis sanity
```

---

### 6. 单类 pre-NMS mismatch

对 THUMOS 20 类不是主因，但它说明 postprocess contract 仍不干净。

**最小验证：**
单类 synthetic batch 中设定低分噪声 proposals，要求 pre-NMS 后 proposal 数量受 `pre_nms_thresh/topk` 控制。

---

### 7. projection / neck / grid-aware route 错误

如果 temporal grid 下采样与 feature pyramid level 对不上，point center 虽然看似正确，实际 feature 对齐可能错。

**最小验证：**
冻结同一输入特征，比较 official dense head 输入 feature shape、mask、level stride、point center、valid length。

---

### 8. 数据 pipeline / eval shortcut / hidden leakage guard 不完整

当前 scanner 缺失，所以不能证明没有 diagnostic GT、teacher/cache、raw prediction shortcut。性能低不代表没有泄漏，只代表不能信任实验归因。

**最小验证：**
补全 scanner 后对所有 train/val/test config 严格扫描，输出 JSON manifest。

---

### 9. loss normalizer / positive count 分布变化

openrange / expanded range 可能产生过多或过少 positives，改变 loss normalizer 与 cls/reg loss 比例。

**最小验证：**
每个 config 记录：

```text
num_pos_per_level
loss_normalizer
cls_loss / reg_loss
assigned GT length distribution
```

并与官方 dense selected-axis sanity 对齐。

---

### 10. optimizer / LR / seed / schedule

这是低优先级。只有 Stage 0-2 通过后，才值得讨论长训超参。

**最小验证：**
dense selected-axis sanity 能稳定回到约 65 后，再做 corrected bridge short smoke 与长训。

---

## E. 详细实验路线

### Stage 0：Linux preflight

**目标：**
证明代码可导入、配置 fail-closed、synthetic verifier 通过；不跑训练。

**命令：**

```bash
git clone --branch codex/sparse-head-clean-20260702 \
  https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702.git sparse
cd sparse
git rev-parse HEAD

python -m py_compile \
  opentad/models/dense_heads/irregular_actionformer_bridge_head.py \
  opentad/models/dense_heads/prior_generator/irregular_point_generator.py \
  opentad/models/detectors/irregular_actionformer.py \
  opentad/models/utils/post_processing/utils.py \
  tools/audit_sparse_head_assignment.py \
  tools/verify_bridge_dense_equivalence.py

python tools/check_fail_closed_config.py \
  configs/adatad/thumos/input_*50pct* --strict

pytest -q tests/test_adapter_native_dense_headv2_contracts.py
python tools/verify_bridge_dense_equivalence.py
```

**当前 gate：FAIL。**
原因：scanner 缺失；claimed fallback/source_axis/single-class/audit diff 未实现。

---

### Stage 1：same-batch official-vs-current assignment audit

**目标：**
在同一 batch、同一 GT、同一点坐标上比较 official dense target builder 与 current bridge target builder。

**命令：**

```bash
python tools/audit_sparse_head_assignment.py \
  --official-config configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
  --current-config configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py \
  --split train \
  --num-batches 4 \
  --seed 0 \
  --official-vs-current \
  --out audits/stage1_absrange_expanded.json
```

**通过 gate：**

```text
positive_mask_diff_count == 0
assigned_class_diff_count == 0
encoded_target_max_abs_diff <= 1e-6
decoded_target_max_abs_diff <= 1e-6
decoded_target_iou_min >= 0.999999
gt_coverage_diff == 0
```

**当前状态：FAIL。**
工具还没有 official-vs-current diff。

---

### Stage 2：official dense selected-axis sanity

**目标：**
证明本仓库在 50% 等间隔 selected-axis dense route 上能复现约 65 Avg-mAP。否则不能把后续下降归因给 sparse/irregular head。

**命令：**

```bash
torchrun --nproc_per_node=4 tools/train.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
  --validate
```

**通过 gate：**

```text
Avg-mAP 接近既有 50% uniform baseline
mAP@0.6/@0.7 不出现异常塌陷
save_dict 中 proposals 的 selected/native/seconds 轴可追踪
scanner 零 forbidden hits
```

**当前状态：WARN。**
配置存在，但未提供运行证据。

---

### Stage 3：corrected bridge candidates short smoke

**目标：**
只在 Stage 0-2 通过后，对 corrected bridge 做短训/短评，验证不再 40/42 崩溃。

**建议 configs：**

```text
input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py
新增 exact-official-range bridge config
新增 uniform native-axis vs selected-axis paired config
```

**命令：**

```bash
torchrun --nproc_per_node=4 tools/train.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py \
  --validate \
  --cfg-options train_cfg.max_epochs=3
```

**输出指标：**

```text
num_pos_per_level
loss_normalizer
cls_loss/reg_loss
proposal recall @ tIoU 0.5/0.6/0.7
pre-NMS proposal count
post-NMS proposal count
decoded segment error histogram
```

---

### Stage 4：gated long training

**允许条件：**

```text
Stage 0 PASS
Stage 1 exact official-compatible route PASS
Stage 2 dense selected-axis sanity PASS
Stage 3 short smoke 无 NaN、无 proposal recall 崩溃
```

**当前：不允许。**

**长训矩阵：**

```text
A. official dense selected-axis sanity
B. exact official-compatible bridge, uniform 50%
C. exact official-compatible bridge, random fixed 50%
D. absrange_expanded bridge
E. levelstride ablation
F. legacy hard/openrange/absrange as negative controls
```

---

### Stage 5：detection-quality / proposal recall / high-IoU localization diagnosis

**目标：**
如果 Avg-mAP 仍低，定位是 recall、localization、score calibration、NMS 还是 seconds conversion 问题。

**指标：**

```text
proposal recall @ top 100/500/1000, tIoU 0.3-0.7
GT coverage by level
center coverage by level
boundary left/right absolute error
pre-NMS recall vs post-NMS recall
score rank of matched proposals
mAP drop by class and by GT duration bucket
```

**关键判据：**

```text
pre-NMS recall 高、post-NMS recall 低 -> NMS/score calibration
pre-NMS recall 已低 -> assignment/range/decode/feature alignment
low IoU only at 0.6/0.7 -> boundary scale/seconds conversion
all IoU low -> proposal center/feature/GT axis mismatch
```

---

## F. 关键实现 patch 汇总

### F1. BridgeHead fallback fail-closed

```python
# irregular_actionformer_bridge_head.py

class IrregularActionFormerBridgeHead(...):
    def __init__(
        self,
        ...,
        allow_center_fallback_inside_gt: bool = False,
        ...
    ):
        ...
        self.allow_center_fallback_inside_gt = bool(allow_center_fallback_inside_gt)

    def _build_candidate_mask(...):
        ...
        missing_gt = ~candidate_mask.any(dim=0)
        if missing_gt.any():
            if self.allow_center_fallback_inside_gt:
                candidate_mask[:, missing_gt] = inside_gt_seg[:, missing_gt]
            else:
                if getattr(self, "debug_assignment", False):
                    self._latest_debug_state["missing_center_gt_count"] = int(
                        missing_gt.sum().item()
                    )
                # no fallback
        return candidate_mask
```

---

### F2. PointGeneratorV2 official mode

```python
# irregular_point_generator.py

if dense_compat_mode == "official_actionformer":
    range_mode = "absolute"
    decode_scale_mode = "level_stride"
    radius_scale_mode = "level_stride"

self.dense_compat_mode = dense_compat_mode
```

并在 `__call__` 结束前加 invariant：

```python
if self.dense_compat_mode == "official_actionformer":
    assert self.range_mode == "absolute"
    assert self.decode_scale_mode == "level_stride"
    assert self.radius_scale_mode == "level_stride"
    # points columns:
    # [center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale]
    assert torch.allclose(points[:, 3], stride_tensor)
    assert torch.allclose(points[:, 4], stride_tensor)
    assert torch.allclose(points[:, 6], stride_tensor)
```

---

### F3. `convert_to_seconds(source_axis=...)`

```python
def convert_to_seconds(segments, meta, source_axis: str):
    if source_axis not in {"selected", "native"}:
        raise ValueError(f"Invalid source_axis={source_axis}")

    dense_segments = segments
    if source_axis == "selected":
        selected_positions = meta.get("irregular_selected_positions")
        valid_len = meta.get("valid_len")
        if selected_positions is not None and valid_len is not None:
            dense_segments = selected_axis_to_dense_axis(
                dense_segments,
                selected_positions.to(dense_segments.device),
                int(valid_len),
            )

    fps = float(meta["fps"])
    snippet_stride = float(meta["snippet_stride"])
    window_start = float(meta.get("window_start", 0))
    return dense_segments * snippet_stride / fps + window_start / fps
```

---

### F4. 单类 post-processing pre-NMS

```python
if num_classes == 1:
    scores_1d = scores.squeeze(-1)
    keep = scores_1d > pre_nms_thresh
    scores_1d = scores_1d[keep]
    segments = segments[keep]

    if pre_nms_topk > 0 and scores_1d.numel() > pre_nms_topk:
        sorted_scores, order = scores_1d.sort(descending=True)
        order = order[:pre_nms_topk]
        scores = sorted_scores[:pre_nms_topk]
        segments = segments[order]
    else:
        scores = scores_1d

    labels = scores.new_zeros(scores.shape, dtype=torch.long)
```

---

### F5. audit diff 输出字段

```python
diff = {
    "positive_mask_diff_count": int((official_pos != current_pos).sum().item()),
    "assigned_class_diff_count": int(
        ((official_cls != current_cls) & (official_pos | current_pos)).sum().item()
    ),
    "encoded_target_max_abs_diff": max_abs_or_zero(
        official_reg[official_pos & current_pos],
        current_reg[official_pos & current_pos],
    ),
    "decoded_target_iou": segment_iou_summary(
        official_decoded[official_pos & current_pos],
        current_decoded[official_pos & current_pos],
    ),
    "decoded_target_max_abs_diff": max_abs_or_zero(
        official_decoded[official_pos & current_pos],
        current_decoded[official_pos & current_pos],
    ),
    "official_per_level_positive_count": official_level_counts,
    "current_per_level_positive_count": current_level_counts,
    "gt_coverage_diff": current_gt_coverage - official_gt_coverage,
}
```

---

## G. 最终目标

最终不是直接证明“sparse sampling 一定导致下降”。当前证据更像：

```text
等间隔 50% sparse observation 本身可以接近 65 Avg-mAP；
当前 irregular bridge/head route 掉到 40/42，
更可能是 head supervision / scale / axis / postprocess contract 偏移。
```

要证明这一点，必须构造一个 **dense-equivalent bridge**：

```text
同一 50% 等间隔输入
同一 selected-axis dense GT 合同
同一官方 ActionFormer assignment
同一 stride/range/center-sampling/encode/decode
同一 seconds conversion / NMS
```

如果 dense-equivalent bridge 能回到接近 65，说明 40/42 不是 sparse sampling 的自然退化，而是当前 irregular head 合同错误。
如果 exact bridge 仍回不到 65，再继续排查 projection/neck/grid-aware feature alignment、data pipeline、seconds/NMS、loss normalizer。
如果始终无法构造 clean dense-equivalent bridge，更优雅路线是：**先不要让 irregular head 重写监督合同**；把 sparse features 映射回 selected-axis dense grid，继续使用官方 ActionFormer/AdaTAD head 与 target builder，只把不规则性限制在 input adapter / feature transport / mask 中。这样论文故事更干净：先证明“选帧不破坏官方检测监督”，再谈 native irregular geometry。

[1]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/branches/all?query=codex%2Fsparse-head-clean-20260702 "Branches · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[2]: https://github.com/sming256/OpenTAD/blob/main/configs/_base_/models/actionformer.py "OpenTAD/configs/_base_/models/actionformer.py at main · sming256/OpenTAD · GitHub"
[3]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/models/dense_heads/irregular_actionformer_bridge_head.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_bridge_head.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[4]: https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/anchor_free_head.py "raw.githubusercontent.com"
[5]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/models/dense_heads/prior_generator/irregular_point_generator.py?plain=1 "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/prior_generator/irregular_point_generator.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[6]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/prior_generator/point_generator.py?plain=1 "OpenTAD/opentad/models/dense_heads/prior_generator/point_generator.py at main · sming256/OpenTAD · GitHub"
[7]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/models/detectors/irregular_actionformer.py?plain=1 "OpenTAD_SparseHeadClean_20260702/opentad/models/detectors/irregular_actionformer.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[8]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/opentad/models/utils/post_processing/utils.py?plain=1 "OpenTAD_SparseHeadClean_20260702/opentad/models/utils/post_processing/utils.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[9]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/tools/audit_sparse_head_assignment.py?plain=1 "OpenTAD_SparseHeadClean_20260702/tools/audit_sparse_head_assignment.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[10]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/tools/verify_bridge_dense_equivalence.py?plain=1 "OpenTAD_SparseHeadClean_20260702/tools/verify_bridge_dense_equivalence.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[11]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/tests/test_adapter_native_dense_headv2_contracts.py "OpenTAD_SparseHeadClean_20260702/tests/test_adapter_native_dense_headv2_contracts.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[12]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[13]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[14]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[15]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[16]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_levelstride_n16r4.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[17]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_uniform_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[18]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[19]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/codex/sparse-head-clean-20260702/root-cause-notes.md "OpenTAD_SparseHeadClean_20260702/root-cause-notes.md at codex/sparse-head-clean-20260702 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
