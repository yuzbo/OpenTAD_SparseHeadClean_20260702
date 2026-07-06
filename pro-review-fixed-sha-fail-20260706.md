## 1. 总判决：**FAIL**

不是说这个 commit 没有进展；相反，`06dce3d5955e9c2d0cf5232eace6b4224831900f` 确实做了大量 hardening，commit 页面显示为 `Harden sparse head route diagnostics`，包含 20 个文件变更，覆盖 config、bridge head、point generator、detector、postprocess、审计脚本和测试。([GitHub][1])

但按你给的 gate——**“当前 sparse/irregular route 是否已经真正正确，是否足以解释或排除 65/63 → 40/42 崩溃根因”**——我判 **FAIL**。

核心原因：

**bridge hard official-compatible 路线已经接近可审计，但 HeadV2/V3 仍然不是官方 ActionFormer 等价实现；selected/native 轴转换仍有 fail-open 边缘；GT remap 仍可能静默改变监督分布；soft assignment / geometry modulation / log1p regression 仍与官方 dense contract 混杂。**

因此，当前不能声明：

> “40/42 mAP 是 sparse/irregular route 的自然退化。”

也不能声明：

> “HeadV3 fixed 或 bridge hard openrange 已经是正确 sparse ActionFormer，只是选帧导致下降。”

目前更合理的结论是：

> **当前 commit 修复了部分高危坐标/scale/route-contract 问题，但尚未完成 sparse/irregular route 的 dense-equivalent 闭环验证。40/42 仍高度可疑，必须继续优先排查 axis、assignment、GT remap、decode/NMS/seconds 和 soft-head 监督合同偏移。**

---

## 2. 已正确实现项

### 2.1 官方 ActionFormer contract 的关键对照源是清楚的

官方 OpenTAD 的 dense ActionFormer/anchor-free 逻辑是硬合同：

官方 `PointGenerator` 生成 `[center, reg_min, reg_max, stride]`，其中 center 是 `arange(T) * stride`，regression range 与 stride 随 level 绑定。([GitHub][2])

官方 assignment 逻辑是：

1. 计算 point 到 GT start/end 的 left/right distance；
2. `inside_gt_seg_mask`；
3. center sampling 使用 `stride * center_sample_radius`；
4. regression range 用 `max(left, right)`；
5. 多 GT 冲突用 shortest GT length；
6. regression target 是基于 point center 的 linear left/right 距离；
7. loss/decode 再通过官方 proposal refinement 路径使用这些 target。([GitHub][3])

这意味着任何 sparse/irregular “dense-equivalent” 路线必须先做到：

```text
same points semantic
same center-sampling radius semantic
same regression range semantic
same shortest-GT assignment
same encode/decode unit
same postprocess/NMS axis
same seconds conversion axis
```

你当前 commit 对其中一部分已经做了正确加固。

---

### 2.2 `IrregularPointGeneratorV2` 的 official dense compat 是正确方向

`dense_compat_mode="official_actionformer"` 现在会强制：

```text
range_mode = "absolute"
decode_scale_mode = "level_stride"
radius_scale_mode = "level_stride"
```

并且只允许 `"official_actionformer"` 这个 compat mode。([GitHub][4])

更重要的是，它会断言 dense compat grid 的 center 必须等于官方 `arange(T) * stride`，否则直接 raise，而不是默默接受 irregular center。([GitHub][4])

这是正确修复。它回答了你第 3 点里的问题：

> official_actionformer dense compat 是否拒绝 irregular center？

**在 V2 point generator 层面：是，已拒绝。**

同时 V2 points layout 明确变为：

```text
[center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale]
```

这比之前混用 `point_scale / full-cell-span` 更可审计。([GitHub][4])

---

### 2.3 bridge hard 分支基本摆脱了旧 fallback 和 full-cell-span 混杂

`IrregularActionFormerBridgeHead` 默认参数已经改成更安全的方向：

```python
reg_denom_mode="left_right_mean"
allow_legacy_full_cell_span=False
allow_center_fallback_inside_gt=False
```

并且 constructor 里会保存这些配置。([GitHub][5])

scale mode 也增加了显式检查：如果使用 `full_cell_span` 但没有打开 `allow_legacy_full_cell_span`，会直接报错，并提示 official-compatible route 应使用 `point_radius` 或 `left_right_mean`。([GitHub][5])

bridge 的 hard target 路径现在是独立 hard assignment，不走 V2/V3 的 `_build_candidate_mask`，并且 debug 明确记录：

```text
bridge_hard_assignment_uses_build_candidate_mask = False
bridge_hard_missing_center_fallback_applied = False
```

这说明 **bridge hard assignment 本身没有再误用 missing-center inside-GT fallback**。([GitHub][5])

所以对你第 3 点：

> hard assignment 是否仍误用 fallback？

我的判定是：

**bridge hard：否，当前 commit 已经修掉。
HeadV2/V3 soft：仍然有 fallback 风险，见后文 S0。**

---

### 2.4 route_contract / config scanner 已经能阻止一部分 legacy dense-equivalent 误报

`check_fail_closed_config.py` 会扫描 bridge head，识别 legacy route，例如 `full_cell_span` 或 center fallback，并阻止 legacy route 声称 `dense_equivalent_claim_allowed=True`。它还会要求 legacy route 带 `route_contract`，并检查 route_contract 与实际 head config 是否一致。([GitHub][6])

这回答你的问题：

> route_contract 是否足以阻止 legacy route 被宣称 dense-equivalent？

**如果这个 checker 被强制纳入 Stage 0 / CI：基本足够。
如果只是一个可选脚本：仍不够。**

它现在是脚本级防线，不是所有训练入口的 runtime hard gate。必须把它变成 preflight 必跑项。

---

### 2.5 detector path 里 NMS 前转 native 的方向是对的

`irregular_actionformer.py` 现在有 axis contract：

```text
gt_axis
proposal_axis
postprocess_axis
irregular_native_axis
irregular_selected_positions
irregular_selected_valid_len
```

并且要求 selected-axis metadata 存在，否则 selected route fail-closed。它还拒绝不合法的 selected/native 组合。([GitHub][7])

在 postprocess 路径中，proposal segments 会先转换到目标 postprocess axis，然后再进入 `batched_nms`，之后才 convert to seconds。([GitHub][7])

所以对你第 3 点：

> NMS 前是否一定转 native？

**在 detector 正常路径、axis contract 生效时：基本是。
但 utility 函数本身仍 fail-open，见 S1。**

---

### 2.6 `audit_sparse_head_assignment.py` 的方向正确

审计脚本支持多 config、同 split、同 batch，记录 assignment mode、regression mode、axis、range/scale mode、hard assignment fallback 状态、per-level positive counts、decode reconstruction、official-vs-current diff 等。([GitHub][8])

这是必要工具。当前实验路线里 Stage 1 “same-batch assignment audit，native/selected 不混 batch” 是正确的。

---

## 3. 仍有错误或高风险项，按严重程度排序

### S0-1：HeadV2/V3 仍不是 official ActionFormer 等价实现

这是当前最大问题。

`IrregularActionFormerHeadV2` 的 candidate mask 仍然存在逻辑：

```python
missing_gt = ~candidate_mask.any(dim=0)
candidate_mask[:, missing_gt] = inside_gt_seg[:, missing_gt]
```

即：当 center sampling 没覆盖到某些 GT 时，fallback 到 inside-GT。([GitHub][9])

这和官方 ActionFormer 不等价。官方如果 center sampling / range mask 没选到某些 GT，不会默默放宽为 inside-GT；它按 mask 和 shortest-GT 规则产生监督。([GitHub][3])

更严重的是，V2 的 target 不是官方硬 shortest-GT assignment，而是 soft/weighted assignment：

```text
assign_weights
weighted_labels
best_gt_idx = assign_weights.max
log1p regression target
```

这和官方 linear regression target + hard shortest-GT assignment 不等价。([GitHub][9])

V3 又在 V2 基础上加入 geometry features / modulation / boundary aux 等结构改动。([GitHub][10])

所以：

```text
HeadV3 fixed Avg mAP = 40.20
```

不能解释为：

```text
官方 ActionFormer sparse/irregular 版本只有 40.20
```

它只能解释为：

```text
V2/V3 soft assignment + log1p regression + geometry modulation + irregular scale/head 改造的混合系统得到 40.20
```

这不是干净实验。

**严重性：S0 blocking。**

---

### S0-2：GT remap 到 selected-axis 仍可能静默改变监督分布

`end_to_end.py` 的 `remap_gt_to_selected_axis=True` 默认存在，并且 `_remap_gt_to_selected_axis` 会将 native GT segment 通过 selected positions 插值映射到 selected-axis。映射后如果 `mapped_end <= mapped_start`，该 GT 会被 drop。([GitHub][11])

这非常危险。

因为你的 mAP 崩溃怀疑点之一就是：

```text
GT remap / selected-native axis 混淆 / assignment contract 偏移
```

如果 selected-axis control 里有 GT 被静默压扁或 drop，那么 detector 看到的监督目标已经不是原始 THUMOS GT。即使 drop 数量很少，也可能集中在短动作或边界敏感动作上，从而造成 @0.6/@0.7 崩溃。

必须把这个改成 fail-closed：

```text
remap 后 dropped GT count > 0
=> 默认 FAIL
=> 只有显式 allow_drop_selected_axis_gt=True 的 legacy diagnostic route 才允许
```

**严重性：S0 blocking。**

---

### S1-1：selected-axis postprocess utility 仍 fail-open

`selected_axis_to_dense_axis` 在 metadata 缺失、positions 缺失、valid_len 缺失或 `irregular_native_axis=True` 时，会直接返回原 coords。([GitHub][12])

`convert_to_seconds(source_axis="selected")` 现在会在 selected metadata 缺失时 raise，这部分是好的；但 `source_axis="auto"` 仍然可能基于 metadata 走自动逻辑。([GitHub][12])

这意味着 detector 主路径可能安全，但 utility 仍不安全。只要某个 eval script、dump script、analysis script、legacy postprocess path 绕过了 detector axis contract，就可能把 selected-axis proposals 当 native-axis proposals 直接转 seconds/NMS。

对于当前 65→40/42 这种级别的崩溃，这类 bug 必须默认怀疑。

**严重性：S1 high。**

---

### S1-2：native irregular cell geometry 与 legacy full-cell-span 仍可能制造 scale 膨胀

native irregular grid 里，cell left/right 是从相邻 selected positions / interval boundary 计算出来的。temporal grid 在缺少显式 cells 时也会根据 adjacent deltas 构造 left/right。([GitHub][13])

这对真正 native irregular head 是合理的，但不能混入 official dense-equivalent claim。

如果某条 route 仍用：

```text
full_cell_span = left + right
openrange
log1p target
irregular center radius
```

那它不再是官方 ActionFormer 的 stride contract，而是另一个 detector。bridge hard 当前已默认禁止 full-cell-span；但 legacy/openrange 仍能作为 ablation 运行。([GitHub][5])

所以：

```text
bridge hard linear openrange Avg mAP = 42.44
```

不能作为 official-compatible sparse head 结论。它最多是 legacy/openrange diagnostic。

**严重性：S1 high。**

---

### S1-3：route_contract 是 config scanner 保护，不是全训练入口 runtime 保护

`check_fail_closed_config.py` 本身设计是对的，但如果用户直接运行 `tools/train.py config.py`，而没有先跑 checker，legacy route 仍可能被训练并被人误报成 dense-equivalent。

这不是代码逻辑 bug，但属于实验协议 bug。

必须把 Stage 0 写成硬门：

```text
check_fail_closed_config.py fail
=> 禁止 Stage 1/2/3/4
=> 禁止提交 mAP claim
```

**严重性：S1 high。**

---

### S2-1：Stage 4 soft assignment analysis 现在还不能作为 mAP 主线

HeadV2/V3 的 soft assignment 分析应该排在 bridge selected/native controls 后面。

当前应先证明：

```text
dense selected-axis control ≈ 历史 dense random-fixed 63.12
bridge hard selected-axis control ≈ dense selected-axis control
bridge hard native corrected ≈ selected/native control
```

否则直接跑 HeadV2/V3，只会继续混杂：

```text
axis bug?
assignment bug?
regression target bug?
NMS bug?
geometry modulation bug?
feature remap bug?
```

**严重性：S2 medium-high。**

---

## 4. 必须立即修改的代码：patch 级关键代码

下面不是完整 diff，但足够明确。

---

### Patch A：postprocess utility 默认 strict，不允许 selected-axis 静默返回 native coords

文件：

```text
opentad/models/utils/post_processing/utils.py
```

关键修改：

```python
def _has_irregular_selected_axis_meta(meta):
    if meta is None:
        return False
    return (
        "irregular_selected_positions" in meta
        or "irregular_selected_valid_len" in meta
        or "irregular_axis_contract" in meta
        or "irregular_gt_axis" in meta
        or "irregular_proposal_axis" in meta
        or "irregular_postprocess_axis" in meta
    )


def selected_axis_to_dense_axis(coords, meta, *, strict=False):
    if meta is None:
        if strict:
            raise ValueError(
                "selected-axis to dense/native conversion requires meta, got meta=None"
            )
        return coords

    if meta.get("irregular_native_axis", False):
        if strict and meta.get("irregular_proposal_axis") == "selected":
            raise ValueError(
                "Invalid axis metadata: proposal_axis=selected but irregular_native_axis=True"
            )
        return coords

    positions = meta.get("irregular_selected_positions", None)
    valid_len = meta.get("irregular_selected_valid_len", None)

    if positions is None or valid_len is None:
        if strict or _has_irregular_selected_axis_meta(meta):
            raise ValueError(
                "Missing irregular_selected_positions / irregular_selected_valid_len "
                "for selected-axis conversion."
            )
        return coords

    # existing interpolation body below
    ...
```

同时修改 `convert_to_seconds`：

```python
def convert_to_seconds(
    segments,
    meta,
    *,
    source_axis="native",
    allow_auto_axis=False,
    ...
):
    if source_axis == "auto":
        if _has_irregular_selected_axis_meta(meta) and not allow_auto_axis:
            raise ValueError(
                "source_axis='auto' is forbidden for irregular routes. "
                "Pass source_axis='native' or source_axis='selected' explicitly."
            )
        # existing auto behavior only for non-irregular legacy paths

    if source_axis == "selected":
        segments = selected_axis_to_dense_axis(segments, meta, strict=True)

    # existing official seconds conversion
    ...
```

PASS gate：

```text
任何 irregular route 中 selected metadata 缺失，必须 raise。
任何 source_axis="auto" + irregular metadata，必须 raise，除非显式 allow_auto_axis=True。
```

---

### Patch B：`IrregularActionFormer` 禁止 metas=None 绕过 axis contract

文件：

```text
opentad/models/detectors/irregular_actionformer.py
```

关键修改：

```python
def _require_metas_for_irregular_route(self, metas, stage):
    if metas is None:
        raise ValueError(
            f"IrregularActionFormer {stage} requires metas for axis contract, "
            "selected/native conversion, and NMS/seconds conversion."
        )
    if not isinstance(metas, (list, tuple)):
        raise TypeError(f"metas must be list/tuple, got {type(metas)}")
```

在这些位置调用：

```python
def forward_train(..., metas=None, ...):
    self._require_metas_for_irregular_route(metas, "forward_train")
    self._assert_axis_contracts(metas, stage="train")
    ...

def forward_test(..., metas=None, ...):
    self._require_metas_for_irregular_route(metas, "forward_test")
    self._assert_axis_contracts(metas, stage="test")
    ...

def post_processing(self, predictions, metas, ...):
    self._require_metas_for_irregular_route(metas, "post_processing")
    ...
```

并确保 selected→native 转换调用 strict utility：

```python
segments = selected_axis_to_dense_axis(segments, meta, strict=True)
```

PASS gate：

```text
metas=None 的 irregular detector forward/test/postprocess 必须 FAIL。
不能 fallback 到 dense/default grid。
```

---

### Patch C：HeadV2/V3 的 missing-center fallback 默认关闭

文件：

```text
opentad/models/dense_heads/irregular_actionformer_head_v2.py
```

constructor 加参数：

```python
allow_center_fallback_inside_gt=False
```

保存：

```python
self.allow_center_fallback_inside_gt = bool(allow_center_fallback_inside_gt)
```

修改 `_build_candidate_mask`：

```python
missing_gt = ~candidate_mask.any(dim=0)

if missing_gt.any():
    if self.allow_center_fallback_inside_gt:
        candidate_mask[:, missing_gt] = inside_gt_seg[:, missing_gt]
        if self.debug_enabled:
            self._latest_debug_state["v2_missing_center_fallback_applied"] = int(
                missing_gt.sum().item()
            )
    else:
        if self.debug_enabled:
            self._latest_debug_state["v2_missing_center_fallback_gt_count"] = int(
                missing_gt.sum().item()
            )
        # no fallback; keep official-style fail-closed behavior
```

V3 继承 V2，也必须显式传入：

```python
allow_center_fallback_inside_gt=False
```

config checker 中加入：

```python
if head_type in {"IrregularActionFormerHeadV2", "IrregularActionFormerHeadV3"}:
    if cfg.get("dense_equivalent_claim_allowed", False):
        raise ConfigError(
            "HeadV2/V3 use soft/weighted assignment and are not dense-equivalent."
        )
    if cfg.get("allow_center_fallback_inside_gt", False):
        require_legacy_route_contract(...)
```

PASS gate：

```text
V2/V3 默认不允许 missing-center fallback。
打开 fallback 的 config 必须标记 legacy / analysis-only / not dense-equivalent。
```

---

### Patch D：selected-axis GT remap 后 drop GT 默认 FAIL

文件：

```text
opentad/datasets/transforms/end_to_end.py
```

constructor 增加：

```python
allow_drop_selected_axis_gt=False
```

保存：

```python
self.allow_drop_selected_axis_gt = bool(allow_drop_selected_axis_gt)
```

在 `_remap_gt_to_selected_axis` 后加：

```python
dropped = getattr(self, "_last_dropped_selected_axis_gt_segments", [])
if self.remap_gt_to_selected_axis and len(dropped) > 0:
    if not self.allow_drop_selected_axis_gt:
        raise RuntimeError(
            f"Selected-axis GT remap dropped {len(dropped)} GT segments. "
            "This changes the supervision contract. "
            "Set allow_drop_selected_axis_gt=True only for legacy diagnostics."
        )
```

同时强制校验 selected positions：

```python
def _validate_selected_positions(self, kept_positions, valid_len):
    kept_positions = np.asarray(kept_positions)
    if kept_positions.ndim != 1:
        raise ValueError("kept_positions must be 1D")
    if kept_positions.size == 0:
        raise ValueError("kept_positions is empty")
    if np.any(np.diff(kept_positions) <= 0):
        raise ValueError("kept_positions must be strictly increasing")
    if kept_positions[0] < 0 or kept_positions[-1] >= valid_len:
        raise ValueError(
            f"kept_positions out of native valid range: "
            f"[{kept_positions[0]}, {kept_positions[-1]}] vs valid_len={valid_len}"
        )
```

PASS gate：

```text
任何 selected-axis control 出现 dropped GT，默认 FAIL。
只有 legacy diagnostic config 可显式 allow。
```

---

### Patch E：把 `check_fail_closed_config.py` 纳入强制 preflight

文件：

```text
scripts / CI / launch scripts
```

所有 train/eval 前必须先跑：

```bash
python tools/check_fail_closed_config.py CONFIG --json-out WORKDIR/fail_closed_config.json
```

并在 launch script 中：

```bash
set -euo pipefail

python tools/check_fail_closed_config.py "$CONFIG" \
  --json-out "$WORK_DIR/fail_closed_config.json"

python tools/audit_sparse_head_assignment.py ...
```

PASS gate：

```text
checker exit != 0 => 不允许 train，不允许 eval，不允许 mAP claim。
```

---

## 5. 必须立即跑的实验：完整命令与 PASS/FAIL gate

下面按你给的 Stage 0–4 排序。

---

### Stage 0：Linux preflight

目标：只检查代码、config contract、测试，不跑长训练。

```bash
git checkout 06dce3d5955e9c2d0cf5232eace6b4224831900f

python -m py_compile \
  opentad/datasets/transforms/end_to_end.py \
  opentad/models/utils/temporal_grid.py \
  opentad/models/utils/post_processing/utils.py \
  opentad/models/detectors/irregular_actionformer.py \
  opentad/models/dense_heads/irregular_point_generator.py \
  opentad/models/dense_heads/irregular_actionformer_bridge_head.py \
  opentad/models/dense_heads/irregular_actionformer_head_v2.py \
  opentad/models/dense_heads/irregular_actionformer_head_v3.py \
  tools/audit_sparse_head_assignment.py \
  tools/check_fail_closed_config.py

python -m pytest -q tests/test_adapter_native_dense_headv2_contracts.py

mkdir -p work_dirs/preflight

python tools/check_fail_closed_config.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py \
  --json-out work_dirs/preflight/fail_closed_config.json
```

**PASS gate：**

```text
py_compile exit 0
pytest exit 0
check_fail_closed_config exit 0
corrected bridge configs:
  allow_legacy_full_cell_span=False
  allow_center_fallback_inside_gt=False
  dense_compat_mode="official_actionformer"
  range_mode="absolute"
  decode_scale_mode="level_stride"
  radius_scale_mode="level_stride"
```

**FAIL gate：**

```text
任何 legacy full_cell_span route 声称 dense-equivalent
任何 center fallback route 声称 dense-equivalent
任何 selected-axis config 缺 metadata contract
任何 HeadV2/V3 config 声称 official dense-equivalent
```

---

### Stage 1：same-batch assignment audit，native/selected 不混 batch

目标：先不看 mAP，只看同一 batch 上 dense official 与 bridge hard 是否 assignment/target/decode 等价。

```bash
mkdir -p work_dirs/audit/stage1_same_batch_bridge

python tools/audit_sparse_head_assignment.py \
  --configs \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_shortgate_n16r4.py \
  --split train \
  --num-batches 8 \
  --seed 20260705 \
  --device cuda \
  --out work_dirs/audit/stage1_same_batch_bridge
```

**PASS gate：**

```text
所有 hard bridge rows:
  hard_assignment_uses_build_candidate_mask == false
  hard_assignment_missing_center_fallback_applied == false

axis:
  gt_axis == proposal_axis
  postprocess_axis ∈ {proposal_axis, native}
  selected postprocess only allowed if followed by selected->native before NMS

dense selected-axis vs bridge selected-axis:
  positive_mask_diff_count == 0
  assigned_class_diff_count == 0
  encoded_target_max_abs_diff <= 1e-4
  decoded_target_max_abs_diff <= 1e-4
  per_level_positive_count_diff == 0
  gt_coverage_diff == 0

decode:
  decode_reconstruction_max_error <= 1e-4

GT remap:
  dropped_selected_axis_gt_count == 0
```

**FAIL gate：**

```text
任何 same-batch assignment diff 非零
任何 decode reconstruction error > 1e-4
任何 selected-axis GT 被 drop
任何 hard bridge fallback applied
```

如果 Stage 1 FAIL，不准跑 Stage 2/3 长训。

---

### Stage 2：official dense selected-axis sanity

目标：证明 selected-axis dataloader / metadata / postprocess 没有破坏官方 dense ActionFormer。

```bash
mkdir -p exps/stage2

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py \
  --work-dir exps/stage2/densehead_random_fixed_selected_axis_seed20260705

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py \
  exps/stage2/densehead_random_fixed_selected_axis_seed20260705/best.pth \
  --eval mAP
```

可选 uniform control：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py \
  --work-dir exps/stage2/densehead_uniform_selected_axis_seed20260705

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py \
  exps/stage2/densehead_uniform_selected_axis_seed20260705/best.pth \
  --eval mAP
```

**PASS gate：**

```text
random-fixed selected-axis densehead:
  Avg mAP >= 62.5
  ideally within ±0.5 of historical 63.12

uniform selected-axis densehead:
  Avg mAP >= 64.5
  ideally within ±0.5 of historical 65.09

No abnormal NMS/seconds warnings
No selected/native conversion fallback
No dropped selected-axis GT
```

**FAIL gate：**

```text
densehead selected-axis 自己掉到 40/42
=> 根因在 dataloader / GT remap / axis / postprocess，不在 irregular head。
```

---

### Stage 3：bridge selected/native controls

目标：隔离 bridge head 是否等价于 dense selected-axis control。

#### 3A. selected-axis bridge hard

```bash
mkdir -p exps/stage3

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py \
  --work-dir exps/stage3/bridge_hard_selected_axis_seed20260705

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py \
  exps/stage3/bridge_hard_selected_axis_seed20260705/best.pth \
  --eval mAP
```

#### 3B. native-axis bridge hard

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py \
  --work-dir exps/stage3/bridge_hard_native_axis_seed20260705

CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py \
  exps/stage3/bridge_hard_native_axis_seed20260705/best.pth \
  --eval mAP
```

**PASS gate：**

```text
bridge selected-axis vs dense selected-axis:
  Avg mAP gap <= 0.5
  assignment audit exact diff == 0

bridge native-axis corrected:
  Avg mAP should be close to random-fixed dense baseline
  target: >= 62.0
  warning band: 60.0–62.0
  fail: < 60.0
```

**FAIL interpretation：**

```text
dense selected-axis PASS, bridge selected-axis FAIL
=> bridge encode/decode/head contract bug

bridge selected-axis PASS, bridge native-axis FAIL
=> selected->native conversion / native temporal grid / NMS seconds bug

both FAIL
=> assignment/GT remap/config/data-loader still broken
```

---

### Stage 4：HeadV2/V3 soft assignment analysis

只能在 Stage 2/3 PASS 后运行。

```bash
mkdir -p work_dirs/audit/stage4_soft_heads

python tools/audit_sparse_head_assignment.py \
  --configs \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_selected_axis_control_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step0b_dense_points_soft_sym.py \
  configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step0b_dense_points_soft_sym_repaired.py \
  configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step3_irregular_points_soft_sym.py \
  configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step4_irregular_points_soft_asym.py \
  --split train \
  --num-batches 16 \
  --seed 20260705 \
  --device cuda \
  --out work_dirs/audit/stage4_soft_heads
```

随后再跑短训/全训。

**PASS gate：**

```text
V2/V3:
  missing_center_fallback_applied == 0
  selected/native axis contract PASS
  no dropped selected-axis GT
  positive count per level not collapsed
  short-GT coverage not worse than bridge hard by large margin
  decode reconstruction error <= 1e-4
  high-IoU oracle decode sanity PASS
```

**FAIL gate：**

```text
V2/V3 mAP 40 while bridge hard selected/native PASS
=> 根因在 soft assignment / log1p target / geometry modulation / V3 architecture，不是 sparse axis route。
```

---

## 6. 后续完整实验路线

### Phase A：先建立 dense-equivalent sparse adapter 闭环

必须按这个顺序：

```text
A0 official dense baseline from official OpenTAD
A1 densehead selected-axis control
A2 bridge hard selected-axis control
A3 bridge hard native-axis corrected control
A4 bridge hard random-fixed / uniform 50% reproduction
```

只有 A1–A4 接近历史 63/65，才说明：

```text
dataloader + selected metadata + assignment + regression + decode + NMS + seconds
```

已经干净。

---

### Phase B：再做 legacy/openrange/full-cell-span 消融

legacy route 只能叫：

```text
legacy irregular scale diagnostic
openrange diagnostic
full-cell-span ablation
```

不能叫：

```text
dense-equivalent
official-compatible
ActionFormer-compatible
```

报告中必须分开：

```text
official-compatible bridge hard
legacy full-cell-span/openrange
V2/V3 soft assignment
V3 geometry modulation
```

---

### Phase C：再引入 HeadV2/V3

每次只改一个因素：

```text
C1 bridge hard linear target
C2 + soft assignment only
C3 + log1p target only
C4 + irregular center/radius only
C5 + geometry modulation only
C6 + boundary aux only
```

每一步都要跑 same-batch audit 和 short/full mAP。

---

### Phase D：最后再讨论 sparse selection 本身

只有当 detector route 已经恢复到 63/65 附近，才能讨论：

```text
uniform 50%
random-fixed 50%
learned sparse
boundary-aware sparse
oracle sparse
```

否则所有 selection 结论都被 detector implementation bug 混杂。

---

## 7. 最终研究目标和可声明结论边界

### 当前 commit 可以声明

可以声明：

```text
1. 本 commit 已经显著加固 sparse/irregular route diagnostics。
2. bridge hard route 已经把 official scale 与 legacy full-cell-span 初步分离。
3. official_actionformer dense compat mode 已经强制 absolute range + level_stride scale，并拒绝 irregular center。
4. bridge hard assignment 不再走 V2/V3 的 missing-center fallback。
5. detector 主路径已经加入 selected/native axis contract，并倾向于在 NMS 前转 native。
6. check_fail_closed_config.py 可以阻止一部分 legacy route 被误报为 dense-equivalent。
```

---

### 当前 commit 不能声明

不能声明：

```text
1. HeadV3 fixed Avg mAP=40.20 证明 sparse/irregular ActionFormer 本身失败。
2. bridge hard linear openrange Avg mAP=42.44 是 official-compatible sparse detector 的真实性能。
3. 65/63 -> 40/42 是随机选帧自然退化。
4. selected-axis GT remap 已经完全正确。
5. selected/native postprocess 在所有脚本和 utility path 中已经全部 fail-closed。
6. V2/V3 是官方 ActionFormer 的 irregular 等价版本。
```

---

### 我对根因的当前排序

如果 Stage 2 dense selected-axis 也掉到 40/42：

```text
第一嫌疑：end_to_end.py 的 GT remap / selected-axis metadata / seconds conversion / NMS axis。
```

如果 Stage 2 PASS，但 Stage 3 bridge selected-axis 掉到 40/42：

```text
第一嫌疑：bridge encode/decode scale、point layout、assignment audit 未覆盖的 target contract。
```

如果 Stage 3 selected-axis PASS，但 native-axis 掉到 40/42：

```text
第一嫌疑：selected->native conversion、native temporal grid cell geometry、NMS 前 axis、duration clipping。
```

如果 Stage 3 全 PASS，但 HeadV3 仍 40.20：

```text
第一嫌疑：V2/V3 soft assignment、missing-center fallback、log1p regression、geometry modulation、boundary aux 混杂。
```

---

## 最终结论

**FAIL。**

这个 commit 不是无效 commit；它修对了不少关键防线，尤其是 bridge hard、official scale mode、dense compat center check、route_contract、assignment audit 方向。

但它还没有达到：

```text
sparse/irregular route 已经 official-compatible
40/42 mAP 可以解释为 selection 自然退化
HeadV3 可以作为 clean detector conclusion
```

的标准。

下一步不要先跑 HeadV3 大实验。先修上面的 strict patches，然后按 Stage 0 → Stage 1 → Stage 2 → Stage 3 证明 bridge hard selected/native control 回到 63/65 附近。只有 bridge hard 闭环恢复后，HeadV2/V3 才值得作为软分配/geometry 改造继续分析。

[1]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/commit/06dce3d5955e9c2d0cf5232eace6b4224831900f "Harden sparse head route diagnostics · yuzbo/OpenTAD_SparseHeadClean_20260702@06dce3d · GitHub"
[2]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/prior_generator/point_generator.py "OpenTAD/opentad/models/dense_heads/prior_generator/point_generator.py at main · sming256/OpenTAD · GitHub"
[3]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/anchor_free_head.py "OpenTAD/opentad/models/dense_heads/anchor_free_head.py at main · sming256/OpenTAD · GitHub"
[4]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/opentad/models/dense_heads/prior_generator/irregular_point_generator.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/prior_generator/irregular_point_generator.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[5]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/opentad/models/dense_heads/irregular_actionformer_bridge_head.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_bridge_head.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[6]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/tools/check_fail_closed_config.py "OpenTAD_SparseHeadClean_20260702/tools/check_fail_closed_config.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[7]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/opentad/models/detectors/irregular_actionformer.py "OpenTAD_SparseHeadClean_20260702/opentad/models/detectors/irregular_actionformer.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[8]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/tools/audit_sparse_head_assignment.py "OpenTAD_SparseHeadClean_20260702/tools/audit_sparse_head_assignment.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[9]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/opentad/models/dense_heads/irregular_actionformer_head_v2.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_head_v2.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[10]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/opentad/models/dense_heads/irregular_actionformer_head_v3.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_head_v3.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[11]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/opentad/datasets/transforms/end_to_end.py "OpenTAD_SparseHeadClean_20260702/opentad/datasets/transforms/end_to_end.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[12]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/opentad/models/utils/post_processing/utils.py "OpenTAD_SparseHeadClean_20260702/opentad/models/utils/post_processing/utils.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[13]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/06dce3d5955e9c2d0cf5232eace6b4224831900f/opentad/models/utils/temporal_grid.py "OpenTAD_SparseHeadClean_20260702/opentad/models/utils/temporal_grid.py at 06dce3d5955e9c2d0cf5232eace6b4224831900f · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
