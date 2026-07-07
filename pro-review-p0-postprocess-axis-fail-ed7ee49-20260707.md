## 0. 固定 commit 与审查边界

我实际看到的固定 commit 是：

`ed7ee49be0ceadb15d886e4f302d37bdf46bae20`

GitHub commit 页可见，提交短 SHA 为 `ed7ee49`，提交信息为 **“Record Stage2 gate brief preflight”**。本次审查基于 GitHub 可见源码与官方 OpenTAD 可见源码对照；我没有声称已经在本地跑通训练、pytest 或 Slurm。([GitHub][1])

**总判定：FAIL / HOLD。**

* **是否允许新的 long training：不允许跑 HeadV2/V3/native/sparse-head claim-oriented long training。**
* **是否允许 Stage2 dense selected-axis sanity long run：只有在先修复/确认下面 P0 postprocess axis 问题，并通过 Stage0/Stage1 gate 后，才允许作为 gate 型 sanity run。**
* **是否允许 dense-equivalent claim：不允许。**

核心原因不是 “HeadV3 还没调好”，而是当前仓库仍存在至少两个 P0 级别的坐标轴/postprocess 合同问题，足以解释高 IoU 从 65 级别掉到 40/42 甚至更低。当前 one-batch assignment audit 有价值，但它只证明了某一批 target construction 可以局部对齐；它没有覆盖完整 dataloader、postprocess、NMS、seconds conversion、multi-batch、多样本、padding、短视频、多 GT、多层 grid、full eval mAP 链路。

---

# 1. 对官方 OpenTAD dense 行为的权威对照

官方 OpenTAD 的 AnchorFreeHead / ActionFormer 风格 dense contract 非常明确：

1. PointGenerator 生成的是 dense temporal centers 与每层 stride/range；head 用这些 points 生成 classification 和 regression。官方 forward train 中，`reg_head` 输出经过 `ReLU(scale(...))`，然后用 `prior_generator(feat_list)` 生成 points。([GitHub][2])

2. 官方 target builder 的核心合同是：

   * 对每个 point 计算到 GT start/end 的 left/right regression target；
   * center sampling 用 `center_sample_radius * point_stride`；
   * regression range gate 用 point 的 regression range；
   * 多 GT 冲突用 shortest duration GT；
   * 正样本分类是 one-hot multi-label；
   * regression loss 在 decoded proposal 与 decoded GT 之间算 IoU/GIoU，而不是只在 encoded target 上做 L1。([GitHub][2])

3. 官方 decode 是：

```python
start = point_center - pred_left * point_stride
end   = point_center + pred_right * point_stride
```

这意味着 dense-equivalent 的任何 bridge / irregular head，只要声称等价，就必须让 center、left/right denom、center radius scale、regression range scale 与 official stride 语义一致。([GitHub][2])

4. 官方 postprocess 是先在模型内部坐标下做 score threshold / topk / batched NMS，然后才 `convert_to_seconds`。官方 `convert_to_seconds` 对 THUMOS sliding window 用的是 `segments * snippet_stride + window_start_frame + offset_frames` 再除以 fps，并 clip 到 duration。([GitHub][3])

所以你当前路线的正确性不是看 “assignment audit 是否 ok” 一项，而是必须同时满足：

```text
dataloader selected/native axis
GT remap axis
point center / stride / regression range
center sampling
positive assignment
regression encode/decode
proposal coordinate axis
NMS coordinate axis
seconds conversion axis
duration clipping
evaluation JSON axis
```

其中任一环节错轴，高 IoU 会先崩，@0.3 可能还能看起来“不完全坏”。

---

# 2. 逐模块审查结论

## 2.1 LoadFrames / GT remap / selected-axis GT：WARN，不是当前最坏点，但必须继续审

当前 `LoadFrames` 已经有 `remap_gt_to_selected_axis=True`、`allow_drop_selected_axis_gt=False`、`target_len`、`selection_unit` 等参数，并实现了 `_remap_gt_to_selected_axis`。这段会把 GT start/end 映射到 selected axis，clip 到 `[0, kept_positions.size]`，如果 remap 后 segment collapse 且不允许 drop，会 raise。这个 fail-closed 比早期版本安全。([GitHub][4])

`LoadFrames` 也会设置 irregular metadata，包括 `irregular_selected_positions`、`irregular_selected_valid_len`、`irregular_native_axis`、`irregular_gt_axis`、`irregular_proposal_axis`、`irregular_postprocess_axis`、`irregular_axis_contract` 等。`Collect` 的默认 meta keys 也包含这些字段，因此理论上 dataloader 不应再静默丢掉轴信息。([GitHub][4])

但这里仍有三个风险：

**风险 1：GT remap no-drop 不等于 GT geometry 无损。**
即使没有 drop，selected-axis remap 仍可能改变 GT endpoint 的 fractional/rounding 位置。对于 @0.7，高 IoU 对 boundary 偏移极敏感。必须新增 roundtrip audit：

```text
native GT seconds
→ selected-axis GT
→ proposal decode selected
→ native dense frame
→ seconds
```

然后逐 GT 计算 remap 前后 IoU、start error、end error，而不是只看 drop count。

**风险 2：selected-axis dense sanity 不是 official dense baseline。**
`input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py` 的语义是 “50% equal-interval input, GT/proposals on selected axis, native-axis post-processing, dense ActionFormer head/prior-generator contract”。这不是原始官方 dense 768 轴，而是把 384 个 selected positions 重建成 dense-like selected axis。它可以作为 sanity gate，但不能直接称为 official OpenTAD dense。配置确实使用 official `ActionFormerHead` / `PointGenerator` 路线，但仍依赖 selected-axis GT remap 与 native-axis postprocess 正确。([GitHub][5])

**风险 3：selected count 与 short video 处理必须进 mAP summary。**
如果短视频 target_len 不等于 384，或者 valid_len/padding 在 eval 中没有一致处理，selected-axis dense sanity 会被污染。当前 metadata 能承载这些信息，但还需要真实 dataloader multi-batch audit。

结论：LoadFrames/Collect 方向是对的，但还不能证明 65→40 不是 GT remap/axis 造成的。

---

## 2.2 temporal_grid / IrregularPointGeneratorV2：WARN，selected-axis dense compat 有进步，但 native irregular 不能 claim dense-equivalent

`IrregularPointGeneratorV2` 的 `dense_compat_mode="official_actionformer"` 会强制设置 `range_mode="absolute"`、`decode_scale_mode="level_stride"`、`radius_scale_mode="level_stride"`，这是正确修复方向。它还会对 temporal grid center 做 official dense compat assertion：在有效 mask 下，grid centers 必须等于 `arange(T) * stride`。([GitHub][6])

但 dense-equivalence 不只等于 center 对齐。官方 dense point layout 本质是：

```text
center = arange(T) * stride
reg_min / reg_max = official absolute regression range
decode stride = stride
center radius stride = stride
```

V2 里 point fields 扩展成：

```text
center
reg_min
reg_max
decode_left
decode_right
range_scale
radius_scale
```

因此必须新增一个强制检查：

```python
if dense_compat_mode == "official_actionformer":
    assert center == arange(T) * stride
    assert reg_min/reg_max == official regression_range[level]
    assert decode_left == stride
    assert decode_right == stride
    assert radius_scale == stride
    assert range_scale is irrelevant or equals stride where used
```

现在代码检查了 center，但我没有看到同等强度地把 decode/radius/range 全字段锁死到 official stride 语义。V2 仍支持 `full_cell_span`、`half_cell_span`、`point_range`、`point_radius` 等 scale 模式；这些在 native irregular 下可能合理，但都不能 claim official dense-equivalent。([GitHub][6])

**判断：selected-axis official_actionformer mode 可以进入 Stage1/Stage2 sanity；native irregular V2 不能直接 claim dense-equivalent。**

---

## 2.3 IrregularActionFormerBridgeHead：WARN/FAIL 混合；hard selected-axis 有希望，native/soft/legacy 不能混

BridgeHead 的默认参数已经朝 fail-closed 修正：`assignment_mode="hard"`、`regression_mode="symmetric_linear"`、`allow_legacy_full_cell_span=False`、`allow_center_fallback_inside_gt=False`。这比旧版 legacy full-cell/openrange route 明显更安全。([GitHub][7])

Bridge hard target builder 也基本复刻了 official 思路：计算 center、left/right、center sampling、regression range、shortest GT conflict、one-hot cls，然后编码 regression target。([GitHub][7])

但仍有几个必须严查的点：

### 2.3.1 regression denom 必须显式等于 official stride

Bridge 的 `_encode_regression_targets` 支持 `symmetric_linear` 和 `asymmetric_log1p`。`symmetric_linear` 使用 `_scale_base(..., self.reg_denom_mode, ...)` 作为 denom；decode 时再乘同一个 denom。([GitHub][7])

如果 `reg_denom_mode="left_right_mean"` 在 selected-axis official compat 下恰好等于 stride，可以认为等价；但这必须由 audit 强制证明。否则 official 是 `left/right / point_stride`，bridge 是 `left/right / custom_scale`，即使 one-batch decoded IoU=1，也可能因为 scale 分布影响 regression training、loss magnitude、score calibration。

**Patch gate：**

```text
dense_compat_mode=official_actionformer
AND bridge.regression_mode=symmetric_linear
AND bridge.reg_denom_mode=level_stride
AND center_radius_scale=level_stride
AND decode_left/right=level_stride
```

如果你坚持保留 `left_right_mean`，必须在每个 level、每个 valid point 上证明：

```python
max_abs(left_right_mean - level_stride) == 0
```

否则不准 dense-equivalent claim。

### 2.3.2 soft assignment 不是 official ActionFormer contract

Bridge 代码里有 soft assignment top-k / cost / normalized weights 路径。它本质上不是官方 dense ActionFormer 的 hard shortest-GT target contract。([GitHub][7])

这不是“小改动”，而是监督定义变化：

```text
official: 一个 point 由 shortest valid GT 决定，hard one-hot cls + one regression target
soft: 一个 point/GT 可能形成 weighted target，正样本分布更宽，score 与 localization quality 更难校准
```

高 IoU 弱非常符合这种 failure mode：@0.3 还能留住一些粗 proposal，@0.7 因 endpoint regression / positive boundary sharpness 不足崩掉。

**结论：Bridge hard selected-axis 可以作为恢复 65 的中间路线；HeadV2/V3 soft route 暂时不能当 dense-equivalent。**

---

## 2.4 HeadV2/V3：FAIL for dense-equivalence；只能作为后续实验变量

你已经有 nogeometry=39.32、reggate=38.32，说明 geometry modulation 和 regression range gate 不是主因。这个判断我同意，但更严格地说：

**HeadV2/V3 的问题不在于某一个开关，而在于它已经偏离 official ActionFormer 的监督合同。**

可能偏离包括：

```text
soft top-k assignment
weighted labels / weighted regression target
log1p regression
geometry modulation
boundary auxiliary
custom score calibration
different positive distribution across levels
different loss normalization
different proposal score interpretation
```

其中 soft top-k + log1p regression 对高 IoU 最危险。因为 TAD 高 IoU 需要 endpoint regression 很尖锐，不能只靠 action coverage；一旦 positive target 被扩散到 interior，或者 regression target 被 log 缩放后 loss 对 boundary error 不敏感，高 IoU 首先崩。Bridge hard linear openrange 只能到 42.44，也说明单纯“恢复多层 positives”不够，完整 inference/postprocess/axis 或 scale contract 仍可能错。

**结论：HeadV2/V3 现在只能放在 Stage5 decomposition；不能再用它们解释 sparse-head 的科学价值。**

---

## 2.5 selected/native/seconds postprocess：P0 FAIL，当前最可疑根因

这是本次审查最严重部分。

### P0-1：SingleStageDetector 仍调用 `convert_to_seconds(..., source_axis=auto)`

目标仓库的 `SingleStageDetector.post_processing` 仍然在 NMS 后直接调用：

```python
segments = convert_to_seconds(segments, metas[i])
```

没有传入 `source_axis`。与此同时，目标仓库的 `convert_to_seconds` 已经扩展出 `source_axis="auto"`，并且在检测到 selected-axis metadata、`irregular_native_axis` 为 false、且 `allow_auto_axis=False` 时会拒绝 auto 推断，要求显式传 `source_axis='selected'` 或 `'native'`。([GitHub][8])

这意味着：

```text
Stage2 official dense selected-axis sanity
    ↓
official ActionFormerHead
    ↓
SingleStageDetector.post_processing
    ↓
convert_to_seconds(auto)
```

这条链路在理论上不是 fail-safe 的。

如果它运行时报错，说明 sanity 根本无法完成。
如果它没有报错，反而更危险：可能说明 selected-axis metadata 没有进入 metas，导致 native-axis postprocess claim 没被审计。

**这个点必须先修，再跑任何 dense selected-axis sanity。**

### P0-2：IrregularActionFormer `_segments_to_seconds` 可见代码只处理 selected，不处理 native

目标仓库的 `IrregularActionFormer` 中，axis contract 默认允许 `selected -> native`，也要求 selected-axis postprocess NMS 显式 opt-in。这个设计方向是正确的。([GitHub][9])

但可见源码中 `_segments_to_seconds` 只看到：

```python
if source_axis == "selected":
    return convert_to_seconds(... source_axis="selected", strict=True)
```

紧接着就是下一个函数定义，未见 `source_axis == "native"` 分支。与此同时，`IrregularActionFormer.post_processing` 会先把 proposal axis 转到 postprocess axis，再 NMS，最后调用：

```python
segments = self._segments_to_seconds(segments, metas[i], postprocess_axis)
```

如果 `postprocess_axis="native"`，这条路径按可见代码会返回 `None` 或进入未定义行为。([GitHub][9])

这可以非常直接解释：

```text
assignment audit ok
train loss不崩
但 eval proposals / seconds / NMS / JSON 出错
→ high-IoU 崩
```

**这是 P0 blocker。必须 patch。**

### P0-3：selected-axis NMS 必须禁止用于 high-IoU claim

当前 IrregularActionFormer 有 `allow_selected_axis_postprocess_nms` 防线；这很好。但从研究 claim 角度，任何 selected-axis NMS 都不能支持 high-IoU localization claim。原因：

```text
selected-axis distance != physical time distance
uniform selected axis 可以近似
random/native irregular selected axis 完全不是 metric time
NMS IoU 在 selected axis 上算，会扭曲 proposal overlap
```

所以最终策略必须是：

```text
GT/proposal training axis: selected or native,按阶段定义
NMS axis: native dense frame or seconds
evaluation axis: seconds
```

不应允许：

```text
NMS axis = selected
```

除非该实验被标注为 diagnostic-only / legacy，不允许论文主表。

---

## 2.6 score calibration / pre_nms_topk / threshold / duration clipping：WARN

官方 SingleStage postprocess 默认 `pre_nms_thresh=0.001`、`pre_nms_topk=2000`，先 threshold/topk，再 batched NMS，然后 convert seconds。([GitHub][3])

你已经补过 single-class irregular postprocess 的 `pre_nms_thresh / pre_nms_topk`，这是必要修复。但仍需检查：

```text
official dense selected-axis
bridge selected-axis
native bridge
HeadV2/V3
```

是否使用完全相同的：

```python
pre_nms_thresh
pre_nms_topk
iou_threshold
min_score
duration clipping
multi_class / single_class flatten logic
score sigmoid placement
visibility_rescore / quality score multiplication
```

否则 mAP 差异可能来自 postprocess calibration，不是 sparse geometry。

---

## 2.7 config legacy route 风险：WARN

`check_fail_closed_config.py` 已经有不少正确防线：拦截 raw prediction/cache shortcut；识别 V2/V3/bridge；禁止 V2/V3 soft/default/fallback claim dense-equivalent；bridge legacy full-cell/fallback 需要显式声明；`allow_drop_selected_axis_gt=True` 在 selected remap 下必须显式 diagnostic marker。([GitHub][10])

但 scanner 目前看起来更像“静态合规检查”，不能保证运行时 postprocess axis 正确。例如：

```text
SingleStageDetector 没传 source_axis
IrregularActionFormer native seconds 分支缺失
NMS 是否真的在 native axis
convert_to_seconds 是否 double conversion
```

这些必须加入 runtime gate，而不是只靠 config scanner。

---

## 2.8 tests：WARN；当前测试偏 contract/static，不足以证明 mAP 链路

仓库中已经有 `test_fail_closed_static_gates.py`、`test_strict_fail_closed_contracts.py`、`test_audit_sparse_head_assignment_contracts.py`、`test_stage2_dense_gate_summary.py` 等测试。tools 里也有 `audit_sparse_head_assignment.py`、`summarize_stage2_dense_gate.py`、`verify_bridge_dense_equivalence.py` 等脚本。([GitHub][11])

但这些测试不能替代：

```text
真实 dataloader batch
多 batch
多样本 hash
多 GT
无 GT 视频
短视频
padding mask
多层 point grid
proposal decode
NMS
seconds conversion
evaluation JSON
```

尤其 `audit_sparse_head_assignment.py` 里的 “official” builder 是当前仓库本地 reimplementation，不是直接调用官方 OpenTAD pinned upstream prepare_targets。它确实实现了 official-like target builder 并比较 positive mask/class/encoded/decoded target，但这仍可能复刻了当前仓库自己的假设。([GitHub][12])

---

# 3. Top 10 blockers，按严重程度排序

## P0-1. SingleStageDetector postprocess 仍未显式 source_axis

**位置**：`opentad/models/detectors/single_stage_detector.py` post_processing。
**问题**：调用 `convert_to_seconds(segments, metas[i])`，没有 `source_axis`。而目标仓库的 `convert_to_seconds(auto)` 在 irregular selected metadata 存在时会 fail-closed。([GitHub][8])
**后果**：official dense selected-axis sanity 可能直接失败，或者 metadata 丢失导致假通过。
**优先级**：最高。先修这个。

## P0-2. IrregularActionFormer `_segments_to_seconds` 缺 native 分支

**位置**：`opentad/models/detectors/irregular_actionformer.py`。
**问题**：可见代码只处理 `source_axis == "selected"`，但 postprocess 默认常需要 native axis。([GitHub][9])
**后果**：native-axis postprocess/eval 可能返回 None 或错转换，高 IoU 必崩。
**优先级**：最高。

## P0-3. selected-axis NMS 不能用于 high-IoU claim

**问题**：selected-axis 不是真实时间 metric；在 irregular selected 下 NMS IoU 会错。
**后果**：@0.7 先崩。
**修复**：NMS 前强制 proposal 转 native dense frame或 seconds。

## P0-4. Stage2 dense selected-axis sanity 尚未通过

在 official dense selected-axis sanity 恢复 near 65、random fixed near 63 之前，不能解释 sparse-head collapse。当前任何 HeadV3/bridge 长训结果都只能视为 contaminated evidence。

## P1-5. 当前 “official_vs_current_assignment_diff” 不是官方上游 prepare_targets 的直接调用

当前 audit 是本地 reimplementation official target builder。它有价值，但不够权威。必须加入 pinned official OpenTAD wrapper 或 vendored official file hash comparison。([GitHub][12])

## P1-6. dense_compat_mode 只 assert center 不够

必须 assert decode scale、radius scale、regression range 全部等价 official stride/range。否则 “center 对齐” 不能推出 dense-equivalent。

## P1-7. Bridge regression denom / center radius scale 未强制 official stride

Bridge 的 `reg_denom_mode`、`center_radius_scale` 只有在严格等于 level stride 时才等价 official。否则 loss scale 与 positive distribution 会变。([GitHub][7])

## P1-8. HeadV2/V3 soft assignment 本质偏离 official dense supervision

soft top-k / weighted target / log1p regression / geometry modulation 都不应进入 dense-equivalence 路径。它们应该只在 Stage5 做 controlled decomposition。

## P1-9. GT remap no-drop 不能证明 endpoint fidelity

必须新增 GT remap roundtrip IoU/start-error/end-error audit。否则 high-IoU 崩溃仍可能来自 remap bias。

## P2-10. tests 还不够真实

当前测试偏 static/contract。必须增加真实 dataloader same-batch audit，覆盖多 batch、多样本、多 GT、多层、mask/padding、postprocess axis。

---

# 4. 为什么 65 → 40/42 仍然发生：root-cause hypothesis tree

## H0：当前仓库的 official dense selected-axis 本身已经崩

**优先级：最高。**

如果 official dense selected-axis sanity 不能恢复：

```text
uniform/equal 50% near 65
random fixed near 63
```

那就不用继续查 HeadV3。根因应优先在：

```text
GT remap
metadata
SingleStage source_axis
convert_to_seconds
NMS axis
duration clipping
evaluation JSON
```

而不是 sparse head。

**证伪方法：**

运行 Stage2 official dense selected-axis sanity。通过 gate：

```text
uniform/equal selected-axis dense Avg-mAP ≈ 65
random fixed selected-axis dense Avg-mAP ≈ 63
mAP@0.7 接近历史水平
```

失败则冻结所有 sparse-head 长训。

---

## H1：random fixed selected-axis dense 明显低于 uniform selected-axis dense

如果 uniform selected-axis 能接近 65，但 random fixed dense 低很多，说明问题可能是 selected-axis GT remap 与 random lattice 的非均匀性：selected axis 上训练看似 dense，但映射回 native 后局部时间尺度扭曲严重。

**证伪方法：**

对比：

```text
uniform/equal selected-axis dense
random fixed selected-axis dense
same postprocess native/seconds
same detector
same selected count
```

如果 random fixed 仍 near 63，则 random lattice 本身不是 collapse 主因。

---

## H2：bridge selected-axis 不能复现 dense selected-axis

如果 official dense selected-axis 过了，但 bridge hard selected-axis 仍在 40/42，则根因在 bridge：

```text
point fields
regression denom
center sampling scale
regression range scale
shortest-GT conflict
loss normalizer
score calibration
decode
postprocess
```

**证伪方法：**

Bridge hard selected-axis 与 official dense selected-axis：

```text
same dataloader
same selected-axis GT
same point centers/stride/range
same assignment
same decoded target
same postprocess native/seconds
```

mAP 应在 1 point Avg-mAP 内接近。否则 bridge 不是 dense-equivalent。

---

## H3：native irregular geometry/postprocess 引入不可忽略边界误差

如果 selected-axis bridge 过了，但 native irregular bridge 崩，说明问题在：

```text
native temporal grid / cell width
selected→native proposal conversion
native NMS
seconds conversion
clipping
irregular cell geometry
```

**证伪方法：**

同一个 selected ledger：

```text
selected-axis bridge
native-axis bridge
native NMS
seconds eval
```

比较每个 proposal 的 start/end error、NMS 前后 proposal IoU 排名、mAP@0.7 drop。

---

## H4：HeadV2/V3 soft top-k / log1p / geometry / boundary aux 贡献 collapse

你已有结果说明 geometry/reggate 不是唯一主因，但还没分解：

```text
hard vs soft assignment
linear vs log1p regression
topk1 binary vs soft weighted
geometry on/off
boundary auxiliary on/off
score calibration
```

**证伪方法：**

在 dense selected-axis 已恢复、bridge selected-axis 已恢复之后，逐项打开 V2/V3 组件。只要某项打开导致 @0.7 掉 3+，就不能放入主方法。

---

## H5：sparse sampling 本身缺失边界证据

这个假设现在还不能优先。只有当 H0-H4 都通过，才可以说性能下降来自 sparse evidence，而不是代码合同错误。

**证伪方法：**

在完全 correct 的 dense/bridge/native chain 下，比较 ledger geometry：

```text
boundary recall@r
endpoint coverage@r
p95 boundary hole
action interior hole
proposal mAP@0.7
```

如果 boundary evidence 缺失与 @0.7 drop 强相关，才说明 sparse evidence 是主因。

---

# 5. Concrete code changes

下面是建议直接 patch 的核心代码结构。

---

## 5.1 selected/native/seconds fail-closed helper

新增文件：

```text
opentad/models/utils/axis_conversion.py
```

核心原则：

```text
不允许 auto
不允许 silent selected/native inference
不允许 selected-axis NMS claim
不允许 double convert
```

代码框架：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch

Axis = Literal["selected", "native", "seconds"]


@dataclass(frozen=True)
class SegmentPacket:
    segments: torch.Tensor
    axis: Axis
    meta: dict[str, Any]
    stage: str


def _require_axis(axis: str) -> Axis:
    if axis not in {"selected", "native", "seconds"}:
        raise ValueError(f"Invalid segment axis: {axis}")
    return axis  # type: ignore[return-value]


def get_axis_contract(meta: dict[str, Any]) -> dict[str, str]:
    contract = meta.get("irregular_axis_contract", {})
    return {
        "gt_axis": meta.get("irregular_gt_axis", contract.get("gt_axis", "native")),
        "proposal_axis": meta.get("irregular_proposal_axis", contract.get("proposal_axis", "native")),
        "postprocess_axis": meta.get("irregular_postprocess_axis", contract.get("postprocess_axis", "native")),
        "nms_axis": meta.get("irregular_nms_axis", contract.get("nms_axis", "native")),
    }


def require_selected_meta(meta: dict[str, Any]) -> tuple[torch.Tensor, float]:
    if "irregular_selected_positions" not in meta:
        raise ValueError("Missing irregular_selected_positions for selected-axis conversion")
    if "irregular_selected_valid_len" not in meta:
        raise ValueError("Missing irregular_selected_valid_len for selected-axis conversion")

    pos = torch.as_tensor(meta["irregular_selected_positions"], dtype=torch.float32)
    valid_len = float(meta["irregular_selected_valid_len"])

    if pos.ndim != 1 or pos.numel() == 0:
        raise ValueError("irregular_selected_positions must be a non-empty 1D array")
    if not torch.all(pos[1:] > pos[:-1]):
        raise ValueError("irregular_selected_positions must be strictly increasing")
    if valid_len <= 0:
        raise ValueError("irregular_selected_valid_len must be positive")

    return pos, valid_len


def selected_to_native(segments: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """Map selected-axis continuous coordinates to native dense frame coordinates.

    Convention:
      selected coordinate i maps to selected_positions[i].
      selected coordinate len(selected_positions) maps to valid_len.
    """
    pos, valid_len = require_selected_meta(meta)
    pos = pos.to(device=segments.device, dtype=segments.dtype)

    xp = torch.arange(pos.numel() + 1, device=segments.device, dtype=segments.dtype)
    fp = torch.cat([pos, torch.tensor([valid_len], device=segments.device, dtype=segments.dtype)])

    x = segments.clamp(min=0, max=float(pos.numel()))
    flat = x.reshape(-1)

    idx = torch.searchsorted(xp, flat, right=False)
    idx = idx.clamp(min=1, max=xp.numel() - 1)

    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]

    t = (flat - x0) / (x1 - x0).clamp(min=1e-6)
    y = y0 + t * (y1 - y0)

    return y.reshape_as(segments)


def convert_packet_axis(packet: SegmentPacket, target_axis: Axis) -> SegmentPacket:
    source_axis = _require_axis(packet.axis)
    target_axis = _require_axis(target_axis)

    if source_axis == target_axis:
        return packet

    if source_axis == "seconds" or target_axis == "seconds":
        raise ValueError(
            "seconds conversion must go through explicit convert_to_seconds; "
            f"got {source_axis}->{target_axis} at {packet.stage}"
        )

    if source_axis == "selected" and target_axis == "native":
        return SegmentPacket(
            segments=selected_to_native(packet.segments, packet.meta),
            axis="native",
            meta=packet.meta,
            stage=f"{packet.stage}:selected_to_native",
        )

    raise ValueError(f"Unsupported axis conversion {source_axis}->{target_axis} at {packet.stage}")
```

---

## 5.2 Patch SingleStageDetector postprocess：强制显式 source_axis

当前官方 dense selected-axis sanity 最大问题是 SingleStageDetector 不知道 selected/native axis。建议 patch：

```python
# opentad/models/detectors/single_stage_detector.py

from opentad.models.utils.axis_conversion import (
    SegmentPacket,
    convert_packet_axis,
    get_axis_contract,
)
from opentad.models.utils.post_processing import convert_to_seconds


def _resolve_proposal_axis(meta: dict, default: str = "native") -> str:
    contract = get_axis_contract(meta)
    return contract.get("proposal_axis", meta.get("irregular_proposal_axis", default))


def _resolve_nms_axis(meta: dict, post_cfg: dict, proposal_axis: str) -> str:
    # Default: native NMS whenever selected metadata exists.
    configured = post_cfg.get("nms_axis", None)
    if configured is not None:
        return configured

    if proposal_axis == "selected":
        return "native"
    return proposal_axis


def _assert_no_selected_axis_nms(nms_axis: str, post_cfg: dict) -> None:
    if nms_axis == "selected" and not post_cfg.get("allow_selected_axis_nms", False):
        raise ValueError(
            "selected-axis NMS is forbidden by default because selected axis is not "
            "metric time. Set allow_selected_axis_nms=True only for diagnostic legacy runs."
        )


# inside post_processing loop, before NMS
proposal_axis = _resolve_proposal_axis(metas[i])
nms_axis = _resolve_nms_axis(metas[i], post_cfg=self.post_processing_cfg, proposal_axis=proposal_axis)
_assert_no_selected_axis_nms(nms_axis, self.post_processing_cfg)

packet = SegmentPacket(
    segments=segments,
    axis=proposal_axis,
    meta=metas[i],
    stage="single_stage_pre_nms",
)

if proposal_axis != nms_axis:
    packet = convert_packet_axis(packet, target_axis=nms_axis)

segments = packet.segments

# run batched_nms on nms-axis segments
segments, scores, labels = batched_nms(segments, scores, labels, **nms_cfg)

# convert to seconds with explicit source_axis
segments = convert_to_seconds(
    segments,
    metas[i],
    source_axis=nms_axis,
    strict=True,
)
```

同时 config 必须显式写：

```python
post_processing = dict(
    pre_nms_thresh=0.001,
    pre_nms_topk=2000,
    nms_axis="native",
    source_axis="selected",
    forbid_auto_axis=True,
    allow_selected_axis_nms=False,
)
```

---

## 5.3 Patch IrregularActionFormer `_segments_to_seconds`

当前可见代码缺 native 分支。应改成：

```python
# opentad/models/detectors/irregular_actionformer.py

def _segments_to_seconds(self, segments, meta, source_axis: str):
    if source_axis not in {"selected", "native"}:
        raise ValueError(f"Unsupported source_axis for seconds conversion: {source_axis}")

    if source_axis == "selected":
        self._require_selected_axis_meta(meta)
        return convert_to_seconds(
            segments,
            meta,
            source_axis="selected",
            strict=True,
        )

    if source_axis == "native":
        return convert_to_seconds(
            segments,
            meta,
            source_axis="native",
            strict=True,
        )

    raise AssertionError("unreachable")
```

并在 postprocess 前增加：

```python
if postprocess_axis == "selected" and not self.allow_selected_axis_postprocess_nms:
    raise ValueError(
        "selected-axis NMS is forbidden for non-diagnostic runs; "
        "convert proposals to native before NMS."
    )
```

---

## 5.4 official OpenTAD target builder wrapper，不要再只靠本地 reimplementation

新增：

```text
tools/audit_official_same_batch_assignment.py
```

关键点：必须 pin official OpenTAD root 和 SHA。不要用当前仓库里的 “official-like” builder 当最终证据。

代码骨架：

```python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch


def git_sha(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OfficialAnchorFreeTargetBuilder:
    """Pinned official OpenTAD prepare_targets reference.

    This wrapper must not import the current repo's opentad package.
    Prefer running it in a subprocess with PYTHONPATH=official_root.
    """

    def __init__(self, official_root: Path, expected_sha: str):
        actual = git_sha(official_root)
        if actual != expected_sha:
            raise RuntimeError(
                f"Official OpenTAD SHA mismatch: actual={actual}, expected={expected_sha}"
            )

        self.official_root = official_root
        self.anchor_free_path = (
            official_root
            / "opentad"
            / "models"
            / "dense_heads"
            / "anchor_free_head.py"
        )
        self.file_hash = sha256_file(self.anchor_free_path)

    def build_by_subprocess(self, payload_path: Path, output_path: Path) -> None:
        helper = Path(__file__).with_name("_run_official_prepare_targets.py")
        cmd = [
            sys.executable,
            str(helper),
            "--official-root", str(self.official_root),
            "--payload", str(payload_path),
            "--output", str(output_path),
        ]
        subprocess.check_call(cmd)


def sample_fingerprint(sample: dict) -> str:
    keys = [
        "video_name",
        "irregular_gt_axis",
        "irregular_proposal_axis",
        "irregular_postprocess_axis",
        "irregular_selected_valid_len",
    ]
    blob = {k: sample.get(k) for k in keys}
    if "irregular_selected_positions" in sample:
        pos = sample["irregular_selected_positions"]
        blob["selected_positions_sha1"] = hashlib.sha1(
            torch.as_tensor(pos).cpu().numpy().tobytes()
        ).hexdigest()
    return hashlib.sha1(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--official-sha", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-jsonl", required=True)
    args = parser.parse_args()

    # 1. build current dataloader from current repo config
    # 2. for each batch:
    #    - collect metas, masks, gt_segments, gt_labels
    #    - collect current points
    #    - serialize official payload
    #    - run official prepare_targets in isolated subprocess
    #    - run current prepare_targets
    #    - compare positive mask, labels, encoded targets, decoded targets
    # 3. write per-sample JSONL and summary JSON

    # Pseudocode placeholders:
    summary = {
        "ok": False,
        "stage": "stage1_same_batch_official_assignment",
        "config": args.config,
        "official_root": args.official_root,
        "official_sha": args.official_sha,
        "num_batches": args.num_batches,
        "num_samples": 0,
        "max_positive_mask_diff": None,
        "max_class_diff": None,
        "max_encoded_target_abs_diff": None,
        "max_decoded_target_abs_diff": None,
        "num_gt_dropped": None,
        "num_padding_positive": None,
        "failures": ["IMPLEMENT_DATALOADER_LOOP"],
    }

    Path(args.out_json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

Subprocess helper `_run_official_prepare_targets.py` 应在 `PYTHONPATH=official_root` 下运行，避免当前 repo 的 `opentad` 污染。

---

## 5.5 real dataloader same-batch audit 输出 schema

`outputs/gates/stage1_same_batch_assignment_summary.json`：

```json
{
  "schema_version": "sparse_head_gate_summary_v1",
  "commit": "ed7ee49be0ceadb15d886e4f302d37bdf46bae20",
  "official_opentad_commit": "<PINNED_SHA>",
  "stage": "stage1_same_batch_assignment_decode",
  "config": "configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py",
  "split": "train",
  "num_batches": 8,
  "num_samples": 32,
  "axis_contract_counts": {
    "gt=selected,proposal=selected,postprocess=native": 32
  },
  "sample_hashes_sha1": "<sha1>",
  "metrics": {
    "max_positive_mask_diff": 0,
    "max_class_target_diff": 0,
    "max_encoded_target_abs_diff": 0.0,
    "max_decoded_target_abs_diff": 0.0,
    "max_target_decode_iou_error": 0.0,
    "num_padding_positives": 0,
    "num_dropped_gt": 0,
    "max_gt_roundtrip_start_error_frames": 0.5,
    "max_gt_roundtrip_end_error_frames": 0.5
  },
  "coverage": {
    "num_no_gt_samples": 2,
    "num_multi_gt_samples": 12,
    "num_short_video_samples": 3,
    "num_levels_with_positive": 5
  },
  "ok": true,
  "failures": []
}
```

Gate 不是 “脚本成功退出”，而是：

```text
max_positive_mask_diff == 0
max_class_target_diff == 0
max_encoded_target_abs_diff <= 1e-6
max_decoded_target_abs_diff <= 1e-5
num_padding_positives == 0
num_dropped_gt == 0
multi-GT samples covered
multi-level positives covered
```

---

## 5.6 configs 必须显式声明 route contract

在所有 dense selected-axis / bridge / native / V2/V3 config 中强制加入：

```python
route_contract = dict(
    route_name="stage2_official_dense_selected_axis_sanity",
    compatibility="official_dense_selected_axis",
    official_opentad_reference_sha="<PINNED_OFFICIAL_SHA>",
    dense_equivalent_claim_allowed=True,

    axis_contract=dict(
        input_axis="selected",
        gt_axis="selected",
        proposal_axis="selected",
        nms_axis="native",
        postprocess_axis="native",
        eval_axis="seconds",
    ),

    selected_axis_contract=dict(
        selected_positions_unit="native_dense_frame_index",
        selected_positions_strictly_increasing=True,
        selected_count_expected=384,
        allow_short_video_selected_count_deviation=False,
        allow_drop_selected_axis_gt=False,
    ),

    postprocess_contract=dict(
        forbid_auto_axis=True,
        forbid_selected_axis_nms=True,
        require_explicit_source_axis=True,
        require_duration_clipping=True,
    ),

    legacy_contract=dict(
        allow_legacy_full_cell_span=False,
        allow_center_fallback_inside_gt=False,
        allow_missing_center_fallback=False,
        allow_openrange_dense_equivalent_claim=False,
    ),
)
```

对于 native irregular：

```python
route_contract = dict(
    route_name="stage4_native_irregular_bridge",
    compatibility="native_irregular_diagnostic",
    dense_equivalent_claim_allowed=False,
    axis_contract=dict(
        gt_axis="native",
        proposal_axis="native",
        nms_axis="native",
        postprocess_axis="native",
        eval_axis="seconds",
    ),
)
```

对于 HeadV2/V3：

```python
route_contract = dict(
    route_name="stage5_headv3_soft_assignment_ablation",
    compatibility="non_official_soft_assignment",
    dense_equivalent_claim_allowed=False,
    claim_scope="diagnostic_only_until_stage5_ablation_passes",
)
```

---

# 6. Experiment roadmap

## Stage 0：Linux preflight / fail-closed scanner / config load / py_compile / pytest

### 运行配置/命令

```bash
git rev-parse HEAD

python -m py_compile \
  tools/check_fail_closed_config.py \
  tools/audit_sparse_head_assignment.py \
  tools/summarize_stage2_dense_gate.py \
  tools/verify_bridge_dense_equivalence.py \
  opentad/models/detectors/single_stage_detector.py \
  opentad/models/detectors/irregular_actionformer.py \
  opentad/models/utils/post_processing.py

python tools/check_fail_closed_config.py \
  configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
  configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py \
  --json-out outputs/gates/stage0_fail_closed_scan.json

pytest -q \
  tests/test_fail_closed_static_gates.py \
  tests/test_strict_fail_closed_contracts.py \
  tests/test_audit_sparse_head_assignment_contracts.py \
  tests/test_stage2_dense_gate_summary.py
```

### Gate

```text
commit == ed7ee49be0ceadb15d886e4f302d37bdf46bae20
scanner no P0/P1 violations
py_compile pass
pytest pass
SingleStage source_axis patch present
IrregularActionFormer native seconds patch present
```

### 失败后诊断

任何失败都不允许进入 Stage2 long sanity。

---

## Stage 1：same-batch official assignment + decode audit

### 必要新增脚本

```text
tools/audit_official_same_batch_assignment.py
tools/_run_official_prepare_targets.py
```

### 命令

```bash
python tools/audit_official_same_batch_assignment.py \
  --config configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py \
  --official-root /path/to/official/OpenTAD \
  --official-sha <PINNED_OFFICIAL_SHA> \
  --split train \
  --num-batches 8 \
  --out-json outputs/gates/stage1_uniform_selected_axis_assignment_summary.json \
  --out-jsonl outputs/gates/stage1_uniform_selected_axis_assignment_samples.jsonl

python tools/audit_official_same_batch_assignment.py \
  --config configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py \
  --official-root /path/to/official/OpenTAD \
  --official-sha <PINNED_OFFICIAL_SHA> \
  --split train \
  --num-batches 8 \
  --out-json outputs/gates/stage1_random_selected_axis_assignment_summary.json \
  --out-jsonl outputs/gates/stage1_random_selected_axis_assignment_samples.jsonl
```

### Gate

```text
positive/class/encoded/decoded diff = 0 or <= numerical tolerance
target decode IoU error = 0
padding positives = 0
dropped GT = 0
multi-GT/multi-level samples included
sample hashes stable
```

### 失败后下一步

* positive mask diff：查 center sampling / regression range / selected GT remap。
* encoded diff：查 stride/denom/range scale。
* decoded diff：查 decode scale。
* GT dropped：查 selected-axis mapping。
* padding positives：查 valid_mask / FPN mask / temporal_grid valid_len。

---

## Stage 2：official dense selected-axis sanity

### 运行配置

```text
configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_adapter_densehead_selected_axis_control_n16r4.py
```

### Gate

```text
uniform/equal-interval 50% selected-axis dense Avg-mAP ≈ 65
random fixed selected-axis dense Avg-mAP ≈ 63
mAP@0.7 不应出现 14/20 级别崩溃
```

### 失败后诊断优先级

1. SingleStage explicit source_axis 是否真正生效；
2. NMS axis 是否 native；
3. convert_to_seconds 是否 selected→native→seconds，且没有 double conversion；
4. GT remap roundtrip error；
5. valid_len/padding；
6. eval JSON duration clipping；
7. score/topk/NMS config 与 official 是否完全一致。

如果 Stage2 失败，**不要查 HeadV3**。

---

## Stage 3：bridge selected-axis vs dense selected-axis

### 新增配置

```text
configs/adatad/thumos/input_uniform_fixed_50pct_bridge_selected_axis_officialcompat_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_bridge_selected_axis_officialcompat_n16r4.py
```

必须锁定：

```python
head=dict(
    type="IrregularActionFormerBridgeHead",
    assignment_mode="hard",
    regression_mode="symmetric_linear",
    reg_denom_mode="level_stride",
    center_radius_scale="level_stride",
    allow_legacy_full_cell_span=False,
    allow_center_fallback_inside_gt=False,
    allow_selected_axis_postprocess_nms=False,
)

prior_generator=dict(
    type="IrregularPointGeneratorV2",
    dense_compat_mode="official_actionformer",
)
```

### Gate

```text
same-batch assignment exact
selected-axis bridge Avg-mAP within <= 1.0 of selected-axis dense
mAP@0.7 within <= 1.5 of selected-axis dense
```

### 失败后诊断

* decoded proposal diff；
* per-level positive count diff；
* score calibration diff；
* regression loss magnitude diff；
* postprocess proposal topk diff；
* NMS proposal survival diff。

---

## Stage 4：native irregular bridge

### 新增配置

```text
configs/adatad/thumos/input_uniform_fixed_50pct_bridge_native_irregular_n16r4.py
configs/adatad/thumos/input_random_fixed_50pct_bridge_native_irregular_n16r4.py
```

### Gate

```text
selected-axis bridge passes first
native bridge Avg-mAP drop <= 2-3 points
mAP@0.7 不出现灾难性掉落
NMS axis = native
eval axis = seconds
```

### 失败后诊断

* native temporal grid center/cell width；
* proposal start/end before NMS；
* selected→native mapping error；
* NMS IoU matrix on native vs selected；
* clipping；
* proposal duration distribution；
* high-score proposal alignment with GT boundary。

---

## Stage 5：HeadV2/V3 decomposition

### 实验矩阵

```text
A0: bridge hard symmetric linear officialcompat
A1: hard + log1p regression
A2: soft topk1 binary
A3: soft topkK weighted
A4: geometry modulation on
A5: boundary auxiliary on
A6: visibility/quality score rescore on
A7: postprocess calibration variants
```

### Gate

每个组件必须报告：

```text
Avg-mAP delta
mAP@0.7 delta
positive count distribution
endpoint error distribution
proposal duration distribution
score calibration
NMS survival
```

如果某组件造成：

```text
Avg-mAP -3 或 mAP@0.7 -3
```

则不得进主线。

---

# 7. Final target：怎样才算恢复可信性能，怎样才算 sparse-head route 有价值

## 7.1 恢复可信性能的最低标准

必须按顺序达到：

```text
Stage2 official dense selected-axis:
  uniform/equal 50% ≈ 65 Avg-mAP
  random fixed ≈ 63 Avg-mAP

Stage3 bridge selected-axis:
  within <= 1 Avg-mAP of Stage2 dense
  high-IoU within <= 1.5

Stage4 native irregular bridge:
  no catastrophic @0.7 drop
  native/seconds postprocess verified

Stage5 HeadV2/V3:
  each non-official change has isolated delta
```

否则，当前 40/42 不能被解释成 sparse sampling 难，而应视为 code-contract failure。

## 7.2 sparse-head route 有价值的证明标准

只有在 dense/bridge/native correctness 全部通过后，才进入真正研究问题：

```text
same detector
same budget
same postprocess
same axis
same mAP evaluator
different selection strategy
```

然后证明：

```text
uniform 384
random fixed 384
oracle-boundary diagnostic
PAction learned 384
GAS-VT 384
Stage2 detector-aware
Stage3 joint
```

在 mAP 与 geometry 上同时成立：

```text
boundary recall@r ↑
endpoint coverage@r ↑
boundary-region p95 hole ↓
selected-frame boundary distance CDF 左移
high-IoU mAP ↑
```

如果 mAP 提升但 boundary geometry 不提升，不能讲 boundary-aware sparse head。
如果 geometry 提升但 mAP 不提升，说明 detector/head/postprocess 没把 boundary evidence 用起来。
如果 Stage2/Stage3 与 p_action-only 几乎同分布，不能讲 detector-aware。

---

# 8. 最终 Verdict

**Verdict：FAIL / HOLD。**

当前 commit 可见，工程防线比早期更强，但仍不允许进入新的 sparse-head long training，也不允许 dense-equivalent claim。

最关键的两个 P0 修复是：

1. **SingleStageDetector postprocess 必须显式 source_axis / nms_axis。**
2. **IrregularActionFormer `_segments_to_seconds` 必须支持 native axis，并禁止 selected-axis NMS 进入主实验。**

在这两个问题修复并通过 Stage0/Stage1 后，才允许跑 Stage2 official dense selected-axis sanity。若 Stage2 不能恢复 near 65/63，后续所有 HeadV3、native irregular、soft assignment、geometry modulation 训练都没有解释价值。

[1]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/commit/ed7ee49be0ceadb15d886e4f302d37bdf46bae20 "Record Stage2 gate brief preflight · yuzbo/OpenTAD_SparseHeadClean_20260702@ed7ee49 · GitHub"
[2]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/anchor_free_head.py "OpenTAD/opentad/models/dense_heads/anchor_free_head.py at main · sming256/OpenTAD · GitHub"
[3]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/detectors/single_stage.py "OpenTAD/opentad/models/detectors/single_stage.py at main · sming256/OpenTAD · GitHub"
[4]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/ed7ee49be0ceadb15d886e4f302d37bdf46bae20/opentad/datasets/transforms/end_to_end.py "OpenTAD_SparseHeadClean_20260702/opentad/datasets/transforms/end_to_end.py at ed7ee49be0ceadb15d886e4f302d37bdf46bae20 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[5]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/ed7ee49be0ceadb15d886e4f302d37bdf46bae20/configs/adatad/thumos/input_uniform_fixed_50pct_official_dense_selected_axis_sanity_n16r4.py "raw.githubusercontent.com"
[6]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/ed7ee49be0ceadb15d886e4f302d37bdf46bae20/opentad/models/dense_heads/prior_generator/irregular_point_generator.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/prior_generator/irregular_point_generator.py at ed7ee49be0ceadb15d886e4f302d37bdf46bae20 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[7]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/ed7ee49be0ceadb15d886e4f302d37bdf46bae20/opentad/models/dense_heads/irregular_actionformer_bridge_head.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_bridge_head.py at ed7ee49be0ceadb15d886e4f302d37bdf46bae20 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[8]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/ed7ee49be0ceadb15d886e4f302d37bdf46bae20/opentad/models/detectors/single_stage.py "OpenTAD_SparseHeadClean_20260702/opentad/models/detectors/single_stage.py at ed7ee49be0ceadb15d886e4f302d37bdf46bae20 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[9]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/ed7ee49be0ceadb15d886e4f302d37bdf46bae20/opentad/models/detectors/irregular_actionformer.py "OpenTAD_SparseHeadClean_20260702/opentad/models/detectors/irregular_actionformer.py at ed7ee49be0ceadb15d886e4f302d37bdf46bae20 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[10]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/ed7ee49be0ceadb15d886e4f302d37bdf46bae20/tools/check_fail_closed_config.py "OpenTAD_SparseHeadClean_20260702/tools/check_fail_closed_config.py at ed7ee49be0ceadb15d886e4f302d37bdf46bae20 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[11]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/tree/ed7ee49be0ceadb15d886e4f302d37bdf46bae20/tools "OpenTAD_SparseHeadClean_20260702/tools at ed7ee49be0ceadb15d886e4f302d37bdf46bae20 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[12]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/ed7ee49be0ceadb15d886e4f302d37bdf46bae20/tools/audit_sparse_head_assignment.py "OpenTAD_SparseHeadClean_20260702/tools/audit_sparse_head_assignment.py at ed7ee49be0ceadb15d886e4f302d37bdf46bae20 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
