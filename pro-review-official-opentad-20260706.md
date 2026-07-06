## 1. Executive Summary

**结论：当前 route 是“部分实现正确，但研究/监督合同存在高风险偏离”，不能认为已经正确实现了官方 AdaTAD dense baseline 的 sparse/irregular 等价版本。** 当前仓库的 `_base_/models/actionformer.py` 看起来仍接近官方 dense ActionFormer 配置；但一进入 `input_random_fixed_50pct_adapter_irregular_*`，它已经不是官方 dense AdaTAD 的简单稀疏输入版本，而是同时改变了 **采样轴、GT 坐标轴、projection/neck、point generator、assignment、regression encode/decode、loss normalizer 和 temporal geometry contract**。官方 dense ActionFormer 的权威参考必须来自 OpenTAD 官方实现，而不能来自当前仓库内的 dense control。官方 base 使用 `ActionFormer + Conv1DTransformerProj + FPNIdentity + ActionFormerHead + PointGenerator`，并固定 dense assignment/regression 逻辑；官方 AdaTAD THUMOS VideoMAE-S 768 配置使用 dense 768 window、train random trunc、val/test sliding window、VideoMAE adapter 后接 ActionFormer projection/head。([GitHub][1])

**性能从等间隔 50% Average-mAP≈65 崩到 40–42，不是单一 traceback/OOM/NaN 训练失败，而是“native-axis irregular sparse detection 合同未闭合”。** 你给的结果和仓库 `root-cause-notes.md` 一致：HeadV3 fixed 40.20、nogeometry 39.32、reggate 38.32、openrange 42.44，说明 geometry modulation 和单独加 regression-range gate 都不是主因；openrange 恢复多层正样本后仍只有 42.44，说明 regression range collapse 是已证实问题，但不足以解释全部崩溃。([GitHub][2])

**主根因排序：**

1. **BLOCKER：supervision/assignment contract 偏离官方 dense ActionFormer。** 官方是 hard positive、center sampling、regression range、shortest-GT conflict resolution、linear distance/stride regression；当前 V2/V3 是 per-GT soft top-k、soft cls target、soft reg weight、pos_mass normalizer、asymmetric log regression。这个差异足以系统性损害高 IoU。([GitHub][3])
2. **BLOCKER：IrregularPointGeneratorV2 的 range scale 语义错误或至少不等价于官方 dense prior。** 官方 range 是按 level stride 定义；当前 `range_mode="hard"` 将 regression_range 乘以局部 `cell_left + cell_right`，导致随机 sparse native grid 下 duration prior 与 level assignment 崩坏。代码确实把 `point_scale = cell_left + cell_right` 用于 `reg_min/reg_max`，V2 又把 left/right scale 作为 decode scale 单独传入。 
3. **HIGH：native-axis sparse route 比 selected-axis/equal-interval dense route 难得多。** 等间隔 50% 或 selected-axis remap 会把稀疏输入重新伪装成均匀序列，检测器只需在 384 个等距 index 上定位；native-axis random irregular 要在真实 768 坐标上从不规则观测点回归未观测边界，尤其高 IoU 最脆弱。
4. **HIGH：projection/backbone/neck 仍大量继承 uniform sequence 假设。** 官方 ConvTransformer 的卷积、stride pooling/local window attention 都按 token index 运算，不理解真实时间间隔；当前即使传了 temporal grid，若主干特征混合仍按 slot-index，则 head 的 native geometry 只能后补，无法恢复边界精度。官方 Transformer block 明确用 stride/pool 在序列位置上下采样，属于 uniform-grid 结构。([GitHub][4])
5. **MEDIUM：boundary auxiliary/quality calibration 是次要但会影响 @0.7。** 当前 commit 的 HeadV3 配置已把 `boundary_loss_weight=0.0`，这避免训练无用 aux，但也去掉了高 IoU 边界补偿；它不能解释 20+ mAP 崩溃，但能解释 openrange 后 @0.7 仍弱。([GitHub][5])

---

## 2. 官方 OpenTAD dense AdaTAD 实现摘要

### 2.1 官方输入 pipeline

官方 AdaTAD THUMOS VideoMAE-S 768 config：

* base 是官方 THUMOS dataset config + 官方 `actionformer.py` model config；
* `window_size=768`，VideoMAE chunk 数 `768 / 16 = 48`；
* train pipeline 是 `LoadFrames(method="random_trunc", trunc_len=768, trunc_thresh=0.75, crop_ratio=[0.9,1.0])`；
* val/test 是 `LoadFrames(method="sliding_window")`；
* backbone 是 `VisionTransformerAdapter`，`total_frames=768`，adapter 插在 12 层；
* VideoMAE 输出经过 `Reduce -> Rearrange -> Interpolate(size=768)`，变成 `B,C,T=768` 的 dense temporal feature，再进入 ActionFormer projection/head。([GitHub][6])

官方 README 给的 AdaTAD THUMOS VideoMAE-S 768×1 160 结果是 Avg-mAP 69.03，mAP@0.7 为 48.27；这说明官方 dense route 是强 baseline，而不是当前 sparse 仓库内任意 dense control 可以替代的参考。([GitHub][7])

### 2.2 官方 GT 坐标轴

官方 dense route 的 GT 坐标是 **dense feature/window axis**。train 时 random trunc 后 GT 被裁剪/平移到当前 dense window；val/test sliding window 后 proposals 在 window feature 坐标中 decode，再由 detector post-processing 转为秒级坐标并做 NMS/eval。官方 `SingleStageDetector` 在 inference 后把 proposals、scores、labels 收集，经过 `batched_nms`，最后 `convert_to_seconds`。([GitHub][8])

### 2.3 官方 dense points：center / stride / regression range

官方 `PointGenerator` 对每个 FPN level 生成：

```python
center = arange(T_level) * stride
point = [center, reg_range_min, reg_range_max, stride]
```

也就是说，官方 `regression_range` 本身是 **dense feature coordinate 下按 level stride 设计的 duration prior**，不是按每个 sample 的局部 cell span 动态缩放。([GitHub][9])

官方 base ActionFormer config 的 prior 是：

```python
strides=[1, 2, 4, 8, 16, 32]
regression_range=[(0,4), (4,8), (8,16), (16,32), (32,64), (64,10000)]
```

并使用 `center_sample="radius"`、`center_sample_radius=1.5`。([GitHub][1])

### 2.4 官方 assignment：center sampling / range gate / shortest-GT

官方 `AnchorFreeHead.prepare_targets` 的关键逻辑是：

1. 对每个 point 计算到每个 GT 的 left/right distance；
2. `inside_gt_seg_mask = reg_targets.min(-1) > 0`；
3. 如果启用 center sampling，则用 `center_sample_radius * point_stride` 得到 GT center 附近有效区域；
4. 用 `max(left,right)` 通过 `regression_range` gate；
5. 对一个 point 能匹配多个 GT 的情况，用 **最短 GT** 解决冲突；
6. cls target 是 hard one-hot / multi-hot，reg target 是 `(left,right) / stride`。([GitHub][3])

这是当前 irregular route 必须首先复现的监督合同。没有这个 parity，不能把性能差距归因到 sparse sampling 本身。

### 2.5 官方 regression encode/decode 和 NMS

官方 regression decode 是线性的：

```python
start = center - reg_left * stride
end   = center + reg_right * stride
```

loss 用 decoded segment 和 GT segment 计算 DIOU；classification 对 valid mask 下所有点做 focal loss，regression 只对 positive points 做，normalizer 是 EMA positive count。([GitHub][3])

官方 inference 先 flatten multi-level scores，按 `pre_nms_thresh` 和 `pre_nms_topk` 取候选，再按 `post_processing.nms` 做 batched Soft-NMS / NMS，最后转秒级坐标。官方 AdaTAD config 使用 Soft-NMS `sigma=0.7`、`max_seg_num=2000`、`multiclass=True`、`voting_thresh=0.7`。([GitHub][8])

### 2.6 官方 AdaTAD adapter + ActionFormer head 接口

官方 AdaTAD adapter 的接口是：

```text
raw frames
 -> VideoMAE-S VisionTransformerAdapter
 -> Reduce/Rearrange/Interpolate to B,C,T=768
 -> Conv1DTransformerProj
 -> FPNIdentity
 -> ActionFormerHead
```

官方 `FPNIdentity` 对各 level 做 norm/identity，不引入 native irregular geometry；mask 跟随 feature level 传递。([GitHub][10])

---

## 3. 当前仓库相对官方的差异表

| 项目                                      | 官方 OpenTAD dense                                                                         | 当前仓库情况                                                                                                                                                                                          | 审查结论                                                              |
| --------------------------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `configs/_base_/models/actionformer.py` | `ActionFormer + Conv1DTransformerProj + FPNIdentity + ActionFormerHead + PointGenerator` | 当前同路径文件从公开 commit 看与官方 base 结构一致                                                                                                                                                                | 这部分可作为“待审查 dense 对象”，但仍需 checksum；不能因为当前仓库有它就把它当权威。([GitHub][1])  |
| 官方 AdaTAD config                        | dense 768，train random_trunc，val/test sliding_window，selected/dense feature axis         | 当前 `input_random_fixed_50pct_adapter.py` 改为 384 selected tokens from 768 source                                                                                                                 | 已经不是官方 dense AdaTAD，只是 selected-axis sparse control。([GitHub][6]) |
| irregular base                          | 官方无 native irregular route                                                               | 当前 `input_random_fixed_50pct_adapter_irregular_actionformer_base.py` 用 `random_fixed_subsample`，`remap_gt_to_selected_axis=False`，model type 改为 `IrregularActionFormer`，projection/neck/head 全换 | 这是新模型路线，不是 dense baseline。([GitHub][11])                          |
| ActionFormerHead / AnchorFreeHead       | 官方 hard dense assignment/linear regression                                               | 当前 irregular V2/V3 新增 soft top-k/log regression/geometry/boundary                                                                                                                               | 监督合同偏离；高概率主因。([GitHub][3])                                        |
| PointGenerator                          | 官方 `[center, reg_min, reg_max, stride]`，range 不按 sample local gap 变                      | 当前 `IrregularPointGeneratorV2` 用 `cell_left+cell_right` 做 point_scale，并用它缩放 range                                                                                                               | BLOCKER：duration prior 语义偏移。                                      |
| Projection/neck                         | 官方 uniform token ConvTransformer/FPN                                                     | 当前换 `IrregularConvTransformerProj` / `GridAwareConv1DTransformerProj` / `IrregularFPN` / `GridAwareFPNIdentity`                                                                                 | 必须证明 geometry-aware，不然仍是 slot-index model。([GitHub][11])          |
| Inference/post-processing               | 官方 proposals 是 dense window 坐标，转 seconds                                                 | 当前 native-axis sparse proposals 必须保持 dense/native axis；selected-axis proposals 必须 inverse remap                                                                                                 | 这是高风险转换点；需要 proposal dump + round-trip tests。                     |

---

## 4. 当前 irregular route 相对官方 dense 的差异表

| 维度                       | 官方 dense                                  | 等间隔 50% selected-axis                                | 当前 random_fixed native irregular                                 |
| ------------------------ | ----------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------- |
| 输入采样                     | dense 768 连续窗口                            | 从 768 取等间隔 384，通常 remap 到 selected axis              | 从 768 随机取 384，保留 native/dense axis                               |
| GT 坐标轴                   | dense window axis                         | selected-axis，动作被压到 384 坐标                           | native-axis，GT 仍在 768 坐标                                         |
| temporal grid            | implicit uniform `center=i*stride`        | implicit uniform 384                                 | explicit `center=kept_positions`，有 `cell_left/right`             |
| backbone/adapter         | 假设 dense uniform chunks                   | 仍可近似 uniform                                         | 输入 token 间隔不均，VideoMAE/adapter 本身未充分理解不规则时间                      |
| projection               | Conv/attention over uniform token index   | 合理                                                   | 若只是带 geometry side-channel，仍不足以修复 token mixing                   |
| FPN 下采样                  | stride 2 index pooling                    | 合理                                                   | 必须合并真实 cell span；不能只按 index pair 下采样                             |
| point center/scale/range | center=index×stride，range fixed per level | center=selected index                                | center=native position；当前 range 被 local cell span 动态缩放           |
| assignment               | hard center/range/shortest-GT             | hard dense-like 可成立                                  | V2/V3 soft top-k；bridge hard 但 range/scale 仍有问题                  |
| regression encode/decode | linear distance / stride                  | linear selected-axis                                 | V2/V3 log1p/expm1；bridge linear 但 denominator 用 point/cell scale |
| loss normalizer          | positive count EMA                        | positive count EMA                                   | V2/V3 pos_mass；bridge hard 接近 dense                              |
| inference/NMS            | dense proposal -> seconds                 | selected proposal 需 inverse remap 或 selected seconds | native proposal -> seconds；边界落在未观测 gap 时高 IoU 难                  |

---

## 5. 逐文件审查结果

### `configs/_base_/models/actionformer.py` — LOW / fairness-critical

**位置：** model dict。
**差异：** 当前 commit 中该文件看起来与官方 base 基本一致：`ActionFormer`、`Conv1DTransformerProj`、`FPNIdentity`、`ActionFormerHead`、`PointGenerator`、同样 strides/ranges/loss。([GitHub][12])
**问题：** 这只能说明当前仓库里有一个类似官方的 dense config，不能证明当前 dense code 未改。
**修复：** 加 checksum/AST parity test：当前 `anchor_free_head.py`、`actionformer_head.py`、`point_generator.py`、postprocess 与官方 commit 做 byte/semantic diff；dense baseline 只引用官方仓库结果。

### `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py` — MEDIUM

**位置：** dataset/model/optimizer/post_processing。
**差异：** 官方 config 是 dense 768 AdaTAD route；当前 sparse 50% config 继承后把 window 改成 384、source 改成 768。官方 dense AdaTAD 不做 random_fixed_subsample。([GitHub][6])
**问题：** 若把 `input_random_fixed_50pct_adapter.py` 的 65 当“官方 dense”，会混淆 dense 768、selected-axis 50%、native-axis irregular 三个不同问题。
**修复：** 实验矩阵中必须保留 `official_dense_768`、`selected_axis_50`、`native_axis_irregular_50` 三个独立基线。

### `configs/adatad/thumos/input_random_fixed_50pct_adapter.py` — MEDIUM

**位置：** `LoadFrames(method="random_fixed_subsample", keep_ratio=0.5, target_len=384, source_len=768)`。
**差异：** 这是 50% sparse selected-axis adapter baseline，不是官方 dense AdaTAD。
**问题：** 它可以达到 63–65，说明“只丢 50% token + selected-axis dense detector”并不崩；但不能说明 native-axis sparse detector 也应同等容易。
**修复：** 显式命名为 `selected_axis_random_fixed_50pct_adapter`，不要叫 dense official baseline。

### `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_actionformer_base.py` — BLOCKER for attribution

**位置：** train/val/test pipeline 和 model。
**差异：** 该 config 明确 `remap_gt_to_selected_axis=False`，并把 model 改成 `IrregularActionFormer`，projection 改成 `IrregularConvTransformerProj`，neck 改成 `IrregularFPN`，head 改成 `IrregularActionFormerHead`。([GitHub][11])
**问题：** 它同时改变了输入采样、GT axis、detector type、projection、neck、head，不能用来做单一 attribution。
**修复：** 拆成四个 control：`selected-axis official head`、`native-axis official head with explicit inverse mapping`、`native-axis bridge head only`、`native-axis bridge + grid-aware proj/neck`。

### `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x.py` — HIGH

**位置：** `GridAwareConv1DTransformerProj`、`GridAwareFPNIdentity`、`IrregularActionFormerHeadV3`。
**差异：** 当前 commit 的 HeadV3 使用 V2/V3 soft assignment 路线，`soft_assign_topk=9`，geometry encoder，`IrregularPointGeneratorV2`，并且 `boundary_loss_weight=0.0`。([GitHub][5])
**问题：** 该 head 与官方 dense ActionFormer 的 hard assignment/linear regression 不等价；高 IoU 弱是预期风险。
**修复：** 暂停把 V3 soft route 当主线，先用 bridge hard dense-like route 建 parity。

### `configs/...bridge_hard_linear_n16r4.py` — HIGH

**位置：** `IrregularActionFormerBridgeHead(assignment_mode="hard", regression_mode="symmetric_linear")`。
**差异：** 它比 V3 更接近官方 dense，但 prior 仍是 `IrregularPointGeneratorV2(range_mode="hard")`。([GitHub][13])
**问题：** hard assignment 本身正确方向，但 `regression_range` 乘 local full cell span 后导致 level positive collapse。你的 same-batch audit `[111,21,0,0,0,0]` 就是这个问题。`root-cause-notes.md` 也记录 current hard 只在前两层有正样本。([GitHub][2])
**修复：** bridge hard 应该默认 `range_mode="absolute"` 或 `range_scale="level_stride"`，而不是 local cell span。

### `configs/...bridge_hard_linear_openrange_n16r4.py` — MEDIUM diagnostic only

**位置：** regression_range 全部 `(0,10000)`。([GitHub][14])
**差异：** 它移除了官方 dense 的 duration prior。
**问题：** openrange 是好诊断，不是好最终模型。它证明 range gate 会杀掉多层 positives；但全开 range 会破坏 level specialization，产生更多重复、粗糙、难校准 proposals。
**修复：** 保留为 diagnostic；不要作为 paper route。下一步优先 abs/level range。

### `configs/...bridge_hard_linear_absrange_n16r4.py` — HIGH recommended but incomplete

**位置：** `prior_generator.range_mode="absolute"`。([GitHub][15])
**差异：** 修正了“range 随 local gap 动态缩放”的问题。
**问题：** 你给的 audit `[9,9,23,29,11,0]` 说明 naive official absolute ranges 太窄/分布不匹配，召回不足。
**修复：** 用 expanded absolute bands，例如 `(0,8),(4,16),(8,32),(16,64),(32,128),(64,10000)`，并让 center radius 使用 level stride 或 robust local median，不用极端 random gap。

### `opentad/datasets/transforms/end_to_end.py` — HIGH

**位置：** `LoadFrames.__call__` 的 `random_fixed_subsample` 分支、`_remap_gt_to_selected_axis`、`_set_irregular_axis_meta`。
**差异：** 当前 loader 同时服务 selected-axis 和 native-axis。
**问题：** 这本身可行，但风险极高：

* `remap_gt_to_selected_axis=True` 时，GT/proposal/postprocess 必须走 selected-axis contract；
* `False` 时，GT/proposal 必须始终保持 dense/native axis；
* 同一 repo 中两类 config 混跑，极容易出现 double remap 或 missing inverse remap。
  **修复：** 增加每 batch 的 axis contract assert：`gt_axis`, `proposal_axis`, `postprocess_axis` 必须一致；eval 时禁止出现 selected-axis proposal 直接按 dense seconds 转换。

### `opentad/models/utils/temporal_grid.py` — HIGH

**位置：** `build_temporal_grid`、`downsample_temporal_grid`。
**差异：** 官方无 explicit irregular grid；当前使用 `center/cell_left/cell_right/level_scale`。
**问题：** 当前文档记录 `level_scale = 0.5*(cell_left+cell_right)`，而 `IrregularPointGeneratorV2` 用 `cell_left+cell_right` 做 `point_scale`；range、decode、radius 三种 scale 混在一起，会系统性改变正样本分配。([GitHub][2])
**修复：** 明确拆分：`decode_scale_left/right`、`range_scale`、`radius_scale`、`cell_span`。不要让一个 `point_scale` 同时承担四个角色。

### `opentad/models/detectors/irregular_actionformer.py` — HIGH

**位置：** temporal grid 构建、projection/neck/head 调用、forward_test postprocess。
**差异：** 官方 `ActionFormer` 不需要 native grid；当前 detector 必须在多级 feature 中保持 grid 对齐。
**问题：** 若任何 level 的 mask/grid/features 长度不一致，或者 postprocess 把 native proposal 当 selected proposal，@0.7 会直接崩。
**修复：** 增加 forward_test dump：每个 video 前 20 个 proposal 的 `{center, left, right, start,end, seconds_start,end, axis}`；与 selected-axis/equal-interval round trip 对齐。

### `opentad/models/projections/irregular_actionformer_proj.py` — BLOCKER until proven

**位置：** irregular attention / conv projection。
**差异：** 官方 ConvTransformer 是 uniform token-index model；当前如果只是把 geometry feature 拼进去，但主混合仍按 index/local window，就不是真正 metric-aware。
**问题：** random irregular tokens 的邻接关系不等于真实时间邻接，尤其大 gap 附近边界证据缺失。
**修复：** 至少实现一种：metric relative position bias、time rasterizer、或 geometry-aware convolution。见第 8 节 patch。

### `opentad/models/necks/irregular_fpn.py` — HIGH

**位置：** 下采样/多级 grid 生成。
**差异：** 官方 FPN levels 是 uniform stride；当前必须合并真实 cell span。
**问题：** 如果 level k 的 center/cell/range 只是每隔两个 token取一个，duration prior 和 decode 都会错。
**修复：** FPN 下采样应合并 `[center-cell_left, center+cell_right]` 的 support interval，重新计算 center/left/right，而不是直接 index stride。

### `opentad/models/dense_heads/anchor_free_head.py` — HIGH fairness guard

**位置：** `prepare_targets`、`get_refined_proposals`、`losses`。
**差异：** 官方 dense 权威在这里；当前仓库同名文件若被改，则 dense control 不再可信。
**问题：** 任何 focal loss、normalizer、range gate、decode、NMS 修改都能污染 baseline。
**修复：** 与官方 `AnchorFreeHead` 做 strict diff；dense baseline 必须跑 official code 或 byte-identical copy。

### `opentad/models/dense_heads/irregular_actionformer_head_v2.py` — BLOCKER

**位置：** `_build_candidate_mask`、`_build_assignment_weights`、`_prepare_targets_soft`、`get_refined_proposals`。
**差异：** 当前 V2 默认 `use_regress_range=False`，soft top-k per GT，soft cls target，soft reg weight，log1p/expm1 regression。([GitHub][16])
**问题：** 这不是官方 dense assignment。soft positives 扩散会让分类和回归目标变软、边界回归不锐利，典型表现就是 low IoU 尚可、高 IoU 断崖。
**修复：** 不要继续在 V2 soft path 上调参；先加 `assignment_mode="dense_hard"` parity path。

### `opentad/models/dense_heads/irregular_actionformer_head_v3.py` — HIGH

**位置：** geometry encoder、boundary aux、继承 V2 target。
**差异：** V3 主要是在 V2 上加 geometry modulation 和 boundary aux。
**问题：** `nogeometry=39.32` vs fixed 40.20 已经说明 geometry 不是一阶主因；V3 继承 V2 supervision 问题，所以继续调 geometry 不会解决 65→42。([GitHub][2])
**修复：** V3 暂停为主线；只在 bridge hard parity 恢复后，再作为 geometry/boundary add-on。

### `opentad/models/dense_heads/irregular_actionformer_bridge_head.py` — HIGH

**位置：** `_prepare_targets_hard`、`_encode_regression_targets`、`get_refined_proposals`、`losses`。
**差异：** bridge hard 已接近官方：center sampling、range gate、shortest GT、linear regression。
**问题：** 它仍把 `point_scale`、left/right cell scale、range scale、center radius scale 混用；这会导致 openrange 后仍 high-IoU 弱。
**修复：** bridge head 应改成显式 fields：`range_min/max` 用 absolute/level scale；decode denominator 用 `decode_scale_left/right`；center radius 用 `radius_scale`。

### `opentad/models/dense_heads/prior_generator/irregular_point_generator.py` — BLOCKER

**位置：** `IrregularPointGeneratorV2.__call__`。
**差异：** 官方 PointGenerator 的 range 是 level prior；当前 hard mode 的 range 是 `official_range * (cell_left+cell_right)`。
**问题：** 这是 regression range collapse 的直接代码原因。
**修复：** 默认改为 `range_mode="absolute"` 或 `range_mode="level_stride"`；保留 `hard_local_cell` 仅作 ablation，不得作为默认。

### `tools/audit_sparse_head_assignment.py` — MEDIUM / necessary

**位置：** same-batch assignment audit。
**差异：** 当前已经能暴露 pos_by_level collapse。
**问题：** 还不够定位 high-IoU 弱。
**修复：** 增加 `center_fail/range_fail/shortest_conflict/unmatched_gt/reg_target_p50/p95/decoded_iou_oracle/per_level_gt_len_hist`。

### `tests/test_adapter_native_dense_headv2_contracts.py` — MEDIUM

**位置：** config/static tests。
**差异：** 测试已覆盖 bridge/openrange/absrange/cross-over/axis contract。
**问题：** 测试可能把当前错误 contract 固化为“正确”，例如 full cell span point_scale。
**修复：** 增加 official parity tensor tests：uniform grid 下 irregular bridge target/decode 必须与官方 AnchorFreeHead 完全一致。

### `root-cause-notes.md` — LOW

**位置：** root cause summary。
**评价：** 当前笔记方向基本正确：记录了训练未崩、range gate alone 不够、geometry 不是主因、openrange 恢复 positives 但高 IoU 仍弱。([GitHub][2])
**修复：** 把“openrange 不是 fix，只是 diagnostic”写成硬门；把 absrange expanded + official parity test 设为下一步 gate。

---

## 6. 根因归因：已证实 / 高概率 / 待验证 / 已排除

### 已证实

1. **不是训练崩溃。** fixed/nogeometry/reggate/openrange 都完成，没有 traceback/OOM/non-finite loss 主导问题。([GitHub][2])
2. **GT axis 比较不纯。** selected-axis dense control 与 native-axis HeadV3/Bridge 不是 head-only 对照。([GitHub][2])
3. **regression range collapse 存在。** current bridge hard 的 positives 只集中 L0/L1；openrange 恢复全层 positives。([GitHub][2])
4. **range collapse 不足以解释全部崩溃。** openrange 完整训练仍只有 42.44，@0.7 只有 15.20。([GitHub][2])
5. **V2/V3 supervision 与官方 dense 不等价。** 官方是 hard dense assignment，当前 V2/V3 是 soft top-k/log regression。([GitHub][3])

### 高概率

1. **native-axis sparse detection 本身比 selected-axis/equal-interval 难很多。** random sparse native axis 要从不规则观测点定位未观测边界，@0.6/@0.7 极难。
2. **projection/neck 仍未充分 metric-aware。** 官方 ConvTransformer/FPN 是 uniform index model；当前 grid-aware 若未改变主 token mixing，只靠 head 后补几何不够。([GitHub][4])
3. **openrange 高 IoU 弱来自 level prior 消失 + regression denominator/cell scale 不稳 + gap boundary 不可见。**

### 待验证

1. selected-axis random_fixed + official dense head 在当前 commit/env 是否能复现 63–65。
2. equal-interval 50 + current bridge hard 是否能接近 selected-axis dense。
3. native-axis postprocess 是否存在 seconds conversion 或 window offset 错误。
4. absrange expanded 是否比 openrange 更稳。
5. boundary auxiliary/refiner 是否能把 @0.7 从 15 提到合理区间。

### 已排除或降级

1. **geometry modulation 不是一阶主因。** nogeometry 39.32 只比 fixed 低约 0.88。([GitHub][2])
2. **单独补 regression range gate 不是 fix。** reggate 38.32 更低。([GitHub][2])
3. **boundary auxiliary 不是 20+ mAP 崩溃主因。** 它最多解释少量高 IoU/Avg 差距。

---

## 7. 推荐实验矩阵

### 路线 A：公平 sanity / 定位崩溃来源

| 优先级 | 实验                                                                               | 目的                               | 预期解释                                               |
| --: | -------------------------------------------------------------------------------- | -------------------------------- | -------------------------------------------------- |
|  A0 | 官方 OpenTAD AdaTAD dense 768 原 config                                             | 确认环境/数据/评估无问题                    | 若显著低于官方/历史，先修数据和 eval                              |
|  A1 | `equal_interval_50pct + official ActionFormerHead + PointGenerator + remap=True` | 复现 65 selected-axis baseline     | 若失败，当前 sparse loader/postprocess 有 bug             |
|  A2 | `random_fixed + selected-axis/remap=True + official dense head`                  | 验证 random sampling 本身是否导致崩       | 若仍 63–65，说明 native irregular route 才是问题            |
|  A3 | `equal_interval_50pct + bridge hard linear + selected-axis uniform points`       | 验证 bridge head 是否能 dense parity  | 若低，bridge head target/decode 有 bug                 |
|  A4 | `random_fixed native + bridge hard + absrange_expanded`                          | 验证 native sparse corrected range | 若高于 openrange，range scale 修正有效                     |
|  A5 | 每个实验先 same-batch assignment audit                                                | 避免长训浪费                           | `pos_by_level`、GT coverage、decoded oracle IoU 必须过门 |

**最小 config patch：selected-axis official dense head control**

```python
# configs/adatad/thumos/input_random_fixed_50pct_adapter_selected_axis_official_dense_n16r4.py
_base_ = ["./input_random_fixed_50pct_adapter.py"]

def _force_selected_axis(pipeline):
    for step in pipeline:
        if step.get("type") == "LoadFrames" and step.get("method") == "random_fixed_subsample":
            step["remap_gt_to_selected_axis"] = True

for _split in ("train", "val", "test"):
    _force_selected_axis(dataset[_split]["pipeline"])

model = dict(
    type="ActionFormer",
    projection=dict(
        _delete_=True,
        type="Conv1DTransformerProj",
        in_channels=384,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=384,
        input_pdrop=0.0,
    ),
    neck=dict(
        _delete_=True,
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        _delete_=True,
        type="ActionFormerHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
    ),
)

work_dir = "exps/thumos/adatad/rf50_selected_axis_official_dense_n16r4"
```

### 路线 B：修正 irregular geometry contract

当前不应假设“只要 head 使用 native centers 就能学好”。projection/neck 至少需要一个真正 metric-aware 机制：

1. **Metric relative position bias**：attention logits 加 `f(log(|t_i-t_j|), log(cell_span_i), log(cell_span_j))`，不是 fixed local window index。
2. **Time rasterizer**：先把 sparse tokens scatter/rasterize 到 dense 768 grid，再用官方 dense head；这是最强 sanity，可以分离“特征缺失”和“head geometry bug”。
3. **Geometry-aware conv**：邻域按真实时间半径取样，不按 index kernel 取样。
4. **FPN 下采样真实 cell merge**：两个 token 合并时合并 support interval，不是简单 stride-2 取点。

**FPN grid merge patch：**

```python
# opentad/models/utils/temporal_grid.py

def downsample_temporal_grid_by_interval(grid, factor=2, min_scale=1e-4):
    center = grid["center"]
    valid = grid["valid_mask"].bool()
    left = grid["cell_left"].clamp_min(min_scale)
    right = grid["cell_right"].clamp_min(min_scale)
    fresh = grid.get("fresh_mask", valid).bool()

    B, T = center.shape
    out_T = (T + factor - 1) // factor

    out_center, out_left, out_right, out_valid, out_fresh = [], [], [], [], []

    for b in range(B):
        c_row, l_row, r_row = [], [], []
        v_row, f_row = [], []
        for s in range(0, T, factor):
            e = min(s + factor, T)
            mask = valid[b, s:e]
            if not mask.any():
                c = center[b, max(s - 1, 0)]
                c_row.append(c)
                l_row.append(left[b, max(s - 1, 0)])
                r_row.append(right[b, max(s - 1, 0)])
                v_row.append(False)
                f_row.append(False)
                continue

            idx = torch.arange(s, e, device=center.device)[mask]
            support_l = center[b, idx] - left[b, idx]
            support_r = center[b, idx] + right[b, idx]
            merged_l = support_l.min()
            merged_r = support_r.max()
            merged_c = 0.5 * (merged_l + merged_r)

            c_row.append(merged_c)
            l_row.append((merged_c - merged_l).clamp_min(min_scale))
            r_row.append((merged_r - merged_c).clamp_min(min_scale))
            v_row.append(True)
            f_row.append(bool(fresh[b, idx].any().item()))

        out_center.append(torch.stack(c_row))
        out_left.append(torch.stack(l_row))
        out_right.append(torch.stack(r_row))
        out_valid.append(torch.tensor(v_row, device=center.device, dtype=torch.bool))
        out_fresh.append(torch.tensor(f_row, device=center.device, dtype=torch.bool))

    out = dict(
        center=torch.stack(out_center),
        cell_left=torch.stack(out_left),
        cell_right=torch.stack(out_right),
        valid_mask=torch.stack(out_valid),
        fresh_mask=torch.stack(out_fresh),
    )
    out["level_scale"] = 0.5 * (out["cell_left"] + out["cell_right"])
    out["level_scale"] = (out["level_scale"] * out["valid_mask"].to(out["level_scale"].dtype)).sum(1) / (
        out["valid_mask"].sum(1).clamp_min(1).to(out["level_scale"].dtype)
    )
    return out
```

### 路线 C：修正 head / assignment / regression

**推荐公式：**

* point fields 改为：

```text
[center, reg_min, reg_max, decode_scale_left, decode_scale_right, range_scale, radius_scale]
```

* range 推荐：

  * 第一阶段：`absolute expanded ranges`；
  * 第二阶段：`level_stride ranges`；
  * 不建议默认使用 `cell_left+cell_right` local span。
* regression encode/decode：

  * 先用 official-like linear：

    ```python
    target_left  = (center - gt_start) / decode_scale_left
    target_right = (gt_end - center) / decode_scale_right
    ```
  * decode：

    ```python
    start = center - pred_left * decode_scale_left
    end   = center + pred_right * decode_scale_right
    ```
  * log1p/expm1 作为 ablation，不作为主线。

**IrregularPointGeneratorV2 patch：**

```python
# opentad/models/dense_heads/prior_generator/irregular_point_generator.py

@PRIOR_GENERATORS.register_module()
class IrregularPointGeneratorV2:
    def __init__(
        self,
        strides,
        regression_range,
        range_mode="absolute",
        decode_scale_mode="level_stride",
        radius_scale_mode="level_stride",
        min_scale=1e-6,
    ):
        self.strides = list(strides)
        self.regression_range = list(regression_range)
        self.range_mode = range_mode
        self.decode_scale_mode = decode_scale_mode
        self.radius_scale_mode = radius_scale_mode
        self.min_scale = float(min_scale)

    def _level_tensor(self, center, value):
        return torch.full_like(center, float(value))

    def __call__(self, feat_list, temporal_grid_list):
        pts_list = []
        for level_idx, (feat, grid, reg_range) in enumerate(
            zip(feat_list, temporal_grid_list, self.regression_range)
        ):
            center = grid["center"].to(dtype=feat.dtype, device=feat.device)
            cell_left = grid["cell_left"].to(dtype=center.dtype, device=center.device).clamp_min(self.min_scale)
            cell_right = grid["cell_right"].to(dtype=center.dtype, device=center.device).clamp_min(self.min_scale)
            cell_span = (cell_left + cell_right).clamp_min(self.min_scale)
            stride = float(self.strides[level_idx])

            if self.decode_scale_mode == "level_stride":
                decode_left = self._level_tensor(center, stride)
                decode_right = self._level_tensor(center, stride)
            elif self.decode_scale_mode == "cell":
                decode_left, decode_right = cell_left, cell_right
            elif self.decode_scale_mode == "cell_span":
                decode_left = decode_right = cell_span
            else:
                raise ValueError(f"Unsupported decode_scale_mode={self.decode_scale_mode}")

            if self.radius_scale_mode == "level_stride":
                radius_scale = self._level_tensor(center, stride)
            elif self.radius_scale_mode == "cell_geom_mean":
                radius_scale = torch.sqrt((cell_left * cell_right).clamp_min(self.min_scale**2))
            elif self.radius_scale_mode == "cell_span":
                radius_scale = cell_span
            else:
                raise ValueError(f"Unsupported radius_scale_mode={self.radius_scale_mode}")

            lo, hi = float(reg_range[0]), float(reg_range[1])
            if self.range_mode == "absolute":
                reg_min = torch.full_like(center, lo)
                reg_max = torch.full_like(center, hi)
            elif self.range_mode == "level_stride":
                reg_min = torch.full_like(center, lo * stride)
                reg_max = torch.full_like(center, hi * stride)
            elif self.range_mode == "local_cell_span":
                # Deprecated ablation only.
                reg_min = lo * cell_span
                reg_max = hi * cell_span
            elif self.range_mode == "open":
                reg_min = torch.zeros_like(center)
                reg_max = torch.full_like(center, 10000.0)
            else:
                raise ValueError(f"Unsupported range_mode={self.range_mode}")

            points = torch.stack(
                [center, reg_min, reg_max, decode_left, decode_right, cell_span, radius_scale],
                dim=-1,
            )
            pts_list.append(points)
        return pts_list
```

**Dense-like hard assignment patch：**

```python
# opentad/models/dense_heads/irregular_actionformer_bridge_head.py

def _point_fields_v2(self, point):
    center = point[..., 0]
    reg_min = point[..., 1]
    reg_max = point[..., 2]

    if point.shape[-1] >= 7:
        decode_left = point[..., 3].clamp_min(self.reg_denom_floor)
        decode_right = point[..., 4].clamp_min(self.reg_denom_floor)
        radius_scale = point[..., 6].clamp_min(self.reg_denom_floor)
    elif point.shape[-1] >= 5:
        decode_left = point[..., 3].clamp_min(self.reg_denom_floor)
        decode_right = point[..., 4].clamp_min(self.reg_denom_floor)
        radius_scale = torch.sqrt((decode_left * decode_right).clamp_min(self.reg_denom_floor**2))
    else:
        stride = point[..., 3].clamp_min(self.reg_denom_floor)
        decode_left = decode_right = radius_scale = stride

    return center, reg_min, reg_max, decode_left, decode_right, radius_scale


@torch.no_grad()
def _prepare_targets_dense_hard(self, points, gt_segments, gt_labels):
    point_list = self._points_per_sample(points, len(gt_segments))
    gt_cls, gt_reg, reg_weight_list = [], [], []

    for point, gt_segment, gt_label in zip(point_list, gt_segments, gt_labels):
        num_pts = point.shape[0]
        center, reg_min, reg_max, dec_l, dec_r, radius_scale = self._point_fields_v2(point)

        if gt_segment.numel() == 0:
            gt_cls.append(point.new_zeros((num_pts, self.num_classes)))
            gt_reg.append(point.new_zeros((num_pts, 2)))
            reg_weight_list.append(point.new_zeros((num_pts,)))
            continue

        num_gts = gt_segment.shape[0]
        gt = gt_segment[None].expand(num_pts, num_gts, 2)
        left = center[:, None] - gt[..., 0]
        right = gt[..., 1] - center[:, None]
        reg_targets = torch.stack([left, right], dim=-1)

        inside_gt = reg_targets.min(dim=-1).values > 0

        if self.center_sample == "radius":
            gt_center = 0.5 * (gt[..., 0] + gt[..., 1])
            radius = self.center_sample_radius * radius_scale[:, None]
            cb_l = center[:, None] - torch.maximum(gt_center - radius, gt[..., 0])
            cb_r = torch.minimum(gt_center + radius, gt[..., 1]) - center[:, None]
            inside_center = torch.minimum(cb_l, cb_r) > 0
        else:
            inside_center = inside_gt

        max_dist = reg_targets.max(dim=-1).values
        inside_range = (max_dist >= reg_min[:, None]) & (max_dist <= reg_max[:, None])

        gt_len = (gt_segment[:, 1] - gt_segment[:, 0]).clamp_min(1e-6)
        lens = gt_len[None].repeat(num_pts, 1)
        lens.masked_fill_(~inside_gt, float("inf"))
        lens.masked_fill_(~inside_center, float("inf"))
        lens.masked_fill_(~inside_range, float("inf"))

        min_len, min_inds = lens.min(dim=1)
        pos = torch.isfinite(min_len)

        cls_target = point.new_zeros((num_pts, self.num_classes))
        if pos.any():
            cls_target[pos, gt_label[min_inds[pos]].long()] = 1.0

        reg_encoded = torch.stack(
            [
                (left / dec_l[:, None]).clamp_min(0.0),
                (right / dec_r[:, None]).clamp_min(0.0),
            ],
            dim=-1,
        )
        reg_target = reg_encoded[torch.arange(num_pts, device=point.device), min_inds.clamp_min(0)]
        reg_target = torch.where(pos[:, None], reg_target, torch.zeros_like(reg_target))

        gt_cls.append(cls_target)
        gt_reg.append(reg_target)
        reg_weight_list.append(pos.to(point.dtype))

    return gt_cls, gt_reg, reg_weight_list
```

**Decode patch：**

```python
def get_refined_proposals(self, points, reg_pred):
    point_tensor = self._concat_points(points)
    reg_tensor = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)
    center, _, _, dec_l, dec_r, _ = self._point_fields_v2(point_tensor)

    left = reg_tensor[..., 0].clamp_min(0.0) * dec_l
    right = reg_tensor[..., 1].clamp_min(0.0) * dec_r

    start = center[None] - left
    end = center[None] + right
    return torch.stack([start, end], dim=-1)
```

**Audit extension：**

```python
# tools/audit_sparse_head_assignment.py

def summarize_assignment(candidate, inside_gt, inside_center, inside_range, assigned, gt_segments, points):
    out = {}
    out["num_points"] = int(points.shape[0])
    out["inside_gt"] = int(inside_gt.any(dim=1).sum().item())
    out["inside_center"] = int(inside_center.any(dim=1).sum().item())
    out["inside_range"] = int(inside_range.any(dim=1).sum().item())
    out["assigned"] = int(assigned.sum().item())

    fail_center = inside_gt & (~inside_center)
    fail_range = inside_gt & inside_center & (~inside_range)
    out["center_fail_pairs"] = int(fail_center.sum().item())
    out["range_fail_pairs"] = int(fail_range.sum().item())

    gt_covered = assigned[:, None] & candidate
    out["gt_coverage"] = int(gt_covered.any(dim=0).sum().item())
    out["gt_total"] = int(gt_segments.shape[0])

    max_dist = candidate.new_zeros(candidate.shape, dtype=torch.float32)
    # fill with max(left,right) before calling this in real code
    return out
```

---

## 8. 最终行动计划

### 立即要修的代码错误 / contract 错误

1. **把 `IrregularPointGeneratorV2` 默认 range 语义从 local cell span 改成 absolute/level stride。** `local_cell_span` 只能保留为 ablation。
2. **BridgeHead 拆分 scale：range scale、decode scale、radius scale 不得混用。**
3. **加 official parity test：uniform grid 下 irregular bridge 的 target/decode/loss mask 必须与官方 AnchorFreeHead 一致。**
4. **加 axis contract test：selected-axis proposal 不得直接走 native seconds；native-axis GT 不得 remap。**
5. **加 proposal dump：每次 long run 前做 same-batch proposal coordinate sanity。**

### 先做 same-batch audit / smoke

1. `selected_axis_official_dense`：确认当前仓库能复现 63–65。
2. `bridge_dense_hard_uniform_grid_parity`：确认 BridgeHead 不是 head bug。
3. `native_bridge_absrange_expanded`：确认 pos_by_level 不 collapse，GT coverage ≥ 95%。
4. `native_bridge_openrange` 只保留对照，不再作为主线。
5. dump 100 proposals/video，验证 native seconds 与 GT seconds 的 tIoU 分布。

### 值得长训的配置

1. `random_fixed + selected-axis + official dense head`：公平 sanity。
2. `equal_interval + bridge hard + selected-axis/uniform grid`：head parity。
3. `random_fixed native + bridge hard + absrange_expanded + level_stride decode/radius`：第一条真正 native sparse 候选。
4. 在 3 达到明显高于 42 后，再加 boundary auxiliary/refiner。

### 不建议继续投入的方向

1. 不继续把 `HeadV3 soft topk + log regression + range_mode=hard` 当主线。
2. 不把 `openrange` 当最终修复；它只是证明 range gate 有问题。
3. 不继续单独跑 `nogeometry` / `reggate` 长训，它们已经回答了窄问题。
4. 不在 official dense parity 未通过前宣称 sparse route 设计失败或成立。
5. 不把当前仓库内 dense code 当官方 baseline，必须持续对官方 OpenTAD 做 diff。

**最终判定：** 当前 implementation 不是完全错误，也不是模型路线已被证伪；它是一个 supervision/geometry contract 尚未闭合的 native-axis sparse detector。性能 40–42 的最强解释是：**regression range collapse + V2/V3 soft supervision 偏离官方 dense + native irregular temporal modeling 不充分**。下一步不是继续盲目长训，而是先把官方 dense parity、selected-axis sanity、BridgeHead hard assignment 和 absolute/level range contract 全部打通。

[1]: https://raw.githubusercontent.com/sming256/OpenTAD/main/configs/_base_/models/actionformer.py "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/c3716eb480983d2e4533a003774faef1b96cfaea/root-cause-notes.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/anchor_free_head.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/bricks/transformer.py "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/c3716eb480983d2e4533a003774faef1b96cfaea/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x.py "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/sming256/OpenTAD/main/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py "raw.githubusercontent.com"
[7]: https://raw.githubusercontent.com/sming256/OpenTAD/main/configs/adatad/README.md "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/detectors/single_stage.py "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/dense_heads/prior_generator/point_generator.py "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/sming256/OpenTAD/main/opentad/models/necks/fpn.py "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/c3716eb480983d2e4533a003774faef1b96cfaea/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_actionformer_base.py "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/c3716eb480983d2e4533a003774faef1b96cfaea/configs/_base_/models/actionformer.py "raw.githubusercontent.com"
[13]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/c3716eb480983d2e4533a003774faef1b96cfaea/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py "raw.githubusercontent.com"
[14]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/c3716eb480983d2e4533a003774faef1b96cfaea/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_openrange_n16r4.py "raw.githubusercontent.com"
[15]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/c3716eb480983d2e4533a003774faef1b96cfaea/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_n16r4.py "raw.githubusercontent.com"
[16]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/c3716eb480983d2e4533a003774faef1b96cfaea/opentad/models/dense_heads/irregular_actionformer_head_v2.py "raw.githubusercontent.com"
