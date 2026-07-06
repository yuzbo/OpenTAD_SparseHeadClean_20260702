---
date: 2026-07-06
source: GPT/Pro review attachment
status: recorded
scope: Strict review of 65-to-40/42 sparse-head performance collapse, official OpenTAD comparison, and route correctness
source_attachment_sha256: 9e43544dc36f49cf4d181c0b94f7fdffdc7e43239b4836b48086bfb5f79fca11
reviewed_repository_commit: d7da39315f80405253bf705997d7c26e63c66ab0
boundary: Reviewer reported no local clone/training and used GitHub source/raw pages plus official OpenTAD static review.
---

# Pro Review: Monitoring Collapse and Route Correctness 2026-07-06

## Record Boundary

This file preserves the external review response verbatim below. The hash above is for the original attachment content, not this wrapper file.

## Original Response

## 总判决

我不会把当前 **65 → 40/42 Average-mAP** 解释成“random 50% sparse 采样自然退化”。这更像是 **监督坐标合同、point scale、NMS/post-process 轴、feature temporal metric** 没有完全闭合造成的系统性崩溃。

我没有在本地完成 clone/训练；容器无法解析 GitHub 域名，所以本次结论来自对你指定 commit `d7da393` 的 GitHub 源码页面/Raw 内容，以及官方 OpenTAD 源码的静态审查。目标仓库 commit 已定位到 `d7da393`，官方对照来自 OpenTAD 官方仓库与其 AdaTAD/ActionFormer 文件。([GitHub][1])

最严厉的结论是：

**当前 sparse / irregular route 还不能声称与官方 dense AdaTAD / ActionFormer 的监督合同等价。**
`absrange_expanded` 值得做 **短训 + 强 gate**，但不值得在未修正 axis/postprocess/audit 前直接长训并解释论文结论。下一步优先级应是：

1. 先做 **official dense selected-axis sanity**；
2. 再做 **bridge hard dense-equivalence sanity**；
3. 然后才是 native-axis irregular head；
4. 最后才讨论 metric-aware projection / neck。

---

## 1. 逐文件 / 逐逻辑块问题

### A. `opentad/datasets/transforms/end_to_end.py`

#### A1. `irregular_gt_axis / irregular_proposal_axis / irregular_postprocess_axis` 被强行设成同一轴

`_set_irregular_axis_meta` 根据 `remap_gt_to_selected_axis` 设置 `axis = "selected"` 或 `"native"`，然后把 `irregular_gt_axis`、`irregular_proposal_axis`、`irregular_postprocess_axis` 全部写成同一个值。([GitHub][2])

这是一个 **confirmed design/contract bug**。

正确合同不应该强制三者相等。尤其 selected-axis sanity 中，合理流程应该是：

```text
GT supervision axis: selected
proposal decode axis: selected
NMS / postprocess / eval axis: native 或 seconds
```

当前 metadata 设计无法表达这个合同。它会把 selected-axis route 锁死在“selected 轴上 NMS，然后再 convert_to_seconds”的错误流程里。

#### A2. `_remap_gt_to_selected_axis` 对退化 segment 造出 1e-3 长度伪 GT

`_remap_gt_to_selected_axis` 用 `np.interp` 将原始 GT start/end 映射到 selected index axis；如果 `end <= start`，代码会把 end 改成 `start + 1e-3`。([GitHub][2])

这是 **confirmed risk / likely bug**。
对于 TAD，尤其 high-IoU 定位，造出极短伪 segment 会污染：

* center sampling；
* regression target；
* DIoU/GIoU 类损失；
* level assignment；
* high-IoU eval 的边界学习。

更合理做法是：退化 GT 要么丢弃，要么标为 ignore，要么保留 native GT 并记录 visibility/partial-observation mask；不要创造 1e-3 的训练目标。

#### A3. `random_fixed_subsample` 分支需要显式化，避免 silent fallback

文件中 method 白名单包含 `"random_fixed_subsample"`，并且 `_select_random_fixed_positions` 已定义。([GitHub][2])
但在可见主采样分支里，显式出现的是 stratified、uniform、pseudo/BATA 等路径；由于 GitHub 页面截断，我不能 100% 排除后续存在 fallback 调用。([GitHub][2])

这不是我能确认的运行 bug，但它是 **高风险实现风格**。建议改成显式分支：

```python
if self.method == "random_fixed_subsample":
    keep_positions = self._select_random_fixed_positions(
        valid_len=valid_len,
        frame_num=frame_num,
        sample_key=sample_key,
    )
elif self.method == "uniform_fixed_subsample":
    ...
elif self.method == "stratified_random_fixed_subsample":
    ...
else:
    raise NotImplementedError(self.method)
```

不要让核心实验路线依赖隐藏 fallback。当前项目目标是证明 sparse/irregular route，任何 silent fallback 都会让归因不可信。

---

### B. `opentad/models/detectors/irregular_actionformer.py`

#### B1. `_assert_axis_contract` 错误地要求 gt/proposal/postprocess 三轴完全相等

当前 detector 会从 meta 读 `irregular_gt_axis`、`irregular_proposal_axis`、`irregular_postprocess_axis`，并要求三者组成的 set 长度为 1。也就是说，三者必须完全相等。([GitHub][3])

这是 **confirmed bug**。

TAD 检测合同应拆成至少三层：

```text
training target axis
proposal decode axis
postprocess / NMS / eval axis
```

在 selected-axis route 中，训练和 decode 可以在 selected index 轴，但 NMS/eval 必须在 native/seconds 轴，否则随机非线性采样下 IoU 排序和 suppress 逻辑不成立。

#### B2. `post_processing` 在 `convert_to_seconds` 之前做 NMS

当前 `post_processing` 对 `segments` 先 threshold/top-k，然后在非 sliding-window 情况下直接调用 `batched_nms(segments, scores, labels, ...)`，之后才 `convert_to_seconds(segments, metas[i])`。([GitHub][3])

这对 native-axis proposal 可能勉强成立，但对 selected-axis proposal 是错误的。因为 selected index 轴下的 segment length / overlap 不等价于真实 native time overlap。目标仓库里其实已经有 `selected_axis_to_dense_axis` 和 `convert_to_seconds`，后者在 `irregular_native_axis=False` 时会先把 selected-axis 坐标映射回 dense/native 坐标。([GitHub][4])

问题是：**转换发生在 NMS 之后**。这会直接打击 high-IoU，尤其 @0.6/@0.7。

#### B3. `_build_center_grid_from_positions` 的 `cell_left/cell_right` 不是半 support，而是 full neighboring gap

native-axis grid 中，代码把左侧距离设为前后 position 差，右侧距离也设为相邻差；首尾也用整段 gap/native_end。([GitHub][3])
`temporal_grid.build_temporal_grid` 也采用类似 full-gap left/right 逻辑。([GitHub][5])

这在 irregular point generator 中后果很严重。对于 uniform stride-2 native positions：

```text
selected native centers: 0, 2, 4, 6, ...
当前 left=2, right=2
left+right=4
```

但 dense-equivalent stride 应该是 2 native units，或 selected-axis 下的 1 selected unit。当前 `full_cell_span = left + right` 会比合理 stride 放大约 2 倍。若 gap 不规则，放大更混乱。

这会污染：

* regression range；
* center sampling radius；
* regression normalization denominator；
* FPN level assignment；
* decoded proposal width。

这与 openrange “positive 很多但 high IoU 仍弱”的症状高度一致。

---

### C. `opentad/models/dense_heads/prior_generator/irregular_point_generator.py`

#### C1. V2 point layout 是新合同，不是官方 PointGenerator 合同

官方 `PointGenerator` 输出 `[center, reg_min, reg_max, stride]`，center 是 `torch.linspace(0, T-1, T) * stride`，regression_range 以 feature grid / stride 语义组织。([GitHub][6])

当前 `IrregularPointGeneratorV2` 输出：

```text
[center, reg_min, reg_max, decode_left, decode_right, range_scale, radius_scale]
```

这本身可以成立，但它已经不是官方合同。它要求 head、assignment、decode、postprocess 全部重新证明。

#### C2. `range_mode="open"` 实际上与 `absolute` 同分支

V2 中 `range_mode in {"absolute", "open"}` 时，`range_scale = 1`，reg_min/reg_max 使用配置中的绝对值；是否“open”完全由配置 range 决定，不是一个独立的数学模式。([GitHub][7])

因此 `openrange` 的实验结果不能被解释为“open range 机制证明 assignment 合理”。它更像是一个 **放宽 range 的候选生成实验**。

#### C3. `range_mode="hard"` 与当前 full-cell-span 组合会严重偏离官方 dense

bridge hard base 配置中，prior V2 使用 official strides/ranges，但 `range_mode="hard"` 会把 range 按 local cell span 重新缩放。([GitHub][8])
如果 cell span 已经是 full gap left+right，那么 official range `(0,4),(4,8),...` 不再是官方 dense 的语义。

这会解释 same-batch audit 里：

```text
openrange: positives 全层很多，GT coverage=38/38
absrange: positives 少但 GT coverage=38/38
levelstride: GT coverage 掉到 20/38
```

这些结果说明 assignment 候选数量很敏感，但还不能说明监督合同正确。

---

### D. `opentad/models/dense_heads/irregular_actionformer_bridge_head.py`

#### D1. hard assignment 形式像官方，但语义不等价

官方 `AnchorFreeHead` 的核心监督逻辑是：

* center point 到 GT start/end 的 left/right distance；
* center sampling radius = `stride * center_sample_radius`；
* regression range 用 point 的 reg_min/reg_max；
* conflict 用 shortest GT；
* regression target 除以 stride。([GitHub][9])

当前 bridge head 也实现了 center sampling、range filtering、shortest GT conflict、regression encode/decode。([GitHub][10])
但它的 scale 来自 V2 point fields：`decode_left/right`、`range_scale`、`radius_scale`，默认 `center_radius_scale="full_cell_span"`、`reg_denom_mode="full_cell_span"`。([GitHub][10])

所以结论是：

```text
形式上 hard assignment 接近官方；
语义上不等价。
```

除非你在 dense no-sparse 情况下证明：

```text
positive mask 一致
assigned GT 一致
encoded reg target 一致
decode 后 proposal 一致
mAP 一致
```

否则不能说 bridge hard dense-like 已经复刻官方监督。

#### D2. `symmetric_linear` regression 本身可用，但 denom 选择危险

`symmetric_linear` encode/decode 如果 denom 是官方 stride，理论上接近官方 ActionFormer。
但当前默认 denom 是 `full_cell_span`，native irregular 下这个值可能比实际 token stride 大很多。

`absrange_expanded` 把 `reg_denom_mode` 改为 `left_right_mean`，并把 decode/radius 改为 `level_stride`，这是朝正确方向走。([GitHub][11])
但这仍然不是 official dense selected-axis sanity。

---

### E. `opentad/models/utils/temporal_grid.py`

#### E1. `cell_left/cell_right` 语义应改成 Voronoi half-cell support

当前 `build_temporal_grid` 在缺省情况下用相邻 center 差值填 left/right。([GitHub][5])
`downsample_temporal_grid` 后续用：

```text
interval_start = center - cell_left
interval_end   = center + cell_right
```

再 merge interval。([GitHub][5])

这个 downsample 逻辑本身可以成立，但前提是 `cell_left/cell_right` 表示真实 half support。当前 full-gap left/right 会让 interval 过宽、互相重叠，downsample 后更宽，进而污染 range/radius/denom。

建议把基础 cell 改成 Voronoi cell：

```text
edge[i+1] = 0.5 * (center[i] + center[i+1])
left[i]  = center[i] - edge[i]
right[i] = edge[i+1] - center[i]
```

首尾用半个首/尾 gap 或 clip 到 window 边界。

---

### F. `tools/audit_sparse_head_assignment.py`

这个 audit 是有价值的。它统计 per-level positives、candidate count、inside GT、range/center fail、valid mask、conflict、radius base、encoded reg stats、decode reconstruction error、GT coverage。([GitHub][12])

但它 **不足以证明 assignment 合理**。它缺少三个关键指标：

1. **assigned-positive decode 后与 assigned GT 的 IoU**；
2. **proposal axis → native/seconds 转换后的 IoU**；
3. **NMS 前后 high-IoU recall**。

所以你现在的 same-batch audit：

```text
openrange positives 多
absrange_expanded coverage 满
```

只能证明“有正样本”和“GT 被覆盖”，不能证明 high-IoU 边界监督正确。TAD 65→40/42 的崩溃恰恰可能发生在“有正样本但边界尺度/坐标/NMS 错”的区域。

---

### G. `scripts/verify_official_dense_reference.py`

这个脚本方向正确：它试图把 local dense reference 与 upstream 官方 raw 文件做 diff。当前覆盖 `anchor_free_head.py`、`point_generator.py`、`fpn.py` 三个文件。([GitHub][13])

但它不足以防止“误信仓库内 dense 代码”：

* 使用的是 moving `main`，不是 pinned official commit；
* 只校验 3 个文件；
* 没校验官方 AdaTAD THUMOS config；
* 没校验 base ActionFormer config；
* 没校验 dataset transform / post-processing / NMS / eval；
* 没强制本地 dense reproduction。

它应改为 pinned official SHA + SHA256 manifest，并把官方 config 也纳入校验。

---

### H. `configs/...absrange_expanded...py`

`absrange_expanded` 做了几个比 openrange 更合理的选择：

* `range_mode="absolute"`；
* expanded ranges `[(0,8),(2,16),(4,32),(8,64),(16,128),(32,10000)]`；
* `decode_scale_mode="level_stride"`；
* `radius_scale_mode="level_stride"`；
* `center_radius_scale="point_radius"`；
* `reg_denom_mode="left_right_mean"`。([GitHub][11])

这比默认 hard/full-cell-span 更接近“bounded native-axis”候选。root-cause notes 也把它解释为当前最强 bounded native-axis candidate，并记录了 same-batch audit 结果：`absrange_expanded: pos_by_level=[17,30,44,45,22,4], GT coverage=38/38`。([GitHub][14])

但它仍然只是 **候选配置**，不是已证明正确路线。原因是它没有解决：

* selected-axis sanity；
* NMS 转换顺序；
* temporal grid cell 语义；
* feature mixer 是否理解 irregular time；
* audit 缺少 decode-to-GT IoU。

---

### I. `root-cause-notes.md`

这份 notes 总体比代码更谨慎。它已经承认 projection/backbone/neck 可能仍是 slot-index based，而 official dense 假设 uniform token spacing；如果 projection/neck 不理解真实 temporal distance，head 收到的 feature geometry 会和 native labels 不一致。([GitHub][14])

它也建议把 upstream official 作为 authority、推进 official dense parity / selected-axis / native-axis sanity，并扩展 audit。([GitHub][14])

所以我认为 notes 的方向没有大问题。需要收紧的是：
**不要把 absrange_expanded 的 same-batch audit 解读成“可以长训证明路线”。它最多是“值得 gated short run”。**

---

## 2. 当前实现与官方 dense AdaTAD / ActionFormer 的监督合同是否等价？

**不等价。**

官方合同非常清楚：

官方 ActionFormer base 使用 `PointGenerator`，strides 为 `[1,2,4,8,16,32]`，regression ranges 为 `[(0,4),(4,8),(8,16),(16,32),(32,64),(64,10000)]`，center sample radius 为 `1.5`。([GitHub][15])

官方 `PointGenerator` 生成 `[center, reg_min, reg_max, stride]`，center 是 feature grid index 乘 stride。([GitHub][6])

官方 `AnchorFreeHead`：

* 用 point center 到 GT start/end 的距离构造 regression targets；
* center sampling radius = point stride × radius；
* regression range 用 point 的 reg_min/reg_max；
* conflict 用 shortest GT；
* regression target 除以 point stride；
* decode 时 `start = center - reg_left * stride`，`end = center + reg_right * stride`。([GitHub][9])

当前 sparse/irregular route 至少有四个不等价点：

| 项目           | 官方 dense                      | 当前 sparse/native route                             |
| ------------ | ----------------------------- | -------------------------------------------------- |
| point center | uniform feature grid × stride | selected native frame positions / irregular grid   |
| stride/scale | 单一 stride                     | decode_left/right、range_scale、radius_scale 多字段     |
| GT axis      | dense feature-time axis       | native 或 selected，可被 remap                         |
| NMS/eval     | dense/seconds 一致轴             | selected/native/postprocess 轴合同未拆开，NMS 在 convert 前 |

这不是“官方 dense 的稀疏输入版本”，而是一个新的 irregular detector。这个方向可以研究，但不能用“dense-like hard assignment”直接背书。

---

## 3. 65 → 40/42 性能崩溃最可能根因排序

### Rank 1：坐标轴合同与 post-processing 顺序没有闭合

尤其是 selected-axis 情况下，NMS 在 selected axis 上做，之后才 convert_to_seconds。这个顺序对 random fixed sparse 是错误的。目标仓库虽然有 selected→dense/native 转换函数，但 detector post-processing 调用顺序不对。([GitHub][3])

即使当前 `random_fixed_50pct` base 设置 `remap_gt_to_selected_axis=False`，也就是 native/native/native，axis contract 仍然阻止你做真正的 selected-axis sanity。base 配置中 train/val/test 均使用 `random_fixed_subsample`，`remap_gt_to_selected_axis=False`。([GitHub][16])

### Rank 2：cell scale / range / radius / denom 语义错误

`cell_left/cell_right` 使用 full adjacent gap，V2 point generator 和 bridge 默认又使用 full-cell-span。这会让 assignment range、center radius、reg denom 系统性偏大。([GitHub][3])

这可以解释：

* openrange positives 很多但 high-IoU 仍弱；
* absrange 太窄；
* levelstride coverage 掉；
* absrange_expanded 看起来较合理但还没证明。

### Rank 3：GT remap 与 partial observation 处理会制造边界噪声

`np.interp` remap GT 到 selected axis 是近似投影。对于 random fixed 非线性采样，它会改变动作长度、边界位置、短动作可见性。退化时再造 1e-3 segment 会进一步污染监督。([GitHub][2])

### Rank 4：projection / neck 仍可能是 slot-index mixer，不理解 irregular temporal distance

官方 neck/FPN/FPNIdentity 默认处理的是规则 feature sequence。官方 FPN 只是 top-down interpolate 和 mask 传播，FPNIdentity 也不引入 irregular temporal metric。([GitHub][17])
目标仓库 notes 已经指出 projection/backbone/neck 可能仍是 slot-index based，而 native labels 是真实时间坐标，两者 geometry 不一致。([GitHub][14])

这可能是最终根因之一，但在 axis/assignment/postprocess sanity 没通过之前，不应优先重做 neck。

### Rank 5：audit 指标被过度解释

`GT coverage=38/38` 和 per-level positive count 只能说明“有正样本”，不能说明 proposal high-IoU、decode 正确、NMS 正确。当前 audit 工具还缺少 assigned-GT IoU 和 postprocess-axis IoU。([GitHub][12])

### Rank 6：可能存在 random_fixed 路径显式性不足 / fallback 风险

我不能确认它已经造成结果错误，但核心路线不应依赖不可见 fallback。应显式化并加测试。

---

## 4. `absrange_expanded` 是否值得长训？

**值得 short gated run，不值得直接 blind long train。**

它比 openrange 更有希望，因为它限制了 range、恢复 full GT coverage，并把 decode/radius/denom 从危险的 full-cell-span 拉回更接近 level-stride / local mean 的设置。([GitHub][11])

但长训前必须加 gate。

### 必须通过的 pre-train audit gate

1. `GT coverage >= 95%`，最好 100%；
2. total positives 不应接近 openrange flood；
3. per-level positives 不能只堆在 L0/L1；
4. encoded regression target 必须 finite，p90 不应异常大；
5. decode reconstruction error `< 1e-4`；
6. 新增：assigned-positive decode 后，与 assigned GT 的 IoU：

   * mean IoU > 0.95；
   * p5 IoU > 0.90；
   * 或 native-axis 边界误差 p95 < 1 feature unit。

### 必须通过的 short-train gate

建议先把 val_start 临时提前到 epoch 5 或 10。当前 workflow 主要训练到 60 epoch，val_start_epoch=40，太晚，不适合作为 route gate。([GitHub][16])

停止条件：

* epoch 10/20 Average-mAP 仍不接近或超过 openrange 的 42.44；
* @0.6/@0.7 没有明显改善；
* decoded proposal length bias 严重；
* positive count 训练中漂移；
* NMS 前后 high-IoU recall 差异异常；
* loss 降但 @0.7 不动。

通过条件：

* short run 已经超过或接近 42.44；
* @0.6/@0.7 明显优于 openrange；
* assigned-GT oracle decode IoU 高；
* NMS axis ablation 不再改变结论。

---

## 5. 下一步更优雅路线

我建议的优先级非常明确：

### Step 1：回到 official dense selected-axis sanity

先不要继续堆 native-axis bridge knob。
先做一个 selected-index axis 的 sanity：

```text
random/uniform fixed 50% selected positions
GT remap 到 selected axis
使用官方 PointGenerator / AnchorFreeHead 语义
proposal decode 在 selected axis
NMS 前映射回 native/seconds
```

uniform 50% selected-axis 应接近你说的等间隔 50% baseline。若这里都崩，说明 remap/postprocess/eval 有 bug。

### Step 2：bridge hard dense-equivalence sanity

在 **非 sparse dense 输入** 上，把 `IrregularActionFormerBridgeHead` 配成 official-equivalent：

```text
absolute official regression ranges
level_stride radius
level_stride denom
center positions = official dense grid
```

要求：

```text
positive mask 与官方一致
assigned GT 与官方一致
encoded target 与官方一致
decode proposal 与官方一致
mAP 与官方接近
```

如果这一步不过，bridge head 本身不能用于 sparse 结论。

### Step 3：再比较 selected-axis random fixed vs native-axis absrange_expanded

只有在 selected-axis sanity 和 bridge-equivalence 都过后，`absrange_expanded` 才有解释空间。

### Step 4：最后才重做 projection/neck

如果 selected-axis route 可以接近 60+，但 native-axis route 仍 40+，说明 native-axis feature mixer 不理解 irregular distance。那时再做：

* metric-aware temporal attention；
* relative time bias；
* interval-aware downsampling；
* irregular-to-regular resampling bridge；
* native-axis proposal head。

现在直接重做 neck，容易把 assignment/postprocess bug 藏进更复杂模型里。

---

## 6. 代码级修改建议

### 6.1 拆开 axis contract

```python
def _axis_contract_from_meta(meta):
    native = bool(meta.get("irregular_native_axis", False))
    default_axis = "native" if native else "selected"

    gt_axis = meta.get("irregular_gt_axis", default_axis)
    proposal_axis = meta.get("irregular_proposal_axis", gt_axis)

    # postprocess should be allowed to differ
    postprocess_axis = meta.get(
        "irregular_postprocess_axis",
        "native" if proposal_axis == "selected" else proposal_axis,
    )
    return gt_axis, proposal_axis, postprocess_axis


def _assert_axis_contract(meta):
    gt_axis, proposal_axis, postprocess_axis = _axis_contract_from_meta(meta)
    allowed = {"native", "selected"}

    if gt_axis not in allowed:
        raise ValueError(f"bad gt_axis={gt_axis}")
    if proposal_axis not in allowed:
        raise ValueError(f"bad proposal_axis={proposal_axis}")

    # train target and proposal decode must match
    if gt_axis != proposal_axis:
        raise ValueError(
            f"gt_axis and proposal_axis must match during training/decode: "
            f"{gt_axis} vs {proposal_axis}"
        )

    # postprocess may differ
    if postprocess_axis not in {"native", "seconds", proposal_axis}:
        raise ValueError(f"bad postprocess_axis={postprocess_axis}")
```

### 6.2 dataset meta 不要三轴同值硬写

```python
def _set_irregular_axis_meta(data_dict, remap_gt_to_selected_axis):
    if remap_gt_to_selected_axis:
        data_dict["irregular_gt_axis"] = "selected"
        data_dict["irregular_proposal_axis"] = "selected"
        data_dict["irregular_postprocess_axis"] = "native"
        data_dict["irregular_native_axis"] = False
    else:
        data_dict["irregular_gt_axis"] = "native"
        data_dict["irregular_proposal_axis"] = "native"
        data_dict["irregular_postprocess_axis"] = "native"
        data_dict["irregular_native_axis"] = True
```

### 6.3 NMS 前先转 native/seconds 轴

```python
def proposal_to_native(segments, meta, proposal_axis):
    if proposal_axis == "native":
        return segments
    if proposal_axis == "selected":
        return selected_axis_to_dense_axis(segments, meta)
    raise ValueError(proposal_axis)


def dense_or_native_to_seconds(segments_native, meta):
    fps = meta.get("fps", -1)
    if fps == -1:
        return segments_native

    snippet_stride = meta.get("snippet_stride", 1)
    offset = meta.get("offset_frames", 0)
    window_start = meta.get("window_start", 0)

    return (segments_native * snippet_stride + window_start + offset) / fps
```

在 `post_processing` 里改成：

```python
gt_axis, proposal_axis, postprocess_axis = self._axis_contract_from_meta(meta)

segments_native = proposal_to_native(segments, meta, proposal_axis)

# NMS must operate on comparable temporal coordinates
if post_cfg.nms is not None:
    segments_native, scores, labels = batched_nms(
        segments_native, scores, labels, **post_cfg.nms
    )

segments_seconds = dense_or_native_to_seconds(segments_native, meta)
```

不要把 selected-axis segments 直接送进 NMS。

### 6.4 `_remap_gt_to_selected_axis` 不要制造 1e-3 GT

```python
def _remap_gt_to_selected_axis(gt_segments, kept_positions, valid_len):
    remapped = []
    ignored = []

    for start, end in gt_segments:
        s = map_coord(start, kept_positions, valid_len)
        e = map_coord(end, kept_positions, valid_len)

        if e <= s + 1e-6:
            ignored.append((start, end, "collapsed_after_selected_axis_remap"))
            continue

        remapped.append((s, e))

    return np.asarray(remapped, dtype=np.float32), ignored
```

并把 `ignored` 写进 meta/debug，不要 silent drop。

### 6.5 修正 temporal grid cell 语义

```python
def centers_to_voronoi_cells(pos, native_end=None):
    # pos: [K], sorted native centers
    K = pos.numel()
    if K == 1:
        left = pos.new_tensor([0.5])
        right = pos.new_tensor([0.5])
        return left, right

    edges = pos.new_empty(K + 1)
    edges[1:-1] = 0.5 * (pos[:-1] + pos[1:])

    first_gap = pos[1] - pos[0]
    last_gap = pos[-1] - pos[-2]

    edges[0] = pos[0] - 0.5 * first_gap
    edges[-1] = pos[-1] + 0.5 * last_gap

    if native_end is not None:
        edges[0] = edges[0].clamp_min(0)
        edges[-1] = edges[-1].clamp_max(float(native_end))

    left = pos - edges[:-1]
    right = edges[1:] - pos

    return left.clamp_min(1e-6), right.clamp_min(1e-6)
```

然后 `downsample_temporal_grid` 才能合理 merge intervals。

### 6.6 bridge hard 默认不要再用 `full_cell_span`

建议把 risky default 改掉：

```python
center_radius_scale = "point_radius"      # or "level_stride"
reg_denom_mode = "left_right_mean"        # or "level_stride"
```

并强制 config 显式声明：

```python
assert center_radius_scale != "full_cell_span" or allow_legacy_full_cell_span
```

### 6.7 扩展 audit：加入 assigned-GT oracle IoU

伪代码：

```python
with torch.no_grad():
    cls_targets, reg_targets, assigned_gt_inds = head.prepare_targets(points, gt_segments, gt_labels)

    decoded = head.decode_from_targets(points, reg_targets)

    decoded_native = proposal_to_native(decoded, meta, proposal_axis)

    assigned_gt = gather_assigned_gt(gt_segments, assigned_gt_inds)
    if gt_axis == "selected":
        assigned_gt = selected_axis_to_dense_axis(assigned_gt, meta)

    ious = segment_iou(decoded_native[pos_mask], assigned_gt[pos_mask])

report = {
    "assigned_decode_iou_mean": ious.mean(),
    "assigned_decode_iou_p05": ious.quantile(0.05),
    "assigned_decode_iou_p50": ious.quantile(0.50),
    "assigned_decode_iou_p95": ious.quantile(0.95),
}
```

如果这个 IoU 不接近 1，说明 assignment/regression encode/decode 轴就已经错了，没必要长训。

### 6.8 `verify_official_dense_reference.py` 改成 pinned manifest

```python
OFFICIAL_COMMIT = "固定的 OpenTAD 官方 commit sha"

FILES = {
    "opentad/models/dense_heads/anchor_free_head.py": "sha256...",
    "opentad/models/dense_heads/prior_generator/point_generator.py": "sha256...",
    "opentad/models/necks/fpn.py": "sha256...",
    "configs/_base_/models/actionformer.py": "sha256...",
    "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py": "sha256...",
}
```

只 diff 三个 Python 文件不够。官方 AdaTAD THUMOS config 明确规定了 `window_size=768`、train random_trunc、val/test sliding_window、ActionFormer neck/head/post-processing 等关键合同。([GitHub][18])

---

## 7. 最小实验矩阵

| 实验                                    | 验证变量                                  | 配置文件 / 路线                                                                                  | 预期观察                                                       | 成功 gate                                 | 失败后转向                              |
| ------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | --------------------------------------- | ---------------------------------- |
| E0 官方 dense reproduction              | 数据、eval、环境是否可信                        | 官方 AdaTAD THUMOS config                                                                    | 接近官方 VMAE-S AdaTAD avg 69.03，或至少接近你本地 dense/equal baseline | 与参考差距 ≤2–3 mAP                          | 先修数据/eval/ckpt，不碰 sparse           |
| E1 bridge dense-equivalence           | bridge hard 是否复刻官方 head               | dense input + IrregularBridgeHead official-equivalent                                      | positive mask、assigned GT、reg target、decode proposal 与官方一致 | same-batch target 一致率 >99%，mAP 差 ≤0.5–1 | 修 bridge head，不跑 sparse            |
| E2 uniform 50 selected-axis sanity    | selected-axis remap/postprocess       | `uniform_fixed_subsample`, `remap_gt_to_selected_axis=True`, NMS 前转 native                 | 接近等间隔 50% baseline                                         | avg ≥60，@0.7 不崩                         | 修 remap / NMS / convert_to_seconds |
| E3 random 50 selected-axis sanity     | random sparse 本身损失                    | `random_fixed_subsample`, selected-axis official-style head                                | 比 uniform 掉一些，但不应掉到 40                                     | 与 E2 差距 ≤5–8 mAP                        | 检查 random sampling/GT visibility   |
| E4 native absrange_expanded short run | bounded native-axis assignment        | `input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py` | high-IoU 优于 openrange                                      | epoch 10/20 超过 42.44 或 @0.6/@0.7 明显提升   | 修 temporal grid / scale，不长训        |
| E5 postprocess axis ablation          | NMS/seconds 转换是否致命                    | 同一 checkpoint，对比 NMS 前/后 native conversion                                                 | selected-axis 应明显受益；native-axis 应基本一致                      | @0.6/@0.7 差异解释清楚                        | 若差异大，先修 postprocess                |
| E6 metric-aware projection/neck       | feature mixer 是否理解 irregular distance | selected-axis strong baseline vs native metric-aware neck                                  | native route 才有资格超过 selected-axis                          | high-IoU 稳定提升                           | 若无提升，放弃 native-head 主线             |

---

## 最终建议

**不要继续把 bridge hard native-axis 参数一组组长训当作主路线。**
当前最可能不是“某个 range 配置还没调好”，而是：

```text
axis contract 未拆开
NMS 轴错误
cell scale 语义错误
bridge hard 只形式像官方
audit 没测真正 high-IoU 边界合同
projection/neck 与 native time metric 可能不一致
```

我建议的执行顺序是：

1. 修 axis meta 与 NMS-before-conversion；
2. 修 temporal grid half-cell；
3. 扩展 audit 到 assigned-GT decode IoU；
4. 跑 official dense selected-axis sanity；
5. 跑 bridge dense-equivalence sanity；
6. 只有这些通过后，才让 `absrange_expanded` 进入 bounded short run；
7. short run 过 gate 后再长训。

若 selected-axis official sanity 都不能回到 60+，native-axis irregular head 不应该继续作为主路线。

[1]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/tree/d7da39315f80405253bf705997d7c26e63c66ab0 "GitHub - yuzbo/OpenTAD_SparseHeadClean_20260702 at d7da39315f80405253bf705997d7c26e63c66ab0 · GitHub"
[2]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/d7da39315f80405253bf705997d7c26e63c66ab0/opentad/datasets/transforms/end_to_end.py "OpenTAD_SparseHeadClean_20260702/opentad/datasets/transforms/end_to_end.py at d7da39315f80405253bf705997d7c26e63c66ab0 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[3]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/d7da39315f80405253bf705997d7c26e63c66ab0/opentad/models/detectors/irregular_actionformer.py "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/d7da39315f80405253bf705997d7c26e63c66ab0/opentad/models/utils/post_processing/utils.py "raw.githubusercontent.com"
[5]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/d7da39315f80405253bf705997d7c26e63c66ab0/opentad/models/utils/temporal_grid.py "OpenTAD_SparseHeadClean_20260702/opentad/models/utils/temporal_grid.py at d7da39315f80405253bf705997d7c26e63c66ab0 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[6]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/prior_generator/point_generator.py "OpenTAD/opentad/models/dense_heads/prior_generator/point_generator.py at main · sming256/OpenTAD · GitHub"
[7]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/d7da39315f80405253bf705997d7c26e63c66ab0/opentad/models/dense_heads/prior_generator/irregular_point_generator.py "raw.githubusercontent.com"
[8]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/d7da39315f80405253bf705997d7c26e63c66ab0/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_n16r4.py at d7da39315f80405253bf705997d7c26e63c66ab0 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[9]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/dense_heads/anchor_free_head.py "OpenTAD/opentad/models/dense_heads/anchor_free_head.py at main · sming256/OpenTAD · GitHub"
[10]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/d7da39315f80405253bf705997d7c26e63c66ab0/opentad/models/dense_heads/irregular_actionformer_bridge_head.py "OpenTAD_SparseHeadClean_20260702/opentad/models/dense_heads/irregular_actionformer_bridge_head.py at d7da39315f80405253bf705997d7c26e63c66ab0 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[11]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/d7da39315f80405253bf705997d7c26e63c66ab0/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_bridge_hard_linear_absrange_expanded_n16r4.py at d7da39315f80405253bf705997d7c26e63c66ab0 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[12]: https://raw.githubusercontent.com/yuzbo/OpenTAD_SparseHeadClean_20260702/d7da39315f80405253bf705997d7c26e63c66ab0/tools/audit_sparse_head_assignment.py "raw.githubusercontent.com"
[13]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/d7da39315f80405253bf705997d7c26e63c66ab0/scripts/verify_official_dense_reference.py "OpenTAD_SparseHeadClean_20260702/scripts/verify_official_dense_reference.py at d7da39315f80405253bf705997d7c26e63c66ab0 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[14]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/d7da39315f80405253bf705997d7c26e63c66ab0/root-cause-notes.md "OpenTAD_SparseHeadClean_20260702/root-cause-notes.md at d7da39315f80405253bf705997d7c26e63c66ab0 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[15]: https://github.com/sming256/OpenTAD/blob/main/configs/_base_/models/actionformer.py "OpenTAD/configs/_base_/models/actionformer.py at main · sming256/OpenTAD · GitHub"
[16]: https://github.com/yuzbo/OpenTAD_SparseHeadClean_20260702/blob/d7da39315f80405253bf705997d7c26e63c66ab0/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_actionformer_base.py "OpenTAD_SparseHeadClean_20260702/configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_actionformer_base.py at d7da39315f80405253bf705997d7c26e63c66ab0 · yuzbo/OpenTAD_SparseHeadClean_20260702 · GitHub"
[17]: https://github.com/sming256/OpenTAD/blob/main/opentad/models/necks/fpn.py "OpenTAD/opentad/models/necks/fpn.py at main · sming256/OpenTAD · GitHub"
[18]: https://github.com/sming256/OpenTAD/blob/main/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py "OpenTAD/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py at main · sming256/OpenTAD · GitHub"
