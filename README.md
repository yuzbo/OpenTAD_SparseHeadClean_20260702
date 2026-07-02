# OpenTAD Sparse-Head Clean Route

这是从 `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_Back` 抽出的稀疏 head / 不规则时间轴路线干净代码库。它只保留 OpenTAD 库主体、当前稀疏头改造相关配置、少量启动脚本和合约测试；历史 wiki、日志、图表、checkpoint、压缩包和同步缓存都不进入版本库。

## 当前目标

当前研究目标是让 THUMOS14 时序动作检测在非均匀、稀疏、可变预算输入下仍能稳定工作。固定 384/768 的 50% 输入是第一阶段可控锚点，用来验证不规则时间轴、稀疏 head、assignment/loss、post-processing 和高 IoU 定位是否可靠；最终目标是走向任务感知动态采集，让简单或冗余片段少算，边界敏感或困难片段多算，同时保持或提升检测精度。

重点路线：

- `IrregularActionFormer` detector 使用真实时间轴元数据，而不是假定均匀 dense axis。
- `IrregularActionFormerHeadV2/V3`、`IrregularPointGeneratorV2`、boundary-aware head 处理稀疏点集和物理时间距离。
- adapter sparse route 保留 VideoMAE/Adapter 特征，同时让 projection、neck、head 理解不规则时间几何。
- native dense headv2 safe route 作为稀疏输入下的 dense-axis 对照和安全回退。

## 目录

- `opentad/`: OpenTAD 库主体，含稀疏/不规则时间轴改造。
- `configs/_base_/`: OpenTAD 基础数据集和模型配置。
- `configs/adatad/thumos/`: 当前稀疏 head 路线配置。
- `tools/train.py`, `tools/test.py`: 训练与评测入口。
- `scripts/`: 稀疏路线启动、排队和数据合约检查脚本。
- `tests/test_adapter_native_dense_headv2_contracts.py`: 当前保留的最小合约测试。
- `RTK.md`: 本仓库项目规则和远端边界。

## 本地使用

```powershell
cd E:\DeskTop\TAD\OpenTAD_SparseHeadClean_20260702
pip install -r requirements.txt
python -m py_compile tools/train.py tools/test.py scripts/check_p0426_r05a_data_contract.py scripts/verify_denseadapter_coords.py
python -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q
```

示例配置加载或训练：

```powershell
python tools/train.py configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py --id 0
python tools/test.py <config> <checkpoint>
```

## 推荐配置入口

- `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py`: N16R4 推荐 adapter sparse headv3 配置。
- `configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_dense_control_pdrop0_n16r4.py`: N16R4 dense-control 对照。
- `configs/adatad/thumos/input_random_fixed_50pct_adapter_native_dense_headv2_safe.py`: native dense headv2 安全门控配置。
- `configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_headv3_x.py`: native irregular ActionFormer headv3 配置。

两个 `*_n16r4.py` 配置默认使用 `/data/run01/sczc063/yuzibo/thumos14`。如果远端数据根不同，直接改配置顶部的 `thumos_root`，不要在配置里重新引入 `import os`。

## N16R4 远端环境

本地登录用 PowerShell，不要从 WSL 发起远端控制：

```powershell
ssh -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa -o HostkeyAlgorithms=+ssh-rsa -i C:\Users\skywalker\.ssh\id_rsa -p 22 -l "sczc063@BSCC-N16R4" ssh.cn-zhongwei-1.paracloud.com
```

远端建议：

```bash
BASE=/data/run01/sczc063/yuzibo
cd "$BASE"
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"

export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export HF_HOME="$BASE/hf_cache"
export THUMOS_ROOT="$BASE/thumos14"
```

需要外网下载时，在登录节点设置代理：

```bash
export http_proxy='http://u-MtfrT7:vH5orjDV@10.244.6.36:3128'
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

THUMOS14 路径约定：

```bash
$THUMOS_ROOT/annotations/thumos_14_anno.json
$THUMOS_ROOT/annotations/category_idx.txt
$THUMOS_ROOT/train
$THUMOS_ROOT/test
```

正式训练必须用 Slurm，例如：

```bash
#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J sparse_head
#SBATCH -o logs/%x-%j.out

BASE=/data/run01/sczc063/yuzibo
cd "$BASE/OpenTAD_SparseHeadClean_20260702"
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"
export THUMOS_ROOT="$BASE/thumos14"
export OMP_NUM_THREADS=8

python tools/train.py configs/adatad/thumos/input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0_n16r4.py --id 0
```

## 协议红线

验证/测试不能使用 GT、teacher cache、raw prediction shortcut 或旧运行缓存。所有结果解释必须区分完整训练、短 smoke、诊断 run 和 oracle diagnostic；短 run 只能证明可启动、稳定性和合约健康，不能当作路线最终成败。
