# RTK: Sparse-Head OpenTAD Clean Repo

## 项目规则

- 本库只保留 OpenTAD 代码库主体、稀疏/不规则时间轴 head 改造、对应 THUMOS14 配置、少量启动和校验脚本。
- 不放历史 `research-wiki/`、`logs/`、`figures/`、实验压缩包、checkpoint、数据集或远端同步缓存。
- 固定 384/768 的 50% 输入只是可控实验锚点，不是最终目标。最终目标是面向时序动作检测的任务感知动态采集和稀疏检测头适配。
- 验证/测试阶段严禁使用 GT、teacher cache、raw prediction shortcut 或隐藏缓存决策。只有明确标记为 `oracle_diagnostic` 的诊断配置可以使用 oracle 信息。
- 稀疏 head 路线的归因必须说明改动面：输入采样、时间轴元数据、projection、neck、head、loss/assignment、post-processing 之中改了哪一层。
- 新增长训前先跑配置加载、形状/时间轴合约和最小 smoke；正式训练必须走 Slurm 或受控远端队列，不能在登录节点直接训练。
- 本项目发起 Pro 讨论时，遵守下方 `Chrome 9223 Pro 讨论共享规则`。

## Chrome 9223 Pro 讨论共享规则

- 多个 agents 共享 `9223` 的核心是：共享端口，但不共享同时控制权。
- Chrome 只开一个：`chrome.exe --remote-debugging-port=9223 --user-data-dir=C:\path\to\shared-profile`。
- Agent 操作前必须抢全局锁：`E:\DeskTop\TAD\OpenTAD_SparseHeadClean_20260702\.codex\chrome-9223.lock`。
- 拿到锁后调用 `http://127.0.0.1:9223/json/list`，找到或新建自己的页面，并记录 `targetId`、`page id`、`webSocketDebuggerUrl`、`url`、`agent name`、`pid`。
- Agent 只操作绑定的 `target/page id`，不要临时切到当前激活页。
- 操作完成后释放锁；若 agent 崩溃，锁必须包含 TTL/heartbeat，过期后允许下一个 agent 清理。
- 推荐锁内容包含 `owner`、`pid`、`targetId`、`pageId`、`startedAt`、`expiresAt`。
- 稳定性层级：最稳是每个 agent 独立 Chrome 端口 + 独立 profile；次稳是共享 `9223` 但每个 agent 独立 target/page 并加全局锁；最不稳是共享 `9223`、共享页面、不加锁。

## 关键代码面

- `opentad/datasets/transforms/end_to_end.py`: 50% 固定采样、不规则时间轴元数据、GT 是否重映射。
- `opentad/models/utils/temporal_grid.py`: dense/native axis 的 temporal grid 合约。
- `opentad/models/detectors/irregular_actionformer.py`: 不规则时间轴 detector 路由。
- `opentad/models/projections/irregular_actionformer_proj.py`: grid-aware / dense-passthrough projection。
- `opentad/models/necks/irregular_fpn.py`: grid-aware / dense-passthrough FPN。
- `opentad/models/dense_heads/irregular_actionformer_head*.py`: 稀疏 head、soft assignment、boundary-aware 或诊断 head。
- `opentad/models/dense_heads/prior_generator/irregular_point_generator.py`: 不规则点集生成。

## 本地检查

```powershell
cd E:\DeskTop\TAD\OpenTAD_SparseHeadClean_20260702
python -m py_compile tools/train.py tools/test.py scripts/check_p0426_r05a_data_contract.py scripts/verify_denseadapter_coords.py
python -m pytest tests/test_adapter_native_dense_headv2_contracts.py -q
```

## 远端边界

- Windows 本地不要用 WSL 做 SSH/SCP/rsync 或远端控制；用 PowerShell 和 `C:\Windows\System32\OpenSSH\ssh.exe` / `scp.exe`。
- N16R4 远端只在 `~/run/yuzibo` 或 `/data/run01/sczc063/yuzibo` 下放代码、环境、数据、日志和输出。
- 登录节点只做编辑、编译、轻量检查和提交 Slurm；训练必须用 `sbatch`。
- THUMOS14 数据默认放在 `$BASE/thumos14`，其中 `BASE=/data/run01/sczc063/yuzibo`。
