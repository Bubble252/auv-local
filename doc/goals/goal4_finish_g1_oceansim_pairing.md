# 任务：完成 G1 OceanSim 配对误差门并收口 G1–G3

记录日期：2026-10-28

状态：**PARTIAL（2026-10-28 执行后：配对完成，`max` 门未通过）**

关系：本文件是 `doc/goals/goal3_mvp_finish_g1_g3.md` 之后的新执行提示词。G2、G3 的本地实现和验收已经完成，本轮只补齐 G1 的 OceanSim 真实配对误差门、更新证据并输出最终报告。不要再重做已通过的 G2/G3。

工作目录：

```text
/home/bubble/水下机器人/settled_局部规划器_反应式
```

一句话目标：在具备 OceanSim 的服务器上执行现有配对脚本，验证解析 13 扇区与 OceanSim P1-A 的 13 个扇区距离是否通过冻结误差门；通过后只更新 G1、M4.4.1 和最终报告，然后停止。本轮不是 G4，不测大规模环境数，不启动训练 campaign。

---

## 0. 执行前先读

必须阅读：

- `doc/goals/goal2_analytic_13sector_multienv.md`
- `doc/goals/goal3_mvp_finish_g1_g3.md`
- `doc/02_执行计划.md` 的 M4.4.1、M4.4.2、M4.4.3
- `scripts/calibrate_p1_analytic_sensor.py`
- `tests/test_m4_analytic_calibration.py`
- `src/auv_rl/perception.py`
- `src/auv_rl/observation.py`

先审计当前工作树，输出一张差距表：

```text
项目 | 已完成证据 | 当前缺口 | 本轮动作
```

以代码、测试和 artifact 为准。不要因为旧文档未勾选就重写已经通过的实现。

---

## 1. 当前基线

截至 2026-10-28：

- G0 文档和冻结契约已完成。
- G1 本地解析后端、13 扇区观测、配对脚本和专项测试已完成。
- G2 `P1TensorVectorEnv` 已实现，`N=1` 固定动作与旧单环境逐项一致。
- G3 每环境 history/GRU hidden、批量 replay、PPER、独立 NoisyNet 噪声和 `train_parallel_ddpg` 已实现。
- G3 smoke 已验证：16 个环境转移、12 个梯度步、eligible `update_to_data_ratio=1.0`、total `0.75`。
- 本地测试记录为 `144 passed`；ruff、pyright、mypy 已通过。
- G1 dry-run 记录为 S1 100 个位姿、S2 421 个位姿、6773 个扇区行；S1 解析扇区全部饱和到 `20 m`。

唯一未完成项：**尚未在 OceanSim 服务器上执行真实配对采集和误差门。**

注意：现有 `artifacts/m4/p1/parallel/dry_run/` 可能在最后一次脚本格式修改前生成，`code_hash` 已过期。引用 dry-run 结果前必须重新运行。

---

## 2. 冻结契约

不得修改：

- 动作 `[u_dot, r]`，范围 `±0.5 m/s²`、`±15°/s`。
- RL 步长 `0.2 s`，积分步长 `0.02 s`。
- 17 维观测：13 个前视扇区距离 + 目标距离 + 目标相对艏向 + `u` + `r`。
- 130° 前视场，13 个 10° 扇区。
- 局部有效视野 `20 m`，原始物理量程 `120 m`。
- 1300 条 `0.1°` 射线，传感器原点 `[0.8, 0, 0] m`，垂直切片 `±0.5 m`。
- 扇区内取最小距离、奖励、碰撞、越界、成功判定、episode 超时和归一化口径。
- 训练来源 `analytic_13sector`，最终评估来源 `oceansim_simplified_pcl`。

两种感知来源必须共用 `ForwardSectorObservationBuilder` 和同一套归一化。G1 失败时只允许修射线方向、原点、坐标变换、扇区边界、距离基准或量程裁剪；不得放宽门禁、缩小动作范围、挑选有利位姿或把 oracle 放入评估路径。

---

## 3. 执行清单

### [ ] 3.1 工作树与本地回归

```bash
cd "/home/bubble/水下机器人/settled_局部规划器_反应式"
git status --short
git diff --check
../.venv/bin/python -m pytest -q
../.venv/bin/python -m ruff check src tests scripts/calibrate_p1_analytic_sensor.py
../.venv/bin/python -m ruff format --check src tests scripts/calibrate_p1_analytic_sensor.py
../.venv/bin/python -m pyright
../.venv/bin/python -m mypy
```

要求：

- 若测试数与“144 passed”不同，以真实输出为准并记录。
- 不执行 `git checkout`、`git reset`、`git clean`。
- 不自动 commit、不自动 push。
- 不终止服务器上的其他用户任务。

### [ ] 3.2 重新生成本地 dry-run

```bash
cd "/home/bubble/水下机器人/settled_局部规划器_反应式"
../.venv/bin/python scripts/calibrate_p1_analytic_sensor.py \
  --dry-run \
  --output artifacts/m4/p1/parallel/dry_run
```

必须核对：

- S1 位姿数不少于 100；S2 位姿数不少于 400。
- 扇区行数应为 `(S1 位姿数 + S2 位姿数) × 13`。
- 当前期望为 `100 + 421 = 521` 个位姿、`6773` 个扇区行。
- S1 的 13 个解析扇区全部饱和到 `20 m`。
- `observation_source` 为 `analytic_13sector`。
- 新的 `dry_run.json` 中 `code_hash` 与当前脚本一致。

如果数量不符，先查明位姿生成或场景解析问题，不要直接进入服务器采集。

### [ ] 3.3 服务器连接与资源预检

连接信息：

```text
host: 152.136.183.51
port: 43127
user: yhx
remote workdir: /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
python: /data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python
```

凭据只能由当前交互 shell 临时提供。不得把密码写入文件、文档、日志、环境示例或 Git。优先使用用户已经配置好的 SSH key；若必须使用密码，只在当前 shell 设置临时的 `SSHPASS` 并通过 `sshpass -e` 使用，完成后取消设置。

认证失败时：

- 记录原始错误，不猜测密码。
- 不修改服务器环境，不尝试绕过权限。
- 报告“需要服务器凭据”并停止；不要伪造 G1 结果。

连接成功后先检查：

```bash
nvidia-smi
pgrep -af "train_p1|probe_p1_vector|python" | head -50
df -h .
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
git status --short
/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python -c \
  "import isaacsim, oceansim, torch; print('ok', torch.__version__, torch.cuda.is_available())"
```

不得 kill 其他用户进程。若 GPU 显存不足，先报告并等待，不得争抢资源。

### [ ] 3.4 执行真实 OceanSim 配对

```bash
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python \
  scripts/calibrate_p1_analytic_sensor.py \
  --output artifacts/m4/p1/parallel
```

要求：

- 同一 `pose_id` 必须使用完全相同的状态分别采集 OceanSim 与解析后端。
- OceanSim 来源必须标记为 `oceansim_simplified_pcl`，解析来源必须标记为 `analytic_13sector`。
- 不得修改 P1-A 的声呐参数，不得把解析距离写成 OceanSim 距离。
- 保存 stdout/stderr 到 `artifacts/m4/p1/parallel/calibration.log`，但日志中不得出现密码或 token。

### [ ] 3.5 检查 artifact 完整性

必须存在：

```text
artifacts/m4/p1/parallel/analytic_vs_oceansim.csv
artifacts/m4/p1/parallel/analytic_equivalence.json
artifacts/m4/p1/parallel/pairing_poses.jsonl
```

核对：

- `analytic_equivalence.json` 的 schema 为 `auv_p1_analytic_equivalence.v1`。
- S1 位姿数不少于 100；S2 位姿数不少于 400。
- `sector_rows` 等于 `(S1 位姿数 + S2 位姿数) × 13`。
- CSV 每个 `pose_id` 有 13 个扇区，扇区索引无缺失、无重复。
- 记录 `ocean_config_hash`、`analytic_config_hash`、`code_hash` 和命令。
- 检查 `pairing_poses.jsonl` 中 OceanSim 扫描有效性；若大量 `ocean_scan_valid=0`，不得把这种数据当作通过证据。

### [ ] 3.6 判定 G1 门

只有以下条件同时成立，G1 才能标记完成：

- 扇区距离 `MAE <= 0.10 m`。
- 绝对误差 `P95 <= 0.25 m`。
- 最大绝对误差 `<= 0.75 m`。
- 危险漏检数为 0。危险漏检定义为 `ocean_range_m < 19.9` 且 `analytic_range_m >= 19.9`。
- S1 的 13 个解析扇区全部饱和到 `20 m`。
- 17 维观测顺序、归一化和 dtype 一致。
- OceanSim 配对数据不是空数据、全无效扫描或被解析结果替换。

同时报告总体和 S1/S2 分场景的 MAE、P95、max 与危险漏检数。

### [ ] 3.7 门失败时的处理

只允许做几何与采集正确性修复，最多进行 3 轮“修复 → 重新采集”，每轮都要记录：

- 假设是什么；
- 改了哪个坐标、原点、射线方向、扇区边界或裁剪逻辑；
- 修改前后的误差指标；
- 是否出现新的危险漏检。

不得用以下方式“通过”：

- 放宽容差或改危险漏检定义；
- 删除不利位姿；
- 改动作范围、奖励或成功判定；
- 调策略参数；
- 用解析结果覆盖 OceanSim 字段；
- 用 P1-A 参数之外的数据冒充配对证据。

三轮后仍失败则停止，报告 `PARTIAL`，保留原始 artifact 和失败原因。不得宣称 G1 或整个 G1–G3 完成。

### [ ] 3.8 通过后的文档更新

只有拿到真实 artifact 后才更新：

- `doc/goals/goal2_analytic_13sector_multienv.md`
  - 勾选“完成 OceanSim 配对数据采集和误差报告”。
  - 写入 artifact 路径、样本数、MAE、P95、max、危险漏检数和各门结论。
- `doc/02_执行计划.md`
  - 勾选 M4.4.1 的 `analytic_vs_oceansim.csv` 和 `analytic_equivalence.json` 输出项。
  - 写入同样的实测结果。
- 必要时更新 `doc/01_技术栈.md` 中的“服务器配对待执行”状态。

每项勾选必须对应一条测试、一个 artifact 或一条可复现命令。不得把“代码已写”写成“验收已通过”。

---

## 4. 输出与 Git 边界

允许写入：

```text
artifacts/m4/p1/parallel/
```

不得覆盖：

```text
outputs/m4/p1/ablations/
artifacts/m4/p1/convergence/
```

不进入 G4，不启动长训练，不测试大规模环境数，不启动 OceanSim 多环境声呐训练，不进入 P2。

不要执行破坏性 Git 操作。不要自动 commit 或 push。最终只给出建议的提交文件列表和 commit message，等待用户确认。

---

## 5. 停止条件

出现任一情况立即停止并报告：

- 服务器凭据不可用或认证失败；
- OceanSim/Isaac Sim 无法导入；
- GPU 资源被他人占用且无法安全执行；
- 配对 artifact 缺失、扫描无效或样本不足；
- 三轮几何修复后误差门仍失败；
- 需要修改冻结观测、动作、奖励或成功定义；
- 需要进入 G4、训练 campaign、OceanSim 多环境声呐训练或 P2。

---

## 6. 最终报告格式

1. 状态：`COMPLETE` / `PARTIAL` / `BLOCKED`
2. 修改文件列表
3. 本地测试、ruff、pyright、mypy 结果
4. dry-run 位姿数和扇区行数
5. G1 配对样本数、总体及 S1/S2 的 MAE、P95、max
6. 危险漏检数、S1 饱和检查和 17 维观测一致性结论
7. G2 回归结果
8. G3 smoke 与关键隔离测试结果
9. 未解决风险
10. 建议的 commit message 和文件列表（不自动执行）

最终报告最后两行必须严格为：

```text
P2 未开始。
多环境训练仅限解析 13 扇区；OceanSim 多环境声呐训练未开始。
```
