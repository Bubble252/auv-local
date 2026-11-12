# 任务：MVP 收尾，完成 G1–G3 的证据闭环

记录日期：2026-10-28

状态：**SUPERSEDED（2026-10-28）**

后续执行入口：`doc/goals/goal4_finish_g1_oceansim_pairing.md`。本文件保留为 G1–G3 的历史任务书；G2/G3 本地验收已完成，剩余的真实 OceanSim 配对与 G1 收口只按新文件执行。

关系：本文件是 `doc/goals/goal2_analytic_13sector_multienv.md` 的续跑任务书，不取代 goal2 的冻结契约。只完成 G1、G2、G3 的剩余工作和验收，不进入 G4。

工作目录：

```text
/home/bubble/水下机器人/settled_局部规划器_反应式
```

一句话目标：在现有工作树上继续，完成解析 13 扇区与 OceanSim 的配对误差门、复核张量向量环境、修复向量化 DDPG 的 `act_batch` 历史语义并补齐 G3 测试；G3 验收通过后立即停止，不启动长训练，不进入 P2。

---

## 0. 先读，不要凭记忆重做

必须阅读：

- `doc/goals/goal2_analytic_13sector_multienv.md`
- `doc/00_项目背景.md`
- `doc/01_技术栈.md`
- `doc/02_执行计划.md`
- `src/auv_rl/perception.py`
- `src/auv_rl/observation.py`
- `src/auv_rl/envs/parallel_planar.py`
- `src/auv_rl/ddpg/agent.py`
- `src/auv_rl/ddpg/networks.py`
- `src/auv_rl/ddpg/replay.py`
- `src/auv_rl/ddpg/vector_trainer.py`
- `tests/test_m4_perception.py`
- `tests/test_m4_observation.py`
- `tests/test_m4_planar_env.py`
- `tests/test_m4_ddpg.py`
- `scripts/calibrate_p1_sonar.py`

第一步不是写新功能，而是审计当前工作树并输出一张差距表：

```text
项目 | 已有实现 | 已有证据 | 缺口 | 本轮动作
```

如果审计结果与下面的“当前基线”不一致，以代码和测试为准，并在最终报告中记录差异。不要因为旧文档未勾选就重写已经通过测试的代码。

---

## 1. 当前基线

截至 2026-10-28，工作树中已有以下前序工作：

- G0 文档和协议已完成。
- G1 本地解析感知链路已实现：
  - `AnalyticSectorPerceptionBackend`
  - 1300 条射线、13 扇区、17 维观测
  - sphere、cylinder、box 求交
  - 世界 AABB 不当作回波障碍
  - `observation_source="analytic_13sector"`
- G1 本地测试已覆盖空场景、S2 圆柱、左右扇区、2–20 m、量程裁剪、观测顺序和批量一致性。
- G2 `P1TensorVectorEnv` 已实现，且 `N=1` 固定动作与旧 `PlanarAvoidanceEnv` 在 S1/S2 上逐项一致。
- G3 已新增：
  - `NoisyLinear` 的 `per_sample_noise`
  - NumPy 预分配 ring replay 与 `add_batch`
  - `begin_episode_for_env`、`act_for_env`、`act_batch`
  - `observe_for_env`、`observe_batch`
  - `train_parallel_ddpg`
- G3 裸 smoke 曾得到：16 个环境转移、`learning_starts=4`、12 个梯度步、eligible ratio 1.0、total ratio 0.75。

已知未完成项：

1. `act_batch` 当前只用“当前步观测 + 持久 GRU hidden”，而旧 `act` 使用“该环境完整 history 序列 + `hidden=None`”。两者在 `N=1` 时可能不等价，必须修复。
2. G3 新增接口尚未补齐专项测试。
3. G1 还没有 OceanSim P1-A 配对采集脚本和服务器误差报告。
4. `goal2` 中的勾选和结果尚未根据最终证据更新。

不要把“代码已存在”当成“验收已通过”。本轮必须用测试和报告证明。

---

## 2. 冻结契约

不得修改：

- 动作为 `[u_dot, r]`，范围 `±0.5 m/s²`、`±15°/s`。
- RL 步长 `0.2 s`，积分步长 `0.02 s`。
- 观测为 17 维：13 个前视扇区距离 + 目标距离 + 目标相对艏向 + `u` + `r`。
- 130° 前视场，13 个 10° 扇区。
- 局部有效视野 `20 m`，原始物理量程 `120 m`。
- 1300 条 `0.1°` 射线，传感器原点 `[0.8, 0, 0] m`，垂直切片 `±0.5 m`。
- 扇区内取最小距离。
- 奖励、碰撞、越界、成功判定、episode 超时和归一化口径。
- 训练来源固定为 `analytic_13sector`。
- 最终评估来源固定为 `oceansim_simplified_pcl`。

两种感知来源必须共用同一个 `ForwardSectorObservationBuilder` 和同一套 17 维归一化。不得为训练和评估各写一套观测构造。

不得通过缩小动作范围、放宽成功阈值、降低碰撞/越界判定、把 oracle 放入评估路径来通过验收。

---

## 3. G1：完成解析与 OceanSim 的配对误差门

### 3.1 新增配对采集脚本

新增：

```text
scripts/calibrate_p1_analytic_sensor.py
```

脚本职责：

1. 在同一组机器人位姿和场景下，同时调用：
   - 现有 OceanSim P1-A 感知链路；
   - 新的 `AnalyticSectorPerceptionBackend`。
2. 输出 13 个扇区的配对距离，不是只比较最近障碍中心距离。
3. 记录机器人状态、场景、目标、扇区索引、传感器配置哈希和代码哈希。
4. 支持固定随机种子，保证同一批位姿可复现。
5. 不修改 P1-A 的声呐参数，不把解析结果伪装成 OceanSim 结果。

建议输出：

```text
artifacts/m4/p1/parallel/analytic_vs_oceansim.csv
artifacts/m4/p1/parallel/analytic_equivalence.json
artifacts/m4/p1/parallel/pairing_poses.jsonl
```

CSV 至少包含：

```text
scenario,pose_id,x_m,y_m,yaw_rad,sector_index,sector_center_deg,
ocean_range_m,analytic_range_m,abs_error_m,
ocean_saturated,analytic_saturated,dangerous_miss,
observation_source_eval,observation_source_train,
sensor_config_hash,code_hash
```

JSON 至少包含：

- 样本数、扇区数、场景数；
- MAE、P95、max error；
- 危险漏检数；
- S1 是否全部饱和到 20 m；
- 每道门的布尔结果；
- 采集命令、时间戳、配置哈希和代码哈希。

### 3.2 采集规模

- S1 空场景：不少于 100 个位姿。
- S2 单圆柱：不少于 400 个位姿。
- S2 必须覆盖：
  - 起点；
  - 正面接近；
  - 左右绕行；
  - 扇区边界；
  - 2–20 m 距离。
- 同一 `pose_id` 必须使用完全相同的机器人状态分别采集两种感知。
- 不要把同一个位姿重复 400 次充数；按距离、方位和扇区边界分层。

### 3.3 验收门

- 13 扇区距离 `MAE <= 0.10 m`。
- 绝对误差 `P95 <= 0.25 m`。
- 最大绝对误差 `<= 0.75 m`。
- 危险漏检数为 0。这里的危险漏检定义为：OceanSim 扇区距离明显小于 20 m，而解析扇区饱和到 20 m；建议阈值 `ocean_range_m < 19.9` 且 `analytic_range_m >= 19.9`，并在 JSON 中写清楚。
- S1 的 13 个解析扇区全部饱和到 20 m。
- 17 维观测的顺序、归一化和 dtype 完全一致。

如果门失败：

1. 先检查射线方向、传感器原点、坐标变换、扇区边界、距离基准和量程裁剪；
2. 修几何后重新采集；
3. 不得通过放宽门、改策略、缩小动作范围或挑选有利位姿来通过；
4. G1 未通过时不要宣称 G1 完成，也不要进入最终报告。

### 3.4 服务器执行

本地只做脚本、单元测试和 dry-run。OceanSim 配对在服务器执行：

```text
ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python
```

运行前必须检查：

```bash
nvidia-smi
pgrep -af "train_p1|probe_p1_vector|python"
```

不得 kill 其他用户的任务。密码只能从当前 shell 临时读取，不能写入文件、文档、日志或 Git。配对失败时先报告，不要改别人的进程或环境。

---

## 4. G2：只做回归复核，不重写

G2 的主要实现已经存在。本轮只需：

1. 重新运行 `N=1` 固定动作等价测试。
2. 重新运行 S1/S2 状态、奖励、终止、观测、最小间隙、路径长度和 simulation time 的一致性检查。
3. 复核：
   - 环境 A reset 不影响环境 B；
   - `terminated`/`truncated` 分离；
   - 自动 reset 后新 episode 观测不写回旧 transition；
   - 多环境固定动作结果与单独运行一致。
4. 如果 G2 已有测试全部通过，不要为了“统一风格”重写 `P1TensorVectorEnv`；只修实际失败点。

---

## 5. G3：修复 `act_batch` 并补齐测试

### 5.1 必须修复的语义问题

旧 `act` 的行为是：

```text
把当前观测追加到该 episode 的 history
→ 取最近 history_length 步
→ 组成 sequence
→ 调用 actor(sequence, hidden=None)
→ 取最后一个时间步动作
```

当前 `act_batch` 的行为是：

```text
只用当前步观测
→ 使用持久 hidden
→ 单步前向
```

这不等价，可能改变 `N=1` 的动作，也可能改变多环境训练分布。必须修成以下语义之一：

- 首选：按每个环境自己的 history 构造完整 sequence；不同历史长度可按长度分组批量前向，或使用严格的左填充 + mask，保证最后一个有效步输出正确。
- 备选：证明“持久 hidden + 单步输入”与“完整 history + hidden=None”在 GRU 上数学等价，并用逐项测试证明；不能只写注释。

无论采用哪种实现，必须满足：

- `N=1` 时，连续多步 `act_batch([0], obs)` 与旧 `act(obs)` 在 `evaluate=True` 下动作逐项一致。
- 每个环境有独立 history；env A 的 history 不进入 env B。
- 每个环境有独立 GRU hidden；env A reset 只清空 env A。
- 训练时 NoisyNet 噪声按环境独立采样；不能整个 batch 共享一份噪声。
- 评估时关闭噪声，结果确定。

### 5.2 必须补充的测试

在 `tests/test_m4_ddpg.py` 中至少增加或确认：

- `N=1` 多步 `act_batch` 与旧 `act(evaluate=True)` 逐项一致，覆盖 GRU 和非 GRU 配置。
- 两个环境使用不同 history 时，分别与各自单独运行的结果一致。
- env A reset 后 env B 的动作、history、episode state 不变。
- NoisyNet 每环境噪声独立；固定权重和 RNG 后，16 行动作不能被一份共享噪声解释。
- replay 序列不跨 env、不跨 episode。
- batch PPER priority 与逐条 update 的结果在容差内一致。
- truncated transition 使用 `done=False` 做 bootstrap。
- terminated transition 使用 `done=True`，不 bootstrap。
- 自动 reset 后，新 episode 的观测不覆盖上一条 transition 的 `next_observation`。
- checkpoint 保存/重载后，批量动作与确定性评估结果一致。
- `train_parallel_ddpg` 输出 `manifest.json`、`training_metrics.json`、`gradient_updates.csv`、`model.pt`。
- 小规模 smoke：16 个环境转移、`learning_starts=4`、`update_to_data_ratio=1.0`，应得到 12 个梯度步，eligible ratio 1.0，total ratio 0.75。

如果发现 `observe_batch` 或 `train_parallel_ddpg` 把 truncated 当成 terminated，或者把 reset 后的观测写进 replay，必须修训练器，不只修测试。

### 5.3 输出边界

G3 新输出只能放到：

```text
outputs/m4/p1/parallel/
artifacts/m4/p1/parallel/
configs/rl/parallel/
```

不要覆盖：

```text
outputs/m4/p1/ablations/
artifacts/m4/p1/convergence/
```

---

## 6. 验证命令

本地至少运行并通过：

```bash
cd "/home/bubble/水下机器人/settled_局部规划器_反应式"
../.venv/bin/python -m pytest -q
../.venv/bin/python -m ruff check src tests
../.venv/bin/python -m ruff format --check src tests
../.venv/bin/python -m pyright
../.venv/bin/python -m mypy
```

如果某条命令因环境缺少工具失败，记录原始错误和已尝试的修复，不要伪报通过。下载依赖失败时优先使用 `7897` 端口代理或国内镜像；不要在文档或日志中写密码。

服务器端至少运行：

- G1 配对采集；
- G1 误差汇总；
- G2 回归测试；
- G3 小规模 smoke。

---

## 7. 文档更新

只有拿到证据后才更新：

- `doc/goals/goal2_analytic_13sector_multienv.md`
- `doc/02_执行计划.md` 的 M4.4
- 必要时 `doc/01_技术栈.md`

更新要求：

- 每项勾选对应一条测试或一个 artifact；
- 写清楚命令、样本数、MAE/P95/max、危险漏检数；
- 写清楚 N=1 等价和多环境隔离结果；
- G1 未通过服务器门时，不能勾选 G1 配对项；
- 不把“代码已写”写成“验收已通过”。

---

## 8. 工作树和 Git 边界

- 当前工作树有大量前序未提交修改，不要执行 `git checkout`、`git reset`、`git clean`。
- 不要删除或回滚别人的改动。
- 不要自动 `git commit`，不要自动 `git push`。
- 需要提交时只给出建议的 commit message 和文件列表，等待用户确认。

---

## 9. 停止条件

出现以下任一情况时停止并报告：

- G1 配对误差门失败，完成几何修正后仍无法通过；
- S1/S2 出现危险漏检；
- `N=1` 在修复后仍无法与旧单环境保持状态、奖励、终止和观测语义一致；
- 多环境出现 episode、history、GRU hidden 或 replay 串扰；
- 为通过测试需要修改 17 维观测、动作范围、奖励或成功定义；
- 需要启动 G4、长训练 campaign、OceanSim 多环境声呐训练或 P2。

完成 G3 验收后也立即停止，不因为“还有时间”而继续跑大规模实验。

---

## 10. 最终交付

输出一份报告，包含：

1. 状态：COMPLETE / PARTIAL / BLOCKED
2. 修改文件列表
3. 本地 pytest、ruff、pyright、mypy 结果
4. G1 配对样本数、MAE、P95、max、危险漏检数和每道门的结论
5. G1 S1 全部饱和检查
6. G2 N=1 语义等价和多环境隔离结果
7. G3 `act_batch` 修复说明、history/GRU/Noise 隔离结果、replay 和 PPER 测试结果
8. G3 smoke 的梯度步数、更新比例和吞吐字段
9. 未解决风险和下一步建议

最后两行必须为：

```text
P2 未开始。
多环境训练仅限解析 13 扇区；OceanSim 多环境声呐训练未开始。
```
