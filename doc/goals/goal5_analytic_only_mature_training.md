# 任务：analytic-only 多环境训练到成熟策略并完成 OceanSim 测评

记录日期：2026-10-28

状态：**PARTIAL（已执行至 P7；P4 未筛出通过候选，P5/P6 学习门未通过）**

工作目录：`/home/bubble/水下机器人/settled_局部规划器_反应式`

关系：本文件是 `doc/goals/goal2_analytic_13sector_multienv.md` 的续跑任务书。G2/G3 已完成，不得重做；G1 保持 `PARTIAL`，本轮采用用户已确认的 `analytic-only` 路线，不再尝试通过冻结的 `max <= 0.75 m` 门。完成后停在 P1 测评与报告，不进入 P2。

执行方式：读完本文件后直接按阶段执行。除“必须停止”条件外，不要因为单个实验耗时而中断，也不要在阶段之间等待额外确认。所有结论必须来自测试、artifact、日志或 checkpoint，不能把“代码已写”写成“验收已通过”。

## 1. 一句话目标

在 OceanSim 服务器的 `oceansim` 环境中，用解析 13 扇区多环境训练链完成从吞吐测量、工程 smoke、候选筛选、成熟训练到 OceanSim P1-A 单环境测评的全流程，产出一个通过预注册学习门的最佳 P1 checkpoint、可复现的训练/评估证据和最终报告。

## 2. 已冻结的决策

- 训练感知固定为 `analytic_13sector`。
- 最终评估感知固定为 `oceansim_simplified_pcl`，使用 OceanSim P1-A 单环境链路。
- G1 的边界量化差异保留为已知限制；不删除失败位姿、不改门限、不改 13 扇区语义、不把解析结果伪装成 OceanSim 结果。
- 训练执行位置固定为 OceanSim 服务器，不在本地 CPU 上产出正式训练结果。
- 多环境实现使用当前自研批量环境；本 goal 不迁移到 IsaacLab，不启动 OceanSim 多环境声呐训练。
- 17 维观测、动作范围 `[u_dot, r]`、奖励终端项、碰撞/越界/成功定义、`0.2 s` 决策步长和 `0.02 s` 积分步长全部保持不变。
- `update_to_data_ratio` 默认保持 `1.0`；任何改变必须作为预注册消融，不能让环境数增加稀释梯度更新。
- 不覆盖旧结果：`outputs/m4/p1/ablations/`、`artifacts/m4/p1/convergence/`、`artifacts/m4/p1/parallel/` 和旧单环境输出全部保留。

## 3. “成熟训练”的判定

这里的“成熟”不是训练时间越长越好，而是同时满足以下工程门：

- 完成 G4：端到端吞吐、内存和稳定性测试通过。
- 完成 G5：至少一个预注册候选达到正式训练预算，并保存可重载的最佳 checkpoint。
- 全部训练与评估无 NaN/Inf、无静默环境降级、无 episode/GRU/replay 串扰。
- 最佳 checkpoint 在固定的解析 6 条验证 episode 上通过学习门：
  - critic 健康门通过；
  - `u_dot` 与艏摇动作饱和率均 `< 0.5`；
  - reset 状态 `tanh_grad_norm > 1e-4`；
  - S1 成功 `>= 2/3`；
  - S2 成功 `>= 1/3`；
  - 训练最后 20% 的回合中，S1 和 S2 至少各有一次成功；
  - 无全零动作、无观测来源混淆。
- 同一 checkpoint 在 OceanSim P1-A 固定 6 条评估 episode 上通过 S1/S2 成功门，并记录来源为 `oceansim_simplified_pcl`。
- 最佳 checkpoint 的 20 条扩展验证（S1/S2 各 10 条）不得出现灾难性退化；扩展集只用于报告，不用于重新挑选 checkpoint。

如果训练到最后仍未通过，必须如实报告“未成熟”，不能放宽门限。

## 4. 预算与 campaign 阶梯

所有预算均指**总环境转移数**，不是每个环境各自训练多少步。

| 阶段 | 每个候选总环境转移数 | seed | 用途 |
|---|---:|---|---|
| smoke | `20,000` | 0 | 只验证 runner、日志、checkpoint 和自动 reset |
| screening | `100,000` | 0 | 预注册候选筛选，最多 4 个候选 |
| mature | `300,000` | 0 | 最佳候选正式训练 |
| confirmation | `300,000` | 1、2 | 只有在 seed 0 通过后运行 |

硬上限：单个 seed 最多 `600,000` 环境转移。只有 seed 0 在 `300k` 时 critic、饱和率和成功率均呈改善趋势，才允许一次性延长到 `600k`；不得无记录地继续加预算。

## 5. 目录与 artifact

新 campaign 目录：

```text
configs/rl/vector/
outputs/m4/p1/analytic_multienv/
artifacts/m4/p1/analytic_multienv/
```

建议结构：

```text
outputs/m4/p1/analytic_multienv/
  g4_throughput/
  smoke/
  screening/
  mature/
    seed_000/
    seed_001/
    seed_002/
  oceansim_eval/

artifacts/m4/p1/analytic_multienv/
  g4_throughput.json
  campaign_manifest.json
  screening.csv
  mature_training.json
  final_evaluation.json
  final_report.md
```

模型、完整 CSV、大型数组和视频不进入普通 Git；只提交小型 JSON、CSV 摘要、配置和报告所需的小型证据。

## 6. 执行阶段

### [x] P0：启动前检查

- [ ] 确认服务器可登录、`oceansim` 环境存在、仓库路径正确。
- [ ] 检查 `nvidia-smi`、内存和 `/data` 空间，选择空闲 GPU；不终止其他用户任务。
- [ ] 记录服务器 hostname、GPU、Python、torch、NumPy 版本和日期。
- [ ] 运行本地现有测试、ruff、format、pyright、mypy，确认 G2/G3 基线没有回归。
- [ ] 检查工作树中已有未提交修改，只增量修改相关文件，不执行 `git checkout`、`git reset` 或 `git clean`。

服务器路径：

```text
/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
```

服务器 Python：

```text
/data/yhx/underwater-robotics/envs/conda_envs/oceansim/bin/python
```

### [x] P1：G4 吞吐与规模测量

实现或修复一个只读 probe，测量 `N = 1, 4, 16, 64, 256, 1024`。每档至少 200 个控制步，重复 3 次，必须包含：

- 解析 1300 射线与 13 扇区观测；
- 50 Hz 子步积分、碰撞/越界/奖励和自动 reset；
- DDPG 前向、replay add、梯度更新；
- `env_steps/s`、`control_steps/s`、`gradient_steps/s`、Peak RSS/显存。

门禁：

- `N=64` 相对旧单环境基线至少 `10x` 端到端吞吐；
- 每个环境保持 5 Hz 控制频率；
- 无 OOM、NaN、静默降级和跨环境串扰；
- 若 `N=64` 不通过，先 profile；只允许优化传感器批处理、环境循环、replay 或更新批处理，不得先改模型结构或训练目标。

输出：

```text
artifacts/m4/p1/analytic_multienv/g4_throughput.json
artifacts/m4/p1/analytic_multienv/g4_throughput.csv
```

选择规则：若 `N=64` 通过且更大规模仍有正收益，按实测选一个稳定值；不预设必须使用 1024。若吞吐在某一档平台化，选平台前的最大稳定档。

### [x] P2：正式向量训练 runner

新增或完善可复现 CLI，至少支持：

- `--config`、`--num-envs`、`--total-env-transitions`；
- `--seed`、`--device`、`--output`；
- `--eval-every-env-transitions`，默认 `20000`；
- `--eval-seeds 0 1 2`，固定 S1/S2，共 6 条；
- `--resume`、`checkpoint_latest.pt`、`checkpoint_best.pt`；
- `--dry-run` 和 `--smoke`；
- 训练结束自动写 manifest、metrics、episode 日志、梯度日志和最佳 checkpoint。

必须修复或确认以下语义：

- `terminated` 与 `truncated` 分离；
- reset 观测不写回上一条 transition；
- 每环境独立 GRU/history/NoisyNet 噪声；
- replay 不跨环境、不跨 episode；
- checkpoint 重载后，固定观测上的动作与保存前完全一致；
- 训练 manifest 明确写 `observation_source=analytic_13sector`；
- 评估 manifest 明确写 `observation_source=oceansim_simplified_pcl`。

最佳 checkpoint 选择规则固定为字典序：

1. 学习门通过；
2. 固定 6 条解析评估成功数；
3. 碰撞/越界数更少；
4. 两个动作维度饱和率最大值更小；
5. 超时数更少。

不得用 OceanSim 评估结果反向挑选 checkpoint。

### [x] P3：工程 smoke

先用 `N=16` 或 G4 选出的稳定小规模，跑 `20,000` 总环境转移、seed 0：

- 至少完成一个自动 reset；
- 至少产生 1 个有效梯度步；
- `update_to_data_ratio` 与配置一致；
- 生成 `manifest.json`、`training_metrics.json`、`episodes.csv`、`gradient_updates.csv`、`checkpoint_latest.pt`；
- 重载 checkpoint 后固定动作一致；
- 无 NaN/Inf。

smoke 未通过时禁止进入筛选。

### [x] P4：候选筛选（执行完成；未筛出通过学习门的候选）

最多运行 4 个预注册候选，每个 `100,000` 总环境转移、seed 0。候选必须从现有配置的可解释变体中选择，默认包含：

- `V0`：C2 等价，GRU + SiLU；
- `V1`：C3 等价，GRU + SiLU + LayerNorm；
- `V2`：SAT7 等价，关闭 actor 输出噪声并使用小输出初始化；
- `V3`：R5 等价，奖励 v2 四项全开；

如果某个旧配置无法直接映射到向量训练器，只移植配置值并在 manifest 写明，不改变冻结契约。不得临时增加未登记候选。

筛选先看 critic 健康、饱和率和无 NaN，再看成功数；不能用 OceanSim 结果调参。

输出：

```text
artifacts/m4/p1/analytic_multienv/screening.csv
```

若所有候选在 `100k` 都未通过，选择“最接近通过”的一个进入成熟训练，但必须在报告中标记“未筛选出已通过候选”。

### [x] P5：成熟训练（seed 0 执行完成；学习门未通过，未启动 seed 1/2）

用筛选最佳候选，按以下顺序运行：

- seed 0：`300,000` 总环境转移；必要时按第 4 节规则延长到最多 `600,000`；
- 若 seed 0 通过，运行 seed 1、2，各 `300,000`；
- 三个 seed 使用相同配置、场景、评估 episode 和预算；
- 每 `20,000` 环境转移做一次固定 6 条解析评估；
- 保存每轮评估、critic 健康、饱和时间线、最佳 checkpoint。

训练期间不得：

- 因为环境数增加而降低 `update_to_data_ratio`；
- 删除失败 episode；
- 只看总奖励而不看成功/碰撞/越界/超时；
- 把训练回合成功率当成最终测评；
- 用 BC、oracle 或特权 critic 替换纯 RL 主结果。

输出：

```text
outputs/m4/p1/analytic_multienv/mature/seed_*/
artifacts/m4/p1/analytic_multienv/mature_training.json
```

### [x] P6：OceanSim P1-A 测评（主评估执行完成；主成功门未通过）

加载 P5 选出的最佳 checkpoint，在 OceanSim P1-A 单环境链路上运行：

- 主评估：`s1`、`s2` × seeds `0,1,2`，共 6 条；
- 每个 episode 记录成功、碰撞、越界、超时、最小 clearance、路径长度、动作饱和率、奖励分量和观测元数据；
- 评估必须使用 `jitter=False` 或与旧评估一致的固定 reset；
- 记录 `observation_source=oceansim_simplified_pcl`；
- 主评估通过后，再运行 S1/S2 各 10 条扩展评估，扩展评估不用于选择 checkpoint。

如果主评估未通过，报告失败原因并停止，不通过修改成功定义或换 seed 重试。

### [x] P7：证据、文档和最终报告

- [ ] 更新 `doc/02_执行计划.md` 的 P1 多环境状态。
- [ ] 更新 `doc/01_技术栈.md` 的 analytic-only 路线和最终结果。
- [ ] 在 `doc/goals/goal5_analytic_only_mature_training.md` 追加执行记录，勾选实际完成项。
- [ ] 生成 `artifacts/m4/p1/analytic_multienv/final_report.md`。
- [ ] 报告 G1 仍为 `PARTIAL`，并写明其边界量化风险不影响本轮 analytic-only 训练，但不能宣称 OceanSim 等价已完成。
- [ ] 不提交 `outputs/`、checkpoint、大型 CSV、视频或点云。

## 7. 每阶段 Git 检查点

每个阶段结束都先做：

```bash
git status --short
git diff --check
git diff --stat
```

建议提交信息：

| 阶段 | 建议提交 |
|---|---|
| P1 | `feat(rl): add analytic vector throughput probe` |
| P2 | `feat(rl): add resumable analytic vector training runner` |
| P3 | `test(rl): add vector training smoke acceptance` |
| P4 | `feat(rl): add analytic multienv screening configs` |
| P5 | `docs: record analytic multienv mature training` |
| P6/P7 | `docs: record analytic-only P1 evaluation` |

提交命令只在用户明确要求时执行：

```bash
git add <已审阅文件>
git commit -m "<建议提交信息>"
git push origin main
```

不要自动 push；不要提交模型、大型 artifact 或旧结果目录。

## 8. 必须停止并报告的情况

出现以下任一情况时停止当前阶段，保留日志，报告现象、已排查项、证据和两个可行方案：

- 服务器不可登录、`oceansim` 环境损坏或没有安全可用 GPU；
- G4 在 `N=64` 达不到 `10x` 或 5 Hz，且 profile 后仍无法定位；
- 出现 OOM、NaN/Inf、静默环境降级、episode/GRU/replay 串扰；
- checkpoint 重载不能复现固定动作；
- 训练到 `600k` 硬上限仍未通过，且没有持续改善趋势；
- 需要修改 17 维观测、动作范围、奖励口径、成功定义或 13 扇区契约；
- 需要 OceanSim 多环境声呐训练、P2 或全局规划器改动；
- 需要覆盖旧结果或删除已有 artifact。

## 9. 最终报告格式

1. 状态：`COMPLETE` / `PARTIAL` / `BLOCKED`
2. 服务器、GPU、环境版本和执行日期
3. 修改文件与提交列表
4. 本地测试、ruff、format、pyright、mypy
5. G4 环境数—吞吐—内存表
6. 训练来源、评估来源、总环境转移数、环境数和更新比例
7. 候选筛选结果
8. 成熟训练三 seed 结果
9. 最佳 checkpoint 的解析 6 条和 OceanSim P1-A 6 条评估
10. 扩展 20 条评估
11. 奖励分量、动作饱和、碰撞/越界/超时和最小 clearance
12. G1 已知边界限制与未解决风险
13. 下一步建议

## 10. 执行记录（2026-10-29）

| 阶段 | 结果 | 证据 |
|---|---|---|
| P0 | 通过 | `artifacts/m4/p1/analytic_multienv/p0_preflight.json`；158 tests、ruff/format/mypy/pyright 通过 |
| P1/G4 | 通过 | `g4_throughput.json`；`N=64` 461.60 transitions/s、7.21 Hz/env、39.25x，选定 `N=64` |
| P2 | 通过 | 可恢复 `training_state.pt`；中断恢复与连续训练动作、replay、计数一致；服务器恢复实测通过 |
| P3 | 通过 | `p3_smoke_validation.json`；20,032 转移、19,032 梯度步、317 回合、64 环境均自动 reset；checkpoint 重载 max diff 0 |
| P4 | 执行完成，无通过候选 | `screening.json`；4 个候选均未过学习门，按规则选 `v2_sat7` 进入成熟训练 |
| P5 | seed 0 执行完成，学习门失败 | `mature_training.json`；300,032 转移、终点 0/6、饱和率 1.0；未延长 600k，未跑 seed 1/2 |
| P6 | 主评估执行完成，门失败 | `final_evaluation.json`；OceanSim S1 3/3、S2 0/3；未跑 20 条扩展 |
| P7 | 完成 | 本报告、`doc/01_技术栈.md`、`doc/02_执行计划.md` 与本记录 |

本轮没有 commit 或 push；`outputs/`、checkpoint 和大型 CSV 未进入 Git。
最终报告为 `artifacts/m4/p1/analytic_multienv/final_report.md`，状态 `PARTIAL`。

报告正文完成后，以下两行必须逐字作为文件最后两行：

P2 未开始。
多环境训练仅限解析 13 扇区；OceanSim 多环境声呐训练未开始。
