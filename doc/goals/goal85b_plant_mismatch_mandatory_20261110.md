# Goal85b：强制 plant mismatch 预注册（drag/thrust/fin，再到 added mass）

**日期**：2026-11-10<br>
**状态**：`B1_IMPLEMENTATION_PASS / MISMATCH_TOO_SMALL / B2_NOT_TRIGGERED / B3_IMPLEMENTED`<br>
**上游**：Goal85 `unbiased_noise_v1` 位置契约<br>
**性质**：强制后续 Goal；不能因为 Goal85 通过而取消。先做 eval-only 敏感性，再做同分布训练；不换算法。

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal85b --goal-doc doc/goals/goal85b_plant_mismatch_mandatory_20261110.md --strict
```

`required: false`：这是预注册实现 Gate。Gate 记录前不得改随机化范围、seed、
训练/评价分布或 held-out；失败后补 blocker 和 ledger decision。

## 1. 为什么必须做

Goal85 只关闭导航测量系统偏差和传感器延迟。它保留了 current 场和执行器动态，
但 `src/auv_lowlevel/batch/dynamics.py:178` 仍然给所有环境使用同一套固定
REMUS100 参数。严格来说，当前“动力学参数估计误差”是 0；`current` 是外部扰动，
不是模型参数误差。

用户已明确要求 plant mismatch 必须做。因此本 Goal 不是可选项，而是 Goal85 的
强制后续；Goal85 通过后立即启动，不得用“先看算法结果”跳过。

## 2. 范围与顺序

| 阶段 | 内容 | 是否训练 | 完成条件 |
|---|---|---|---|
| B1 | drag/thrust/fin 逐 episode 乘性随机化，eval-only 敏感性 | 否 | 同 seed 可复现；效应大于 seed 噪声；无 NaN/发散 |
| B2 | 用 B1 的同一分布训练和评价 | 是 | 与 nominal 训练同协议配对；held-out 关闭 |
| B3 | added mass ±5% 逐 episode 随机化 | 视 B2 | 必须实现并通过稳定性测试；不能省略 |

`added mass` 是 Goal85b 的完成条件之一，不是可选项。实现上它比 drag/thrust/fin
贵，因为需要 per-env 质量矩阵；可以先做标量近似，再做完整 6×6，但最终必须留下
可复现实现和测试。

## 3. 假设

### H85b-A：小范围 plant mismatch 可测但不压垮控制器

在 frozen checkpoint、相同 seed、相同 action 序列下，drag/thrust/fin
`±5–10%` 的 paired tracking/terminal 差异应大于 seed 噪声，但不出现大面积
timeout、safety 或发散。

成立应看到：paired 效应 CI 不包含 0，且失败模式仍是 terminal/authority，不是
数值崩溃。
失败应看到：效应小于 seed 噪声，或出现系统性发散。
最便宜证伪：B1 eval-only，20 个 dev seeds × 2 个 scenario，同一 checkpoint 配对。

### H85b-B：plant mismatch 不制造 p_nav/eta 位置偏差

plant mismatch 只改变真实动力学，不改变 sensor bias/delay。Goal85 的
`abs(mean(p_nav - eta)) < 0.05 m` 应保持成立。

成立应看到：p_nav 偏差不变，tracking/terminal 变化来自动力学。
失败应看到：p_nav 偏差随 plant mismatch 改变，说明随机化误伤了传感器链路。
最便宜证伪：同一 seed 下分别开/关 plant mismatch，比较 p_nav-eta 均值和趋势。

### H85b-C：同分布训练的鲁棒性优于只在 nominal 上训练

在 B2 中，plant-mismatch 训练的策略在 mismatch 评价集上的 paired success
应不低于 nominal 训练策略，且不出现 nominal 评价集的大幅回退。

成立应看到：mismatch 评价集 success 提升，nominal 评价集回退有界。
失败应看到：两边都不改善，或 nominal 回退超过预注册阈值。
最便宜证伪：B2 单因素训练，固定算法、reward、动作参数化和 seed 预算。

## 4. 数据与隔离

| 项目 | 固定值 |
|---|---|
| dev seeds | `20–59` |
| held-out | `100–149` 关闭 |
| 评价规模 | B1: 60 runs / 2400 episodes；B2 与 Goal85 对齐 |
| checkpoint | Goal85 使用的 frozen checkpoints |
| 算法 | 固定 PPO；不启动 SAC/REDQ/DroQ |
| 随机化 seed | 与 episode seed 绑定；所有方法共用 common random numbers |
| 训练/评价分布 | B2 必须同分布；评价不得使用 clean nominal 替代 |

禁止用 mismatch 评价结果反向选择 checkpoint、seed 或随机化范围。B1 的范围在
本文件中预注册；如需扩大，必须新开 Gate 并记录原因。

## 5. 方法

### 5.1 参数化

新增 `PlantRandomizationConfig`：

```text
enabled: bool
seed: int
drag_scale: [min, max]          # 默认 [0.95, 1.05] 和 [0.90, 1.10] 两档
thrust_scale: [min, max]
fin_scale: [min, max]
added_mass_scale: [min, max]    # B3，默认 [0.95, 1.05]
actuator_time_scale: [min, max] # 可选，默认 [1.0, 1.0]
```

每个 episode 采样一次，episode 内固定。所有环境共享同一 nominal 参数，但每个
环境有自己的乘性因子；控制器/策略不知道这些因子。

随机化只作用于真实 plant，不作用于传感器、终止坐标或 actor observation：

| 因子 | 精确作用位置 | 不作用的位置 |
|---|---|---|
| `drag_scale` | 六自由度 damping diagonal、parasitic drag、cross-flow resistance | lift 本身、传感器、current |
| `thrust_scale` | 螺旋桨轴向 thrust | propeller torque、舵面力 |
| `fin_scale` | rudder/stern lift force、对应 drag 和力矩 | 命令限幅、执行器时间常数 |
| `added_mass_scale` | 整个 6×6 added-mass diagonal 的标量乘子 | rigid-body mass、restoring |
| `actuator_time_scale` | 舵机与螺旋桨一阶时间常数 | 力项、传感器 |

实现使用 episode seed 的确定性 64-bit hash 采样，不推进全局 RNG。相同 seed、
相同随机化配置一定得到相同因子；baseline、candidate、zero-residual 和三档
mismatch 共用 common random numbers。`enabled=false` 时五个因子严格为 `1.0`，
必须与 Goal85 旧动力学逐位一致。

### 5.2 实现顺序

1. 先实现标量 `drag_scale`、`thrust_scale`、`fin_scale`，分别乘到
   `hydrodynamic_damping`、`thrust`、`rudder/stern` 力项；
2. 再加 per-env `added_mass_scale`；如果完整 6×6 成本过高，先实现对角近似，
   但必须在文档中标注近似，并在 B3 前补完整测试；
3. `actuator_time_scale` 作为可选扩展，不阻塞 B1/B2；
4. 所有因子随 episode seed 可复现，记录到 episode metadata。

实现入口：`src/auv_lowlevel/batch/dynamics.py:143`、`src/auv_lowlevel/batch/pc_env.py`
的 reset/step 路径、`src/auv_lowlevel/config.py` 的新 config block。

## 6. 执行计划

- [x] Step 0：冻结本文件；记录 Goal85 的 profile、checkpoint 和 eval SHA；
- [x] Step 1：实现 drag/thrust/fin 随机化与 metadata；
- [x] Step 2：B1 eval-only：0%、±5%、±10% 三档，40 dev seeds 配对；
- [x] Step 3：Gate B1 判 `MISMATCH_TOO_SMALL`；实现检查通过，单因子机制诊断完成；
- [x] Step 4：B2 未触发；预注册分支要求 B1 detected 后才训练，避免低先验 formal；
- [x] Step 5：B3 实现 added mass；补稳定性和复现测试；
- [x] Step 6：写 result、decision、LEDGER、STATE；不自动 commit/push。

### 6.1 B1 正式矩阵（测量前冻结）

B1 使用 Goal85 的四个 frozen Goal59 checkpoint（`s240/s752/s880/s1008`）、
四个 zero-residual 对照（2 roots × 2 scenarios）和 dev seeds `20–59`。
三档随机化范围固定如下：

| 档位 | drag | thrust | fin | added mass | actuator time |
|---|---|---|---|---|---|
| `m0` | `[1.00, 1.00]` | `[1.00, 1.00]` | `[1.00, 1.00]` | `[1.00, 1.00]` | `[1.00, 1.00]` |
| `m05` | `[0.95, 1.05]` | `[0.95, 1.05]` | `[0.95, 1.05]` | `[0.95, 1.05]` | `[1.00, 1.00]` |
| `m10` | `[0.90, 1.10]` | `[0.90, 1.10]` | `[0.90, 1.10]` | `[0.90, 1.10]` | `[1.00, 1.00]` |

`m05/m10` 的四个 scale 共用同一随机化 seed，但每个因子使用不同 hash salt，
因此不会所有参数同向绑定。B1 的 run 数如下：

| 组成 | run 数 | episode 数 |
|---|---:|---:|
| policy：2 roots × 2 scenarios × 4 checkpoints × 3 levels | 48 | 1920 |
| zero-residual：2 roots × 2 scenarios × 3 levels | 12 | 480 |
| **合计** | **60** | **2400** |

每个 run 固定 40 个 seeds，不挑 checkpoint、不按结果补 seed。B1 是 eval-only；
不训练、不打开 held-out、不改 reward/action/controller。

B1 主判据（测量前固定）：

1. **实现 Gate**：同 seed 因子完全一致；各因子落在上表范围；`m0` 与 Goal85
   动力学等价；无 NaN/Inf；`abs(mean(p_nav-eta)) < 0.05 m`。
2. **敏感性 Gate**：对 `m05-m0` 或 `m10-m0`，按 seed 配对计算
   `path_three_d_rmse_m` 与 `terminal_eta_distance_m` 的 95% CI。主判据使用
   zero-residual 控制器（避免 checkpoint 选择混淆）；若任一主指标的 CI 不含 0，
   该 scenario/root 单元记为 detected。`m10` 至少 `3/4` 个 scenario/root 单元
   detected，或 `m05` 至少 `2/4` 个单元 detected，判 `B1_DETECTED`。四个
   checkpoint 的 policy 效应作为次级诊断报告，不用于挑选 best checkpoint。
3. 次级指标：`eta_tight_at_done`、plane-miss、timeout、safety、material
   saturation。任何 safety/timeout 恶化超过 `2 pp` 必须单独报告，不用放宽阈值救。
4. 若未 detected，先检查因子是否进入对应力项和实际 episode metadata；判
   `MISMATCH_TOO_SMALL` 或 `INVALID_IMPLEMENTATION`，不进入算法改动。

B1 产物目录固定为：

```text
/data/yhx/underwater-robotics/src/plan_p1/goal85b_plant_mismatch_20261110/
  b1/{m0,m05,m10}/{scenario}/{root}/{action}/{checkpoint}/
```

### 6.2 B1 单因子机制诊断（不改变 B1 Gate）

B1 主矩阵完成后、在看到单因子结果前冻结此诊断。它只解释主矩阵，不重新裁决
`MISMATCH_TOO_SMALL`。固定条件：

| 项目 | 固定值 |
|---|---|
| 控制器 | zero-residual |
| 档位 | 单因子 `±10%`，其余因子 `1.0` |
| 因子 | drag、thrust、fin、added mass |
| 场景 / root | `default/strict125 × baseline/candidate` |
| seeds | `20–59` |
| 规模 | 16 runs / 640 episodes |

输出目录 `b1_component/{factor}/{scenario}/{root}/zero-residual/s240/`。
报告 path RMSE 的 mean/median、terminal distance、success；用于确认每个乘子确实进入
闭环，而不是挑选新的 B1 成功口径。

## 7. Gate

| check | 阈值 |
|---|---:|
| 同 seed 同参数 | `100%` 可复现 |
| 随机化范围 | 不超出本文件预注册范围 |
| `p_nav - eta` 偏差 | 仍 `< 0.05 m` |
| NaN / invalid / 发散 | `0` |
| B1 paired 效应 | 95% CI 不包含 0，且大于 seed 噪声 |
| B2 mismatch success | 不低于 nominal 训练；nominal 回退有界 |
| B3 added mass | 已实现、可复现、稳定性通过 |
| held-out opened | `false` |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 参数不可复现 | `INVALID_IMPLEMENTATION` | 修 seed/采样，不分析结果 |
| 效应小于 seed 噪声 | `MISMATCH_TOO_SMALL` | 先检查乘子是否真正进入力项；不换算法 |
| 出现大面积发散 | `MISMATCH_TOO_STRONG` | 回退范围，写 blocker；不改 reward 救结果 |
| p_nav 偏差变化 | `SENSOR_CONTAMINATION` | 判 INVALID，隔离 sensor 与 plant 路径 |

## 8. Git

禁止 `git add -A`。只 add 本 Goal 的文档、plant config、dynamics、env、测试、
分析脚本和 ledger/STATE；不要提交 artifacts、checkpoint、日志或缓存。提交前运行
`git diff --cached --check`。用户明确要求前不 commit / push。

## 9. Goal 提示词

```text
你现在执行 Goal85b：强制 plant mismatch。

先读：
- doc/goals/goal85_unbiased_nav_no_delay_20261110.md
- doc/goals/goal85b_plant_mismatch_mandatory_20261110.md
- doc/goals/goal84_result_20261110.md

硬约束：
1. 不换算法；PPO/reward/action/seed 预算固定。
2. 先 B1 eval-only，再 B2 训练，最后 B3 added mass；不能跳过 added mass。
3. 训练和评价同分布；held-out 100-149 关闭。
4. 不改 sensor profile 的零均值/无延迟契约。
5. 不改历史 config；新增随机化 profile 和 config。
6. 读并执行 `research/AUTORESEARCH_LOOP.md`；Gate 完成后不要停下来等我逐步指令。
7. 如果 plant mismatch 失败或出现 hard blocker，按 loop 写 blocker、至少两个可证伪
   hypothesis、用 `paper-search` / `deep-research` 查论文、下载 PDF/TXT、定位可用代码、写 reading note、跑最便宜 falsification、写 ledger；不改本
   Gate 的随机化范围或成功定义。
8. 如果 pivot，依据指标自动写新的 Goal/prompt 并在预注册预算内继续；持续迭代到 Gate PASS 或触发停止条件，不要因为一次 FAIL 结束；只有改变任务成功定义、
   打开 held-out、改变主算法族、超过预算或 commit/push 时才问我。不自动 commit/push。

执行顺序：
- 实现 drag/thrust/fin 逐 episode 乘性随机化；
- B1 0/5/10% 配对敏感性；
- Gate 通过后 B2 同分布训练；
- B3 added mass 实现与稳定性测试；
- 写 result、ledger、STATE。
```
