# Goal171：route-health-centered residual training

**日期**：2026-11-12<br>
**状态**：`PREREGISTERED / REWARD_IMPLEMENTED / CURRICULUM_LOCKED / FULL_EVAL_MATRIX_LOCKED / C4_DEPTHFIX_PENDING / OLD_RUN_INVALID / ROUTE_HEALTH_PRIMARY / PPO_SAC_FIXED / HELDOUT_CLOSED`<br>
**上游**：Goal170 `GATE170_B_FAIL`；Goal169 `TERMINAL_HOMING_50M_SELECTED`；Goal167/166 `ACTION_AUTHORITY`；Goal140 `VERTICAL_REWARD_STANDBY`

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal171 --goal-doc doc/goals/goal171_route_health_training_20261112.md --strict --require-result
```

## 1. 为什么现在做这个 Goal

Goal170 已经证明：在 50 m guidance 下直接 replay 旧 checkpoint，不能得到整条路线收益。
主指标不是 success，而是整条路线、五段和 10% keypoint：

- A/B 的 route delta 只有约 `+0.002…+0.008 m`，且只在末段有符号变化；
- T2-SAC 的 route delta 为 `+0.23…+0.38 m`，五段全部退化；
- T2-PPO 的 route delta 约 `0.0001 m`，转向物理权限接近零；
- PPO base 和 policy 还违反 `2.5 m/s` 速度硬上限。

因此下一步不是继续 replay，也不是换算法救 success，而是训练一个**真正以路线健康为目标**的
residual policy，并用同一协议比较 PPO/SAC。

## 2. 固定约束

- 成功定义、termination、goal radius、sensor delay 不变；
- 速度硬上限 `2.5 m/s`，sampler 仍需满足 `v_cmd + current + variation <= 2.5`；
- 每个训练环境必须启用冻结的 C3 课程：`C3=0.70 / C2=0.15 / C0=0.10 / C4=0.05`；
- 任何评价都必须跑完整 `8 arms × 4 levels × 2 radii + 3 类 zero baseline × 4 levels × 2 radii`
  = `64 + 24 = 88` 个 run，不得退化为单 level、单 radius 或省略 zero baseline；
- held-out `100–149` 关闭；dev seeds `20–59`；
- 保留 `z_t`/BC 架构、ALOS base、50 m terminal guidance；
- PPO 和 SAC 都是允许的主算法，但不混算 seed、预算或 Gate；
- 不把 success 当主 Gate；success/final distance/plane-miss 只作二级诊断；
- 任何 PPO arm 必须先证明 `ground_speed_max_mps <= 2.5`，否则记为 `INVALID_SPEED`。

## 3. 可证伪假设

### H171-A：中段 route reward 能把 residual 的收益从终段推到整条路线

- 成立：至少一个 arm 在 current10 default/strict125 的 3D route RMSE 改善
  `>=5%` 且绝对改善 `>=0.05 m`；五段至少 3 段改善；20%–90% keypoint 至少 5 个改善。
- 失败：只有 90%–100% 的终点附近改善，或 pooled RMSE 改善但中段/垂向退化。
- 最便宜证伪：1 seed、200k transitions、2 个 radius；只看 route metrics 和速度包线。

### H171-B：PPO 的 route 无效应主要来自转向通道 residual collapse

- 成立：在相同 reward/观测下，给 rudder/stern 通道独立且可测的初始化/scale 后，
  physical steering delta 达到可影响 cross-track 的量级，route 指标出现方向性改善。
- 失败：增大转向通道后仍只有 rpm 变化，或出现饱和/路线退化；此时再考虑样本效率方法。
- 最便宜证伪：纯 eval 的通道缩放 + 短 training pilot；不改算法族。

### H171-C：始终开启 residual 与分阶段开启的差异来自探索/信用分配，而不是动作接口

- 成立：在相同 reward、相同 checkpoint 初始化下，`always`、`ramp`、`late` 三种 schedule
  产生不同的中段 route error；late 的优势能被 ramp 复制。
- 失败：三种 schedule 的分段曲线几乎相同；此时不再把“晚开 residual”当成独立机制。
- 最便宜证伪：先复用固定策略做 eval-only schedule replay；再做同预算 training control。

### H171-D：SAC 相比 PPO 只在样本效率/探索上占优，不应在 route reward 未验证前切换

- 成立：PPO 在相同 reward/action 下保持可用 residual 但路线不动，SAC 在相同预算下
  出现更快的 route 改善，且不是靠提高速度/放大动作。
- 失败：SAC 仍然以速度/横向误差换取短期 reward，或 route 不如 PPO；此时不应以
  “换算法”替代 reward/动作修复。
- 最便宜证伪：1 seed/200k 的 PPO vs SAC 同协议对照，不混算 success。

## 4. 奖励设计（只改 reward，不改 termination/success）

现有环境已经提供 `path_cross`、`path_vertical`、`path_progress` 和 path error。Goal171
新增的 reward 只作用于训练目标，不改变 evaluator 的成功/终止：

1. **主项：连续整路线误差**
   - `w_path_cross * huber(cross_track, path_cross_huber_delta_m)`；
   - `w_path_vertical * huber(vertical, path_vertical_huber_delta_m)`；
   - 中间 20%–80% 弧长使用平滑 gate，最高为 `1 + route_middle_gate`，
     避免终点附近几何支配；
2. **进度项**：`w_path_progress * progress_ratio`，其中
   `progress_ratio = clip(progress / (reference_speed * dt), -1, 1)`；
   它防止策略用停滞换取低误差，也避免投影跳变产生无界 advantage；
3. **保留现有限制项**：action smoothness、saturation、roll/pitch rate、energy；
4. **success/terminal**：保持 Goal169/Goal85 原契约；goal bonus 只作稀疏 guard，
   不因本 Goal 提高；
5. 所有新 reward 项默认 0，旧配置逐位 parity；新 arm 才启用。

关键点**不直接作为稀疏奖励**。原因是一个关键点奖励只会告诉策略“到过”，不能约束
两个关键点之间是否偏离；当前实现用连续 Huber 误差约束整条航线，keypoint 只用于评价
和 Gate。若后续证明需要关键点 credit，再单独预注册，不与本轮 reward 同时改。

已落地实现（2026-11-12）：

- `RewardConfig` 新增 `w_path_cross`、`w_path_vertical`、`w_path_progress`、
  `route_middle_gate`、`path_cross_huber_delta_m`、`path_vertical_huber_delta_m`；
- `pc_route_health_terms()` 提供纯张量、可单测的 dense route reward；
- `BatchedPCLowLevelEnv._path_reward()` 同步输出
  `reward_route_cross_huber`、`reward_route_vertical_huber`、
  `reward_route_middle_gate`、`reward_route_progress_ratio`、`reward_route_reward`；
- `tests/test_goal171_route_reward.py` 覆盖默认关闭、零权重 parity、Huber 对称性、
  中段 gate、进度方向、20-step finite 和 `2.5 m/s` 包线。

不在同一 Gate 内同时扫 reward 权重、residual scale、schedule 和算法；每轮只改变一个
因素，失败的 arm 保留。

## 5. 实验矩阵

### Phase 0：实现与 parity

| 项目 | 内容 |
|---|---|
| reward fields | `w_path_cross`、`w_path_vertical`、`w_path_progress`、`route_middle_gate` |
| parity | 所有新字段为 0 时，旧 reward/trace 逐位一致 |
| tests | reward finite、shape、zero-weight parity、速度包线、PPO speed guard |
| eval | 不改 checkpoint；先跑 1 个 20-step smoke |
| status | `DONE`：实现和 20-step smoke 已通过，下一步进入 Phase 1 |

### Phase 1：1 seed / 200k screen（先证伪，不追正式结果）

固定：50 m guidance、BC/ALOS base、actor residual、dev seeds 20–59、current10。

| arm | base | algorithm | reward | 目的 |
|---|---|---|---|---|
| `R0_BC_PPO` | frozen BC | PPO | old route baseline | 同预算 control |
| `R0_BC_SAC` | frozen BC | SAC | old route baseline | SAC 同预算 control |
| `R0_ALOS_PPO` | ALOS | PPO | old route baseline | 无 BC control |
| `R0_ALOS_SAC` | ALOS | SAC | old route baseline | 无 BC control |
| `R1_BC_PPO` | frozen BC | PPO | route-health | PPO route 信号 |
| `R1_BC_SAC` | frozen BC | SAC | route-health | SAC 同协议对照 |
| `R1_ALOS_PPO` | ALOS | PPO | route-health | 传统 base + RL residual |
| `R1_ALOS_SAC` | ALOS | SAC | route-health | 同协议对照 |

每个 arm 都必须跑完整 `4 levels × 2 radii`：`nominal/current07/current10/current15`
与 `default(2.0 m)/strict125(1.25 m)`，即 `8 × 4 × 2 = 64` 个评价 run；只记录方向、
分段和速度包线，不把 1 seed 结果写成 formal。Phase 1 统一
`1 seed / ~200k transitions`：SAC 为 200k，PPO 为
`256 envs × 100 steps × 8 iterations = 204,800`。

训练与评价都必须先通过 `scripts/check_goal171_protocol.py`；训练环境缺少课程块、
8-arm 不完整、level/radius 不完整、dev seed 或 held-out 边界错误都会直接中止。
训练结束后还必须通过 `scripts/audit_goal171_artifacts.py --phase train`，检查每个 arm
确实写出了 `curriculum_episodes.jsonl`、四个 stage 都出现、stage 比例没有漂移、深度和
`speed+current` 没有越界；评价结束后必须通过 `--phase eval`，检查 88 个 cell 的
`summary.json`、40 个 dev seed 和 trace 全部存在。任一项缺失都不得进入下一阶段。

正式产物根目录为
`artifacts/lowlevel/goal171_route_health_curriculum_depthfix_20261112`。

### C4 深度包线 bug 与旧运行作废

2026-11-12 的 `artifacts/lowlevel/goal171_route_health_curriculum_20261112` 整个根目录
标记为 `INVALID_C4_DEPTH_OUT_OF_BOUNDS_20261112`，不能用于 Gate171。根因不是算法，
而是环境契约错误：C4 课程深度范围为 `150–250 m`，但旧训练配置的
`plant.bounds_m[2][1]` 和 `termination.max_depth_m` 都是 `200 m`。C4 行 reset 后第一步
就触发 `OUT_OF_BOUNDS`，PPO 日志出现 `mean_episode_length=1.0`、`return_mean=-50.245`
和 `success_rate=0`，训练实际没有收集有效课程轨迹。

修复后固定：

- `plant.bounds_m[2][1] = 300.0`；
- `termination.max_depth_m = 300.0`；
- 预注册算法、reward、seed、level/radius 和 held-out 均不变；
- 旧 root 不删除，只保留取证；所有正式入口和 analysis 都改用 depthfix root。

因此这不是换算法，也不是用新 seed 救失败，而是修正配置/契约后按同一协议重跑。

2026-11-12 曾启动过一批未带课程块的环境配置；其 checkpoint 已标记为
`INVALID_NONCURRICULUM_20261112`，只保留取证，不进入任何 Gate。重新训练使用新的
课程锁定配置和新的产物根目录。

### Phase 2：3 seeds / 1M screen

只推进 Phase 1 中至少一项同时满足 route 方向和速度 guard 的算法；每个 arm
`3 seeds × 1M`，仍只做 screen。

### Phase 3：5M formal（仅在 Phase 2 通过后）

固定 3 seeds、仍强制完整 `4 levels × 2 radii`，并补充路径形状/长度/速度/噪声压力；
最终确认至少 5 seeds 才能写正式结论。

### Phase 4：residual schedule 机制

在同一个 winner 上比较：

- `always`：全程开启；
- `ramp`：前 20% 从 0 线性到 1；
- `late`：前 20% 关闭，再 ramp。

保持 reward、动作接口、算法和 seeds 不变；分析晚开是否只是降低探索噪声，还是改变了
中段 credit。若 `ramp` 能复制 `late`，优先保留 `ramp`，因为它更符合“经常开着 residual”的
部署要求。

## 6. 评价协议

主指标（所有实验统一）：

- 整条路线 `route_e3d_t_rmse_m`、`route_e3d_s_rmse_m`；
- `route_cross_abs_p90_m`、`route_vert_abs_p90_m`；
- 五段 `0–20/20–40/40–60/60–80/80–100` 的 3D/cross/vertical RMSE 与 p90；
- 10% 与 5% 弧长 keypoint 的 3D/cross/vertical error 与 reached；
- pre-terminal `0–90%` 的 3D/cross/vertical RMSE 与 p90；
- `completion_fraction`、`progress_efficiency`、mean/p95/max ground speed；
- timeout、safety、overspeed、action saturation、residual channel RMS。

二级诊断：success、final distance、goal-plane miss、终点附近 distribution。

## 7. Gate171

### Gate171-A：实现/契约

- reward 新字段默认 0 时与旧行为 parity；
- 所有 run 无 NaN/Inf，速度峰值 `<=2.5 m/s`；
- PPO policy/zero 均通过 speed guard；
- dev seeds 完整，无 held-out，无训练中途换 reward。

### Gate171-B：route-health screen

在 current10 default/strict125 上，至少一个 arm：

1. 3D route RMSE 改善 `>=5%` 且绝对 `>=0.05 m`；
2. pre-terminal `0–90%` 3D RMSE 改善 `>=0.03 m`，且 cross/vertical p90
   退化不超过 `0.05 m`；
3. 五段至少 3 段改善，最坏退化不超过 `0.15 m`；
4. 5% dense keypoint 在 20%–90% 至少 10/15 个改善；10% keypoint 至少 5/8 个改善；
5. `dense_keypoint_e3d_mean_m` 与 `dense_keypoint_e3d_p90_m` 均改善；
6. completion 不下降超过 `0.01`；
7. 2/3 training seeds 同向；速度、timeout、safety 不恶化。

### Gate171-C：部署/机制

- `ramp` 或 `always` 至少一个在多个 level/radius 保持路线收益；
- success 可以不变，但不得系统性退化；
- terminal 只作为二级诊断；若 success 改善而 route 不变，判 `TERMINAL_ONLY_SIGNAL`；
- 若路线改善但 success 退化，记录 `ROUTE_HEALTH_TRADEOFF`，不直接判失败，先查 terminal capture。

## 8. 执行步骤

- [x] Step 0：读 Goal170 result/blocker/ledger、Goal140、Goal167 和现有 route analyzer；
- [x] Step 1：实现 reward 字段、zero-weight parity 和单测；
- [x] Step 2：本地 20-step smoke，检查 reward finite、动作量级、速度包线；
- [ ] Step 3：Phase 1 `1 seed/200k`，启用 C3 课程，先通过 train artifact audit，
      再跑完整 `4 levels × 2 radii + 3 zero baselines`；
- [ ] Step 4：route/keypoint/segment 分析与 Gate171-B；
- [ ] Step 5：只推进方向正确的 arm 到 3 seeds/1M；
- [ ] Step 6：按结果决定 `continue/pivot/stop`；失败写 blocker、至少 2 hypothesis、
      reading note 和 ledger；
- [ ] Step 7：只有 screen 通过才进 5M formal；不自动 commit/push。

## 9. 执行入口

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_goal171_route_reward.py
python scripts/make_goal171_configs.py
.venv/bin/python scripts/check_goal171_protocol.py --phase train
bash scripts/run_goal171_train.sh
.venv/bin/python scripts/audit_goal171_artifacts.py \
  --phase train \
  --root artifacts/lowlevel/goal171_route_health_curriculum_depthfix_20261112 \
  --seeds 86201 --min-episodes 100
bash scripts/run_goal171_eval.sh
python scripts/analyze_goal170_route_transfer.py \
  --root artifacts/lowlevel/goal171_route_health_curriculum_depthfix_20261112 \
  --arms R0_BC_PPO R0_BC_SAC R0_ALOS_PPO R0_ALOS_SAC \
         R1_BC_PPO R1_BC_SAC R1_ALOS_PPO R1_ALOS_SAC \
  --levels nominal current07 current10 current15 \
  --radii default strict125 \
  --checkpoint-seeds 86201 \
  --zero-arm zero_sac50 \
  --ppo-zero-arm zero_alos50 \
  --ppo-bc-zero-arm zero_bc50
```

服务器入口：

```bash
ssh -p 43127 yhx@152.136.183.51
cd /data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111
ROOT=artifacts/lowlevel/goal171_route_health_curriculum_depthfix_20261112
mkdir -p "$ROOT/logs"
nohup setsid bash scripts/run_goal171_train.sh \
  >"$ROOT/logs/launcher.log" 2>&1 </dev/null &
nohup setsid bash scripts/run_goal171_eval.sh \
  >"$ROOT/logs/eval_launcher.log" 2>&1 </dev/null &
```

## 10. Goal171 提示词

```text
你现在执行 Goal171：route-health-centered residual training。

先读：
- doc/goals/goal171_route_health_training_20261112.md
- doc/goals/goal170_result_20261112.md
- research/blockers/2026-11-12-goal170-route-health-no-transfer.md
- research/paper_reading/2026-11-12-goal170-route-health-metrics.md
- doc/goals/goal140_vertical_tracking_reward_plan_20261111.md

硬约束：
1. 保留 success/termination/goal radius 和 z_t/BC 契约；
2. 速度硬上限 2.5 m/s；PPO 必须先通过 speed guard，否则 INVALID_SPEED；
3. 主指标是整条路线、pre-terminal 0–90%、五段以及 5%/10% keypoint，success 只是二级诊断；
4. 同一 Gate 只改一个因素：reward、action scale、schedule、算法不同时变；
5. 先 1 seed/200k falsification，再 3 seeds/1M，最后才 5M formal；
6. 不打开 held-out，不为失败 arm 扩 seed；
7. 失败后写 blocker、至少两个可证伪 hypothesis、reading note 和 ledger decision；
8. 不自动 commit/push。
9. 所有训练必须启用 C3 课程；训练后必须通过 `curriculum_episodes.jsonl` 审计；所有评价必须跑完整 `4 levels × 2 radii + 3 zero baselines`（88 runs），缺少任一项即 `INVALID_INCOMPLETE_EVAL`。
10. 旧 `..._curriculum_20261112` root 已因 C4 深度越界作废，禁止用于 Gate；只使用 `..._depthfix_20261112`。
```
