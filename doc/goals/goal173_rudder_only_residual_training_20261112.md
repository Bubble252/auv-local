# Goal173：rudder-only residual 的同协议 PPO/SAC 训练对照

**日期**：2026-11-12<br>
**状态**：`SCREEN_CANDIDATE_PASS_NOT_FORMAL / USER_REQUESTED_PAUSE_AFTER_SCREEN / CURRICULUM_LOCKED / FULL_EVAL_MATRIX_LOCKED / HELDOUT_CLOSED`<br>
**上游**：Goal172 eval-only 归因（全矩阵重跑）；Goal171 单 seed route-health screen；Goal86 PPO/SAC 残差对照

> **后注（2026-11-12）**：本 Goal 的 ALOS 基座是 legacy `ALOS-CRAB-EST`
> （`compensate_current=false, current_crab_enabled=true`）。新的 canonical 默认是
> `ALOS-COMP`（estimated-current `compensate_current`）；两者不得混表。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal173 --goal-doc doc/goals/goal173_rudder_only_residual_training_20261112.md --strict --require-result
```

## 1. 为什么现在做这个 Goal

Goal172 的 current10 诊断提示：`R1_BC_PPO` 在 learner rudder-only、effective
residual scale `0.75` 时，default/strict125 的整条 3D route RMSE 约有
`-0.13 m / -7.4%`，且五段同向改善、无 overspeed；同一 checkpoint 的全通道放大
则退化，stern/rpm-only 也退化。这个结果还不能直接当作训练结论，因为：

1. 还需要 Goal172 四 levels × 两 radii 的完整矩阵确认它不是 current10 偶然；
2. 这只是 eval-only 放大，不是“训练时固定 rudder-only”的新 checkpoint；
3. 需要同协议 PPO/SAC 对照，排除“只是换了 residual scale”的解释。

因此 Goal173 不换算法族，只训练 rudder-only residual，并做严格的控制臂。

## 2. 固定契约

- 观测、`z_t` 编码、BC/ALOS base、reward、termination、success、goal radius、
  sensor delay 全部沿用 Goal171/172；本 goal 只改 learner residual 的 action
  channel permission 与 scale。
- 所有训练必须启用冻结 C3 课程：`C3=0.70 / C2=0.15 / C0=0.10 / C4=0.05`；
  逐 episode 写 `curriculum_episodes.jsonl`，训练后课程审计失败即
  `INVALID_NONCURRICULUM`。
- 速度硬上限 `2.5 m/s`；训练 sampler 必须满足
  `speed + current + variation <= 2.5 m/s`。
- 所有评价必须跑完整 `arms × 4 levels × 2 radii + 3 zero baselines × 4 levels
  × 2 radii`；本 goal 8 个 policy arms 固定为 `64 + 24 = 88` 个 cell。
- dev seeds `20–59`；held-out `100–149` 关闭；不因单 seed 结果扩 seed 或改阈值。
- 主指标是整条路线 3D/cross/vertical、五段、10% keypoints、completion/progress；
  success、terminal capture、final distance 只作二级诊断。

## 3. 可证伪假设

### H173-A：固定 rudder-only permission 训练会把 eval-only 的路线收益变成可训练收益

- 成立：至少一个 `R1_BC_PPO` rudder-only arm 在 screen 的 current10 达到
  `>=5%` 且 `>=0.05 m` 的 3D route RMSE 改善，五段至少 3 段改善，20–90%
  的 10% keypoints 至少 5/8 改善，completion 不下降超过 `0.01`。
- 失败：放大 scale 但训练后 route 仍不动，或只有 success/final distance 改善。
- 最便宜证伪：1 seed、1M transitions，不换 reward，先只跑 current10 两个 radius。

### H173-B：收益来自 rudder 通道，而不是单纯把 residual scale 放大

- 成立：`rudder-only @0.75` 优于 `all-channel @0.75`，且 `rudder-only @0.25`
  只解释一小部分收益；stern/rpm 不出现同向路线改善。
- 失败：`all-channel @0.75` 与 rudder-only 同等甚至更好；此时必须撤掉
  “rudder 是瓶颈”的解释。
- 最便宜证伪：四个 1M screen arms 的全矩阵对照，不额外扫 seed。

### H173-C：SAC 若更好，只能是样本效率，不是接口或安全包线换来的

- 成立：相同 reward、checkpoint 初始化、curriculum、scale 下，SAC 比 PPO 更早
  出现 route 改善，且 `speed_max <= 2.5 m/s`、overspeed/timeout/safety 不恶化。
- 失败：SAC 靠更大动作幅度、更高速度或更差 timeout 换 success；此时不把 SAC
  写成算法优势。
- 最便宜证伪：PPO/SAC 同预算 1 seed screen，先比较 route 曲线与速度包线。

### H173-D：只改善终点附近不算成功

- 成立：改善必须出现在 pre-terminal `0–90%`、五段或中段 keypoints，而不只是
  最后 10%；否则记为 `TERMINAL_ONLY_SIGNAL`，不进入 formal。
- 失败：路线主指标无改善，success 只是终段捕获波动。

## 4. Screen 训练矩阵

先固定 1 个训练 seed（`86201`）、1M transitions/arm，不直接上 3M/5M。PPO 与
SAC 使用同一 frozen Goal86 BC residual base、同一环境、同一 C3 stage mix、同一
reward 和同一 residual scale 语义。SAC 的 feature extractor 因 command-aware BC
head 接口限制，仍用 Goal26/D3 command-free BC prior 初始化；该差异会写进 manifest，
不能把 base action 差异误判成算法差异。

| arm | algorithm | base | learner channels | residual scale | 作用 |
|---|---|---|---:|---:|---|
| `P_ALL_025` | PPO | BC | all | `0.25` | 原协议控制 |
| `P_ALL_075` | PPO | BC | all | `0.75` | scale-only 控制 |
| `P_RUD_025` | PPO | BC | rudder-only | `0.25` | channel-only 控制 |
| `P_RUD_075` | PPO | BC | rudder-only | `0.75` | **主候选** |
| `S_ALL_025` | SAC | BC | all | `0.25` | SAC 原协议控制 |
| `S_ALL_075` | SAC | BC | all | `0.75` | scale-only 控制 |
| `S_RUD_025` | SAC | BC | rudder-only | `0.25` | channel-only 控制 |
| `S_RUD_075` | SAC | BC | rudder-only | `0.75` | **SAC 主候选** |

`rudder-only` 的 learner channel gains 固定为 `[1.0, 1e-6, 1e-6]`；frozen base
residual 不被修改。SAC 的同一参数由新增
`--actuator-residual-channel-gains` 入口传入，避免 PPO/SAC 接口不一致。

> 2026-11-12 运行记录：第一次 Goal173 启动时发现 SAC config 缺少
> `base_residual_checkpoint`，实际会变成 ALOS + learned residual，与 PPO 的 BC base
> 不一致。该批 partial run 已停止并移到
> `artifacts/lowlevel/goal173_rudder_only_residual_20261112_aborted_baseshare_20261112`；
> 重新生成 config 后 PPO 与 SAC 共用同一个 Goal86 BC residual base 与 SHA256。
> 该 aborted root 不进入任何分析或结论。

> 2026-11-12 第二次运行记录：config 中虽然写入了冻结 `C3/C2/C0/C4`，但训练器仍无条件
> 构造 `CurriculumPromotionController`，dev eval 连续通过后自动把运行时 stage 改到
> `C5/C6`。检查到 `P_RUD_075/seed86201/curriculum_state.json` 的 `stage=C5`、
> `stage_mix=[C5 0.70, C4 0.15, C0 0.10, C6 0.05]`，且 episode provenance 已包含
> C5/C6。该批 root 已停止并移到
> `artifacts/lowlevel/goal173_rudder_only_residual_20261112_invalid_autopromotion_20261112`，
> 标记为 `INVALID_NONCURRICULUM`，不得评价、分析或用于 Gate。
>
> 修复：`curriculum.promotion_enabled=false` 现在贯穿 config、训练 hook、manifest、
> `curriculum_state.json` 和 artifact audit；Goal173 协议检查器强制该字段为 false，
> 并检查最终 `stage=C3`、冻结 mix 一致、`finalized=true`、没有 promotion decisions。
> 重新训练后必须先从零开始生成新的有效 root，不能复用本批 checkpoint。

> 2026-11-12 第三次运行记录：第一次清理时旧 SAC launcher 收到 `SIGTERM` 后没有终止，
> 旧进程继续用旧代码写 state；随后新 SAC 也写同一目录，造成 dual-writer 污染。该 root
> 已移到
> `artifacts/lowlevel/goal173_rudder_only_residual_20261112_invalid_dualwriter_20261112`，
> 同样标记为 `INVALID_NONCURRICULUM`。`run_goal173_train.sh` 现在使用 `flock` 和
> stale-process 预检；正式重跑前必须确认没有任何进程引用目标 root。

## 5. 完整评价矩阵

每个 policy arm 都必须在以下 8 个 cell 上评价，不能只跑 current10：

```text
levels: nominal / current07 / current10 / current15
radii:  default=2.0 / strict125=1.25
seeds:  20–59 (40 个 dev seeds)
```

另跑 3 类 zero baseline：`zero_sac50 / zero_bc50 / zero_alos50`，每种同样覆盖
4 levels × 2 radii。总 cell 数：

```text
8 policy arms × 4 levels × 2 radii = 64
3 zero baselines × 4 levels × 2 radii = 24
total = 88
```

缺 cell、缺 `summary.json`/`trace.npz`、seed 集不完整或混入 held-out，直接
`INVALID_INCOMPLETE_EVAL`，不进入分析。

## 6. Gate173

### Gate173-A：工程与课程

- 8 个训练 run 都有 `curriculum_episodes.jsonl`，C3/C2/C0/C4 全出现，比例
  在冻结容差内；
- 训练/评价速度峰值 `<=2.5 m/s`；无 NaN/Inf、无 flew-out；
- `P_ALL_025` 与 Goal171 的同配置 baseline 行为一致。

### Gate173-B：screen route gate

候选 `P_RUD_075` 或 `S_RUD_075` 至少一个满足：

| check | 阈值 |
|---|---:|
| current10 3D route RMSE 改善 | `>=5%` 且 `>=0.05 m` |
| 五段改善 | `>=3/5`，最坏退化 `<=0.15 m` |
| 20–90% 10% keypoints 改善 | `>=5/8` |
| pre-terminal 0–90% 3D RMSE | 改善 `>=0.03 m` |
| completion | 退化不超过 `0.01` |
| 速度/安全 | 无 overspeed，timeout/safety 不恶化 |

通过后才允许把该 arm 扩到 3 seeds/3M；单 seed screen 不能写成 formal。

### Gate173-C：部署与迁移

- 候选在 4 levels × 2 radii 的至少 6/8 个 cell 保持同向路线收益；
- 不是只靠 default 或 strict125 的单 radius；
- 若 route 改善但 success 下降，记录 `ROUTE_HEALTH_TRADEOFF`，先查 terminal
  capture，不直接换算法；
- 若失败原因明确是探索/样本效率，才进入 SAC 预算或 REDQ/DroQ 的新预注册 goal。

## 7. 执行步骤

- [x] Step 0：Goal172 全矩阵结果出来，确认 `P_RUD_075` 候选及控制臂方向；
- [x] Step 1：生成 Goal173 configs、manifest、protocol checker 和 artifact audit；
- [x] Step 2：本地/远端 tiny smoke，验证 PPO/SAC 的 channel gains/scale、共享 BC base 和
  课程 provenance 真正进入环境与 manifest；
- [x] Step 3：1 seed/1M screen，先通过课程审计和完整矩阵审计；
- [x] Step 4：按 Gate173-B 选候选；`P_RUD_075` 通过，状态 `PASS_NOT_FORMAL`；
- [ ] Step 5：通过后 3 seeds/3M，最后才考虑 5M formal；**按用户要求暂停，不自动执行**；
- [x] Step 6：写 result、ledger、blocker/reading note（若失败），不自动 commit/push。

结果与完整 88-cell 分析见 `doc/goals/goal173_result_20261112.md`。本轮有效产物为
`artifacts/lowlevel/goal173_rudder_only_residual_20261112`；三个
`aborted_baseshare` / `invalid_autopromotion` / `invalid_dualwriter` root 不得使用。

## 8. 复现入口（已按 Goal172 结果冻结）

```bash
cd settled_局部规划器_反应式
.venv/bin/python -m pytest -q tests/test_goal173_residual_channel_config.py
.venv/bin/python scripts/check_goal173_protocol.py --phase train
bash scripts/run_goal173_pipeline.sh
.venv/bin/python scripts/analyze_goal173_rudder_only.py \
  --root artifacts/lowlevel/goal173_rudder_only_residual_20261112
```

正式入口不再提供“只训练、不跑完整评价”的收尾方式。`run_goal173_pipeline.sh`
会先做训练协议检查，训练后做课程审计，再做评价协议检查、88-cell 评价和评价审计。
只跑 `default/strict125`、`current10` 或单个 arm 只能作为诊断，不能写成 Gate 结果。

服务器：`ssh -p 43127 yhx@152.136.183.51`，仓库
`/data/yhx/underwater-robotics/src/plan_p1/goal153_reward_curriculum_20261111`。

## 9. Goal173 提示词

```text
你现在执行 Goal173：rudder-only residual 的同协议 PPO/SAC 训练对照。

先读：
- doc/goals/goal173_rudder_only_residual_training_20261112.md
- doc/goals/goal172_result_20261112.md
- doc/goals/goal171_result_20261112.md

硬约束：
1. 所有训练必须启用 C3/C2/C0/C4=0.70/0.15/0.10/0.05 课程，并保留逐 episode
   curriculum_episodes.jsonl；缺失即 INVALID_NONCURRICULUM。
2. 所有评价必须跑 8 arms × 4 levels × 2 radii + 3 zero baselines，共 88 cells；
   不能只跑 default/strict125 或 single level。
3. 速度上限 2.5 m/s；held-out 100–149 关闭；dev seeds 20–59。
4. 先 1 seed/1M screen，再 3 seeds/3M；不直接上 5M formal。
5. 不换算法族；PPO/SAC 使用同接口、同 channel gains、同 residual scale 语义。
6. 失败后先归因，至少写两个可证伪 hypothesis、reading note 和 ledger decision；
   不通过换 seed、放宽阈值或改 held-out 救结果。
7. 主指标是整条路线、五段、keypoints、completion/progress；success 只作二级诊断。
8. 不自动 commit/push。
```

