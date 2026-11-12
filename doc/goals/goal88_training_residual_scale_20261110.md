# Goal88：训练时执行器残差尺度 1.5×/2× 的预注册对照

**日期**：2026-11-10<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`<br>
**上游**：Goal87 H87-O2 / H87-O4、Goal86 PPO/SAC screen、Goal85 unbiased contract<br>
**性质**：预注册训练对照；固定 `z_t64`、frozen BC、PPO、reward、terminal、success、goal radius 和 dev seeds，只改训练时 `pc.actuator_residual_scale`。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal88 --goal-doc doc/goals/goal88_training_residual_scale_20261110.md --strict
```

`required: true`：本 Goal 会改变训练时动作参数化。Gate 记录前不得改 reward、
terminal、success、goal radius、seed 预算或 held-out；不得在同一 Goal 中途换
PPO/SAC 主算法族。

## 1. 背景与立项证据

Goal87 已经排除网络接线、checkpoint 错配、NoisyNet 主导、reward/terminal
强错位和 stale old-log-prob。H87-O2 进一步证明：

- 显式非零初始化真实生效，初始 std `1e-3/1e-2/3e-2` 对应最终
  residual/base ratio `0.86%/2.06%/6.37%`；
- 但 default success 只有 `30/30/31`，strict125 全是 `17/40`；
- `3e-2` 三个训练 seed 都只把 default 的边界 dev seed 33 翻成 success，
  strict125 翻转 `0/40`。

H87-O4 的 eval-only 反事实给出方向性证据：

| eval 残差乘数 | default success | strict125 success | magnitude clip |
|---:|---:|---:|---:|
| 1× | 31/40 | 17/40 | 0.659 |
| 1.5× | 31/40 | 20/40 | 0.741 |
| 2× | 33/40 | 21/40 | 0.802 |
| 5× | 23/40 | 10/40 | 0.950 |
| 10× | 15/40 | 8/40 | 0.993 |

2× 在 `1e-3/1e-2/3e-2` 共 7 个 checkpoint 上把 strict125 提到 `20–21/40`。
这说明策略方向可用，训练出的物理幅度约是可用最优值的一半；5× 以后执行器
包络饱和，继续放大反而有害。

因此本 Goal 只问一个便宜的问题：**训练时直接把残差尺度提到 1.5×/2×，
策略会不会自己学到足够的物理幅度，并把 strict125 从 17/40 推过 BC？**

## 2. 假设

### H88-A：动作参数化是主因，训练时放大尺度会直接改善闭环

- 支持：`scale=1.5/2.0` 训练后，确定性 eval 的 physical residual
  `actuator_residual_abs_mean` 明显高于 scale1，且 strict125 success
  至少一个 arm 达到 `>=20/40`，3 seeds 中至少 2 个方向一致；path 3D RMSE
  退化不超过 `0.02 m`，plane-miss/timeout 不恶化。
- 否定：normalized residual ratio 随训练尺度按比例缩小，physical residual
  仍停在 scale1 的量级；strict125 仍 `17/40` 或更低。
- 最便宜证伪：`scale=2.0`、seed86201、1M、dev40 两场景；若 success 不动，
  再跑 seed86202/03 确认不是单 seed。

### H88-B：如果训练放大后均值仍缩回一半，瓶颈是 on-policy 探索/样本效率

- 支持：`scale=2.0` 的 normalized ratio 从 `0.06` 降到约 `0.03`，
  physical residual 与 scale1 基本相同，success 不改善；同时 KL/clip 没有
  爆炸。此时才允许新开 SAC 同协议对照。
- 否定：`scale=2.0` 的 normalized ratio 保持约 `0.06`，physical residual
  翻倍并带来 success 改善；说明不需要 SAC，先继续动作参数化。
- 最便宜证伪：比较 scale1/1.5/2.0 的 `residual_to_base_l2_ratio_mean`
  与 `actuator_residual_abs_mean`；不需要额外训练。

### H88-C：2× 训练会把策略推到饱和区，1.5× 才是最优

- 支持：`scale=2.0` 的 magnitude clip `>0.90`，success/path RMSE 比
  `scale=1.5` 差；`scale=1.5` 的 success 更高且 clip 仍 `<0.85`。
- 否定：`scale=2.0` 的 clip 仍约 `0.80`，success 不低于 1.5×。
- 最便宜证伪：同一 1M screen 同时跑 1.5×/2.0×，比较 clip 与 success。

## 3. 固定契约

### 3.1 不变项

- 架构：项目原生 `z_t64`，PPO features `[z_t64, command3, zero_dade64]=131`；
- 基座：frozen D3 BC，`base_policy.checkpoint` 与 SHA 不变；
- 算法：PPO，`1000 envs × 25 steps × 40 iterations = 1M`；
- reward：Goal85 `pc_command_track_v2`，不改 `goal_bonus/timeout/w_path_3d`；
- terminal：`position_source=eta`，零传感器延迟，`unbiased_noise_v1`；
- success：`success_frame=both`，goal radius 不变；
- dev seeds `20–59`；held-out `100–149` 关闭；
- 训练 seeds `86201/86202/86203`；
- 初始化：`--residual-output-init-std 0.03`，与 Goal87 `3e-2` 一致。

### 3.2 唯一变量

训练配置的 `pc.actuator_residual_scale`：

| arm | 训练 scale | 训练 seeds | 预算 | 评价 |
|---|---:|---|---:|---|
| `S1.0` | 1.0 | 86201/02/03 | 1M | 已有 Goal87 `3e-2`，作为同协议参照 |
| `S1.25` | 1.25 | 86201/02/03 | 1M | Pareto screen；1M 后若通过 Gate 则进 5M formal |
| `S1.5` | 1.5 | 86201/02/03 | 1M | dev40 default+strict125 |
| `S1.75` | 1.75 | 86201/02/03 | 1M | Pareto screen |
| `S2.0` | 2.0 | 86201/02/03 | 1M | dev40 default+strict125 |

`base_policy.residual_scale` 仍为 `1.0`；只改 `pc.actuator_residual_scale`，
不混用这两个字段。

`S1.25`/`S1.75` 是在 1.5×/2.0× 出现 path RMSE tradeoff 后加入的 Pareto
细化 arm；Gate 阈值、seed、dev 集和预算都没有改变。选择 formal arm 时仍按
原 Gate 的 `path RMSE <=0.02 m`，不因为 success 更高就放宽 path 门禁。

## 4. 指标

主指标：

- default / strict125 的 `success_rate`；
- 逐 seed 相对 BC 的 success 翻转（gain/loss）和 McNemar；
- `path_three_d_rmse_m`、`plane_miss_rate`、`timeout_rate`；
- `actuator_residual_abs_mean`（物理幅度）；
- `residual_to_base_l2_ratio_mean`（策略归一化输出，用于判断是否按比例缩回）；
- `actuator_residual_magnitude_clip_rate`、`scale_clip_rate`；
- `approx_kl`、`clip_fraction`、actor 末层 `weight_norm`。

过渡统计：

- goal proximity 分桶与 capture-ever（`1.25/1.5/2.0/2.5 m`）；
- 终止帧 eta/p_nav 距离与成功余量；
- 3 seeds 的配对方向一致性。

## 5. 执行计划

- [ ] Step 0：写本 Goal 文档和 Research Contract；创建 `configs/lowlevel/goal88/` 的
      `scale1.5/scale2.0` 配置，只改 `pc.actuator_residual_scale`。
- [ ] Step 1：本地单测和 1-iteration smoke，确认新 checkpoint 的
      `training_manifest.json` 记录 `actuator_residual_scale=1.5/2.0`，且初始
      residual 与 Goal87 `3e-2` 一致。
- [ ] Step 2：远端 6×4090 并行跑 `S1.5`/`S2.0` 各 3 seeds，1M；日志写
      `logs/goal88/`，checkpoint 写 `artifacts/lowlevel/goal88_scale_20261110/`。
- [ ] Step 3：12 个 dev40 eval（6 checkpoint × default/strict125），单线程
      CPU/float64 并行；按 seed 与 BC 配对。
- [ ] Step 4：Gate 判定；若 screen 通过，用通过 path 门禁的 Pareto arm
      （当前为 `S1.25`）跑 5M formal（`training.iterations=200`）；formal 前不打开 held-out。
- [ ] Step 5：写 `doc/goals/goal88_result_20261110.md`，更新 blocker/STATE/LEDGER，
      运行 checker；不自动 commit/push。

训练命令模板（远端仓库根目录）：

```bash
CUDA_VISIBLE_DEVICES=0 /data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/train_batch_lowlevel_ppo.py \
  --training-config configs/lowlevel/goal88/pa_bc_frozen_residual_scale1.5_upd25x40.json \
  --iterations 40 --num-envs 1000 --num-steps 25 --num-mini-batches 100 \
  --residual-output-init-std 0.03 --seed 86201 \
  --output artifacts/lowlevel/goal88_scale_20261110/scale1.5_seed86201 \
  --device cuda --record-only --overwrite
```

评价命令模板：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/goal88/pa_bc_frozen_residual_scale1.5_upd25x40.json \
  --checkpoint artifacts/lowlevel/goal88_scale_20261110/scale1.5_seed86201/final.pt \
  --output artifacts/lowlevel/goal88_scale_20261110/scale1.5_seed86201_eval_default_dev40 \
  --seed-start 20 --seed-count 40 --device cpu --dtype float64 --success-frame both
```

strict125 追加：

```bash
--environment-config configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json
```

## 6. Gate

### 6.1 1M screen Gate

| check | 阈值 |
|---|---:|
| strict125 success | 至少一个 arm 在 seed86201 上 `>=20/40`，且不低于 BC `17/40` |
| default success | 不低于 BC `30/40` |
| 3-seed 方向 | 至少 2/3 seeds 的 strict125 相对 BC 净 gain `>=0`，且合计 net gain `>=3/120` |
| path 3D RMSE | 相对 BC 退化 `<=0.02 m` |
| 安全性 | plane-miss/timeout 不恶化；0 error / 0 NaN |
| 归一化幅度 | 若 physical residual 翻倍但 normalized ratio 按比例缩回，记录为 H88-B 支持 |

判定：

- `PASS_SCREEN`：进入 5M formal；
- `PARTIAL`：只扩 seed，不改 Gate；
- `FAIL`：写 blocker；若明确指向探索/样本效率，按预注册开 SAC 同协议对照；
- `INVALID`：实现/契约错误，修好后重跑同 arm，不换 seed 救结果。

### 6.2 5M formal Gate

只在 screen 通过后执行；当前预注册 formal arm 为 `S1.25`（1M screen 中唯一
同时满足 success 提升与 path RMSE `<=0.02 m` 的 arm）：

- 3 seeds、5M transitions；
- strict125 success 相对 BC 提升且配对 95% CI 不跨 0；
- path RMSE 不退化，plane-miss/timeout 不恶化；
- 3 seeds 方向一致；
- formal 记录后才允许打开 held-out `100–149`。

## 7. 判读与下一步

| 观察 | 判读 | 下一步 |
|---|---|---|
| 1.5×/2× 的 normalized ratio 保持约 0.06，physical residual 翻倍，success 上升 | 动作尺度是主因 | 继续 scale 扫描/formal，不换 SAC |
| normalized ratio 随 scale 按比例缩回，physical residual 不变 | 策略把物理幅度自我封顶，指向探索/样本效率 | 新开 SAC 同协议对照 |
| 2× clip >0.90 且 success 下降 | 2× 过饱和 | 用 1.5× 或更细的 1.25/1.75 |
| 3 seeds 方向不一致 | seed 噪声 | 不扩 formal，写 `seed_noise_ambiguous` blocker |

只有 `decision=pivot` 才允许换主线；本 Goal 不直接跳到 REDQ/DroQ。

## 8. Git

禁止 `git add -A`。只 add 本次相关文件；提交前运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal88 \
  --goal-doc doc/goals/goal88_training_residual_scale_20261110.md --strict
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志和缓存。不自动 commit/push。

## 9. Goal 提示词

```text
你现在执行 Goal88：在 frozen BC + z_t64 + PPO 的同一协议下，只把训练时
pc.actuator_residual_scale 从 1.0 提到 1.5/2.0，验证是否能修掉 Goal87
观察到的“方向对但物理幅度约小一半”。

先读：
- doc/goals/goal88_training_residual_scale_20261110.md
- doc/goals/goal87_network_history_parity_20261110.md
- doc/goals/goal87_residual_credit_assignment_20261110.md
- research/blockers/2026-11-10-goal87-ppo-no-movement.md

硬约束：
1. 固定 z_t64、frozen BC、PPO、Goal85 unbiased_noise_v1、position_source=eta、
   零传感器延迟、reward、terminal、success、goal radius；
2. dev seeds 20-59；held-out 100-149 关闭；
3. 训练 seeds 86201/86202/86203；1M screen，5M formal 只在 screen 通过后；
4. 只改 pc.actuator_residual_scale，不改 base_policy.residual_scale；
5. 不打开 held-out，不自动 commit/push，不换 REDQ/DroQ。

执行顺序：
- 创建 scale1.5/2.0 配置和单测；
- 1-iteration smoke，确认 manifest 的 scale 和初始 residual；
- 6 卡并行训练，12 个 dev40 eval；
- 逐 seed 配对、KL/clip/actor norm、physical vs normalized residual；
- 按 Gate 判 PASS/PARTIAL/FAIL，写 result md、ledger、STATE；
- 只有 H88-B 被支持时才新开 SAC 同协议对照。
```

## 10. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。每次失败先归因，不直接换算法或扫 seed；
每个 blocker 至少写两个可证伪 hypothesis；用 `paper-search` / `deep-research`
查论文和代码并写 reading note；先跑最便宜的 falsification，再写
`continue / pivot / stop`。成功标准、停止条件和自动执行边界以
`research/AUTORESEARCH_LOOP.md` 为准。
