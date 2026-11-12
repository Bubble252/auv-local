# Goal87-PPO 诊断：更新次数、零输出初始化与探索

**日期**：2026-11-10<br>
**状态**：`SCREEN_RECORDED / PPO_LOCKED / NOT_A_GATE / HELDOUT_CLOSED`<br>
**上游**：Goal86 T1/T2 PPO 1M screen、Goal87 残差信用分配<br>
**性质**：固定 PPO on-policy 算法，只改变训练调度、初始化和探索参数；不改变 success 定义、reward、terminal capture 或 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal87 --goal-doc doc/goals/goal87_residual_credit_assignment_20261110.md --strict
```

本文件是 Goal87 的 PPO 子诊断，不修改 Goal86 已冻结的 Gate。任何结果先标
`PPO_DIAG / NOT_GATE`；只有 3 seeds 方向一致且通过预注册方向性阈值，才允许
把它提升为新的 screen 结论。

## 1. 已观察到的 PPO 失败形态

Goal86 T2-PPO 1M screen：

| 项目 | 观察 |
|---|---:|
| 配置 | `1000 envs × 125 steps × 8 iterations = 1,000,000` |
| actor warmup | `5 iterations` |
| actor 可见 iteration | `3` |
| 每 iteration mini-batch 更新 | `100` |
| actor 最终层权重范数 | 约 `0.008` |
| actor 前两层权重范数 | 约 `9.1 / 6.3` |
| 最后 iteration 的 `approx_kl` | 约 `2.3e-5` |
| 训练 stochastic success | 约 `0.748` |
| 确定性 dev40 success | `0.675`（zero-residual 也是 `0.675`） |
| 确定性 strict125 success | `0.35`（zero-residual 也是 `0.35`） |

这不是“完全没更新”：每次 PPO iteration 会跑 `num_mini_batches=100` 次更新。
问题是 actor 只有 3 个可见 iteration，且最后层梯度/更新幅度极小；训练时带
随机扰动的策略成功率更高，但确定性均值没有吸收这部分收益。当前优先排除：

1. 1M 内有效 PPO iteration 太少；
2. `initial_residual_zero=true` 把 actor 放在一个强零解局部点；
3. 探索幅度太小，或者 PPO 的 `clip_param=0.02`/学习率过于保守；
4. 以上都不是，才进入 reward shape/terminal capture。

## 2. 可证伪假设

| ID | 假设 | 成立应看到 | 失败应看到 | 最便宜证伪 |
|---|---|---|---|---|
| `H87-P1` | 1M 内 actor 可见 iteration 太少 | `steps=25, iterations=40` 在同一 1M 下 residual 和 success 同时移动 | 仍近零残差、KL 仍约 `1e-5` | 单 seed 1M screen |
| `H87-P2` | 零输出初始化是强的零解局部点 | `initial_residual_zero=false` 的随机非零初始化能产生非零均值并保持/提升 success | 随机初始化也迅速塌回零或明显变差 | 单 seed 1M screen |
| `H87-P3` | 探索尺度与物理残差效果不匹配 | `init_noise_std=0.3/0.5` 使 residual 增长，且 success 或目标附近分布改善 | 只增大动作噪声，success 不改善或恶化 | 单 seed 1M screen |
| `H87-P4` | PPO 更新过于保守 | `clip_param=0.2`、`lr=1e-3`、`epochs=4` 或去掉 warmup 后 actor 明显移动 | 所有更新参数都不改变 final layer/KL | 单 seed 1M screen |
| `H87-P5` | BC 头非零先验能打破零解 | P-A `BC head + PPO` 的 residual 非零且 dev40 优于 zero-residual | BC 头被迅速压回零或只复制 BC 的弱收益 | 单 seed 1M screen |

## 3. 固定契约

- 架构：项目原生 `z_t`；PPO features `[z_t64, command3, zero_dade64] = 131`；不退回 flatten+MLP。
- 环境：Goal85 `unbiased_noise_v1`、`position_source=eta`、零传感器延迟、path12。
- 训练预算：每个 arm 恰好 `1,000,000` env transitions；不因结果好坏偷加预算。
- 评价：dev seeds `20–59`，default 和 strict125 两个场景；held-out `100–149` 关闭。
- 评价方式：同 seed 配对；至少记录 success、true-eta、plane-miss、timeout、path 3D RMSE、terminal eta 分布、residual L2 ratio、action abs max。
- 训练 seed：第一批统一 `86201`；只有 screen 出现方向性移动才扩到 `86202/86203`。
- 禁止：不改 success/goal radius/reward/terminal；不把 PPO 结果和 SAC 混算；不打开 held-out。

## 4. 第一批实验矩阵

每个 arm 只改一个主变量；`baseline` 是 Goal86 `T2-PPO` seed `86031`，并额外报告同 seed 的 zero-residual。

| arm | 配置主变量 | 目的 | 输出 |
|---|---|---|---|
| `baseline` | `steps125, iters8, warmup5, std0.1, clip0.02` | 复现 Goal86 PPO 锁定零残差 | `screen/t2_ppo` |
| `upd25x40` | `steps25, iters40` | 同 1M 下增加 actor 可见 iteration | `ppo_diag/upd25x40` |
| `upd50x20` | `steps50, iters20` | 控制 rollout 分段，验证不是单纯 iteration 计数 | `ppo_diag/upd50x20` |
| `random_init` | `initial_residual_zero=false` | 检验零输出初始化局部点 | `ppo_diag/random_init` |
| `noise0.3` | `init_noise_std=0.3` | 检查探索幅度 | `ppo_diag/noise0.3` |
| `noise0.5` | `init_noise_std=0.5` | 检查更大探索幅度 | `ppo_diag/noise0.5` |
| `epochs4` | `num_learning_epochs=4` | 同一 rollout 多用数据 | `ppo_diag/epochs4` |
| `critic0` | `critic_warmup_iterations=0` | 检验 warmup 是否吃掉了 1M 预算 | `ppo_diag/critic0` |
| `clip0.2` | `clip_param=0.2` | 标准 PPO 裁剪是否解除保守更新 | `ppo_diag/clip0.2` |
| `lr1e-3` | actor/critic lr `1e-3` | 检查优化步长 | `ppo_diag/lr1e-3` |
| `adv_norm` | `normalize_advantage_per_mini_batch=true` | 检查 advantage 尺度 | `ppo_diag/adv_norm` |
| `pa_bc_head` | P-A + BC head 非零热启动（`--warmstart-actor`） | 检验 BC 先验（H87-P5） | `ppo_diag/pa_bc_head` |

第一批先并行 `upd25x40`、`random_init`、`noise0.3`。原因是它们分别对应
更新预算、初始化局部点和探索三个互不替代的解释，且训练成本相同。其余 arm
只在第一批没有方向性信号时启动，避免无边界扫参。

## 5. 预注册判读

单 seed screen 只做方向性判读，不写 formal 结论：

| 观察 | 判读 | 下一步 |
|---|---|---|
| residual L2 ratio `>= 0.05`，且 default 或 strict125 success 比 zero-residual 高 `>= 2.5 pp`，path RMSE 不退化 `> 0.05 m` | `MOVE` | 补 seed `86202/86203`，保持同一 arm |
| residual ratio `>= 0.02`，但 success 未移动 | `WEAK_MOVE` | 先查 reward/terminal 对残差的敏感性，不扩 formal |
| residual ratio `< 0.02` 且 final layer norm `< 0.02` | `LOCKED` | 进入下一条解释（初始化/更新/探索） |
| success 下降 `>= 5 pp` 或 plane-miss/timeout 明显恶化 | `HARMFUL` | 停止该 arm，不通过加预算救 |
| 与 baseline 的差异小于 40-seed 配对噪声 | `INVALID_SCREEN` | 不写成胜负，只扩大 seed 或改评价 |

## 6. 运行命令

远端仓库根目录：

```bash
export PYTHONPATH="src:/data/yhx/underwater-robotics/src/stable-baselines3"
export PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" scripts/train_batch_lowlevel_ppo.py \
  --training-config configs/lowlevel/goal86/ppo_diag_20261110/upd25x40.json \
  --output artifacts/lowlevel/goal86/ppo_diag/upd25x40 \
  --device cuda --overwrite
```

每个训练结束后固定评价 default/strict125 dev40：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" scripts/evaluate_pc_ppo.py \
  --training-config configs/lowlevel/goal86/ppo_diag_20261110/upd25x40.json \
  --checkpoint artifacts/lowlevel/goal86/ppo_diag/upd25x40/final.pt \
  --output artifacts/lowlevel/goal86/ppo_diag/upd25x40_eval_default_dev40 \
  --seed-start 20 --seed-count 40 --device cpu --dtype float64
```

strict125 评价必须显式加
`--environment-config configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json`；
不要使用 `--allow-sensor-override` 冒充场景覆盖。

## 7. Git 与产物

- 训练配置、训练 manifest、iteration records 是实验产物，不提交 `artifacts/`。
- 若新增配置生成脚本或诊断脚本，只 add 相关文件：

```bash
git add configs/lowlevel/goal86/ppo_diag_20261110/<file>.json
git add doc/goals/goal87_ppo_update_diagnostic_20261110.md
git diff --cached --check
git commit -m "goal87: preregister PPO update diagnostic"
```

禁止 `git add -A`；不提交 checkpoint、日志、缓存或大文件。

## 8. 停止条件

- 三个主解释中任一个出现 `MOVE`：保留该方向，补 3 seeds；不扩大到无边界参数扫描。
- 三个主解释全部 `LOCKED`：进入 Goal87 的 reward shape / terminal capture 审计，并写 blocker 与 reading note。
- PPO 在所有解释下都锁定零残差，而 SAC 已有非零残差：只能在 blocker 中提出“on-policy 样本效率/优化问题”，仍不直接换算法；另开 Goal 做预注册 SAC 对照。
- 任何阶段都不打开 held-out，不以换 seed、加预算或改 success 补救。


## 9. 2026-11-10 实际结果

### 9.1 第一批与扩展 arm

| arm | default success | strict125 success | actor 末层范数 | 末次 KL | clip fraction | mean episode reward (default) |
|---|---:|---:|---:|---:|---:|---:|
| zero-residual | 0.675 | 0.350 | — | — | — | — |
| upd25x40 | 0.675 | 0.350 | 0.0186 | 1.7e-4 | 0.199 | -3.5099 |
| upd50x20 | 0.675 | 0.350 | 0.0123 | 9.9e-5 | 0.106 | -3.5098 |
| random_init | 0.625 | 0.350 | 0.982 | 1.0e-3 | 0.620 | -3.7312 |
| noise0.3 | 0.675 | 0.350 | 0.0130 | 2.4e-5 | 0.009 | -3.5094 |
| noise0.5 | 0.675 | 0.350 | 0.0187 | 1.8e-5 | 0.008 | -3.5109 |
| epochs4 | 0.675 | 0.350 | 0.0117 | 7.8e-5 | 0.095 | -3.5101 |
| critic0 | 0.675 | 0.350 | 0.0127 | 3.2e-5 | 0.025 | -3.5096 |
| clip0.2 | 0.675 | 0.350 | 0.0219 | 1.1e-3 | 0.003 | -3.5103 |
| lr1e-3 | 0.675 | 0.350 | 0.0183 | 2.1e-4 | 0.245 | -3.5107 |
| adv_norm | 0.675 | 0.350 | 0.0077 | 3.6e-5 | 0.028 | -3.5101 |

这些 T2-PPO arm 都是 `base_policy.mode=none` 的直接 actuator 策略，不是
“冻结 BC + 残差”，因此没有 `residual_to_base_l2_ratio` 字段；不能把它们的
`actuator_residual_abs_mean` 与 frozen residual 的 `residual/base L2` 直接比较。
共同现象是：default/strict125 success 与 zero-residual 基本相同，末层范数
`<0.022`（random_init 除外），KL `1e-5–1e-3`。

其中 `random_init` 也是直接 actuator 策略，不是 frozen BC residual；它不能作为
H87-O2（frozen residual 小非零输出初始化）的结果。H87-O2 仍未运行。

### 9.2 BC head 3-seed 与 frozen residual

| arm | default success | strict125 success | 说明 |
|---|---:|---:|---|
| BC-only / model_0 | 0.750 | 0.425 | 冻结 BC 基座 |
| BC head seed86201 | 0.700 | 0.425 | 8 iter |
| BC head seed86202 | 0.775 | 0.450 | 8 iter |
| BC head seed86203 | 0.725 | 0.425 | 8 iter |
| BC head upd25x40 seed86201 | 0.475 | 0.350 | 40 iter，被推坏 |
| frozen residual 0.25 | 0.750 | 0.425 | 三个 seed 几乎不动 |
| frozen residual 1.0 | 0.750 | 0.425 | success 仍不动 |
| frozen scale1 upd25x40 | 0.750 | 0.425 | success 仍不动 |

### 9.3 结论

- `pa_bc_head` 的收益来自 BC 先验，不是 PPO 学会了有效增量；
- 增加 PPO iteration 的 `upd25x40` 反而把 BC 先验推坏；
- frozen residual 在 8/40 iter、scale 0.25/1.0 下都锁定在毫米级；
- H87-O1 已证伪：减少 mini-batch 没有让残差长大，success 翻转 `0/40`；
- H87-O3 不支持 stale old-log-prob：KL 峰值在第二个 post-warmup iteration；
- H87-O2 仍开放：配置里的 `initial_residual_zero=false` 被 loader 强制零初始化覆盖，
  不能算作已测试；
- 当前没有“残差方向对、只是探索不够”的证据，也没有“样本效率不足”的证据；
- 下一步先做 `research/blockers/2026-11-10-goal87-ppo-no-movement.md`
  中的 H87-O2 小非零初始化对照；H87-O2 也否定后，才允许新开预注册 Goal 考虑
  latent `delta_z` 或 SAC 同协议对照，不直接换 REDQ/DroQ。
