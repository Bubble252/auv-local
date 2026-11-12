# Blocker：Goal87 PPO 残差确定性均值锁在毫米级，终态捕获不变

日期：2026-11-10<br>
Goal：Goal87 `PPO_DIAG / NOT_GATE`<br>
状态：`ACTIVE / OPTIMIZATION_CREDIT_ASSIGNMENT / H87_O2_MECHANISM_MOVE / H87_O4_ACTION_SCALE_POSITIVE / NEXT_GOAL88`

## 1. 现象

在同一冻结 D3 BC 基座、同一 Goal85 `unbiased_noise_v1 + position_source=eta`
契约、同一 1M transitions 预算下：

| arm | 训练 seed | default success | strict125 success | 确定性 residual / base L2 | actor 末层范数 | 末次 KL |
|---|---:|---:|---:|---:|---:|---:|
| zero-residual | 86031 | 0.675 | 0.350 | 0 | — | — |
| frozen residual 8it | 86201/02/03 | 0.750 | 0.425 | 0.0014 | 0.015–0.016 | 7e-5 |
| frozen residual 40it | 86201/02/03 | 0.750 | 0.425 | 0.0018 | 0.018–0.026 | 7e-5 |
| frozen scale1 40it | 86201/02/03 | 0.750 | 0.425 | 0.0057 | 0.022–0.027 | 1.4e-4 |
| BC head 8it | 86201/02/03 | 0.700–0.775 | 0.425–0.450 | 非零（BC 先验） | 0.314 | 0.09 |
| BC head 40it | 86201 | 0.475 | 0.350 | 被推坏 | 0.248 | 0.16 |

terminal/capture 审计（40 dev seeds）显示：

- default：BC 与 scale1 都是 30/40 success、10/40 plane miss；
- strict125：BC 与 scale1 都是 17/40 success、23/40 plane miss；
- scale1 相对 BC 的成功终止帧 eta 距离只改善 `0.65–1.50 mm`；
- 1.25/1.5/2.0/2.5 m 的 capture-ever 计数在 40 个 seed 上全部不变；
- 成功 episode 的终止余量只有 `0.8–31 mm`，14/30 < 1 cm，24/30 < 2 cm；
- scale1 的 reward 改善仅约 `0.0045–0.0048` mean episode reward（千分之几），
  逐 seed 方向不一致。

网络结构历史对照（`doc/goals/goal87_network_history_parity_20261110.md`）已经完成：

- 当前 `PCActorCritic` 与 Goal80 主干同构；差异只有 path12 拆分、frozen BC
  分支和 reserved-slot 补零修复；
- D3 BC 与 path12/cmd3 兼容 checkpoint 的 `actor.2/actor.4`、`actor.0`
  的 z/DADE 列、state encoder 和 path 前 8 列逐元素一致；
- 没有发现新的接线、维度或 checkpoint 错配。

## 2. 证据

- `doc/goals/goal87_terminal_capture_audit_result_20261110.md`
- `doc/goals/goal87_ppo_update_diagnostic_20261110.md`
- `doc/goals/goal87_residual_credit_assignment_20261110.md`
- `artifacts/lowlevel/goal87_terminal_audit_20261110/{bc_default,scale1_default,bc_strict125,scale1_strict125}/summary.json`
- `artifacts/lowlevel/goal87_terminal_audit_20261110/*/trace.npz`
- `artifacts/lowlevel/goal86/ppo_diag/pa_bc_frozen_residual_seed86201/iteration_records.jsonl`
- `artifacts/lowlevel/goal86/ppo_diag/pa_bc_frozen_residual_upd25x40_seed86201/iteration_records.jsonl`
- `artifacts/lowlevel/goal86/ppo_diag/pa_bc_frozen_residual_scale1_upd25x40_seed86201/iteration_records.jsonl`
- `artifacts/lowlevel/goal86/ppo_diag/pa_bc_head/iteration_records.jsonl`
- `artifacts/lowlevel/goal86/ppo_diag/pa_bc_head_upd25x40/iteration_records.jsonl`
- `artifacts/lowlevel/goal87_o2_init_20261110/std*/training_manifest.json`
- `artifacts/lowlevel/goal87_o2_init_20261110/std*_eval_*_dev40/summary.json`
- `artifacts/lowlevel/goal87_o2_init_20261110/std*_scale2_eval_*_dev40/summary.json`
- `artifacts/lowlevel/goal87_o2_init_20261110/std3e-2_seed86201_scale{1.5,5,10}_eval_*_dev40/summary.json`
- `artifacts/lowlevel/goal87_o2_init_20261110/paired_analysis.json`

## 3. 已排除项

1. **不是 NoisyNet 主导**：Goal86/87 的 PC 配置没有 NoisyNet；训练随机性来自
   Gaussian policy `init_noise_std=0.1`。
2. **不是 residual scale 包线太小**：scale=1.0 仍只有 `0.57%` base L2；
   `actuator_residual_abs_mean` 的大数值是冻结 BC 相对 ALOS 的基座差。
3. **不是 reward/terminal 契约错位**：reward 与 terminal 都几乎不变；成功与
   plane-miss 在 40/40 seed 上不变。
4. **不是 p_nav/eta capture frame**：Goal85 trace 中终止距离逐行 p_nav == eta；
   old frame gap 已从终止链路移除。
5. **不是网络主干接错**：当前 z_t 主干与 Goal80 代码一致，仅增加 path8/endpoint4
   拆分、frozen BC 分支和 reserved-slot 修复；BC 兼容 checkpoint 的权重范数与
   初始动作等价。
6. **不是单纯的探索幅度**：noise0.3/0.5 没有带来 success 改善；当前
   Gaussian 噪声约是确定性残差 L2 的 49 倍，训练采样确实由噪声主导。
7. **不是动作权限绝对不足**：eval-only 2× 残差放大能把 strict125 从 17/40
   提到 21/40，5×/10× 才因执行器包络饱和崩掉；说明方向可用，训练出的物理
   幅度约小一半。

## 4. 可证伪 hypothesis

### H87-O1：actor 更新被 100 个 mini-batch 的累计步长/学习率压成毫米级

- 支持：在相同 1M transitions、相同 frozen BC、相同 seed 下，把
  `num_mini_batches` 从 100 降到 10/25，或把 actor lr 从 3e-4 降到 1e-4，
  末层范数 `>0.05`、确定性 residual/base L2 `>0.02`，且至少一个场景的
  terminal 桶计数发生变化。
- 否定：末层范数仍 `<0.03`、KL 仍 `<1e-4`，terminal 计数仍完全不变。
- 最便宜证伪：单 seed 1M，2–3 个更新调度 arm，dev40 两场景。

结果：`FALSIFIED`。`mb10/mb25` 与 `mb100` 的 success 都是 default `0.750`、
strict125 `0.425`；四个新 arm 的 success 翻转均为 `0/40`，scale1 的
residual/base L2 反而从 `0.57%` 降到 `0.25%–0.31%`。详见
`doc/goals/goal87_network_history_parity_20261110.md` 第 5 节。

### H87-O2：零初始化/极小输出让 old-log-prob 与 advantage 信用分配失真

- 支持：把残差输出层初始化为小的非零尺度（例如 `1e-3–1e-2`）或改为
  `initial_residual_zero=false + 输出层缩放`，在相同预算下 residual 增长且
  success 至少一个场景 `>=2.5 pp`，同时 KL 稳定在 `1e-3–1e-2`。
- 否定：非零初始化仍被压回毫米级，或 BC head 式的非零初始化继续造成
  KL/clip 爆炸和 success 下降。
- 最便宜证伪：单 seed 1M，两个初始化尺度，dev40 两场景。

当前状态：`MECHANISM_MOVE / OUTCOME_WEAK`。已新增显式
`--residual-output-init-std` 路径，manifest 记录真实初始 residual L2。结果：

| init std | final residual ratio | default success | strict125 success |
|---|---:|---:|---:|
| `1e-3` | 0.0086 | 30/40 | 17/40 |
| `1e-2` | 0.0206 | 30/30/31 | 17/17/17 |
| `3e-2` | 0.0637 | 31/31/30 | 17/17/17 |

非零初始化确实打破零解锁定，且呈剂量响应；但 default 只把边界 seed 33
翻成 success（`3e-2` 三个 seed 都是），strict125 翻转 `0/40`。因此机制层
假设支持，闭环效果不支持。

### H87-O4：训练出的残差物理幅度约为可用最优值的一半

- 支持：eval-only 把残差乘数从 1× 提到 2×，success 提升且执行器未饱和；
  5×/10× 才饱和崩坏。
- 否定：2× 也不动，或 1.5× 已饱和。
- 最便宜证伪：同一 checkpoint，dev40，default+strict125，1×/1.5×/2×/5×/10×。

结果：`POSITIVE_DIAGNOSTIC`。`3e-2 seed86201` 的 default success 为
`31/31/31/23/15`，strict125 为 `17/20/21/10/8`（1×/1.5×/2×/5×/10×）；
magnitude clip 为 `0.66/0.74/0.80/0.95/0.99`。2× 在 `1e-3/1e-2/3e-2`
共 7 个 checkpoint 上把 strict125 提到 `20–21/40`。这是 eval-only 反事实，
不是正式 Gate。

### H87-O3：BC-head 的 95% clip / 大 KL 是 warmup 后 stale old-log-prob

- 支持：在 actor warmup 期间也记录/重算 old log prob，或把 warmup 改为
  “只更新 critic、rollout 不跨 warmup”，BC-head 的 KL 从 `0.09` 降到
  `<0.01`，success 不再从 0.75 掉到 0.475。
- 否定：重算后 KL 仍大，说明问题在 optimizer/mini-batch 累计步长而非
  stale log prob。
- 最便宜证伪：单 seed 1M，BC-head 8it，记录 warmup 前后的 old/new log prob
  差异；不需要新 Gate。

当前状态：`NOT_SUPPORTED`。`rsl_rl` 每次 rollout 都重新写入
`transition.actions_log_prob = policy.get_actions_log_prob(actions)`；warmup
只是把 actor lr 置零。BC-head 的 KL 轨迹为 iteration 6 `0.0253`、7 `0.0124`、
8 `0.0926`，峰值不在第一个 post-warmup iteration，不符合 stale log-prob。

## 5. 下一步

`decision = continue`（继续 Goal87 的 optimization 诊断，不改主算法族）。

1. H87-O1 已完成并证伪，不再扩 mini-batch/LR 扫描；
2. H87-O3 已完成只读审计，不支持 stale old-log-prob；
3. H87-O2 已完成：非零初始化解除零解锁定，但闭环只动在 default 边界，
   strict125 不动；
4. H87-O4 已完成：2× 放大是正向诊断，5×/10× 是饱和负向诊断；说明策略方向
   可用、训练时物理幅度约小一半；
5. 下一步新开 `Goal88`：同 frozen BC、同 `z_t64`、同 PPO、同 1M/dev40/
   default+strict125，只把训练时 `pc.actuator_residual_scale` 提到 `1.5/2.0`，
   3 seeds；不改 reward、terminal、success、goal radius、seed 或 held-out；
6. 若 Goal88 仍把均值缩回一半，才按预注册开 SAC 同协议对照；不直接换
   REDQ/DroQ。
