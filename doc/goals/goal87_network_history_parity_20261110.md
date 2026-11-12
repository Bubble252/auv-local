# Goal87 网络结构历史对照：主干无接线错误，H87-O2 解除零解锁定但幅度不足

**日期**：2026-11-10<br>
**状态**：`READ_ONLY_PARITY / NO_WIRING_BUG / H87_O2_MECHANISM_MOVE / MAGNITUDE_UNDERSIZED / NOT_A_GATE`<br>
**上游**：Goal9 V-0D R-PPO、Goal58/59 DDPG residual、Goal80 terminal reference、Goal86 PPO/SAC screen、Goal87 PPO update diagnostic<br>
**性质**：只读代码、checkpoint 和 iteration records 对照；不训练新算法、不改 reward/terminal/action 参数化、不打开 held-out。

## 1. 结论

当前 `z_t` 网络没有发现新的接线、维度或 checkpoint 错配：

- 当前 `PCActorCritic` 与 Goal80 的主干一致：state encoder `[128,128]`、`d_z_total=64`、`n_z_reserved=4`、actor/critic `[256,128]`、ELU、DADE 关闭；
- Goal87 新增的只是 path12 → path8 + endpoint4、frozen BC 分支，以及 reserved-slot 补零修复；
- D3 BC 兼容 checkpoint 的已有权重范数逐项一致，零残差时动作与 BC 等价；
- 当前 PPO 的 `actor.4` 输出层确实在更新，问题不是“梯度断开”，而是更新方向在不同 iteration 间相互抵消。

更重要的是，同样的失败在旧网络上已经出现过：Goal9 V-0D 的 `z_t + DADE` PPO 在 `41.9M` transitions 下只学到 `1.18%–1.83%` 的 residual/base L2，闭环 3D RMSE 只改善 `0.10%–0.60%`。因此当前现象不是 Goal86/87 新引入的网络 bug，而是 **action residual 这条训练路径的跨版本复现问题**。

但“不是接线错误”不等于“所有初始化解释都已排除”。当前完成度是：

| 项目 | 状态 | 说明 |
|---|---|---|
| 网络主干/checkpoint 对照 | `NO_WIRING_BUG` | Goal80、Goal9 和当前 `z_t` 主干同构；兼容 BC checkpoint 逐元素一致 |
| `H87-O1` 更新调度/学习率 | `FALSIFIED` | 减少 mini-batch 没有让残差长大，success 翻转 `0/40` |
| `H87-O2` frozen residual 小非零初始化 | `MECHANISM_MOVE / OUTCOME_WEAK` | 非零初始化真实生效并呈剂量响应，但闭环只翻转边界 seed；strict125 不翻转 |
| `H87-O3` warmup stale old-log-prob | `NOT_SUPPORTED` | KL 最大值出现在第二个 post-warmup iteration，不符合 stale log-prob 轨迹 |
| `H87-O4` 动作权限/eval-only 残差放大 | `POSITIVE_DIAGNOSTIC` | 2× 放大把 default/strict125 提升到 33/21；5×/10× 因包络饱和崩掉 |

因此当前最准确的判断是：**网络结构问题基本排除；非零初始化能打破零解锁定，但 PPO 学到的物理残差幅度只有可用最优值的一半左右。2× 放大是正向诊断，5× 以上是饱和负向诊断。**

## 2. 历史架构对照

| 版本 | 输入/特征 | actor 主干 | 探索/输出 | 残差接口 | 训练预算 | 残差幅度 | 闭环结果 |
|---|---|---|---|---:|---:|---:|---|
| Goal9 V-0D R-PPO | `z_t64 + DADE64`，path8，无 command | `[256,128]` ELU | Normal `std=0.1`，无 NoisyNet | frozen BC + action residual，scale `0.25/1.0` | `41.9M` | L2 ratio `1.18% / 1.83%` | 3D RMSE `-0.10% / -0.60%`，success 不变 |
| Goal58/59 DDPG | 26/30D history + GRU | MLP + GRU | NoisyNet，tanh/softsign | action residual | `5M` | gate-active L2 `10.5%–22.5%` | deterministic 全饱和，success/plane-miss 恶化 |
| Goal86 SAC T2 | `[z_t64, zero DADE64, command3]` | squashed Gaussian | SAC learned `log_std≈0.17–0.22` | actuator residual | `1M` | ratio `25.6%` | default `+2.5 pp`、strict `+7.5 pp`，配对 CI 仍跨 0 |
| Goal86/87 PPO | `[z_t64, command3, zero DADE64]` | `[256,128]` ELU | Normal `std=0.1`，无 NoisyNet | frozen BC + action residual，scale `0.25/1.0` | `1M` | L2 ratio `0.14% / 0.57%` | success、plane-miss、capture 计数全部不变 |

这张表说明：

- NoisyNet 不是当前问题；Goal86/87 没有 NoisyNet；
- “网络太小”也不是解释；旧 Goal9 和当前主干同量级，训练到 42M 仍只有不到 2% 残差；
- SAC 能学到非零均值，说明 `z_t` 特征和动作接口本身可用；
- 当前 PPO 的异常点是 **Gaussian mean 不积累**，而不是观测、路径编码器或 actor 宽度接错。

### 2.1 代码级 diff：Goal80 到当前实现

远端只读对照使用以下两份代码：

| 代码 | 远端路径 | SHA256 |
|---|---|---|
| Goal80 基线 | `goal80_terminal_reference_20261110/repo/src/auv_lowlevel/batch/pc_policy.py` | `6e868325...6f4a4` |
| 当前 Goal87 | `settled_局部规划器_反应式/src/auv_lowlevel/batch/pc_policy.py` | `c231b098...b79a` |

`diff -u` 显示的功能性差异只有以下几类，没有 actor/critic 主干重写：

1. `PCActorCritic.__init__` 新增 `path_encoder_zero_init`、`bc_path_dim`、`endpoint_encoder_*` 参数；actor 仍是 `256 → 128 → 3`，critic 仍是 `256 → 128 → 1`，激活仍是 ELU。
2. `_fuse_actor_path()` 把 path12 拆成 `path8`（BC 已见）和 `endpoint4`（新分支），两个分支都加到同一个 `z_t64` 上；`d_z_total=64`、`n_z_reserved=4` 未变。
3. frozen BC 分支新增 `frozen_bc_path_encoder` 和 `restore_pc_bc_frozen_current_encoder_alpha()`；并修复 reserved-slot 补零条件：只有 encoder 输出仍是 `d_z_active` 时才补 `n_z_reserved`，避免对已经 64 维的 frozen 输出再补零。
4. PPO 顶层输入仍是 `[z_t64, command3, DADE64] = 131`；Goal80 的 z_t/actor/critic 宽度、激活、`std` 参数和动作头语义没有变化。

结论：新增的是 **接口扩展和 frozen BC 分支**，不是新的网络主干。

## 3. checkpoint 与零初始化 parity

### 3.1 BC 兼容 checkpoint

| 项目 | 原始 D3 BC | path12/cmd3 兼容 checkpoint |
|---|---|---|
| 远端路径 | `artifacts/lowlevel/goal86/pa_residual_bc_scale100_20261110/pc_bc_actor.pt` | `artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt` |
| `path_dim` | `8` | `12`（前 8 列原样，后 4 列 endpoint） |
| `actor_command_dim` | `0` | `3`（新增列零填充） |
| current encoder | off | 零输出占位 |
| actor `0/2/4` 权重范数 | `17.1671 / 14.1374 / 0.318514` | 逐项一致 |
| SHA256 | `44bad72b...604fad` | `80024ef5...ff9dd` |

逐元素复核结果：

- `actor.2.weight`、`actor.4.weight`：`torch.equal=True`，`maxdiff=0.0`；
- `actor.0.weight` 的前 64 列（z_t）与最后 64 列（DADE）：`maxdiff=0.0`；
- 新增 3 列 command：`max_abs=0.0`；
- `actor_encoder` 三层：`torch.equal=True`；
- `path_encoder` 前 8 列：`maxdiff=0.0`；新增 endpoint 4 列：`max_abs=0.0`；
- 新增 `actor_current_encoder`：所有权重和偏置均为零，输出为零。

因此 frozen base 的初始动作与 D3 BC 等价，不是“加载了另一套网络”。兼容 checkpoint 的扩展是严格的前缀复制 + 新列零填充，不改变已有 BC 动作。

### 3.2 frozen branch 的零初始化

`pa_bc_frozen_residual.json` 中 `training.initial_residual_zero=false`，但
`load_pc_bc_frozen_actor_prior()` 会无条件把可训练 `actor` 的最后一层置零：

```text
residual_zeroed = _zero_last_linear(policy.actor)
```

实际 manifest 记录为 `initial_residual_zero_applied=true`、
`initial_residual_abs_max=0.0`。所以本组实验事实上仍是零初始化；配置文件里的
`false` 只是被 loader 覆盖，不能把它解释成“随机非零初始化已经测过”。这是一个
配置语义上的歧义，不是本次数值错误。

还需要区分 Goal87 diagnostic 里的 `random_init` arm：它是 `base_policy.mode=none`
的 **T2 直接 actuator 策略**，不是 frozen BC + residual。它的 `random_init` 结果不能
算作 H87-O2 的证据，也不能用来证明 frozen residual 的非零初始化无效。

## 4. 为什么说“参数在动但均值不积累”

以 `pa_bc_frozen_residual_scale1_upd25x40_seed86201` 为例，`actor.4.weight`：

| checkpoint | iteration | 权重范数 | 相对上一个 checkpoint 的增量范数 | 与上一次增量方向的 cosine |
|---|---:|---:|---:|---:|
| `model_0` | 0 | 0.000000 | 0.000000 | — |
| `model_10` | 10 | 0.008894 | 0.008894 | — |
| `model_20` | 20 | 0.007892 | 0.011396 | `-0.724` |
| `model_30` | 30 | 0.014626 | 0.016167 | `-0.274` |
| `model_39` | 39 | 0.022205 | 0.029028 | `-0.624` |

连续增量的 cosine 为负，表示新一步主要在“抵消”上一步，而不是沿同一方向累积。
这与 iteration records 中的 `actor.4` 每轮更新范数约 `0.01–0.03`、但权重范数长期
停在 `0.01–0.02` 完全一致。梯度并没有断开；当前 PPO 的 on-policy 梯度方差和
100 个 mini-batch 的顺序更新没有形成稳定均值。

## 5. H87-O1 的直接结果

同一 frozen BC、同一 1M transitions、同一 seed `86201`、dev40 两场景：

| arm | mini-batches | scale | default success | strict125 success | residual/base L2 | actor 末层范数 | 末次 KL | clip fraction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mb100_scale025` | 100 | 0.25 | 0.750 | 0.425 | 0.001355 | 0.0147 | 7.0e-5 | 0.087 |
| `mb10_scale025` | 10 | 0.25 | 0.750 | 0.425 | 0.001719 | 0.0150 | 1.9e-5 | 0.009 |
| `mb25_scale025` | 25 | 0.25 | 0.750 | 0.425 | 0.001128 | 0.0231 | 2.8e-5 | 0.017 |
| `mb100_scale1` | 100 | 1.0 | 0.750 | 0.425 | 0.005659 | 0.0222 | 1.4e-4 | 0.175 |
| `mb10_scale1` | 10 | 1.0 | 0.750 | 0.425 | 0.002468 | 0.0093 | 8.4e-5 | 0.110 |
| `mb25_scale1` | 25 | 1.0 | 0.750 | 0.425 | 0.003137 | 0.0097 | 7.5e-5 | 0.100 |

逐 seed 相对 BC 配对：四个新 arm 的 success 翻转数都是 `0/40`；default/strict125
的 3D RMSE 平均差都在 `1e-3 m` 以内。

结论：

```text
H87-O1 = FALSIFIED
减少 mini-batch 没有让残差长大，反而让 scale1 residual ratio 从 0.57% 降到 0.25%–0.31%；
success / plane-miss 仍然零翻转。
```

### 5.1 H87-O2：已解除零解锁定，但闭环只动在边界

实现上新增了显式路径，避免再次被 loader 的强制零初始化覆盖：

- `training.initial_residual_output_std` / `--residual-output-init-std`；
- loader 对 actor 最后一层按指定 std 正态初始化，bias 置零；
- manifest 记录实际 `initial_residual_output_std`、`initial_residual_l2_mean`、
  `initial_residual_to_base_l2`、`initial_residual_abs_max`；
- 单测和 1-iteration smoke 已确认非零初始化真实生效。

同 frozen BC、1M transitions、dev40、default + strict125；`1e-3` 只跑 seed86201，
`1e-2/3e-2` 跑 seed86201/86202/86203：

| init std | 训练 seeds | 初始 residual/base L2 | 最终 residual ratio | default success | strict125 success | KL med/max | clip med/max | actor.4 norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `1e-3` | 86201 | 0.000827 | 0.0086 | 30/40 | 17/40 | 5.6e-5 / 2.2e-4 | 0.055 / 0.247 | 0.0207→0.0171 |
| `1e-2` | 86201/2/3 | 0.008267 | 0.0206 | 30/30/31 | 17/17/17 | 7.3e-5 / 2.8e-4 | 0.093 / 0.364 | 0.207→0.066 |
| `3e-2` | 86201/2/3 | 0.024801 | 0.0637 | 31/31/30 | 17/17/17 | 1.5e-4 / 1.3e-3 | 0.202 / 0.683 | 0.622→0.407 |

逐 seed 配对相对 BC 零残差：

- `3e-2` default：三个训练 seed 都把同一个 dev seed 33 从 fail 翻成 success；
  seed86203 另外把 seed 49 从 success 翻成 fail，总计 `+3/-1`，净 `+2/120`；
- `3e-2` strict125：三个 seed 都是 `17/40`，翻转 `0/40`；
- `1e-2` default：`30/30/31`，只有 seed86203 的 seed 33 翻转；strict125 全 `17/40`；
- `1e-3`：success 模式与 BC 完全相同。

边界证据：BC 的 dev seed 33 在 default 的 `p_nav_done=2.087 m`，而 default goal
radius 约 `2.0 m`，只差约 `9 cm`；strict125 的失败点 `p_nav_done=1.63–2.09 m`，
radius 是 `1.25 m`，差 `0.4–0.8 m`。所以 6% 残差只能翻 default 的边界 seed，
对 strict125 的失败点没有足够权限。

判读：

```text
H87_O2 = MECHANISM_MOVE / OUTCOME_WEAK
```

非零初始化确实打破了零解锁定，且初始 std 与最终残差呈剂量响应；但没有出现
KL/clip 爆炸，也没有在 strict125 产生任何 success 翻转。因此“零初始化导致
完全学不动”只能算机制层支持，不能算闭环有效。

### 5.2 动作权限诊断：2× 是正向，5×/10× 是饱和负向

为区分“策略方向不对”和“策略方向对但幅度太小”，新增 eval-only
`--actuator-residual-scale-override`。它只改评估时残差乘数，不改训练、不改
checkpoint、不写回 config hash；覆盖值记录在 summary。

固定 `3e-2 seed86201`，default/strict125 各 40 seeds：

| eval 残差尺度 | default success | strict125 success | default gain/loss | strict gain/loss | magnitude clip |
|---:|---:|---:|---:|---:|---:|
| 1× | 31/40 | 17/40 | +1/0 | 0/0 | 0.659 |
| 1.5× | 31/40 | 20/40 | +2/1 | +3/0 | 0.741 |
| 2× | 33/40 | 21/40 | +3/0 | +5/1 | 0.802 |
| 5× | 23/40 | 10/40 | +4/11 | +3/10 | 0.950 |
| 10× | 15/40 | 8/40 | +2/17 | +5/14 | 0.993 |

2× 在其他训练 seed 和初始化尺度上同样成立：

| init std | 训练 seeds | default success @2× | strict125 success @2× |
|---|---|---:|---:|
| `1e-3` | 86201 | 33/40 | 20/40 |
| `1e-2` | 86201/2/3 | 33/33/33 | 20/20/21 |
| `3e-2` | 86201/2/3 | 33/31/32 | 21/21/21 |

`3e-2` 三个 seed 在 2× 下 strict125 都是 `21/40`，每个都是 `+5/-1`；default
合计 `+8/-2`。这说明策略学到的**方向**是可用的，但训练时的物理残差幅度约小了一半；
5× 以后 magnitude clip 到 0.95/0.99，包络饱和反而破坏闭环。

注意：这是 eval-only 反事实诊断，不是正式 Gate。单场景 per-seed McNemar 仍约
`p=0.22`，不能写成“2× 已证明有效”；它只足以支持新开一个训练时尺度的预注册 Goal。

## 6. H87-O3：不是 warmup 边界的 stale log-prob

`rsl_rl` 的 `PPO.act()` 在每次 rollout 时写入
`transition.actions_log_prob = policy.get_actions_log_prob(actions)`；critic warmup
只是把 actor 参数组的 `lr` 置零，并不会让 storage 里的 old log-prob 跨 warmup
残留。BC-head 的 KL 轨迹也不支持 stale 解释：

| iteration | 状态 | approx KL | clip fraction |
|---:|---|---:|---:|
| 6 | warmup 后第一个更新 iteration | 0.0253 | 0.902 |
| 7 | 第二个更新 iteration | 0.0124 | 0.871 |
| 8 | 第三个更新 iteration | 0.0926 | 0.949 |

如果问题来自“warmup 前记录的 old log-prob”，KL 应该在第一个 post-warmup
iteration 最大，之后下降；实际是第一个 `0.025`、第三个才到 `0.093`。因此大 KL
来自同一 rollout 内 100 个 mini-batch 的累计 drift，不是跨 warmup 的 stale
old-log-prob。H87-O3 的实现层假设 **不支持**。

## 7. 判读

```text
network_wiring_bug = false
checkpoint_mismatch = false
noisenet_dominant = false
gradient_disconnected = false
ppo_mean_accumulation = observed
H87_O1 = falsified
H87_O2 = mechanism_move_outcome_weak
H87_O3 = not_supported
H87_O4 = action_authority_2x_positive_5x_saturated
```

当前最合理的下一步不是直接换 SAC，而是先做最便宜的动作参数化修复：

1. 新开 `Goal88`：同 frozen BC、同 `z_t64`、同 PPO、同 1M/dev40/default+strict125，
   只把训练时 `pc.actuator_residual_scale` 从 `1.0` 提到 `1.5/2.0`，3 seeds；
2. 若 2× 训练后策略自己学到足够的物理幅度，strict125 稳定超过 BC，则不需要 SAC；
3. 若训练时放大后策略仍把均值缩回约一半，才说明瓶颈是 on-policy 探索/样本效率，
   此时按预注册开 SAC 同协议对照；仍不直接跳到 REDQ/DroQ。

`Goal87` 的 Gate、reward、terminal、seed、held-out 保持不变；上面 2× 结果只作为
`Goal88` 的立项依据，不写成 `Goal87` 的正式结论。

## 8. 证据

- `goal80_terminal_reference_20261110/repo/src/auv_lowlevel/batch/pc_policy.py`（SHA256 `6e868325...6f4a4`）
- `src/auv_lowlevel/batch/pc_policy.py:547-578`（path12 拆分与 encoder 配置）
- `src/auv_lowlevel/batch/pc_policy.py:983-1046`（`_fuse_actor_path` / frozen BC path 融合）
- `src/auv_lowlevel/batch/pc_bc.py:1880-1955`（checkpoint 前缀复制、current encoder 与强制零初始化）
- `artifacts/lowlevel/goal86/pa_residual_bc_scale100_20261110/pc_bc_actor.pt`（D3 BC，SHA256 `44bad72b...604fad`）
- `artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt`（兼容 checkpoint，SHA256 `80024ef5...ff9dd`）
- `artifacts/lowlevel/goal87_ppo_update_20261110/*_seed86201/training_manifest.json`
- `artifacts/lowlevel/goal87_ppo_update_20261110/*_seed86201/iteration_records.jsonl`
- `artifacts/lowlevel/goal87_ppo_update_20261110/*_eval_{default,strict125}_dev40/summary.json`
- `artifacts/lowlevel/goal87_o2_init_20261110/std*/training_manifest.json`（非零初始化与真实初始 residual L2）
- `artifacts/lowlevel/goal87_o2_init_20261110/std*_eval_*_dev40/summary.json`（1M 三档初始化闭环）
- `artifacts/lowlevel/goal87_o2_init_20261110/paired_analysis.json`（逐 seed 配对摘要）
- `artifacts/lowlevel/goal87_o2_init_20261110/std*_scale2_eval_*_dev40/summary.json`（2× 权限诊断）
- `artifacts/lowlevel/goal87_o2_init_20261110/std3e-2_seed86201_scale{1.5,5,10}_eval_*_dev40/summary.json`（剂量响应）
- `artifacts/lowlevel/goal86/ppo_diag/pa_bc_frozen_residual*`
- `scripts/evaluate_pc_ppo.py` 的 `--actuator-residual-scale-override`（eval-only，不改训练）
- `doc/goals/goal9_v0d_rppo_scale1p0_diagnostic_20261106.md`
- `doc/goals/goal58_result_20261109.md`
- `research/paper_reading/2026-11-08-residual-policy-learning.md`

复现代码 diff 和 checkpoint parity 的最短命令（在远端仓库根目录执行）：

```bash
diff -u \
  /data/yhx/underwater-robotics/src/plan_p1/goal80_terminal_reference_20261110/repo/src/auv_lowlevel/batch/pc_policy.py \
  src/auv_lowlevel/batch/pc_policy.py

/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python - <<'PY'
import torch
old = torch.load('artifacts/lowlevel/goal86/pa_residual_bc_scale100_20261110/pc_bc_actor.pt', map_location='cpu', weights_only=False)['state_dict']
new = torch.load('artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt', map_location='cpu', weights_only=False)['state_dict']
assert torch.equal(old['actor.2.weight'], new['actor.2.weight'])
assert torch.equal(old['actor.4.weight'], new['actor.4.weight'])
assert torch.equal(old['actor.0.weight'][:, :64], new['actor.0.weight'][:, :64])
assert torch.equal(old['actor.0.weight'][:, -64:], new['actor.0.weight'][:, -64:])
print('BC prefix parity: PASS')
PY
```

本文件只记录只读对照，不提交 `artifacts/`、checkpoint、日志或大文件。

