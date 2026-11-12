# Blocker：Goal95 监督残差先验安装成功，但闭环 success 没有提高

日期：2026-11-11<br>
Goal：Goal95 `SCREEN_FAILED`<br>
状态：`ACTIVE / SUPERVISED_PRIOR_INSTALLED / TERMINAL_CAPTURE_SUSPECT / NO_SAC`

## 1. 现象

在 Goal86 P-A frozen BC、Goal95 监督残差初始化、同协议 PPO、1M transitions、
dev40 default+strict125 下：

| arm | default | strict125 | residual/base | mag clip | rate clip | timeout |
|---|---:|---:|---:|---:|---:|---:|
| `P_D3` | 30/40 | 19/40 | 0.5479 | 0.7211 | 0.5351 | 0 |
| `P_BC15` | 30/40 | 19/40 | 0.5564 | 0.7212 | 0.5335 | 0 |

同 seed `S15_s86201` 原始 summary 是 default `31/40`、strict `20/40`；frozen BC
是 default `31/40`、strict `20/40`。两个 P-* arm 相对 S0 都至少丢 1 个 default
和 1 个 strict 成功。

监督先验本身不是零解：

- residual/base 约 `0.55`；
- path/endpoint/current encoder 在 final checkpoint 中全部非零；
- frozen BC drift `0.0`；
- timeout `0`、safety `0`、无 NaN/Inf。

因此 blocker 不是“初始化没装上”，而是“装上以后闭环不赚”。

## 2. 终端证据

trace 的位置距离漏斗：

| arm/scenario/policy | success | 10 m | 5 m | 3 m | 2 m | 1.25 m | near-miss +0.25/+0.50/+1.00 m |
|---|---:|---:|---:|---:|---:|---:|---|
| `P_D3` default frozen | 31 | 40 | 40 | 37 | 31 | — | 2 / 3 / 6 |
| `P_D3` default final | 30 | 40 | 40 | 36 | 30 | — | 2 / 3 / 6 |
| `P_D3` strict frozen | 20 | 40 | 40 | 37 | 31 | 20 | 3 / 11 / 13 |
| `P_D3` strict final | 19 | 40 | 40 | 36 | 30 | 19 | 2 / 5 / 13 |

关键损失：

- default：`3 m -> 2 m` 从 `36` 掉到 `30`；
- strict125：`2 m -> 1.25 m` 从 `30` 掉到 `19`；
- strict125 失败样本中 `13` 个落在 `+1.00 m` 壳层。

路径跟踪不是主因：path 3D RMSE 与 frozen 基本同量级，且 path-zero 没有使
success 或 pre-goal RMSE 恶化。

## 3. 已排除项

1. **不是 NoisyNet 主导**：当前 policy 没有 NoisyNet；PPO 训练噪声是 Gaussian
   `init_noise_std=0.1`，评价为 deterministic inference。Goal58/59 的 NoisyNet
   属于 DDPG 谱系，不是当前 PPO。
2. **不是网络接线/维度/checkpoint 错配**：`d_z_total=64`、path12、command3、
   actor `[256,128]`、state encoder `[128,128]`、ELU、DADE off；frozen BC 的
   22 个 entry 与 Goal86 checkpoint `max abs diff=0.0`。
3. **不是 path encoder 没训练**：`residual_path_encoder`、`residual_path_endpoint_encoder`、
   `residual_actor_current_encoder` 均全非零；训练和加载缺口已在 Goal95 修复。
4. **不是标签代数错误**：离线 target 按
   `r_PPO = ((D3 - ALOS) / full_scale - s_env * r_BC) / (s_env * s_ppo)`
   生成；P_BC15 的更高 loss 可由 `1/s_ppo` 放大解释。
5. **不是动作幅度完全为零**：residual/base 约 `0.55`，magnitude/rate clip 仍高，
   但闭环 success 没动。
6. **不是 seed 波动**：两个 arm、两种场景同向失败；screen 阈值未达，不扩 seed。

## 4. 可证伪 hypothesis

### H96-A：terminal window 内的 residual 方向/权限不对，导致最后几米无法闭合

- 机制：3 m 到 2 m、2 m 到 1.25 m 的失败来自末端 residual 指向错误、幅度不足、
  或 requested residual 被执行器截断，而不是全局 path tracking。
- 成立应看到：在 terminal window 内，success/failure 的
  `action_effect_score`（residual 通道对 radial closure 的经验效应）或
  `cos_prior`（与 BC 先验的对齐）有显著分离；失败样本的
  requested/effective/applied 生存率更低，或逐通道 clip 集中在转向/垂向。
- 失败应看到：terminal window 内方向与成功/失败不可分，且 effective/applied
  接近 1，success 仍不改善。
- 最便宜证伪：直接用 Goal95 已有 8 组 trace，按 `3 m/2 m/1.25 m` 截取末端窗口，
  计算 per-channel direction、requested/effective ratio、clip 与 outcome 的
  分离；不训练、不改环境。

### H96-B：reward/credit assignment 只在 done 行给终端信号，pre-terminal radial closure 没有被有效优化

- 机制：PPO 在最后几米才收到强 terminal reward，前面的 radial closure 被视为
  中性；1M 样本下 advantage 无法把“接近目标”归因到末端动作，导致 PPO 破坏
  BC 先验而不是改进它。
- 成立应看到：reward decomposition 中 terminal 项占比过高；pre-terminal
  radial closure 与 success 的相关性弱；success/failure 在最后 3 m 的 reward
  轨迹没有提前分离。
- 失败应看到：pre-terminal reward 已经单调奖励闭合，且 success/failure 在
  3 m 之前就分离；此时应继续查动作参数化。
- 最便宜证伪：只读 trace + reward 记录，做 reward 分项与 outcome 的 episode-level
  相关/分层，不新增训练。

### H96-C：动作映射在 terminal window 内仍有大量 magnitude/rate clip，残差请求没有变成有效执行

- 机制：总残差 `s_env=1.5` 下，末端 `r_BC+r_PPO` 接近包线；PPO 的末端增量被
  clamp/rate limit 吸收，闭环看到近似零增量。
- 成立应看到：terminal window 的 effective/applied 明显低于 1；failure 比
  success 更常触发 clip；逐通道 clip 主要在 stern/rudder。
- 失败应看到：terminal window 的 effective/applied 接近 1，且 clip 与 outcome
  无分离。
- 最便宜证伪：对已有 trace 直接计算 terminal-window `effective/applied` 和逐通道
  clip loss；不训练。

### H96-D：BC/D3 先验已接近当前契约下的局部最优，PPO 的随机探索和 proxy reward 只是在破坏它

- 机制：initial 与 frozen BC 的 success 已高于 PPO final；PPO 更新把先验推离
  终端捕获 basin，default 从 31 掉到 30。
- 成立应看到：initial/frozen 的 terminal funnel 优于 final；PPO 在 terminal
  window 的 residual 方向与 initial 明显分叉，且分叉与失败 episode 对齐。
- 失败应看到：final 在 terminal window 更接近目标方向，只是被 clip 截断。
- 最便宜证伪：比较 initial、final、frozen 的末端 residual 方向和漏斗；已有
  checkpoint 和 trace，不需要训练。

## 5. 最便宜的 falsification

先做 **只读 Goal96 audit**，不训练、不改 reward/terminal/success/goal radius：

1. 从 Goal95 的 `P_D3`/`P_BC15` initial、final、frozen trace 中截取
   `10–5 m`、`5–3 m`、`3–2 m`、`2–1.25 m` 四个窗口；
2. 计算每个窗口的：
   - `action_effect_score`（residual 通道对 radial closure 的经验效应）与
     `cos_prior`（与 BC 先验的对齐）；
   - requested/effective/applied residual L2 与逐通道比例；
   - magnitude/rate clip 与 survival；
   - success/failure 的方向均值、bootstrap CI；
3. 从 reward 记录中分解 terminal、tracking、progress、action penalty 项，检查
   done 行与非 done 行的 credit 分布；
4. 用现有 terminal funnel 判断 PPO 是否破坏 BC 先验。

判据：

```text
direction_separates_success_failure && effective_survival_high
  -> 才允许评估 SAC 同协议对照；
direction_wrong_or_clipped
  -> 先修动作参数化/末端权限，不换算法；
preterminal_reward_not_aligned
  -> 先修 reward/credit assignment，不换算法；
prior_better_than_final
  -> 先查 PPO 更新是否在破坏先验，不换算法。
```

## 6. 结论与 next

```text
decision = pivot
target   = terminal capture / reward-credit assignment audit
SAC      = not yet
```

不扩 seed、不扫 lr/epoch/gain/seed、不开 SAC/REDQ/DroQ。下一步按
`doc/goals/goal96_terminal_credit_audit_20261111.md` 的预注册执行；在该文档
记录前不改 reward、terminal、success、goal radius，不打开 held-out。

## 7. 证据

- `doc/goals/goal95_supervised_residual_pretrain_20261111.md`
- `doc/goals/goal95_result_20261111.md`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40/*/summary.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40_initial/*/summary.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/eval_dev40_pathzero/*/summary.json`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/traces_dev40/*/trace.npz`
- `artifacts/lowlevel/goal95_supervised_residual_20261111/traces_dev40/*/terminal_proximity.json`
- `doc/goals/goal89_network_parity_and_gate_scope_audit_20261111.md`
- `doc/goals/goal87_network_history_parity_20261110.md`
