# Goal86 SAC 1M Screen 结果

**日期**：2026-11-10<br>
**状态**：`SCREEN_ONLY / T2_3SEED_POSITIVE_BUT_PATH_TRADEOFF / NOT_A_FORMAL_GATE / HELDOUT_CLOSED`<br>
**3-seed 结果**：`doc/goals/goal86_t2_3seed_result_20261111.md`<br>
**协议**：Goal85 `unbiased_noise_v1`、`position_source=eta`、零传感器延迟、path12、dev seeds `20–59`；held-out `100–149` 关闭。<br>
**架构**：SAC features `[z_t64, zero_dade64, command3] = 131`；没有退回 flatten+MLP。

## 1. 训练

两组均完成 `1,000,192` transitions，使用修复后的 `PCHistoryVecEnv` 向量化采样：

| arm | wall time | transitions/s | completed episodes | train success |
|---|---:|---:|---:|---:|
| `T1-SAC` | `2289 s` | `436.9` | `804` | `475/804 = 0.591` |
| `T2-SAC` | `2254 s` | `443.7` | `775` | `484/775 = 0.625` |

checkpoint：

```text
artifacts/lowlevel/goal86/screen/t1_sac/final.zip  sha256 335cecc0…
artifacts/lowlevel/goal86/screen/t2_sac/final.zip  sha256 10044cb0…
```

## 2. dev40 评价

zero-residual 基线：default `0.675`、strict125 `0.350`。所有比较都是同 seed 配对。

| arm | 场景 | success | 配对胜负 | path 3D RMSE | Δ path（95% CI） | residual L2 ratio | action abs max |
|---|---|---:|---:|---:|---:|---:|---:|
| `T1-SAC` | default | `0.500` | `1 胜 / 8 负` | `2.388 m` | `+0.061 ± 0.039` | `0.104` | `0.999` |
| `T1-SAC` | strict125 | `0.150` | `3 胜 / 11 负` | `2.387 m` | `+0.065 ± 0.041` | `0.108` | `1.000` |
| `T2-SAC` | default | `0.700` | `4 胜 / 3 负` | `2.341 m` | `+0.014 ± 0.072` | `0.256` | `0.997` |
| `T2-SAC` | strict125 | `0.425` | `4 胜 / 1 负` | `2.337 m` | `+0.015 ± 0.072` | `0.256` | `0.998` |

## 3. 解释

### 3.1 T1 command residual 明确有害

- default success 少 `17.5 pp`，strict125 少 `20 pp`；
- 配对胜负为 `1/8` 和 `3/11`；
- path 3D RMSE 的配对置信区间不跨 0，说明不是随机波动；
- 动作接近饱和（`action_abs_max ≈ 1.0`），残差把 ALOS command 推到了不利方向。

结论：`T1_COMMAND_RESIDUAL = HARMFUL_IN_SCREEN`。不应该因为 PPO 的 T1 近零就
把 T1 当成安全接口；SAC 一旦真的学出残差，command residual 会伤性能。

### 3.2 T2 actuator residual 是第一个正向信号

- default `+2.5 pp`（4 胜 3 负）；
- strict125 `+7.5 pp`（4 胜 1 负）；
- residual L2 ratio 约 `0.256`，说明策略确实在使用执行器残差；
- path RMSE 略差 `+0.014/0.015 m`，95% CI 跨 0，不能说明路径跟踪变好；
- success 的配对胜负仍不足以排除 seed 噪声。

结论：`T2_ACTUATOR_RESIDUAL = POSITIVE_BUT_NOT_SIGNIFICANT`。值得扩到 3 seeds，
但不能据此写 formal 结论。

### 3.3 不是 NoisyNet 主导

SAC 配置没有 NoisyNet。确定性策略的 `mu` 已经非零：

| arm | `mu` weight norm | `mu` bias norm | `log_std` bias mean | `exp(bias)` mean |
|---|---:|---:|---:|---:|
| `T1-SAC` | `0.749` | `0.068` | `-1.492` | `0.225` |
| `T2-SAC` | `0.648` | `0.167` | `-1.777` | `0.169` |

因此“残差大”不是 NoisyNet 注入的；训练时的随机性来自 SAC 自己的高斯策略。T1 的
问题是确定性残差方向错误；T2 的残差方向可能有帮助，但还不够稳定。

## 4. 当前裁决

```text
T1_COMMAND_RESIDUAL = HARMFUL_IN_SCREEN
T2_ACTUATOR_RESIDUAL = POSITIVE_BUT_NOT_SIGNIFICANT
NOISENET_DOMINANT = false
REWARD_SCALE_100X_CHANGED_SUCCESS = false
ALGORITHM_SWITCH = false
HELDOUT_OPENED = false
FORMAL_STARTED = false
```

## 5. 下一步

- [x] `T2-SAC` seeds `86202/86203` 1M screen；3/3 完成。
- [ ] `P-A SAC + BC head` 1M 诊断（运行中）；
- [x] 3-seed 配对；default 3/3 正、strict 2/3 正，但 success CI 跨 0，path pooled +0.277 m 显著退化，因此不进入 5M formal。
- [x] 判 `T2_3SEED_POSITIVE_BUT_PATH_TRADEOFF`；不换算法，后续转 C-line/terminal-path 审计。

## 6. 目标附近分布（过渡性统计）

`terminal_eta_distance_m` 的分位点和捕获率：

| arm / 场景 | median eta | ≤1.25 m | ≤1.5 m | ≤2.0 m | ≤2.5 m |
|---|---:|---:|---:|---:|---:|
| `T2-SAC` default | `1.991` | `0.000` | `0.000` | `0.700` | `0.800` |
| zero default | `1.994` | `0.000` | `0.000` | `0.675` | `0.850` |
| `T2-SAC` strict125 | `1.588` | `0.425` | `0.475` | `0.700` | `0.800` |
| zero strict125 | `1.694` | `0.350` | `0.475` | `0.675` | `0.850` |

解释：

- strict125 的增益不是单点运气：`≤1.25 m` 从 `0.350` 提到 `0.425`，median
  eta 从 `1.694` 提到 `1.588`；
- default 的 tight capture 略升（`0.675→0.700`），但 loose capture `≤2.5 m`
  反而从 `0.850` 降到 `0.800`，说明策略在目标附近更“贴紧”，但有少量 episodes
  被推远；
- `terminal_p_nav_distance_m` 与 `success_pnav_tight` 不一致，是已知的 p_nav
  估计偏差；正式 success 仍以 eta at done 为准，p_nav 只作诊断，不拿来替代
  success 定义。
