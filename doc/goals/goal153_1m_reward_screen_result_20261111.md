# Goal153 1M reward screen 结果

**日期**：2026-11-11<br>
**状态**：`SCREEN_COMPLETED / R1_R2_NO_GAIN / ACTION_SATURATION_DOMINANT / NOT_FORMAL`<br>
**协议**：BC-init SAC，actuator residual，C1，direct-command off，dev seeds `20–59`，
default `2.0 m` / strict125 `1.25 m`，单训练 seed `86201`。

## 1. 训练内 C1 eval

| arm | default success | strict125 success | default plane miss | strict plane miss |
|---|---:|---:|---:|---:|
| `zero` | `26/40` | `14/40` | 14 | 26 |
| `R0` | `28/40` | `17/40` | 12 | 23 |
| `R1` | `23/40` | `12/40` | 17 | 28 |
| `R2` | `24/40` | `14/40` | 16 | 26 |

判读：

- `R0` 相对 zero 有小幅正向：default `+2/40`，strict `+3/40`；
- `R1` 相对 zero 为负：default `-3/40`，strict `-2/40`；
- `R2` default `-2/40`，strict 持平；
- 没有任何 reward arm 达到预注册的 `+4/40` 门槛。

## 2. Standing A_screen（1M，default radius）

| pressure | zero | R0 | R1 | R2 | 最佳 |
|---|---:|---:|---:|---:|---|
| nominal | `27/40` | `28/40` | `23/40` | `24/40` | R0 `+1` |
| current10 | `18/40` | `21/40` | `16/40` | `18/40` | R0 `+3` |
| noise_n2 | `27/40` | `28/40` | `23/40` | `24/40` | R0 `+1` |
| plant20 | `26/40` | `28/40` | `23/40` | `23/40` | R0 `+2` |

同一批 200k checkpoint 上，R1/R2 在部分压力档还略高于 R0；到 1M 反而回落，
而 R0 继续上升。这个非单调性提示 reward arm 不是“样本不够”，更像是训练后段
动作饱和/策略漂移，而不是奖励信号被正确利用。

### 2.1 Standing A_screen（1M，strict125）

| pressure | zero | R0 | R1 | R2 | 最佳 |
|---|---:|---:|---:|---:|---|
| nominal | `14/40` | `17/40` | `11/40` | `14/40` | R0 `+3` |
| current10 | `12/40` | `10/40` | `7/40` | `8/40` | zero |
| noise_n2 | `14/40` | `18/40` | `11/40` | `14/40` | R0 `+4` |
| plant20 | `13/40` | `16/40` | `10/40` | `13/40` | R0 `+3` |

strict125 下 R0 在 3/4 压力档为正，但 current10 仍比 zero 差 `2/40`；
R1/R2 没有一档超过 zero。R0 的正向主要来自更稳的 residual，而不是纵向 reward。

## 3. 关键异常：策略头饱和

| checkpoint | policy-head pre-scale saturation | residual L2 ratio | 判读 |
|---|---:|---:|---|
| 200k R0/R1/R2 | `0%` | `0.06–0.15` | 残差幅度正常 |
| 1M R0 | `85.9–88.2%` | `0.37–0.38` | 策略头长期顶满 |
| 1M R1 | `58.5–61.2%` | `0.28` | 顶满但 success 更差 |
| 1M R2 | `78.9–84.1%` | `0.32` | 顶满但 success 更差 |

策略头在 1M 大量输出 `|a| > 0.95`，说明问题首先是动作参数化/残差权限，而不是
reward 权重还不够大。继续扫 `F_z/F_zdot/F_predz` 权重会把“动作顶满”误当成奖励不足。

## 4. eval-only residual-scale 扫描

用同一个 1M `R0` checkpoint，只改 evaluator 的 `residual_scale`，不改训练：

| arm | nominal success | current10 success | policy-head sat | 备注 |
|---|---:|---:|---:|---|
| `R0 scale 0.25` | `31/40` | `18/40` | `85%` | nominal 最佳，比 zero `+4/40` |
| `R0 scale 0.50` | `29/40` | `22/40` | `85%` | current10 最佳，比 zero `+4/40` |
| `R0 scale 0.75` | `28/40` | `22/40` | `85%` | 折中 |
| `R0 scale 1.00` | `28/40` | `21/40` | `85%` | 原协议 |
| `R1 scale 0.50` | `26/40` | `16/40` | `62%` | 仍不如 R0 |

结论：

- nominal 与 current10 偏好不同的残差权限；全局固定 scale 不是最优；
- 但策略头本身仍然顶满，eval multiplier 只是把有效残差压小；
- 这正好支持 Goal154 的 authority schedule / 动作参数化审计，而不是换 PPO/SAC 或上 REDQ/DroQ。

## 5. 裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| R1/R2 未达 `+4/40`，且 1M 回退 | `REWARD_NO_GAIN` | 停止扫 reward 权重 |
| 1M policy-head 饱和 `59–88%` | `ACTION_SATURATION_DOMINANT` | 先查动作参数化/残差权限 |
| eval-only scale 出现 `+4/40` | `AUTHORITY_SCHEDULE_SIGNAL` | 进 Goal154 预注册 |

本结果不是 formal：只有 1 个训练 seed，40 个 dev episodes；不能写“R1/R2 永久无效”，
但足以阻止在当前协议下继续加 reward 权重或换算法。
