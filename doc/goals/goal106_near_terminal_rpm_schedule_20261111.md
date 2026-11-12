# Goal106：近终端 rpm 距离门控的 eval-only 确认

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal105 `RPM_ONLY_SIGNAL / STATIC_INTERACTION_FALSIFIED`<br>
**性质**：只读评估型 Gate；固定 checkpoint、success 契约和 dev seeds。不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal106 --goal-doc doc/goals/goal106_near_terminal_rpm_schedule_20261111.md --strict
```

记录前不得修改 profile、距离窗口、seed、scenario、success frame、path 阈值或 checkpoint SHA。

## 1. 背景与证据

Goal104/105 已确认：

- `U050`（全局 rpm gain 0.50）在 strict125 为 `23/40`，gained `25/51/56/59` 跨 Goal104/105 完全复现；
- `U050` 同时降低 path RMSE 约 `2.6%`、pre-goal 3D RMSE 约 `4%`；
- 但全局 rpm 降权让 default 的 `26/33` 掉出成功，default 只有 `32/40`；
- 静态 rudder 交互不能同时恢复 default 和保住 strict：`U050R075` 为 `33/22`，`U050R125` 为 `32/23`；
- 网络 parity、checkpoint SHA、无 NoisyNet 均已复核，当前不是接线错误。

因此本 Goal 只检验一个机制：

```text
远处保持 rpm gain 1.0；进入近终端后平滑降低 rpm residual；
能否保住 default 33，同时保留 U050 的 strict 23。
```

## 2. 假设

### H106-A：rpm 收益主要发生在近终端，距离门控可避免 default 损失

- 机制：全局 rpm 降权改变远处路径与速度；只在近终端降权可保留 default 的 `26/33`，同时保留 terminal capture 收益。
- 成立：至少一个 profile 同时满足 `default>=33`、`strict125>=23`、path、timeout、safety Gate。
- 失败：没有任何门控 profile 同时保住 default 33 与 strict 23。
- 最便宜证伪：5 个 profile 的 dev40 配对评估。

### H106-B：rpm 的收益是全局量，不是近终端局部量

- 机制：如果 U050 的 strict 增益来自整段航程的平均速度/能量，而不是最后几米，缩小到近终端窗口会消失。
- 成立：所有非 C0 profile 的 strict 都低于 23，或 strict gained 与 Goal104 的 `25/51/56/59` 重叠显著减少。
- 失败：至少一个 profile 保持 `>=3` 个 Goal104 gained seeds，且 strict 达到 23。
- 最便宜证伪：逐 seed gained 集合与 `GOAL104_U050_GAINED` 的重叠。

### H106-C：终端捕获瓶颈不是 rpm，而是几何/奖励目标

- 机制：若所有距离窗口都无法把 strict 推过 23，说明限制来自终端几何、奖励或 termination capture，而不是 rpm 的局部 authority。
- 成立：`NO_SIGNAL`，且所有 profile 的 strict final-distance median 仍接近 1.25 m。
- 失败：至少一个 profile 让 strict `>=23` 或显著改善 terminal distance 分布。
- 最便宜证伪：复用 trace 的 final distance、pre-goal RMSE 与逐 seed 翻转。

## 3. 数据与隔离

- checkpoint：`artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt`；
- checkpoint SHA256：`2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616`；
- environment：Goal85 unbiased default / strict125 candidate；
- dev seeds：`20–59`，每 run 40 episodes；
- success frame：`both`；
- eval scale：固定 `2.00`；
- training scale：固定 `1.50`，只用于 checkpoint 契约校验；
- held-out `100–149`：关闭；
- 不训练 PPO/SAC/REDQ/DroQ，不新增训练数据。

## 4. 动作参数化

通道 residual 保持：

```text
delta = residual_normalized * 2.00 * full_scale * channel_gain(d)
channel_gain(d) = far_gain + smoothstep(d; near_m, far_m) * (near_gain - far_gain)
```

其中：

- `d <= near_m`：使用 `near_gain`；
- `d >= far_m`：使用 `far_gain`；
- 中间使用 smoothstep 插值；
- `far_gain` 固定为 `(1.00, 1.00, 1.00)`；
- 只降低 rpm 通道的 near gain，rudder/stern 不变。

### 4.1 冻结 profile 矩阵

| ID | rudder near/far | stern near/far | rpm near/far | near_m | far_m |
|---|---:|---:|---:|---:|---:|
| `C0` | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | disabled | disabled |
| `G075_n5_f10` | 1.00 / 1.00 | 1.00 / 1.00 | 0.75 / 1.00 | 5 | 10 |
| `G050_n5_f10` | 1.00 / 1.00 | 1.00 / 1.00 | 0.50 / 1.00 | 5 | 10 |
| `G050_n3_f8` | 1.00 / 1.00 | 1.00 / 1.00 | 0.50 / 1.00 | 3 | 8 |
| `G050_n2_f5` | 1.00 / 1.00 | 1.00 / 1.00 | 0.50 / 1.00 | 2 | 5 |

新跑矩阵：

```text
1 arm × 2 scenarios × 5 profiles = 10 dev40 eval runs
```

`C0` 必须精确复现：

```text
default 33/40, strict125 20/40, timeout 0, safety 0
```

## 5. Gate

### 5.1 `INVALID`

- C0 不能复现 `33/20`；
- checkpoint SHA 不匹配；
- seed 集合不是 `20–59`；
- 缺 summary / trace；
- NaN/Inf；
- 读取 held-out；
- profile 的 near/far gains 或距离窗口与冻结矩阵不一致。

### 5.2 `DISTANCE_GATED_RPM_SIGNAL`

存在任一非 C0 profile 同时满足：

```text
default_success >= 33
strict125_success >= 23
path_3d_rmse <= 1.05 * C0（两个 scenario 分别计算）
timeout == 0
safety == 0
```

### 5.3 `PARTIAL_DISTANCE_SIGNAL`

没有 profile 达到完整 Gate，但至少一个 profile 满足：

```text
path_3d_rmse <= 1.05 * C0
timeout == 0
safety == 0
且 (default_success >= 33 或 strict125_success >= 23)
```

该状态只允许用于选择下一步审计，不得写成正式 success 改善。

### 5.4 `NO_SIGNAL`

没有 profile 达到上述任一状态，或 strict gained 与 Goal104 `25/51/56/59` 的重叠
少于 2 个 seed。此时停止 rpm 距离门控扫值，转入 reward/terminal capture 审计。

## 6. 执行计划

- [x] Step 0：Run Contract 与冻结 profile；
- [x] Step 1：实现 per-channel 距离门控，schedule 关闭时保持 C0 行为不变；
- [x] Step 2：单测 + C0 1-seed smoke；C0 与 Goal105 完全一致才继续；
- [x] Step 3：远端运行 10 个 dev40 eval；
- [x] Step 4：按冻结 Gate 判断并做逐 seed 配对；
- [x] Step 5：补 result/decision、LEDGER、STATE，运行 checker。

## 7. 失败裁决

| 观察 | 判读 | 下一步 |
|---|---|---|
| C0 不复现 33/20 | 评估链无效 | 记 `INVALID`，先修复链路，不改 profile |
| 只有 default 恢复 | 近端门控改变了 default，但 strict 机制不足 | 记 `PARTIAL`，转 reward/terminal 审计 |
| 只有 strict 保持 | rpm 收益是局部的，但 default 损失未解决 | 记 `PARTIAL`，不做新一轮常数扫值 |
| 两者都失败 | rpm 不是当前终端瓶颈 | 记 `NO_SIGNAL`，停止 rpm 扫值 |
| 两者都通过 | 距离门控是可复现的 eval 信号 | 才允许另开 held-out 确认 Goal；不直接改主算法 |

## 8. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不扩 seed，不打开 held-out；
- 不改 reward、terminal、success、goal radius；
- 不扫静态 rudder/rpm 常数；
- 不因一次 gate 失败就切换算法族；
- 不把 Gate 通过写成训练成功。

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因、写至少两个可证伪 hypothesis、
查论文/代码、跑最便宜的 falsification、写 LEDGER decision；不自动 commit/push，
不打开 held-out，不改变主算法族。

## 10. Goal 提示词

```text
你现在执行 Goal106：近终端 rpm 距离门控的 eval-only 确认。

先读：
- doc/goals/goal106_near_terminal_rpm_schedule_20261111.md
- doc/goals/goal105_result_20261111.md
- doc/goals/goal104_result_20261111.md
- doc/goals/goal85_result_20261110.md
- doc/goals/goal86_screen_status_20261110.md

硬约束：
1. 固定 P_D3、scale 2.0、Goal85 unbiased、dev 20-59；不训练、不打开 held-out。
2. 只跑 C0/G075_n5_f10/G050_n5_f10/G050_n3_f8/G050_n2_f5，不新增 profile。
3. 远处 rpm gain 1.0，近端平滑降到 0.75/0.50；rudder/stern 不变。
4. Gate 为 default>=33、strict125>=23、path<=1.05*C0、timeout/safety=0。
5. C0 不能复现 33/20 判 INVALID；失败后停止 rpm 扫值，按 autoresearch 转 reward/terminal 审计。
```
