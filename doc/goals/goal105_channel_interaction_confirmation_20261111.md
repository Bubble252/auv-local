# Goal105：RPM 降权与 rudder 交互的 eval-only 确认

**日期**：2026-11-11<br>
**状态**：`COMPLETED_RPM_ONLY_SIGNAL / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal104 `CHANNEL_SIGNAL_WEAK / NO_SAC`<br>
**性质**：只读评估型 Gate；复用 Goal85 unbiased 环境、Goal95 P_D3、Goal104 的通道增益实现和评估链。不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal105 --goal-doc doc/goals/goal105_channel_interaction_confirmation_20261111.md --strict
```

记录前不得修改 profile、seed、scenario、success frame、path 阈值或 checkpoint SHA。

## 1. 背景与证据

Goal104 在 13 个单通道 profile 上发现：

- `R075` / `R125` 通过冻结 Gate，但都只新增 1 个 strict seed，方向相反、非单调；
- `U050` 的 strict 为 `23/40`（`+4/-1`），default 为 `32/40`；
- `U075` 的 strict 为 `22/40`，default 为 `32/40`；
- `U150` strict 为 `19/40`，说明 rpm 方向不是随机单调漂移；
- Goal104 的 Gate-qualified strict 增益是 goal-plane 1.25 m 边界翻转，不是全轨迹改善。

Goal85/85b/86 已排除：

- Goal85：零均值 p_nav、无 delay、eta termination 契约通过；
- Goal85b：±10% plant mismatch 只造成约 `3 cm` path 退化，不能解释 ANN 对传统方法的劣势；
- Goal86：PPO 残差近零；SAC T2 只有不显著正向信号；NoisyNet 不是当前主因。

因此本 Goal 只回答一个问题：

```text
rpm 降权提供的 strict 方向，能否通过 rudder 交互保住 default 33？
```

## 2. 假设

### H105-A：`U050+R075` 存在互补交互

- 机制：rpm 0.5 减少终端速度/越过目标的风险；rudder 0.75 减少大 rudder residual
  在 default 上的边界损失。两者组合应同时保住 default `>=33` 并把 strict 推到 `>=23`。
- 成立：至少一个交互 profile 通过全部 Gate，且 strict gained 集合覆盖 Goal104 的
  U050 gained `25/51/56/59` 中至少 2 个。
- 失败：交互 profile 的 default 仍 `<33`，或 strict 没有超过 U050 的 `23`。
- 最便宜证伪：本 Goal 的 10 个 eval run；不训练、不扩 seed。

### H105-B：rpm 降权单独就是唯一有效方向

- 机制：rudder 交互不提供额外收益；U050/U075 的 strict 增益已经代表可用边界。
- 成立：`U050` 在复跑中 strict `>=23`、default `>=32`，且交互 profile 没有更高
  strict 或没有更好 default。
- 失败：交互 profile 同时提高 default 和 strict，说明 rudder 交互有额外贡献。
- 最便宜证伪：同矩阵的配对 success 与 path ratio。

### H105-C：Goal104 的 U050 增益是 seed 边界噪声

- 机制：Goal104 的 strict `+4/-1` 在 40 个 seed 上不显著（exact McNemar `p=0.375`）；
  复跑若不能保持 gained 集合，则不应继续参数化微调。
- 成立：U050 复跑的 strict gained 集合与 Goal104 重叠少于 2 个 seed，或 strict `<22`。
- 失败：U050 复跑保持 `>=23` 且 gained 集合稳定，说明方向可复现。
- 最便宜证伪：同 checkpoint、同 seeds、同环境的复跑配对。

### H105-D：所有改动都只在 goal-plane 边界抖动

- 机制：success 是 1.25 m 硬壳事件；任何 profile 都只是把 final distance 在
  1.24–1.30 m 之间翻转，path/terminal 机制没有改善。
- 成立：所有 Gate-qualified episode 的 final eta distance 都在 `[1.24, 1.30] m`，
  且 path RMSE 没有方向性改善。
- 失败：至少一个 profile 的 success 增益伴随 median terminal distance 或
  pre-goal RMSE 的稳定改善。
- 最便宜证伪：复用 trace 计算 gained seeds 的 final eta distance 与 pre-goal RMSE。

## 3. 数据与隔离

- checkpoint：`artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt`；
- checkpoint SHA256：
  `2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616`；
- environment：Goal85 unbiased default / strict125 candidate；
- dev seeds：`20–59`，每 run 40 episodes；
- success frame：`both`；
- eval scale：固定 `2.00`；
- 训练 scale：固定 `1.50`，仅用于校验 checkpoint 契约；
- held-out `100–149`：关闭；
- 不训练、不新增训练数据、不改 reward/terminal/success/goal radius。

## 4. 动作参数化

复用 Goal104 的 eval-only 通道增益：

```text
delta = residual_normalized * 2.00 * full_scale * channel_gain
channel_gain = (rudder, stern, rpm)
```

### 4.1 Profile 矩阵

| ID | rudder | stern | rpm | 目的 |
|---|---:|---:|---:|---|
| `C0` | 1.00 | 1.00 | 1.00 | B20 复现与配对基线 |
| `U075` | 1.00 | 1.00 | 0.75 | rpm 降权确认 |
| `U050` | 1.00 | 1.00 | 0.50 | rpm 强降权确认 |
| `U050R075` | 0.75 | 1.00 | 0.50 | rpm + rudder 小幅降权 |
| `U050R125` | 1.25 | 1.00 | 0.50 | rpm + rudder 小幅升权 |

新跑矩阵：

```text
1 arm × 2 scenarios × 5 profiles = 10 eval runs
```

`C0` 必须复现 Goal104：

```text
default 33/40, strict125 20/40
```

## 5. Gate

### 5.1 `INVALID`

- `C0` 不能复现 default `33`、strict `20`；
- checkpoint SHA 不匹配；
- seed 集合不是 `20–59`；
- 缺 summary / trace；
- NaN/Inf；
- 读取 held-out。

### 5.2 `INTERACTION_SIGNAL`

存在任一非 C0 profile 同时满足：

```text
default_success >= 33
strict125_success >= 23
path_3d_rmse <= 1.05 * C0（两个 scenario 分别计算）
timeout == 0
safety == 0
```

### 5.3 `RPM_ONLY_SIGNAL`

没有 profile 达到 `INTERACTION_SIGNAL`，但 `U050` 或 `U075` 满足：

```text
default_success >= 32
strict125_success >= 23
path_3d_rmse <= 1.05 * C0
timeout == 0
safety == 0
```

这表示 rpm 方向存在，但不能写成正式 success 改善。

### 5.4 `NO_SIGNAL`

没有任何 profile 达到 `INTERACTION_SIGNAL` 或 `RPM_ONLY_SIGNAL`，或所有变化都小于
seed 边界翻转噪声。停止通道微调，进入新的 reward/terminal 或动作投影 Goal。

## 6. 执行计划

- [x] Step 0：Run Contract；
- [x] Step 1：复用 Goal104 runner/analyzer，新增 Goal105 profile 与 Gate；
- [x] Step 2：单测 + C0 1-seed smoke；
- [x] Step 3：运行 10 个 dev40 eval；
- [x] Step 4：按冻结 Gate 判断；
- [x] Step 5：补 result/decision、LEDGER、STATE，并运行 checker。

## 7. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不扩 seed，不打开 held-out；
- 不改 reward、terminal、success、goal radius；
- 不把单 seed 或单 profile 写成正式成功；
- 不在本 Goal 中引入 joint norm projection、新网络或 plant randomization。

## 8. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因、写至少两个可证伪 hypothesis、
查论文/代码、跑最便宜的 falsification、写 LEDGER decision；不自动 commit/push，
不打开 held-out，不改变主算法族。

## 9. Goal 提示词

```text
你现在执行 Goal105：RPM 降权与 rudder 交互的 eval-only 确认。

先读：
- doc/goals/goal105_channel_interaction_confirmation_20261111.md
- doc/goals/goal104_result_20261111.md
- doc/goals/goal85_result_20261110.md
- doc/goals/goal85b_result_20261110.md
- doc/goals/goal86_screen_status_20261110.md

硬约束：
1. 固定 P_D3、scale 2.0、Goal85 unbiased、dev 20-59；不训练、不打开 held-out。
2. 只跑 C0/U075/U050/U050R075/U050R125，不新增 profile。
3. Gate 为 default>=33、strict125>=23、path<=1.05*C0、timeout/safety=0。
4. 先 smoke，再正式；C0 不能复现 33/20 判 INVALID。
5. 失败后按 autoresearch 记录，不直接换 SAC/REDQ/DroQ 或改 reward。
```
