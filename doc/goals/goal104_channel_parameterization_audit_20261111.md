# Goal104：B20 frozen-base 的分通道动作参数化审计

**日期**：2026-11-11<br>
**状态**：`COMPLETED_CHANNEL_SIGNAL_WEAK / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal103 `CONTROL_AUTHORITY_LIMITED / PARAMETERIZATION_NOT_CAUSALLY_ISOLATED`<br>
**性质**：只读/评估型 Gate；不训练、不改 reward/terminal/success/goal radius，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal104 --goal-doc doc/goals/goal104_channel_parameterization_audit_20261111.md --strict
```

记录前不得修改 profile、seed、scenario、success frame、path 阈值或 checkpoint SHA。

## 1. 背景

Goal103 证明：

- B20 strict 失败不是网络接线、checkpoint 错配或 NoisyNet；
- 两臂在 `frozen-base` 下完全相同；
- 14/20 失败有 authority clip 签名，但成功集 clip 更高，因此“贴边”不是
  outcome 的充分判别器；
- 13/20 失败为 vertical-dominant；
- B20 的 scale 2.0 把 BC 残差放大到标签包线之外（label 1.0、训练 1.5、评估 2.0）。

所以下一步不是训练 PPO，也不是加宽网络，而是把三个 actuator 通道的 effective
authority 分开，检验是否存在可用的方向/包线效应。

## 2. 数据与隔离

- checkpoint：`artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt`；
- 只使用 P_D3 作为 canonical arm：Goal103 已证明 B20 的 P_D3/P_BC15 为同一
  frozen-base 轨迹，重复跑两臂不会增加信息；
- environment：Goal85 unbiased default / strict125 candidate；
- dev seeds `20–59`，每 run 40 episodes；
- success frame `both`；
- held-out `100–149` 关闭；
- 无训练、无新增训练数据、无 reward/terminal/success 改动。

### 2.1 网络结构只读复核（正式测量前）

为排除把参数化问题误归因为网络接线，测量前额外做了不改变协议的 checkpoint
对照：

| 项目 | 结果 | 证据 |
|---|---|---|
| P_D3 checkpoint SHA | `2d57745a...c1a616` | `artifacts/.../P_D3/final.pt` |
| frozen BC checkpoint SHA | `80024ef5...ff9dd` | Goal86 `pc_bc_actor.pt` |
| frozen-BC 张量 | `22/22` `torch.equal` | 对应 `max_abs_diff=0.0` |
| actor / critic | `131→256→128→3` / `131→256→128→1` | P_D3 `model_state_dict` |
| state encoder | `20→128→128→60` | `d_z_total=64`、`n_z_reserved=4` |
| path / endpoint / current | `12→64→64` / `4→32→64` / `3→64→60` | 与 Goal95 配置一致 |
| NoisyNet | 无 `noise`/`sigma` 参数 | deterministic mean 评估 |
| 非有限张量 | `0` | 全 checkpoint 扫描 |

该对照与 Goal87/Goal94 的历史结论一致：当前不是新网络接线错误，也不是
NoisyNet 主导。它不能排除动作参数化或终端几何；这正是本 Goal 要审计的部分。

## 3. 动作参数化

统一固定标量 residual scale 为 `2.00`，只改通道增益：

```text
delta = residual_normalized * 2.00 * full_scale * channel_gain
```

`channel_gain` 依次对应 `(rudder, stern, rpm)`。

### 3.1 Profile 矩阵

| ID | rudder | stern | rpm | 目的 |
|---|---:|---:|---:|---|
| `C0` | 1.00 | 1.00 | 1.00 | B20 复现 |
| `R075` | 0.75 | 1.00 | 1.00 | rudder 小幅降权 |
| `R125` | 1.25 | 1.00 | 1.00 | rudder 小幅升权 |
| `R050` | 0.50 | 1.00 | 1.00 | rudder 强降权 |
| `R150` | 1.50 | 1.00 | 1.00 | rudder 强升权 |
| `S075` | 1.00 | 0.75 | 1.00 | stern 小幅降权 |
| `S125` | 1.00 | 1.25 | 1.00 | stern 小幅升权 |
| `S050` | 1.00 | 0.50 | 1.00 | stern 强降权 |
| `S150` | 1.00 | 1.50 | 1.00 | stern 强升权 |
| `U075` | 1.00 | 1.00 | 0.75 | rpm 小幅降权 |
| `U125` | 1.00 | 1.00 | 1.25 | rpm 小幅升权 |
| `U050` | 1.00 | 1.00 | 0.50 | rpm 强降权 |
| `U150` | 1.00 | 1.00 | 1.50 | rpm 强升权 |

新跑矩阵：

```text
1 arm × 2 scenarios × 13 profiles = 26 eval runs
```

`C0` 必须复现 Goal100 的 B20：

```text
default 33/40, strict125 20/40
```

## 4. 假设

### H104-A：stern 通道限制垂直终端捕获

如果成立：`S125` 或 `S150` 在 strict 增加至少 1 个 seed，default 不低于 33，
且 gained seeds 包含 Goal103 的 vertical-dominant 失败 seed。

如果失败：stern 增益不改变同一批 seed，说明当前垂直误差不是“stern authority 不够”。

### H104-B：rudder 通道限制水平/复合终端捕获

如果成立：`R125` 或 `R150` 翻转水平主导 seed（如 32/47/52/58），且不破坏
vertical-dominant seed。

如果失败：rudder 增益只改变整体路径或没有 success 变化。

### H104-C：rpm/速度通道限制终端闭合

如果成立：`U075` 或 `U125` 改变 final goal distance 和 terminal capture，
且同一 seed 集合发生定向翻转。

如果失败：rpm 增益主要改变 path/energy，不改变 goal-plane capture。

### H104-D：任何单通道都无信号

如果成立：13 个 profile 都没有达到 default `>=33`、strict `>=21`、path 不恶化
`>5%`。此时不能再扫通道；下一步进入 joint direction-preserving projection 或
reward/terminal audit。

### H104-E：评估链/checkpoint 无效

如果成立：C0 不能复现 B20，checkpoint SHA 或 seed 集合不匹配，或出现 NaN/Inf。
记为 `INVALID`，不改阈值。

## 5. Gate

### 5.1 `INVALID`

- `C0` 不能复现 default `33`、strict `20`；
- checkpoint SHA 不匹配；
- seed 集合不是 `20–59`；
- 缺 summary / trace；
- NaN/Inf；
- 读取 held-out。

### 5.2 `CHANNEL_SIGNAL`

存在某个 profile 同时满足：

```text
default_success >= 33
strict125_success >= 21
path_3d_rmse <= 1.05 * C0
timeout == 0
safety == 0
```

### 5.3 `SINGLE_CHANNEL_DIRECTION`

没有 profile 同时满足 default 与 strict，但至少一个通道的两个方向呈现可复现的
相反效果，且同一 seed 的翻转可解释。

### 5.4 `NO_CHANNEL_SIGNAL`

没有任何 profile 达到 `CHANNEL_SIGNAL`，且通道方向差异小于 seed 翻转噪声。
此时停止通道 scale 扫描。

## 6. 执行计划

- [x] Step 0：Run Contract；
- [x] Step 1：实现 eval-only channel gains，默认 `(1,1,1)` 保持行为不变；
- [x] Step 2：单测 + C0 1-seed smoke；
- [x] Step 3：运行 26 个 dev40 eval；
- [x] Step 4：按冻结 Gate 判断；
- [x] Step 5：补 result/decision、LEDGER、STATE，并运行 checker。

## 7. 不做什么

- 不训练 PPO/SAC/REDQ/DroQ；
- 不扩 seed，不打开 held-out；
- 不改 reward、terminal、success、goal radius；
- 不把单 seed 或单 profile 写成正式成功；
- 不在本 Goal 中加入 joint norm projection；若 `NO_CHANNEL_SIGNAL`，另开 Goal。

## 8. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因、写至少两个可证伪
hypothesis、查论文/代码、跑最便宜的 falsification、写 LEDGER decision；
不自动 commit/push，不打开 held-out，不改变主算法族。
