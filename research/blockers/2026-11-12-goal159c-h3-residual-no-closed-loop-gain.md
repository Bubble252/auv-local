# Goal159c blocker：residual 活跃但闭环 success 不移动

**日期**：2026-11-12<br>
**Goal**：Goal159c（Goal159 的修正训练协议）<br>
**状态**：`OPEN / GATE_FAIL / PIVOT_TO_TERMINAL_CAPTURE_ACTION_PARAMETERIZATION`<br>
**结果**：`doc/goals/goal159c_result_20261112.md`

## 1. 现象

3 seeds × 1M，C3/ALOS，H3 controller，current `[0.275, 0.55]`，near10_2 eval：

| level | H3 zero | policy | Δsuccess |
|---|---:|---:|---:|
| current07 | `99/120` | `100/120` | `+1` |
| current10 | `93/120` | `90/120` | `-3` |
| current15 | `51/120` | `49/120` | `-2` |

Gate 要求 current10/15 至少 `+4/120` 且 2/3 seeds 非负，未满足。

## 2. 已排除项

- **不是 direct-command 语义**：3 份 training manifest 都是
  `direct_commands_active=false`，`total_env_transitions_actual=1,000,192`，
  `num_envs=256`、`batch_size=256`、`UTD=0.125`；
- **不是 residual 为零**：`<=2 m` deterministic RMS `0.64–0.99`，`2–10 m`
  `0.48–0.59`，`>10 m` 为 0；
- **不是探索噪声主导**：actor `log_std` 约 `0.168–0.212`，评估为 deterministic；
- **不是单纯幅度过大**：scale `1.0/0.5/0.25` 下 success 基本不动；
- **不是单纯缺少 terminal residual**：mid_only 关闭 2 m 内 residual 后，
  current10 从 `-3` 恶化到 `-9`；
- **不是 residual 权限不足**：关闭 near10_2 gate、允许 full-authority residual 后，
  current10/15 分别降到 `75/120`、`48/120`（scale 1.0），scale 0.5 也只有
  `79/120`、`45/120`，远低于 H3 zero 的 `93/120`、`51/120`。远段 residual 明确有害。

## 3. 可证伪假设

### H1：2 m entry geometry 已经决定结果，residual 无法在 terminal 内修正

失败全部是 `goal_plane_miss`，final distance 集中在 `1.35–1.82 m`；2 m entry 的
横向误差中位数约 `1.80–1.94 m`，已经接近球壳边界。若 H1 成立，则把 success
阈值放宽到 `1.5/1.75/2.0 m` 时 policy 应出现稳定优势，或者 residual 应显著降低
2 m entry 的横向误差。

证伪结果：放宽阈值后 policy 仍不优于 zero（current07 `115/119/120` vs
`117/120/120`；current10 `105/117/119` vs `108/117/120`），因此 H1 不能解释为
“只差一点阈值”；terminal capture 仍是主瓶颈，但需要改 entry/terminal 控制，
不是放宽容差。

### H2：动作参数化/信用分配错误，导致 residual 饱和但不产生有效控制

若 H2 成立，降低 scale 应减少饱和但不一定救 success；通道限制应只有少数通道
有效，且跨 level 不稳定。实测正是如此：

- scale `1.0` action saturation 约 `0.56`，`0.25` 降到 `0`，success 仍不升；
- stern-only 在 current07 `+5`，但 current10 `-2`、current15 `0`；
- residual-error 相关跨 seed 变号，没有稳定的 rudder/stern/rpm 方向。

结论：H2 得到支持。下一步应测试 command-space residual 或受约束的
terminal residual，而不是继续放大 actuator residual。

### H3：样本效率/优化不足，需要 5M 或 REDQ/DroQ

若 H3 成立，1M 的 residual 方向应稳定正确，只是幅度/信用不足；当前没有满足这个
前提：通道相关跨 seed 变号，scale 与 channel 限制都没有稳定收益。Goal156 的 5M
full-residual 仍在跑，作为样本效率的独立检验；在它出结果前不启动 REDQ/DroQ，也
不把 H3 当作已证实的失败原因。

## 4. 最便宜证伪

已完成，均为 eval-only，不训练、不改 Gate：

1. `artifacts/lowlevel/goal159c_scale_diag_20261112`：scale `0.25/0.5/1.0`；
2. `artifacts/lowlevel/goal159c_channel_diag_20261112`：rudder/stern/rpm only；
3. `artifacts/lowlevel/goal159c_mid_only_diag_20261112`：关闭 2 m 内 residual。
4. `artifacts/lowlevel/goal159c_full_diag_20261112`：full-authority residual；
5. `artifacts/lowlevel/goal159c_full_scale05_diag_20261112`：full-authority + scale 0.5。

## 5. 判读

`GATE_FAIL / ACTION_CREDIT_AND_TERMINAL_CAPTURE / NOT_SAMPLE_EFFICIENCY_PROVEN`。

不换算法、不扩 seed、不扫 reward 权重。下一阶段预注册 Goal160：

- 固定 SAC、z_t、H3、Goal85 位置契约、reward/terminal/success/radius；
- 只改动作参数化：优先 command-space residual 或 terminal-only constrained residual；
- 仍用 current07/10/15 + nominal、3 seeds × 1M、dev40、H3 zero 配对；
- 失败后再进入 terminal reward/credit 审计，不直接上 REDQ/DroQ。
