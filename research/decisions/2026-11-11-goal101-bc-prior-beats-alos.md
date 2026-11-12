# Goal101 decision：BC 先验在同 checkpoint 配对中超过 ALOS

**日期**：2026-11-11<br>
**状态**：`BC_PRIOR_BEATS_ALOS / EVAL_ONLY / NO_SAC`

## 1. 触发

Goal100 判 `BC_SCALE_ONLY_SUPPORTED / PPO_INCREMENT_HARMFUL`：`B20` 两臂
default `33/40`、strict `20/40`。但当时缺少同 checkpoint、同环境、同 seeds
的真正 ALOS/zero-residual 配对，不能把历史 Goal85 的 `0.655/0.265` 当正式
对照。

## 2. 结果

4 个新 ALOS run，dev seeds `20–59`，success frame `both`：

| group | ALOS | B20 | delta | path ratio |
|---|---:|---:|---:|---:|
| D3 default | 27 | 33 | +6 | 1.0189 |
| D3 strict125 | 14 | 20 | +6 | 1.0172 |
| BC15 default | 27 | 33 | +6 | 1.0189 |
| BC15 strict125 | 14 | 20 | +6 | 1.0172 |

timeout/safety 全 0；ALOS 两臂逐 seed 与 path 完全一致；B20 复现
`33/20`。Gate 判 `BC_PRIOR_BEATS_ALOS`。

## 3. 判读

1. frozen BC + scale 2.00 的闭环增益是真实的，不是谱系差异。
2. PPO 全开仍然有害：Goal100 的 `P0_20` 从 `33/20` 退到 `31/32`、`18/18`。
3. strict 绝对值仍低：B20 `20/40`，失败仍主要是 goal-plane miss。
4. `+6/40` 的逐场景 exact McNemar p 为 `0.070/0.109`；预注册计数 Gate 已
   通过，但不能写成 5% 统计显著。两臂轨迹相同，不能当 80 个独立样本。
5. 当前没有样本效率 blocker，不开 SAC/REDQ/DroQ。

## 4. 决策

`decision = continue`

下一 Goal（Goal102）预注册 **BC prior + bounded PPO correction**：

- 从 Goal95 checkpoint 出发；
- PPO 从零/近零增量开始；
- 加 action norm、KL-to-BC 或 terminal trust region；
- 先在 dev screen 证明不破坏 B20 的 `33/20`；
- 不改 success/terminal/goal radius，不打开 held-out；
- 只有出现明确样本效率 blocker 才考虑 SAC。

## 5. 证据

- `doc/goals/goal101_alos_same_checkpoint_pairing_20261111.md`
- `doc/goals/goal101_result_20261111.md`
- `artifacts/lowlevel/goal101_alos_pairing_20261111/phase0_summary.json`
- `artifacts/lowlevel/goal101_alos_pairing_20261111/phase0_summary.md`
- `artifacts/lowlevel/goal101_alos_pairing_20261111/run_manifest.json`
- `doc/goals/goal100_result_20261111.md`
