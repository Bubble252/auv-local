# Goal100 decision：BC-only scale 2.0 通过，PPO 增量有害

**日期**：2026-11-11<br>
**状态**：`BC_SCALE_ONLY_SUPPORTED / PPO_INCREMENT_HARMFUL / NO_SAC`

## 1. 触发

Goal99 判决 `SCHEDULE_NOT_SUPPORTED` 后，Goal100 预注册了四个 eval-only
profile：

- `B20`：frozen BC only、PPO 全关、residual scale 2.00；
- `A3/A2/A125`：保留 PPO，但在 `3/2/1.25 m` 内做 near-field attenuation。

## 2. 结果

`B20` 两臂 default `33/40`、strict `20/40`，timeout/safety 全 0，path 相对
`B15` 约 `1.013x`。Gate 判 `BC_SCALE_ONLY_SUPPORTED`。

`A3/A2/A125` 仍是 default `31/32`、strict `18/18`。它们的 attenuation
mean 为 `0.92–0.99`、zero fraction 只有 `0–4.8%`，实际几乎没有关掉 PPO，
因此行为接近 `P0_20`。

逐 seed：

- `B20` 相对 `B15` default gained `27/49`，lost 无；
- strict gained `24/44`、lost `29/33`，净 0；不是逐 seed 完全保留；
- `A` profiles 相对 `B20` strict lost `22/24`，与 Goal99 的 PPO 破坏 seed 一致。

## 3. 判读

1. 当前 PPO checkpoint 的增量没有必要的闭环贡献，且会破坏 strict capture。
2. 问题不是探索不足或样本效率不足；关闭 PPO 后结果更好，因此不开 SAC/REDQ/DroQ。
3. 窄带 near-field attenuation 太弱；不再扫 `3/2/1.25 m`。
4. `B20` 是新的强基线：`33/20`，但 strict seed 集合发生交换，formal 必须
   报 seed 集合，不能只报总数。
5. 与 Goal85 历史 zero-residual candidate 对照，`B20` 的 `0.825/0.500` 明显高于
   `0.655/0.265`；但这是不同 checkpoint 谱系，必须用 Goal101 做同 checkpoint、
   同配置、同 seeds 的正式 ALOS/zero-residual 配对后才能写成正式结论。

## 4. 决策

`decision = continue`

下一 Goal（Goal101）先做 eval-only ALOS/zero-residual 配对，确认 `B20` 相对
传统控制器的真实增益；然后再预注册“BC 先验 + 有界 PPO 修正”的训练 arm。

## 5. 证据

- `doc/goals/goal100_result_20261111.md`
- `artifacts/lowlevel/goal100_terminal_prior_20261111/phase0_summary.json`
- `artifacts/lowlevel/goal100_terminal_prior_20261111/phase0_summary.md`
- `artifacts/lowlevel/goal100_terminal_prior_20261111/run_manifest.json`
- `doc/goals/goal100_terminal_ppo_prior_preservation_20261111.md`
