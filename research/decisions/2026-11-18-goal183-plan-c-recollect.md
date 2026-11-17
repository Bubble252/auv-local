# Goal183：从停止改为方案 C 重采 demo

**日期**：2026-11-18
**决策**：`pivot`
**上游结论**：`BC_RETRAIN_DRIFT`（旧 PE2E demo 的 current 分布与 `unbiased_small_noise_v2` 不匹配）

## 现象与证据

- 旧 empirical BC 在 C3 seeds `20–23`、`endpoint_stop`、`path_dim=5` 下 3D `3.1209 m`，
  fixed 为 `1.9841 m`。
- 旧 demo 的 current 特征在 C3 teacher rollout 中有 `25.33%` 的 `abs(z)>3`，state 为
  `0.12%`，path 为 `0%`。
- 旧 demo 使用 `realistic_calibrated`，BC_ONLY/PPO 环境使用 `unbiased_small_noise_v2`。
- 旧 demo 中 seed `20` 的一个 episode 进入了 fixed BC 的 training split；新数据必须完全排除
  `20–23`。

## 本次 pivot

用户选择方案 C：不停止 empirical 线，也不采用 fixed-current 混合方案 B；重采与 C3
`unbiased_small_noise_v2` 匹配的 PE2E demo，再用原 BC 配方重训并做同一 BC_ONLY 对照。

## 边界

- 不改 normalization 位置；
- 不改 BC 网络、LR、batch、epochs、episode split 规则；
- 不改 PPO 超参数或 Gate；
- held-out `100–149` 保持关闭；
- C 仍失败则停止，不扫 std/LR/seed/route。

## 结果

- 新 demo：209 episodes / 262144 samples，seeds `90000–90063`。
- C 的 BC_ONLY：3D `2.4459 m`、horizontal `2.4265 m`、vertical `0.3075 m`；同环境 teacher
  为 `2.4133 / 2.3955 / 0.2928 m`。
- current `abs(z)>3` 从 `25.33%` 降到 `0.00%`。
- 判为 `SCREEN_PASS_WITH_REFERENCE_GAP`：允许 1-iteration PPO smoke，不写成 formal PASS。

## 下一步

上传 C checkpoint，更新 warmstart SHA，在干净服务器 worktree 跑 1-iteration 1536-env smoke；
通过后才考虑 72-iteration。
