# Decision：NoisyNet 与大残差的归因判别

日期：2026-11-10<br>
关联：Goal7 / Goal58 / Goal59 / Goal67 / Goal69 / Goal79<br>
性质：只读归因记录，不是新 Gate，不改任何冻结参数

## 1. 结论

当前“大残差”不支持由 NoisyNet 主导。

这里必须区分三个量：

1. 训练日志里的 `residual_abs_mean`：stochastic rollout，包含 NoisyNet 采样；
2. 评估时的 applied residual：deterministic mean weights 产生的动作偏移；
3. NoisyNet 噪声：训练时的探索扰动，正式评估通过 `evaluate=True` 关闭。

当前闭环失败对应的是第 2 项。确定性 mean 本身已经很大，关闭噪声后并没有恢复到
小残差；因此不能把“大残差”直接解释成“评估时还在采样噪声”。

## 2. 当前 Step 7 证据

Goal67 的 Step 7 已按 deterministic policy vs forced zero-residual 完成配对：

| scenario | policy lost true | zero-residual lost true |
|---|---:|---:|
| default | `5/160` | `4/40` |
| strict125 | `6/160` | `1/40` |

forced zero-residual 仍然丢失 default `4/40` 和 strict125 `1/40` 的 baseline true；
按比例是 default `10.0%` vs policy `3.1%`、strict125 `2.5%` vs policy `3.8%`；
strict125 的 zero-residual success 比例甚至高于 policy pooled 值（`35.0%` vs
`28.8%`）。因此 residual 既不是失败的充分原因，也不支持“residual 越大，闭环
必然越坏”的单调解读。

Goal79 的剩余 false 也主要不是来自噪声最大的 checkpoint：

| scenario | s240 | s752 | s880 | s1008 | zero |
|---|---:|---:|---:|---:|---:|
| default base / pred tight false | `16/7` | `14/9` | `13/8` | `14/7` | `12/7` |
| strict125 base / pred tight false | `6/2` | `5/3` | `6/5` | `6/2` | `10/8` |

s240/s752/s880 的 `noise/det` 约为 `0.0049/0.0083/0.00049`，s1008 约为
`1.056`；但 s1008 的 base false（default `14/40`、strict125 `6/40`）并不高于
低噪声 checkpoint。预测 correction 后，zero-residual 仍保留 default `7/12`、
strict125 `8/10` 的 false。低噪声和 zero-residual 都复现失败，说明剩余问题不是
NoisyNet 采样。

## 3. 以前确实遇到过 NoisyNet 主导

早期 Goal7 的失败是另一类现象：BC checkpoint 带入的隐藏层探索噪声
`sigma=0.5` 远大于 actor 输出均值，且 `rho` 在训练中继续漂移。表现为：

- deterministic eval 仍能成功，但采样噪声下的训练 rollout 失败；
- 把 hidden sigma 固定到 `0.01`、冻结 actor 噪声参数后恢复；
- 因此当时可以把主因定位为 NoisyNet 探索噪声幅度和可训练性。

当前候选不满足这组条件：Goal59 的 8/8 deterministic mean 都已饱和，hidden/output
sigma 仍在初始化附近的 `0.300/0.100`，`noise/det` 中位数约 `0.019`，只有
s1008 的噪声比例较高，但它的 mean 仍饱和。Goal59 的 screen 还显示，关闭噪声的
`N12` 仍有 `99.9%` any-dim 饱和。也就是说，当前不是“无噪声时会出现小残差”，
而是“没有噪声时 mean-weight 仍停在边界”。

## 4. 可复用的判别表

只有同时满足下面多数条件，才应把 NoisyNet 作为主因继续追：

| 判据 | NoisyNet 主导时 | 当前证据 |
|---|---|---|
| deterministic mean 是否远离边界 | 接近零或明显低于动作边界 | 8/8 deterministic saturation `1.0` |
| `softplus(rho)` 是否膨胀 | 明显高于初始化 `0.3/0.1` | 仍在初始化附近 |
| `noise>det` 状态比例 | 持续很高 | 中位数为 `0`，仅 s1008 约 `0.66` |
| 关闭噪声后残差是否回落 | 应明显回落 | zero-residual 仍保留大量 false |
| 失败是否集中在高噪声 seed | 应集中在高噪声 checkpoint | s240/s752/s880 仍贡献多数剩余 false |

当前只满足“s1008 的噪声比例较高”这一项，不足以支持 NoisyNet 主导。

## 5. 当前更可信的剩余解释

Goal64/67/69 的证据把问题指向 terminal approach 和 tracking，而不是噪声头：

- Goal64：strict125 的丢失 true 全部是 `GOAL_PLANE_MISS`；lost/retained 的
  last-100-step residual L2 中位数几乎相同（`0.351/0.352`）；
- Goal67：lost 的 command-goal cosine 约 `0.987`，但 actual-goal cosine 为负，
  载具以约 `2 m/s` 掠过捕获球后外离；
- Goal69：residual/base 在 lost 组不高于 retained 组，heading/pitch tracking lag
  是两个 scenario 的主要共同通道；
- Goal79：可观测 predictor 只能削掉部分 false，signed terminal bias 仍不可预测。
- Goal80：把 terminal reference 沿 goal tangent 前移后，policy true 明显下降
  （default `80->64`、strict125 `33->27`），plane miss 分别增加 `+0.131/+0.056`；
  但完全 zero-residual 的 arm 也出现 default true `17->16`、strict 持平，
  两个 scenario plane miss 都 `+0.075`。也就是说，没有 residual 仍会被
  plane-miss 终止规则打掉，失败方向与 residual 无关。

这些结果一致指向 terminal reference / termination 几何、command feasibility 和
pre-crossing tracking，而不是“噪声把动作推大”。

## 6. 决策与提醒

- 不通过关闭 NoisyNet、缩小噪声或重扫 residual scale 救当前 Gate；
- 不因为“残差大”重开 Goal27，也不启动 SAC/REDQ/DroQ；
- Goal80 已判定 `TERMINAL_REFERENCE_HURTS`：停止 goal-tangent 前移，不扫
  extension 值；下一步只读审计 plane-miss terminator 和 tight capture 语义；
- 只有新预注册 arm 明确出现 optimization/exploration 失败证据，才考虑 SAC
  同协议对照；只有明确 sample-efficiency 证据，才考虑 REDQ/DroQ；
- 若将来要做训练期 NoisyNet 对照，必须新建预注册 arm，固定总 transitions、UTD、
  seed、eval split 和 checkpoint 规则；不能用 `N12` screen 结果追认当前 Gate；
- 当前没有新增 GPU 训练任务；任何后续训练型 arm 先做
  `num_envs 128/256/512`、batch `256/512` 的单卡吞吐基准，再按 seed 分 8 张
  GPU。扩大采样/批量必须另开预注册 arm，不能用来救 Goal80 的失败结果。

## 7. 证据

- `doc/goals/goal7_bc_ddpg_stabilization.md`
- `doc/goals/goal58_result_20261109.md`
- `doc/goals/goal59_result_20261109.md`
- `doc/goals/goal59_screen_result_20261109.md`
- `doc/goals/goal67_result_20261110.md`
- `doc/goals/goal79_result_20261110.md`
- `doc/goals/goal80_result_20261110.md`
- `research/blockers/2026-11-10-goal69-multichannel-tracking-lag.md`
- `research/blockers/2026-11-10-goal79-observable-predictor-fail.md`
- `research/blockers/2026-11-10-goal80-terminal-reference-hurts.md`
- `artifacts/lowlevel/goal79_observable_bias_predictor_20261110/analysis/goal79.json`
- `artifacts/lowlevel/goal80_terminal_reference_20261110/analysis/goal80.json`
