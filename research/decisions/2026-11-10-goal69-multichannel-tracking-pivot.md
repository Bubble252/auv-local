# Decision：从 residual/NoisyNet 转向 command feasibility / phase-lead

日期：2026-11-10<br>
关联 Goal：Goal69<br>
决策：`pivot`（研究问题 pivot，不是算法切换）

## 依据

- Goal69 在 crossing 前 1 s 发现 heading 误差 `0.930/0.364 rad`、pitch 误差
  `0.618/0.596 rad`，而 speed 通道不支持；
- H69-B 在两个 scenario 都由 pitch 独立支持；H69-C 只在 default 支持；
- H69-A 的 1 s 证据强，但 2s/1s 持续性守卫未过，按预注册不能追认；
- command-goal cosine 仍在 `0.980/0.983`，actual-command cosine 只有
  `0.294/0.659`，说明 command 方向对，actual 跟踪不上；
- residual/base 在 lost 不高于 retained，zero-residual 仍丢 4/1；NoisyNet eval
  走 deterministic mean weights，因此 residual/NoisyNet 不是当前失败线的解释。

## 决策

停止把 residual/NoisyNet 作为主线解释；不重开 Goal27，不启动 SAC/REDQ/DroQ。
下一步只读诊断 command feasibility / phase-lead：command rate、actuator rate
limit、command shaping、terminal guidance，以及 default 的 p_nav 分支。

## 约束

- frozen checkpoint、dev `20–59`、CPU/float64、`endpoint_stop` 不变；
- held-out `100–149` 关闭；
- 不训练、不扫 reward/gamma/seed/controller；
- 扩大 `num_envs`/batch 必须另开预注册 arm，固定总 transitions/UTD/replay 比例、
  评估协议和 checkpoint 选择。
