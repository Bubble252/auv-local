# Goal155 decision：保留 full-authority 训练，gate 只在部署时打开

**日期**：2026-11-12<br>
**Goal**：Goal155<br>
**决策**：`continue`，进入 Goal156；不换算法，不扩 seed。

## 证据

- 正式 verdict：`CAPTURE_TRADEOFF`；
- `near10_2` 把 path 收回到 `+0.0003 m`，但 default success `80/81`；
- `mid_only`/`far_off_only` success 更差（`75/81`）；
- `full` 仍复现 `86/81` 和 `+0.2769 m` path；
- gated arms 的 shell residual 未塌缩，vertical guard 通过。

## 解释

训练时 gate 让 actor 从第一步就看不到远段动作，抑制了探索和 terminal credit；
Goal154 的 eval-only gate 是在 full-trained checkpoint 上做的，保留了 shell 内的大
residual，因此得到 `+10/120`。这两个设置不能混为一谈。

## 决策

1. 不把 Goal155 的 train-time gated arms 推进到 5M；
2. 按 Goal156 预注册，固定 SAC、`z_t`、Goal85 位置契约、reward、terminal、success、
   radius，做 3 seeds × 5M 的 full-residual 训练；
3. 部署/评估时只改变 authority gate，主候选为 `mid_only`，同时报
   `zero/full/near10_2/far_off_only`；
4. 不打开 held-out，不扩 seed，不用 dev 选 checkpoint；
5. 若 Goal156 仍失败，按 blocker H2/H3 转向 terminal capture、动作参数化或
   reward/critic credit，不把失败归因成“需要换 SAC”。

## 下一步

执行 Goal156 formal；完成后先看 default/strict success、path、vertical、residual bins、
terminal distribution 和多环境 A_screen，再决定是否进入 B/C/D tier。
