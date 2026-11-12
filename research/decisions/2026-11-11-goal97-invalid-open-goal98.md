# Decision：Goal97 记 INVALID，开 Goal98 修正 scale grid

日期：2026-11-11<br>
关联：Goal97<br>
决定：`pivot`

## 判据

Goal97 预注册假定 `s=1.00` 是训练基线；实际 Goal95 配置的
`pc.actuator_residual_scale` 是 `1.5`。补跑 no-op 锚点后：

| arm/scenario | no override | override 1.50 |
|---|---:|---:|
| P_D3 default | 30/40 | 30/40 |
| P_D3 strict125 | 19/40 | 19/40 |
| P_BC15 default | 30/40 | 30/40 |
| P_BC15 strict125 | 19/40 | 19/40 |

因此 `1.5` 才是基线；Goal97 的 `0.25–1.25` 全部是缩幅，不能回答原假设。

## 不代表什么

- 不代表 PPO residual 净有害；
- 不代表放大 scale 有效；
- 不代表 NoisyNet 或网络接线有问题；
- 不代表可以开 SAC/REDQ/DroQ。

## 新路线

开 Goal98，修正后的预注册：

- 基线 `1.50`；
- grid `{0.75, 1.00, 1.25, 1.50, 1.75, 2.00}`；
- 先验证 1.50 复现 Goal95 final 的 30/19；
- 仍用 dev seeds 20–59、success-frame both、同一 checkpoint/config；
- Gate 使用 frozen 31/20、path 3D RMSE、timeout/safety、trace 有限性；
- 不打开 held-out，不改 reward/terminal/success/goal radius，不开 SAC。

## 代码防复发

Goal98 runner 必须在运行前读取 training config 的
`pc.actuator_residual_scale`，并断言它等于基线 `1.50`；否则直接报错，不产生
Gate 记录。analyzer 也必须拒绝缺少 `1.50` 的 grid。

## 不做什么

- 不把 Goal97 的无效 sweep 写成 `SCALE_NOT_SUPPORTED`；
- 不用 `s=1.25` 的改善替代 1.50 基线；
- 不在 Goal98 中扫 lr/gain/seed；
- 不打开 held-out `100–149`。
