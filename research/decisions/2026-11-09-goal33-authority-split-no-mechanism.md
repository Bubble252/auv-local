# Goal33 decision：停止 authority-magnitude 机制线

日期：2026-11-09<br>
Goal：Goal33<br>
机制裁决：`NO_SUPPORTED_MECHANISM`<br>
Ledger decision：`pivot`

## 依据

Goal33 只读 Goal31 已冻结的四组 40-seed formal traces，dev seeds `20-59`；
checkpoint SHA256 固定为 `0851496...5d9`，held-out `100-149` 未打开。所有
block 均为 40/40 seeds，配置为 stern `20 deg`、fin rate `150 deg/s`、pitch gains
`2/1`，且没有 `INVALID_DIAGNOSTIC`。

- H-SPEED `FALSIFIED`：水平速度与 p95 vertical closing authority 的 Spearman
  为 `-0.1738--0.0358`；高速组减低速组的 authority 差为
  `-0.0163-+0.0465 m/s`，低于预注册的 `0.05 m/s` 门槛。
- H-STERN `FALSIFIED`：vertical-miss 窗口 stern binding 中位数为
  `0.5208-0.6204`，角度限幅很常见；但 shortfall 与 binding 的 Spearman 为
  `-0.4562--0.1211`，高 shortfall 组的 binding 反而低
  `0.2270-0.4216`，速率绑定率为 `0`。限幅是共享工作区，不是 shortfall 的排序
  变量。
- H-EXCESS `FALSIFIED`：入口 excess 与 required closure 的 Spearman 为
  `0.9746-0.9891`，但 plane time 与 required closure 的 Spearman 为
  `-0.4167--0.3370`，越过预注册 `>= -0.30` 的边界；不能把数量级相关改写成
  支持证据。

## 停止事项

- 停止速度包络、stern 限幅阈值和 authority-magnitude 机制线；
- 不因 stern 限幅常见就直接修改 pitch gain、reward、terminal window 或 reference；
- 不重跑 Goal27 / Goal31，不新增 rollout，不训练，不打开 held-out `100-149`；
- 不把本次 `NO_SUPPORTED_MECHANISM` 写成控制器正式 PASS/FAIL。

## 下一步

转向更上游的全程垂直/相位机制：预注册 Goal34，只读现有 Goal31 traces，分别
检验终端垂直误差是否在距 goal `40/30/20 m` 时已经建立（H-GLOBAL）、
sign-matched vertical velocity 的方向/换向滞后是否比幅值更关键（H-PHASE），
以及路径/参考垂直斜率是否造成持续偏差（H-SLOPE）。只有单一机制通过预注册判据
后，才允许提出控制器候选。
