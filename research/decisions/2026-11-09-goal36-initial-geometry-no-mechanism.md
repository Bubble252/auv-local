# Goal36 decision：停止初始垂直几何排序机制线

日期：2026-11-09<br>
Goal：Goal36<br>
机制裁决：`NO_SUPPORTED_MECHANISM`<br>
Ledger decision：`stop`

## 依据

Goal36 只读 Goal31 已冻结的四组 40-seed formal traces，dev seeds `20-59`；
checkpoint SHA256 固定为 `0851496...5d9`，held-out `100-149` 未打开。四个 block
均为 40/40 seeds，vertical-miss 数为 `25/25/31/32`，baseline/candidate 初始
几何最大差为 `0.0`，没有 `INVALID_DIAGNOSTIC`。

- H-STATE `FALSIFIED`：世界垂直差与初始/terminal 绝对误差的 Spearman 为
  `0.1008-0.1831` / `0.4500-0.5876`，初始/终端中位数比为
  `0.7048-0.7583`；虽然 `72.0%-80.7%` 的 seeds 已超出 capture radius，但世界
  垂直差不能单独排序初始或终端 miss。
- H-REFERENCE `FALSIFIED`：goal-tangent 参考投影中位数为
  `6.7750-6.8424 m`，份额为 `75.7%-85.0%`，数量级很大；但参考绝对量与初始/
  terminal 绝对误差的 Spearman 为 `-0.9556--0.9154` /
  `-0.5823--0.5054`，高参考组反而 terminal 更小，方向与预注册“参考几何制造并
  排序偏置”相反。
- H-START-DYNAMICS `FALSIFIED`：初始 pitch/垂向速度幅度足够，但控制参考几何后
  partial Spearman 为 `-0.2465--0.0749`；initial pitch rate 中位数只有
  `0.0026 rad/s`。没有起始动力学量通过独立排序判据。

## 停止事项

- 停止“世界垂直状态 / goal-tangent 参考投影 / 起始动力学”三选一的排序机制线；
- 不因参考份额大而改写 H-REFERENCE，不修改测量后的阈值或分组；
- 不修改 pitch gain、reward、terminal window、lookahead 或 seed；
- 不训练、不新增 rollout、不打开 held-out `100-149`；
- 不据此预注册控制器候选，也不把 `NO_SUPPORTED_MECHANISM` 写成控制器正式
  PASS/FAIL。

## 保留的描述性事实

参考投影在本样本中是一个近常数、较大的加性项（中位数约 `6.8 m`），而世界垂直差
变化更宽，两者常互相抵消。这个描述可以解释为什么参考份额很大但绝对量排序为负，
但它不是 Goal36 冻结判据下的支持机制。若未来要继续，必须先预注册新的
“常数加性偏置 vs 随 seed 变化分量”诊断，并明确说明重开理由；本 Goal 不授权该
后续实验自动启动。

## 下一步

当前没有获支持的初始垂直几何机制，也没有可预注册的控制器候选。保持主线停止；
只允许用户明确要求后，才新建独立 goal 重新预注册，不得把本次结果直接用于改控制器。
