# Goal32 decision：转向垂直权限与 terminal 时间预算诊断

日期：2026-11-09<br>
Goal：Goal32<br>
Decision：`pivot`

## 依据

Goal32 只读 Goal31 已冻结的四组 40-seed formal traces。所有 seed、checkpoint 和
finite 检查通过，held-out 未打开。

- H-ENTRY `FALSIFIED`：vertical-miss seeds 中 `84.0%-90.6%` 在入口已位于
  捕获半径之外，入口/终点绝对垂直误差 Spearman 为 `0.5565-0.6212`；但终端段内
  仍有 `1.34-2.00 m` 的中位误差闭合，超过预注册的半个捕获半径。入口状态有预测力，
  但不足以把终点结果视为入口已固定。
- H-AUTHORITY `SUPPORTED`：四个 block 的入口 required closure 中位数为
  `0.4284-0.5360 m/s`，episode 实际展示的 p95 closing speed 只有
  `0.1938-0.2146 m/s`；shortfall 中位数 `0.2631-0.3508 m/s`，
  `62.5%-76.0%` 的 seeds 要求大于展示权限，`100%` 的 seeds 实际闭合率低于要求。
  required closure 与 terminal 绝对垂直 miss 的 Spearman 为 `0.6443-0.6948`。
- H-GEOMETRY `FALSIFIED`：candidate 和 baseline 在终端入口的真实垂直状态几乎相同，
  所有 seed 的入口绝对垂直变化都在 `0.10 m` deadband 内；Goal31 candidate 的差异
  发生在入口之后，不是入口前已经形成的参考几何偏移。

## 裁决

把主线从“继续调 terminal reference / lookahead”转到 **只读的垂直权限与 terminal
时间预算诊断**。`authority_p95` 是闭环 episode 展示值，不是开环硬件上限，因此当前
结论是闭环权限不足，不是已证明物理不可达。

## 下一步

- 预注册 Goal33，测量速度相关的垂直权限包络、stern limit/rate、pitch 跟踪带宽、
  plane-crossing time 和 terminal kind；
- 在 Goal33 找到可区分的单一机制前，不训练控制器、不新增 rollout、不扫参数；
- 不重跑 Goal27 / Goal31，不打开 held-out `100-149`；
- 不把 `AUTHORITY_LIMITED_SUPPORTED` 写成控制器 PASS/FAIL。
