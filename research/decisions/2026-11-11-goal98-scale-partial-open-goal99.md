# Decision：Goal98 判 SCALE_PARTIAL，开 Goal99 终端方向审计

日期：2026-11-11<br>
关联：Goal98<br>
决定：`pivot`

## 判据

- 1.50 基线复现 Goal95 的 30/19；
- 2.00 让 default 达到 D3 31/40、BC15 32/40；
- strict125 只有 18/40，低于 frozen 20；
- 所有 strict 失败均为 `goal_plane_miss`；
- strict lost seed 22/29 的 `2–1.25 m` closure 从约 1.0 降到 0.44–0.69；
- strict gained seed 44 的 closure 从 0.23–0.33 升到 2.05；
- default gained seed 27/49 的 `3–2 m` closure 明显上升。

## 结论

scale 不是全局单因素：2.00 在一个 seed 子集上放大有效闭合，在另一个子集上
放大无效/反向 terminal 行为。当前最具体的机制是 **terminal residual 方向或
有效动作是 seed-dependent 的**，而不是“样本效率不足”。

## 新路线

开 Goal99，只读复用 Goal98 的 24 份 trace：

1. 对 1.50 与 2.00 的 terminal window 计算 residual 方向、applied/effective
   ratio、magnitude/rate clip；
2. 对 success/failure 和 gained/lost seed 分别报告；
3. 区分 `DIRECTION_SUPPORTED`、`AUTHORITY_LIMITED`、`GEOMETRY_LIMITED`；
4. 只有 `DIRECTION_SUPPORTED` 才预注册 direction-aware gate；
5. 不训练、不扩 seed、不打开 held-out、不开 SAC/REDQ/DroQ。

## 为什么不换算法

没有出现“方向正确、effective 增量存在、但闭环仍不改善”的证据。相反，2.00
已经能在 default 上超过 frozen，说明 residual 通道本身有可用的信息；问题是
strict 的终端方向和捕获几何没有被正确利用。直接开 SAC 会重复同一个方向/
credit 问题。

## 不做什么

- 不把 2.00 default 的增益写成跨 scenario 的修复；
- 不继续盲扫 2.25/2.50；
- 不训练、不改 reward/terminal/success/goal radius；
- 不打开 held-out `100–149`。
