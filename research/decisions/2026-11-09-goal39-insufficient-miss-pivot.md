# Goal39 decision：停止旧 vertical-miss 子群重裁，转全 seed 终端分量诊断

日期：2026-11-09<br>
Goal：Goal39<br>
诊断裁决：`INVALID_DIAGNOSTIC`<br>
Ledger decision：`pivot`

Goal38 的 provenance 修复使 Goal31 old traces 上 `25/25/31/32` 个 vertical-miss
seeds 变为 `9/7/19/16`。Goal34/35/36 的冻结最小样本数是每个 block 10，因此
default 两个 block 不足，整体不能解释机制。

这不是 trace 损坏：四个 trace 的 SHA、40 seeds、checkpoint、Goal38 provenance
和初始几何一致性都通过。问题在于旧 vertical-miss 子群被 stale path 0 放大，
修正后不足以支撑原来的三个机制问题。

不再降低 10-seed 门槛，也不沿用 Goal34/35/36 的旧机制结论。下一步只允许
预注册 Goal40，用全部 40 seeds 分解 terminal miss 的 horizontal、vertical、
longitudinal 和 `p_nav`-vs-`eta` discordance。Goal40 不得训练、改控制器或打开
held-out `100-149`。
