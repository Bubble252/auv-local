# Goal35 decision：继续初始状态/起始参考几何诊断

日期：2026-11-09<br>
Goal：Goal35<br>
机制裁决：`INITIAL_OFFSET_DOMINANT`<br>
Ledger decision：`continue`

## 依据

Goal35 只读 Goal31 的四组 40-seed formal traces，dev seeds `20-59`，held-out
`100-149` 未打开。四组均通过 seeds、checkpoint、初始/分段/handoff 窗口和 finite
检查。

- H-INITIAL `SUPPORTED`：初始绝对垂直误差中位数 `8.05-8.94 m`，初始距离中位数
  约 `34.2 m`；`84.0%-96.9%` 的 vertical-miss seeds 初始即在捕获半径外；初始
  与 terminal 误差的 Spearman 为 `0.5270-0.6215`，中位数比为
  `1.6752-1.7727`。
- H-TRACK `FALSIFIED`：reference pitch 方向 sign match 只有
  `0.4677-0.5132`，低于 `0.60`；tracking error 与 terminal miss 的 Spearman
  只有 `-0.0517-0.1023`。
- H-HANDOFF `FALSIFIED`：交接 pitch jump 明显（`0.1656-0.1797 rad`），但
  vertical worsening 中位数为 `0`，worsening-terminal Spearman 为
  `-0.1714-0.1876`。

## 决策

继续早期垂直机制，但把主线收窄到**初始状态/起始参考几何**。不改 pitch 内环增益，
不重开 authority/reward/terminal window 扫描。

## 下一步

- 预注册 Goal36 初始状态/起始参考几何只读诊断；
- 只使用 Goal31 现有 traces，dev seeds `20-59`，held-out `100-149` 继续冻结；
- 检查初始 `eta` 垂直位置、初始 goal/path 垂直几何、初始 pitch/velocity 和
  起点参考切线；
- 不训练、不新增 rollout、不扫 reward/terminal window/seed/lookahead。
