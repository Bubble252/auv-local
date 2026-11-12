# Goal34 decision：继续早期全程垂直跟踪机制

日期：2026-11-09<br>
Goal：Goal34<br>
机制裁决：`GLOBAL_CARRYOVER_SUPPORTED`<br>
Ledger decision：`continue`

## 依据

Goal34 只读 Goal31 的四组 40-seed formal traces，dev seeds `20-59`，held-out
`100-149` 未打开。四组均通过 40/40 seeds、checkpoint SHA256、`40/30/20 m`
分段、terminal phase 和 finite 检查。

- H-GLOBAL `SUPPORTED`：四个 block 的 `40 m -> terminal` 垂直误差 Spearman 为
  `0.5270-0.6215`，`20 m -> terminal` 为 `0.6631-0.7077`；`40 m` 绝对垂直误差
  中位数 `8.05-8.94 m`，`20 m` 为 `5.75-6.48 m`，terminal 仍为
  `4.63-5.34 m`。终端 miss 是早期全程垂直误差的延续。
- H-PHASE `FALSIFIED`：sign-matched fraction 中位数为 `0.2467-0.3892`，但有效
  reversal coverage 只有 `0.00-0.1613`，低于预注册 `>= 0.50`；换向延迟无法
  从当前 traces 支持。
- H-SLOPE `FALSIFIED`：斜率与 terminal miss 的描述性相关为
  `0.3533-0.3841`，但 `40 m` 相关只有 `0.1725-0.2059`，未达到预注册
  `>= 0.30`。

## 决策

继续全程垂直跟踪机制，但把问题从 terminal authority 收窄到**更早的全程垂直
跟踪/交接**。不做控制器参数修改，不重开 authority-magnitude 扫描。

## 下一步

- 预注册 Goal35 早期全程垂直跟踪/交接只读诊断；
- 只使用 Goal31 现有 traces，dev seeds `20-59`，held-out `100-149` 继续冻结；
- 在 `40/30/20 m` 之前检查参考 pitch、实际 pitch、垂直速度、误差符号和路径
  交接，不再测 stern authority；
- 不训练、不新增 rollout、不扫 reward/terminal window/seed/lookahead。
