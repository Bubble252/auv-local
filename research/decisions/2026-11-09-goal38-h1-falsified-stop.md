# Goal38 decision：停止 terminal path-preserving 路线

日期：2026-11-09<br>
Goal：Goal38<br>
Gate 裁决：`FAIL / H1_FALSIFIED`<br>
Ledger decision：`stop`

## 依据

Goal38 修复了 `BatchedPCCommandResidualEnv` 的 stale path alias，并用 fresh
40-seed dev rollouts 重跑 Goal31 的四组 formal。四个 block 的 goal、goal
tangent、route tangent、path progress 和 start provenance 均为 `40/40`；
fresh trace 与原始 Goal31 同 seed/step trace 的 `actual_actuator`、`eta`、
`nu`、`requested_action` 最大差均为 `0.0`，`terminal_kind` 完全一致。

这证明 original Goal31 的控制链没有被 wrapper bug 改变，但用 provenance
一致数据按原 Gate 重算后，H1 仍然失败：

- strict125 terminal capture 点估计增加 `+0.100`，但 CI 下界为 `-0.0006`；
- 两 scenario 的 horizontal offset 中位数改善超过 `-0.05 m`，但 CI 都跨零；
- vertical offset 分别恶化 `+0.1773 m` / `+0.2198 m`；
- terminal material saturation 为 `+0.0396` / `+0.0347`，超过 `+0.03` 护栏；
- protocol success 和 plane miss 都只在允许边界上，没有净收益。

## 裁决

```text
H-EVAL-ALIAS = SUPPORTED
H-H1-RECHECK = FALSIFIED
decision     = stop
```

不再把 strict125 的 terminal capture 改善写成支持 H1 的机制证据，也不把
original Goal31 的 FAIL 归因于评测 path 伪差。

## 停止事项

- 停止 terminal path-preserving 路线；
- 不重扫 lookahead、terminal window、reward、gamma/lambda、seed、authority 或
  pitch gain；
- 不训练、不新增 checkpoint、不改变控制率或 policy；
- 不打开 held-out `100-149`；
- 不因 wrapper 修复自动重开 Goal27/Goal30 的已失败参数搜索。

## 下一步

只有在用户明确要求且提出新的可证伪假设时，才允许预注册新的 baseline、eval
split、Gate 和失败边界。Goal38 的 provenance 修复只改变历史 path-derived
数值的可引用性，不授权继续同一控制器路线。
