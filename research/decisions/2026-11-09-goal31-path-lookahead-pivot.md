# Goal31 decision：pivot away from path-lookahead terminal control

日期：2026-11-09<br>
Goal：Goal31<br>
Decision：`pivot`

## 依据

- 40-seed dev formal：`terminal_eta_capture` default `0.125 -> 0.125`，
  strict125 `0.075 -> 0.100`，没有达到预注册 `+0.05` 主效果；
- horizontal-normal terminal offset 只改善 `0.011/0.014 m`，CI 跨零；
- vertical terminal offset 变差 `0.106/0.268 m`，CI 不跨零；
- terminal saturation 超过 `+0.03` 护栏；
- path 3D/vertical RMSE 改善，但这不能替代 terminal capture；
- 控制链 40/40 seeds 有效，不是 `INVALID_CONTROL_CHAIN`；
- 默认关闭的 seed 41 replay 与 Goal30 baseline 的 45 个 trace 数组逐位一致。

## 停止事项

- 不再扫 path-lookahead 参数或终端窗口；
- 不重开 reward/gamma/lambda/seed；
- 不重开 command residual 或 supervised residual；
- 不打开 held-out `100–149`。

## 下一步

先做只读 H2/H3 诊断：

1. 按 seed 分解 `eta` vs `p_nav` 的 terminal horizontal/vertical 差值；
2. 检查 terminal stern/rudder、saturation 与 vertical offset delta 的关系；
3. 判断是显式 goal-plane error feedback 更合理，还是 estimator/lag 修正更合理；
4. 只有形成固定参数、可证伪的机制后，才预注册 Goal32。
