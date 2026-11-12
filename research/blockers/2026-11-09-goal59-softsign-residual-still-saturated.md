# Blocker: Goal59 softsign + terminal gate 仍然饱和

日期：2026-11-09<br>
Goal：Goal59<br>
状态：`OPEN_PIVOT_TO_ACTION_REWARD_TERMINAL_AUDIT`

## 1. 触发条件

Goal59 预注册 Gate 在 8 个 formal seed × 5M transitions 下失败。不是依赖、
路径、配置 typo 或一次 smoke 失败，因此进入 Research Escalation。

## 2. 现象与证据

正式评估使用 dev `20–59`、default/strict125、CPU/float64、`endpoint_stop`，
每个 seed 的 `final.pt` 与 forced zero-residual 严格配对。

| 指标 | default | strict125 |
|---|---:|---:|
| mean Δterminal capture | `-0.04375` | `-0.01875` |
| mean Δsuccess | `-0.05000` | `-0.165625` |
| mean Δplane-miss | `+0.05000` | `+0.165625` |
| capture 95% CI | `[-0.071875, -0.018750]` | `[-0.087500, +0.050000]` |

8 个 seed 中 `0/8` 通过全部 Gate 检查。只有 `752/880` 在 strict125 达到
`+0.05` capture，但 default 不改善，success 下降 `0.075`，plane-miss 上升
`0.075`，并且 eval action saturation 约 `1.0`。

正式 NoisyNet/raw probe：

- 8/8 deterministic saturation `1.0`，8/8 raw saturation `1.0`；
- `noise/det` 中位数 `0.019`，只有 seed `1008` 为 `1.03`；
- `noise>det` 中位数 `0`，只有 seed `1008` 为 `0.66`；
- hidden/output sigma 仍在初始化附近 `0.300/0.100`；
- gate-active residual L2 为 `10.5%–22.5%`。

对 752/880 的 default `path-zero`：capture 和 plane-miss 都不变，RMSE ratio
分别 `0.99992/1.00000`，按协议记为 `INVALID_ATTRIBUTION`。

## 3. 已排除项

- 不是 zero-residual 基线错误：使用同一缓存基线和 40 个 dev seed；
- 不是 held-out 泄漏：`100–149` 未打开；
- 不是 timeout/safety 终止：全部通过；
- 不是非有限训练：8/8 日志 clean，所有 summary `finite=true`；
- 不是 seed 选择或 best-checkpoint 选择：只用预注册 `final.pt`；
- 不是 NoisyNet 评估噪声：deterministic mean 已经饱和；
- 不是“残差没起作用”：gate-active residual 非零且有读数；
- 不是简单 scale clip：scale clip 全为 0，rate clip 不是本节主因。

## 4. 可证伪 hypothesis

### H1：softsign/scale/gate 能维持 bounded residual 并改善 terminal capture

可证伪预测：8/8 probe 的 deterministic/raw saturation `<=0.50`，至少 6/8 seed
在同一场景 capture 改善 `>=0.05`，其余护栏不恶化。

结果：**FALSIFIED**。8/8 raw saturation 为 `1.0`，capture mean 为负，success/
plane 护栏失败。

### H2：12D endpoint context 足以把 residual 转成 terminal capture

可证伪预测：G12 相对 G8 改善，且主效果 seed 的 default path-zero 使 capture
下降 `>=0.05`、plane 上升 `>=0.05` 或 RMSE 恶化 `>=5%`。

结果：**NOT_SUPPORTED**。formal G12 没有建立主效果；752/880 的 path-zero
没有变化，不能把微小 strict 变化归因于 12D context。

### H3：NoisyNet 探索主导 residual

可证伪预测：deterministic 动作接近零或不饱和，而 noise 显著更大，
`noise/det` 和 `noise>det` 很高，sigma 膨胀。

结果：**REJECTED**。mean 已经 100% 饱和，sigma 在初始化附近；seed 1008 噪声
较高但 mean 仍饱和，不能解释整体失败。

### H4：优化/探索不足，换 SAC 就能解决

可证伪预测：residual 非零且不饱和、训练曲线离开零附近，但 capture 不动；
或增加更新后 residual 方向改善。

结果：**NOT_SUPPORTED**。residual 饱和且方向规则地破坏 success/plane，没有
“优化不足”的证据；不应启动 SAC 作为逃生方案。

### H5：动作参数化/终端 reward/终止口径仍未对齐

可证伪预测：如果对 residual 方向与 terminal error、reward 各分项、gate 时序和
终止平面做只读审计，会发现至少一个可操作的符号/尺度/时序错位；修正后应在
不增加算法的前提下改变 residual 方向或 terminal capture。

状态：**待下一 Goal 检验**。这是当前最便宜的剩余解释；不能在本 Goal 内改 reward
或 gate 后重报 Gate。

## 5. 排查顺序（下一 Goal）

按“动作参数化 → reward → terminal capture”顺序，不先换算法：

1. 检查 softsign raw 值为何在 5M 后冲到边界，scale、rate limit、gate 和
   actor output 初始化是否形成“持续最大修正”；
2. 检查 reward-v2 的 path/terminal/goal-plane 项是否与 true-state
   `terminal_eta_capture` 同向，是否把 path RMSE 改善误当成终端捕获；
3. 检查终止平面、`endpoint_stop`、terminal eta 和 protocol success 的口径；
4. 用只读 trace 比较 residual 方向与 terminal error 的方向，先不训练或只做
   1-seed 最便宜 falsification；
5. 只有明确证据显示优化/探索是主因，才预注册 SAC 同协议；只有样本效率是主因，
   才考虑 REDQ/DroQ。

## 6. 禁止

- 不重开 softsign/gated residual 的 seed/reward/gamma/scale rescue；
- 不关闭 NoisyNet 来救 Gate；
- 不打开 held-out `100–149`；
- 不把 `INVALID_ATTRIBUTION` 写成 PASS；
- 不在 next Goal 预注册前启动 SAC/REDQ/DroQ。
