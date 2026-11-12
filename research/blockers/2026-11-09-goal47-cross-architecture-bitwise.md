# Goal47 blocker：跨 CPU 架构的 bitwise replay 不可达

**日期**：2026-11-09<br>
**Goal**：Goal47<br>
**触发**：8×4090 server 的 `default/baseline` instrumentation preflight 与 Goal42
reference 的 summary flags 相同、row count 相同，但 26 个旧 trace 字段不是 bitwise
equal；连续两次 server 检查（32 threads 与 1 thread）得到同一模式。按 Goal47 冻结
Gate，server 结果不能写成 `PASS`。

## 1. 现象

server 为 `zhulab3-AS-4125GS-TNRT`，8×RTX 4090，CPU 为 AMD EPYC 9654；本地为
Intel i9-14900HX。相同代码、checkpoint、config、40 seeds、CPU/float64：

| run | rows | seed/step | summary flags | old fields bitwise |
|---|---:|---|---|---|
| local 40-seed preflight | 40766 vs 40766 | equal | equal | PASS（0 mismatches） |
| server 32 threads | 40766 vs 40766 | equal | equal | FAIL（26 fields） |
| server 1 thread | 40766 vs 40766 | equal | equal | FAIL（26 fields） |
| local 1 thread | 40766 vs 40766 | equal | equal | PASS（0 mismatches） |

server 的最大绝对差为：

```text
eta                       7.11e-15 m
nu                        1.89e-15
actual_actuator           4.55e-13
p_nav                     1.42e-14 m
command                   8.22e-15
```

所有差异都在 float64 的机器精度量级；episode flags、终止类型和 row count 一致。

## 2. 已排除项

- **instrumentation 改动改变动力学**：本地 40-seed preflight 的全部旧字段与
  Goal42 reference bitwise 相同，且新字段 identity error 仅 `6.66e-16 m`；
- **线程数**：server 32 vs 1 thread 的 mismatch 集合相同；本地 1 thread 仍与
  reference bitwise 相同；
- **seed/batch 组成**：preflight 固定 40 seeds；曾用 1 seed 得到不同 sensor draw，
  因为 sampler seed offset 依赖 batch 组成，该 run 已排除；
- **checkpoint/config**：server checkpoint SHA256 与 frozen
  `0851496b15ffebbc0a19bafa5bf40c0493c0d9a3f485267b75fde5f811dde5d9` 一致，
  所有输入 config SHA256 一致；
- **summary 协议**：success、goal-capture、plane-miss 在 server 与 reference 全一致。

## 3. 可证伪 hypotheses

### H-INSTRUMENTATION-SIDE-EFFECT：新增字段改变了控制或 RNG

若成立：本地和 server 都应在 instrumentation 前后出现 row count、terminal flags 或
旧字段差异，而且差异应大于跨机器舍入噪声。

当前证据：**FALSIFIED**。本地 40-seed 全部旧字段 bitwise equal，server 的差异只出现
在浮点数值字段，flags/行数不变。

### H-CROSS-ARCH-ROUNDING：Intel 与 AMD/BLAS 的浮点运算顺序造成 sub-ULP 差异

若成立：本地同架构 bitwise 通过；server 无论 1 还是 32 threads 都在 `1e-15` 到
`1e-13` 量级偏离，且 summary flags 保持相同。

当前证据：**SUPPORTED**。这符合 IEEE-754 对非结合加法和不同指令实现的预期，见
`research/paper_reading/2026-11-09-goal47-cross-architecture-replay.md`。

## 4. 最便宜的证伪实验

已完成，全部在正式 Goal47 analyzer 读取结果之前：

1. 本地 40-seed preflight：bitwise PASS；
2. server 40-seed、32 threads：26 个旧字段最大差 `4.55e-13`；
3. server 40-seed、1 thread：同 26 个字段仍非 bitwise；
4. 本地 40-seed、1 thread：bitwise PASS。

因此不是 instrumentation bug，也不是线程数选择；是跨 CPU 架构的浮点实现差异。

## 5. 裁决与下一步

```text
decision = continue
```

保留 Goal47 预注册的 bitwise Gate，不在看到机制结果后放宽它。正式 analyzer 使用
与 Goal42 reference 同架构的本地 40-seed instrumentation replay，保证所有旧字段
bitwise equal；server replay 只作为“outcome-level corroboration”，明确标记为
`NOT_BITWISE_CROSS_ARCH`，不混入 formal trace、不用于改阈值或选择 seed。

这满足“不把 INVALID 写成 FAIL/PASS”，也保留服务器验证。没有训练、controller 修改、
参数扫描或 held-out 访问。

## 6. 证据

- `doc/goals/goal47_estimator_state_instrumentation_20261109.md`
- local preflight `/tmp/goal47_preflight_20261109/default/baseline/trace.npz`
- local 1-thread `/tmp/goal47_local_onethread_20261109/default/baseline/trace.npz`
- server 32-thread `work/artifacts/.../formal/default/baseline/trace.npz`
- server 1-thread `/tmp/goal47_server_onethread/default/baseline/trace.npz`
- `research/paper_reading/2026-11-09-goal47-cross-architecture-replay.md`
