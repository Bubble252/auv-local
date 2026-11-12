# Goal55 blocker：固定 8.0 s pre-terminal 窗口在新 seed block 上发生 terminal leakage

**日期**：2026-11-09<br>
**Goal**：Goal55<br>
**触发**：server smoke/preflight 发现 seeds `60-99` 中有 episode 在 `8.0 s` 前进入
terminal phase；Goal55 冻结的固定窗口无法作为纯 pre-terminal 输入。<br>
**Ledger decision**：`stop`

## 1. 现象

Goal55 原计划在 seeds `60-99` 上原样复现 Goal54 的 `0 <= t < 8.0 s`、13 个
pre-terminal 特征和四个 block。正式 replay 前，先在 server 隔离目录运行
`default/baseline` 的 40-seed preflight。结果：

```text
seeds                    = 60-99 (40)
trace rows               = 40,545
window rows per episode  = 400
window terminal rows     = 1
seed 67 terminal entry   = 7.98 s
seed 77 terminal entry   = 8.36 s
Goal54 dev minimum       = 8.48 s
formal local replay      = NOT_RUN
heldout_opened           = false
```

seed 67 的 terminal phase 在 `t=7.98 s` 第一次为 true，正好落入
`[0, 8.0)`；episode 正常持续到 `16.08 s`，不是 trace 截断。seed 77 的 terminal
entry 为 `8.36 s`，虽然不在窗口内，但早于 Goal55 冻结的 `8.48 s` 安全边界。

## 2. 证据

- 输入 hash：checkpoint、evaluator、sensors、config、Goal54/55 analyzer 全部与
  Goal55 §7 一致；
- 传感器：`realistic_calibrated`，seed `81300`，40-seed suite reset 成功；
- preflight trace：40 seeds，40,545 rows；每 episode 400 个 `t<8.0` rows；
- terminal phase：在 episode 内单调保持 true；seed 67 在 `7.98 s` 进入，
  episode 结束于 `16.08 s`；seed 77 在 `8.36 s` 进入；
- 没有 NaN/Inf、没有 evaluator exception、没有 held-out seed、没有错误 checkpoint；
- server GPU idle，preflight 只使用 CPU/float64；
- 本地没有 Goal55 formal artifact，没有运行四 block。

## 3. 可证伪 hypotheses

### H1：`7.98 s` terminal entry 是 trace/evaluator 伪影

- 机制：如果 trace 的 `guidance_terminal_phase` 被错误写入、episode 被截断，或
  time axis 有错位，则应看到 terminal phase 非单调、episode 提前结束或 terminal
  timing 与 snapshot/summary 不一致。
- 预测：修 evaluator/trace 后，seed 67 的 terminal entry 会回到 `>=8.48 s`。
- 本轮结果：**不符合**。terminal phase 在 `7.98 s` 后持续为 true，episode 长度
  `805` steps/`16.1 s`，summary 正常，说明不是简单 trace 截断；没有观察到
  time axis 或 array alignment 错误。
- 最便宜证伪：同一 checkpoint/seed 重跑一次 default/baseline，比较 terminal
  entry 和 trace 字段；若可复现，则不是一次性伪影。

### H2：Goal54 的 `8.48 s` 不是新 seed block 的普遍下界

- 机制：terminal phase entry 由 episode 的路径/起点几何和终止规则共同决定；
  `8.48 s` 只是 dev seeds `20-59` 的观测最小值，不是环境保证。
- 预测：在独立 seed block 中应出现 terminal entry `<8.48 s`；其中一部分可能
  落入 `[0, 8.0)`。
- 本轮结果：**支持**。seeds 67/77 分别给出 `7.98 s` 和 `8.36 s`。
- 最便宜证伪：对全部 40 seeds 计算 terminal entry 分布；若所有值都 `>=8.48 s`，
  则 H2 不成立。当前分布已否定该条件。

### H3：cutoff-first / per-episode truncation 可以恢复有效 pre-terminal 窗口

- 机制：如果先确定每个 episode 的 terminal entry，再把 feature cutoff 截到
  `min(8.0, first_terminal_s - 0.5)`，则不会使用 terminal phase 或 post-cutoff
  row；这可能恢复 valid audit，但会改变窗口长度和“fixed-window replication”
  的含义。
- 预测：在新的、未观察的 seed block（例如 `150-189`）上，所有 episode 至少有
  一个有效 pre-cutoff window，且 cutoff 内 terminal rows 为 0。
- 本轮结果：**尚未检验**。这属于新的 pre-registered Goal，不应在 Goal55 内
  看到 preflight 后直接改窗口。
- 最便宜证伪：新 Goal 先只做 cutoff/window provenance smoke，不计算 outcome
  association；若仍有 seed 无法形成有效窗口，则停止该设计。

## 4. 已排除项

按 `数据 → 标签 → 评测 → baseline → 架构 → 文献`：

- **数据**：seeds `60-99` 与 held-out `100-149` 不重叠；trace seed set 精确，
  row count 正常；
- **标签**：preflight 未计算 feature-outcome association；summary flags 只用于
  检查 episode 完整性；
- **评测**：evaluator 正常完成，没有 timeout/NaN/崩溃；terminal phase 单调；
- **baseline**：只跑 default/baseline preflight，没有运行 strict125 或 candidate；
- **架构**：source/checkpoint/sensor hash 全部匹配，GPU idle，CPU/float64；
- **文献**：early-warning 的 cutoff-first truncation 和 lead-time 协议要求先冻结
  prediction cutoff；不能用 dev minimum 代替新 seed block 的有效性证明。

## 5. 决策

```text
decision = stop
```

停止 Goal55 的固定 8.0 s 复现。不得在本 Goal 内：

- 把窗口改成 7.0/7.5 s 或按 seed 排除；
- 删除/替换 seeds 67、77 或换 seed block 以得到 formal；
- 把 smoke outcome 当作 replication result；
- 训练、reward/gamma/seed sweep、恢复 command-residual；
- 打开 held-out `100-149`。

如果继续，必须先写新的 Goal56 预注册文档，使用未观察的独立 seed block
（例如 `150-189`），明确 cutoff 公式、lead time、窗口长度、false-alarm Gate
和 oracle/deployable 边界。不得在本 blocker 内直接执行新 outcome test。
