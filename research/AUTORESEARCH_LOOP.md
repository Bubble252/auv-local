# 自主研究循环：指标驱动、预注册、可检查

**日期**：2026-11-10<br>
**适用范围**：`settled_局部规划器_反应式` 的 Goal85 及其后续 Goal<br>
**目的**：把“持续尝试直到出效果”变成可执行的 research loop，而不是无限重复训练或事后移动阈值。

## 1. 总原则

1. 一次 Gate 失败不等于项目结束；失败后先诊断，再决定 `continue` / `pivot` / `stop`。
2. Gate 记录前不改变主指标、成功半径、seed、eval split、held-out 或 checkpoint 选择规则。
3. 不因为结果不好而换 seed、扩预算、放宽阈值或把 `INVALID` 写成 `FAIL`。
4. 新方法必须先写新 Goal，再实现；不能在同一 Gate 内静默换算法、reward、动作接口或评价口径。
5. 每次迭代必须留下 blocker、至少两个可证伪 hypothesis、reading note、最便宜证伪实验和
   `research/LEDGER.jsonl` decision。
6. 不自动 commit / push；只有用户明确要求时才提交。

## 2. “出效果”的成功标准

正式成功必须在预注册协议下同时满足：

| 项目 | 阈值 |
|---|---|
| 主指标 | 至少一个场景 `eta_tight_at_done` 相对 paired zero-residual 提升 `>=5 pp` |
| 另一场景 | 回退不超过 `2 pp` |
| safety / timeout | 恶化不超过 `2 pp` |
| plane-miss | 系统性恶化不超过 `5 pp` |
| path RMSE | 不恶化超过 `5%` |
| 机制 Gate | zero-residual parity、无 NaN/Inf、无真值观测泄漏、residual 非饱和 |
| 统计 | formal 至少 3 个训练 seed；最终确认至少 5 个训练 seed，方向保持 |
| 判分 | 使用真值 `eta`；`p_nav` 只作部署诊断，不得泄漏进 reward/termination/success |

近目标次级指标不能单独把 FAIL 改成 PASS；它们用于解释失败发生在接近、进入近区、
终端对准还是擦过外离。最终结论必须同时报告 `eta_min`、near-miss shell、annulus entry、
conditional capture、radial closure 和 radius sensitivity curve。

## 3. 每个 outer loop 的顺序

### Step 0：读状态

先读：

- 项目 `AGENTS.md`；
- `research/README.md`、`research/STATE.md`；
- 当前 Goal 文档和最近的 result/ledger；
- 当前 `research/AUTORESEARCH_LOOP.md`。

### Step 1：归因

按顺序检查，不得跳过：

1. 数据/标签：split、teacher、零标签、泄漏、shape；
2. 评价：`p_nav`/`eta` frame、success/termination、seed 配对、checkpoint 选择；
3. baseline：zero-residual、传统控制器、历史 BC；
4. 动作：限幅、rate limit、饱和、零初始化、BC 初始化；
5. reward/terminal：path 项是否压过 terminal，plane-miss 是否与 success 同向；
6. 算法：优化/探索、样本效率、replay、UTD、NoisyNet；
7. plant/estimator：current、噪声、delay、drag/thrust/fin、added mass；
8. 文献：只有前七项不能解释时才引入新方法。

### Step 2：写可证伪假设

每个 blocker 至少两条 hypothesis，每条必须写：

- 机制；
- 成立时应看到什么；
- 失败时应看到什么；
- 最便宜的 falsification；
- 需要的 seed/eval 规模和预算。

### Step 3：查论文与代码

使用 `paper-search` / `deep-research`，必要时用 `scoop-check`：

- 至少查 2–5 篇最接近的论文，写 `research/paper_reading/YYYY-MM-DD-<slug>.md`；
- 优先找到可运行 code；记录 license、框架、输入输出、依赖和不可迁移部分；
- 至少写 1 个“最小可迁移组件”，不要把整篇论文当成一个黑盒；
- 下载的 PDF/TXT 不提交；reading note 必须提交。

### Step 4：最便宜证伪

先跑 1 个 seed、1M transitions 或纯只读分析；只有机制方向稳定后才扩 formal。
禁止直接开 8 卡 × 5M 追结果。

### Step 5：决策

在 `research/LEDGER.jsonl` 写一条：

```json
{
  "date": "YYYY-MM-DD",
  "goal": "GoalXX",
  "trigger": "...",
  "evidence": "...",
  "hypotheses": ["...", "..."],
  "experiment": "...",
  "result": "...",
  "decision": "continue|pivot|stop",
  "next": "..."
}
```

### Step 6：pivot 时写新 Goal

如果 `pivot`，必须新写 `doc/goals/goalXX_<slug>_YYYYMMDD.md`，包含：

- Research Contract；
- 当前失败证据；
- 至少两个可证伪 hypothesis；
- 与旧 Gate 的隔离；
- 固定 seed/eval/held-out/checkpoint/budget；
- 主指标与风险指标；
- 新 Goal 的 prompt；
- 本文件的自主迭代要求。

新 Goal 必须说明“为什么这个新尝试有机会出效果”，不能只写“再试一次”。

## 4. 失败分类与默认分支

| 观察 | 归因 | 默认分支 |
|---|---|---|
| residual 很小且训练曲线平 | 优化/探索或信用分配 | 先查 reward/BC/action；有明确证据才开 SAC 同协议对照 |
| residual 很大且饱和、闭环变差 | 动作参数化/终端目标 | 不改算法；新 Goal 查 scale/rate/terminal |
| path RMSE 改善但 capture 不改善 | reward/terminal capture | 不换算法；查 terminal 目标和 reward composition |
| nominal 可以、plant mismatch 变差 | 模型失配 | Goal85b 或新的同分布鲁棒训练 Goal |
| 两种算法差异小于 seed 噪声 | seed noise ambiguous | 新预注册确认实验；不选 best seed |
| 指标都好但动作/能耗差 | 多目标失败 | 加成本指标，不把 single success 当全部 |
| 评价口径仍有泄漏 | 实现/契约错误 | 判 `INVALID`；先修评价，不分析算法 |

REDQ/DroQ 只在明确样本效率失败时进入新的预注册 Goal；DADE 仍受项目硬约束约束。

## 5. 自动执行边界

以下事情可以自动决定并继续：

- 选择下一个最便宜的 falsification；
- 写 blocker/reading note/ledger；
- 根据失败类型写新 Goal 和 prompt；
- 在已预注册预算内并行 seed/eval；
- 修正明确的实现错误并补测试。

以下事情必须暂停并向用户报告：

- 改变任务成功定义或 `goal_radius`；
- 打开 held-out `100–149`；
- 改变主算法族（PPO/SAC 之外）或重开 DADE；
- 超过预注册训练预算/算力；
- 平台、数据或安全边界变化；
- 需要 commit/push。

## 6. 停止与继续

- 每个 Goal 最多 3 个 outer loop；到上限必须写 `stop` 或 `pivot`，不能静默重复。
- 连续 3 个 Goal 都没有主指标或近目标机制移动时，写 blocker review 并向用户报告；
  用户可授权继续新方向，不能把同一失败实验换 seed 再跑。
- 出现 safety 回归、NaN/Inf、数据泄漏或预算耗尽时立即停。
- 只要还有明确的、可证伪的新方向，且预算允许，就不因一次 FAIL 结束项目。
- “直到出效果”意味着持续提出可检验的新方向，不意味着无限重复同一失败配置。
