# Reading note：早期预测的 cutoff、lead time 与 temporal leakage

**日期**：2026-11-09<br>
**触发**：Goal55 固定 8.0 s 窗口在新 seed block 的 seed 67 上包含 terminal row。<br>
**用途**：约束下一个 pre-terminal 窗口设计；不把外部文献当作本项目机制证据。

## 1. 检索方法

使用 `paper-search` skill：

```bash
/home/bubble/.cc-switch/skills/paper-search/.venv/bin/python \
  /home/bubble/.cc-switch/skills/paper-search/scripts/search_papers.py \
  --queries "label leakage early warning prediction lead time|\
pre-terminal window failure prediction robotics|\
time-to-fault prediction lead time robot" \
  --start-year 2020 --end-year 2026 --max-papers 5 \
  --sources semantic_scholar arxiv crossref open_alex
```

Semantic Scholar 返回 HTTP 429；arXiv、Crossref 和 OpenAlex 返回 44 条 unique
records。另用 OpenAlex/arXiv 摘要接口核验了下面四条与 cutoff/lead time 最相关的
条目。没有把只看标题的条目写成已验证机制。

## 2. 相关条目

### Lê, Abel, Laforge 2026, “When Can We Trust Early Warnings? Leakage-Excluded Early Outcome Prediction from LMS Interaction Logs”

- arXiv `2605.25794`，DOI `10.48550/arxiv.2605.25794`。
- 主题：把 early outcome prediction 形式化为带 temporal availability constraint
  的 cutoff-based prediction；LEAP 协议先做 cutoff-first truncation，再做 join 和
  aggregation，并审计 feature provenance，防止 post-cutoff evidence 进入特征。
- 对本项目：Goal55 的 `[0, 8.0)` 必须对每个 episode 保证 cutoff 前没有 terminal
  row；仅仅复用 dev seeds 的观测最小值 `8.48 s` 不足以保证新 seed block 的
  temporal validity。该文献直接支持“先截断、后计算”，而不是看到 leakage 后再
  选择窗口。

### Sun, Zhu, Gu et al. 2026, “From Noisy Telemetry to Actionable Warnings: GPU Failure Prediction in Industrial Clusters”

- arXiv `2609.34473`。
- 主题：工业 telemetry 的 actionable failure warning，强调 lead time 与 false
  alarm；不是水下 terminal capture 的直接证据。
- 对本项目：early-warning claim 必须把 lead time、窗口和 false-alarm 一起冻结；
  不能只用 association p-value 冒充实现在线预警。

### Ali, Kamal 2025, “Time-to-Fault Prediction Framework for Automated Manufacturing in Humanoid Robotics Using Deep Learning”

- DOI `10.3390/technologies13020042`。
- 主题：用 LSTM 从实时传感器预测机器故障，报告约 10 分钟 lead time 和回归误差。
- 对本项目：time-to-fault 预测的 endpoint 是 lead time/剩余时间；Goal55 的
  terminal true miss 是 terminal outcome，不是同一个任务。该文献只能支持
  “明确 horizon 和 lead time”的方法学要求，不能证明 8 s 或 6 s 特征一定有效。

### 2023, “A framework for understanding label leakage in machine learning for health care”

- DOI `10.1093/jamia/ocad178`。
- 主题：label leakage 不只是简单的 “no time machine rule”，需要同时考虑
  prediction cadence、perspective 和 applicability。
- 对本项目：Goal55 的 fixed window 在新 seed 上出现一个 terminal row，属于
  temporal availability violation；不能因为只泄漏 1 个 row 就忽略，也不能
  事后改变 cutoff 来掩盖。

## 3. 对下一设计的约束

1. **先 cutoff，后 feature**：先确定每个 episode 的 prediction cutoff，再截断
   trace，再计算所有 13 个特征。
2. **cutoff 必须独立于 outcome**：可以用 terminal phase entry 的几何时间构造
   oracle cutoff，但必须明确这不是在线可部署；若声称 online，cutoff 必须只依赖
   当时可见信息。
3. **新 seed block**：seeds `60-99` 已被 Goal55 preflight 看到 outcome/terminal
   timing；任何继续的独立 replication 应使用未观察的 block，例如 `150-189`。
4. **报告 lead time 和 false alarm**：仅报告 family rho/AUC 不足以支持预警；
   还要报告 cutoff 后剩余时间、告警率和 held-out 隔离。
5. **不把 null 写成所有 early warning 无效**：Goal54/55 只否定冻结窗口和特征；
   改变 cutoff、lead time 或 instrumentation 是新问题。

## 4. 引用

```text
Lê, N. L., Abel, M.-H., Laforge, B. (2026). When Can We Trust Early Warnings?
Leakage-Excluded Early Outcome Prediction from LMS Interaction Logs.
arXiv:2605.25794. DOI:10.48550/arxiv.2605.25794.

Sun, Y., Zhu, R., Gu, W., et al. (2026). From Noisy Telemetry to Actionable
Warnings: GPU Failure Prediction in Industrial Clusters. arXiv:2609.34473.

Ali, A., Kamal, H. M. (2025). Time-to-Fault Prediction Framework for Automated
Manufacturing in Humanoid Robotics Using Deep Learning. Technologies.
DOI:10.3390/technologies13020042.

Anonymous/author list per journal metadata (2023). A framework for understanding
label leakage in machine learning for health care. JAMIA.
DOI:10.1093/jamia/ocad178.
```
