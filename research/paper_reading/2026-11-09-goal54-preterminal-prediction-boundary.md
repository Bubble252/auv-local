# Reading note：pre-terminal prediction、lead time 与 Goal54 null 的边界

**日期**：2026-11-09<br>
**触发**：Goal54 的固定 `0-8.0 s` pre-terminal 特征在 strict125 三族全部失败；
default H1/H3 却通过。<br>
**用途**：判断这次 null 能排除什么、不能排除什么，以及是否适合继续做 early
feature/window sweep。只把文献作为设计约束，不把它当作本项目因果证据。

## 1. 检索方法

使用 `paper-search` skill：

```bash
/home/bubble/.cc-switch/skills/paper-search/.venv/bin/python \
  /home/bubble/.cc-switch/skills/paper-search/scripts/search_papers.py \
  --queries "pre-terminal trajectory prediction failure early warning autonomous underwater vehicle|\
early warning failure prediction multivariate time series closed-loop control|\
terminal approach capture failure autonomous underwater vehicle" \
  --start-year 2018 --end-year 2026 --max-papers 6 \
  --sources arxiv crossref
```

返回 31 条 unique records。大部分是 trajectory tracking、generic anomaly detection
或非水下早期预警，不能直接回答本项目问题。下面只列与“lead time、terminal phase、
trajectory anomaly”边界相关的条目；没有把只看标题的条目写成已核验机制。

## 2. 相关条目与解释

### Jiao, Bai, Liu et al. 2023, “Learning Representation for Anomaly Detection of Vehicle Trajectories”

- arXiv `2303.05000`。
- 主题是把 vehicle trajectory 表示为可用于 anomaly detection 的 latent feature。
- 对 Goal54：early anomaly detection 是“先定义 lead time 和异常标签”的独立问题；
  不能因为存在 trajectory representation 方法，就推断 8 s 的 tracking error/
  actuator variance 一定能预测 terminal miss。

### Zheng, Zhang, Zhang 2023, “Lagrange tracking-based long-term drift trajectory prediction method for Autonomous Underwater Vehicle”

- DOI `10.3934/mbe.2023932`。
- 主题是 AUV 长时 drift trajectory prediction，而不是 terminal capture 的二分类
  或 lead-time early warning。
- 对 Goal54：long-term prediction 与 terminal approach failure 不是同一 endpoint；
  该文献支持“预测需要明确的 trajectory model/horizon”，不支持在本数据上继续
  换 early window。

### Zhang 2018, “Research on Terminal Docking of Autonomous Underwater Vehicle Based on Visual Guidance and Rotational Docking Station”

- DOI `10.3901/jme.2018.20.081`。
- 主题是 terminal docking 的专用视觉制导，而非 8 s 前的低维状态预测。
- 对 Goal54：terminal capture 可能需要 terminal-phase 专用 sensing/guidance；
  这与 Goal53 的 late-window association 一致，但不能证明 late association 是
  pre-terminal predictor。

### Sun, Zhu, Gu et al. 2026, “From Noisy Telemetry to Actionable Warnings: GPU Failure Prediction in Industrial Clusters”

- arXiv `2609.34473`。
- 主题是工业 telemetry 的 actionable warning，强调 lead time 与 false alarm；
  与水下 terminal miss 没有直接可迁移证据。
- 对 Goal54：只作为“early warning 必须同时报告 lead time、false alarm 和 outcome”
  的方法学提醒，不作为机制引用。

### Xiang, Yang, Chen 2026, “Dual-Score Association-Discrepancy Enhancement for Early-Warning Multivariate Time-Series Anomaly Detection”

- DOI `10.20944/preprints202607.1547.v1`，preprint，未作为强证据。
- 主题是 multivariate time-series early warning。
- 对 Goal54：说明 early-warning 线通常需要独立的异常定义和时序评估协议；
  Goal54 的固定 binary/continuous endpoint 与 lead-time 设计不同，不能直接套用。

## 3. 对本项目的结论

1. Goal54 的 null 只否定“当前 13 个 early 特征 + 8.0 s 窗口 + 当前 Gate”的
   predictive claim；不能写成“terminal failure 没有任何 early warning 可能”。
2. 文献没有提供理由在本 Goal 内追加窗口、特征、lag 或模型。继续这样做会变成
   post-hoc feature sweep，并再次混淆 concurrent association 与 lead-time
   prediction。
3. 若重新进入 early-warning 研究，必须先冻结独立 seed/checkpoint、lead time、
   false-alarm 预算和 endpoint；若研究机制，则选择 controlled intervention 或
   显式记录一个具体 hidden state。
4. Goal53 的 late-window association 仍可保留为 descriptive/concurrent evidence；
   它不是 Goal54 的 pre-terminal predictor，也不能作为训练授权。

## 4. 引用

```text
Jiao, R., Bai, J., Liu, X., et al. (2023). Learning Representation for Anomaly
Detection of Vehicle Trajectories. arXiv:2303.05000.

Zheng, S., Zhang, M., Zhang, J. (2023). Lagrange tracking-based long-term drift
trajectory prediction method for Autonomous Underwater Vehicle. Mathematical
Biosciences and Engineering. DOI:10.3934/mbe.2023932.

Zhang, T. (2018). Research on Terminal Docking of Autonomous Underwater Vehicle
Based on Visual Guidance and Rotational Docking Station. Journal of Mechanical
Engineering. DOI:10.3901/jme.2018.20.081.

Sun, Y., Zhu, R., Gu, W., et al. (2026). From Noisy Telemetry to Actionable
Warnings: GPU Failure Prediction in Industrial Clusters. arXiv:2609.34473.

Xiang, Z., Yang, L., Chen, W. (2026). Dual-Score Association-Discrepancy
Enhancement for Early-Warning Multivariate Time-Series Anomaly Detection.
Preprint. DOI:10.20944/preprints202607.1547.v1.
```
