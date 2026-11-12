# Reading note：USBL geometry、filter state 与 DVL calibration 对 AUV navigation residual 的含义

**日期**：2026-11-09<br>
**触发**：Goal50 的 13 个 recorded timing/control 特征全部可评估但全部
`FALSIFIED`；需要判断剩余机制是否可能来自未记录的 estimator internal state。<br>
**用途**：为 Goal51 instrumentation replay 选择最小、可证伪的记录项；不把文献
结论直接当作本项目的因果证据。

## 1. 检索方法

使用 `paper-search` skill：

```text
/home/bubble/.codex/skills/paper-search/.venv/bin/python \
  /home/bubble/.cc-switch/skills/paper-search/scripts/search_papers.py \
  --queries "underwater vehicle navigation drift DVL USBL error state observability|\
AUV dead reckoning sensor outage drift trajectory correlation|\
underwater navigation filter consistency innovation covariance diagnostics|\
AUV navigation error trajectory dependent observability" \
  --start-year 2018 --end-year 2026 --max-papers 10 \
  --sources arxiv crossref
```

结果：arXiv 30 条、Crossref 40 条，去重后 66 条。Semantic Scholar 因 HTTP 429
未作为证据；arXiv 第一次查询有连接超时，后续 arXiv/Crossref 查询成功。下面只列
与 Goal50 blocker 直接相关的 6 篇；摘要结论来自 arXiv API 或 Crossref 元数据，
未把仅标题可读的条目当作机制证据。

## 2. 直接相关文献

### Kedia, Makam, Hamann, Sundaram 2026, “BIND-USBL: Bounding IMU Navigation Drift using USBL in Heterogeneous ASV-AUV Teams”

- arXiv `2604.11861`；DOI `10.1109/oceans66983.2026.11616954`；OCEANS 2026。
- 核心结论：没有 external position fix 时，AUV 依赖 inertial dead-reckoning；
  长时 navigation failure 不只取决于单次 USBL 精度，而取决于 fix 的 temporal
  sparsity 和 geometric availability。
- 对本项目：Goal47/50 记录的是 late-window mean 与 `usbl_age_s` 等摘要，但没有
  USBL fix 的几何可用性、更新间隔或 delayed-fusion 状态。`nav_drift_sigma` 与
  late residual 的 association 可能通过“fix 何时可用”而不是“fix 均值”起作用。
- 可复现实验：Goal51 replay 记录每次 USBL due/accepted/rejected 的时间、几何和
  delay；之后只读计算 fix-availability 特征。
- 不能直接迁移：该文是多 ASV 协作定位，本项目是单 AUV + 固定 acoustic setup；
  不能直接套用 multi-ASV scheduler。

### Nitsch, Stenger, Abel 2023, “Automated Tuning of Nonlinear Kalman Filters for Optimal Trajectory Tracking Performance of AUVs”

- arXiv `2304.03565`。
- 核心结论：navigation filter 参数会显著影响 closed-loop GNC tracking；论文用
  Bayesian optimization 调 UKF 参数，报告最高约 50% median tracking error 改善。
- 对本项目：如果 Goal47 traces 中没有 filter covariance/gain/parameter state，
  那么“controller feedback 不足”可能只是 filter tuning 状态未被记录。Goal51
  应记录 gain/covariance 或 normalized innovation，而不是再扫 policy reward。
- 可复现实验：同一 replay 中记录 posterior covariance diagonal / trace 和 Kalman
  gain norm；不改变 filter 参数。
- 不能直接迁移：该文针对 nanoAUV 的 UKF、USBL + depth；本项目是 PC batch
  REMUS100 + synthetic-calibrated sensors，参数和观测模型不同。

### Zhang, Liu, Zhou, Ji, Wang, Xia, Xu 2024, “Sensor Misalignment-tolerant AUV Navigation with Passive DoA and Doppler Measurements”

- arXiv `2402.07218`。
- 核心结论：acoustic array 与 attitude sensor 之间的 misalignment 会恶化 navigation；
  论文联合估计 AUV navigation、beacon localization 和 sensor alignment。
- 对本项目：DVL/acoustic sensor 的 scale 或 misalignment 可能是一个未记录的
  seed-level 状态；Goal46 只检查 DVL outage exposure，没有检查 calibration 状态。
- 可复现实验：Goal51 replay 记录 DVL bottom-track residual 与（若可得）scale /
  misalignment 估计；先只做 read-only 统计。
- 不能直接迁移：该文的 DoA-only 配置与本项目 DVL/USBL 观测不同；只能借用
  “misalignment 是独立状态”这一设计原则。

### Cohen, Levy, Klein 2023, “Leveraging Past DVL Velocity Measurements for Acceleration-Aided AUV Navigation”

- arXiv `2308.11762`。
- 核心结论：从过去 DVL velocity 的 polynomial fit 构造 acceleration measurement，
  作为额外 EKF update，可改善位置/速度/姿态精度并降低 estimator covariance。
- 对本项目：late-window mean 不能表达 velocity history 或 covariance；Goal50
  的 `command_std`/`actuator_std` 也不等于 DVL history 状态。Goal51 应记录
  DVL acceleration/history summary 或 covariance，而不是继续加 actuator mean。
- 可复现实验：从已有 trace 的 DVL ground velocity 序列计算过去窗口 acceleration
  作为 read-only 特征；仍不训练、不改 filter。
- 不能直接迁移：该文使用真实 Snapir AUV 海试数据；本项目是仿真，不能假定相同
  采样噪声和 DVL beam geometry。

### Cohen, Yampolsky, Klein 2022, “Set-Transformer BeamsNet for AUV Velocity Forecasting in Complete DVL Outage Scenarios”

- arXiv `2212.11671`。
- 核心结论：DVL outage 会移除 velocity update，使 inertial solution 无法被 bound；
  论文用 inertial history + past DVL velocity 预测当前 velocity。
- 对本项目：Goal46 观察到所有 DVL outage 都在 early episode，late outage fraction
  为 0；因此这不能解释 strict125 late residual。Goal51 若记录 DVL outage/latency，
  必须区分 early 和 late exposure，不能把 episode-level outage 当作 late 机制。
- 可复现实验：只读输出 early/late outage fraction 与 bottom-track age；若 late
  exposure 仍为零，不再把 DVL outage 当候选。
- 不能直接迁移：该文的目标是 velocity forecasting，本项目不训练新网络，只做
  instrumentation 审计。

### Sahoo, Klein 2026, “NAViLoss: An Underwater Navigation-Aware Dual-Residual Objective for Physics-Consistent Learning”

- arXiv `2610.09690`。
- 核心结论：navigation-aware loss 同时约束 navigation-state velocity residual 与
  DVL beam-consistency residual，并用 adaptive uncertainty 处理 beam geometry；
  在约 10 km semi-synthetic sea-trial 数据上报告 velocity accuracy 改善。
- 对本项目：beam geometry / measurement uncertainty 是当前 trace 未记录的中间状态；
  Goal51 应至少记录 DVL beam-consistency 或 innovation uncertainty 的代理，而不是
  只看 RPM/舵角 mean。
- 可复现实验：Goal51 replay 记录 per-step DVL residual norm 与 beam-consistency
  residual（若 evaluator 可得到）；先验证 replay equivalence，再做 association。
- 不能直接迁移：该文训练 DeepONet；本项目禁止新训练，只借用状态设计。

## 3. 已核验但不用于机制裁决的条目

**Cui, Wang, Wu, Pan 2025**, “Observability Analysis on DVL and Range Assisted AUV
Navigation in Presence of Unknown Current and Range Drift”, DOI
`10.1007/978-981-96-2244-3_45`。Crossref 标题、作者和 DOI 已核验；摘要未从
Crossref/OpenAlex 取得，因此本 note 不把该文的具体结论当作证据。其标题只支持
一个弱设计提示：unknown current 与 range drift 会让 observability 依赖 trajectory。

## 4. 综合判断

1. 文献没有支持“再换一个 late-window mean 特征”就能解释 strict125 residual；
   相反，USBL fix geometry、filter tuning/covariance、DVL history/uncertainty 和
   sensor misalignment 都是**当前 trace 未记录的状态**。
2. Goal50 的 full-coverage null 把 recorded timing/control 这一层基本排除了；
   继续在 Goal50 加特征会变成 fishing。
3. 最便宜且可证伪的下一步不是训练，而是 Goal51 instrumented replay：同一策略、
   同一 seeds，只增加 opt-in trace 字段，并用 replay equivalence Gate 保证旧字段
   bitwise invariant。
4. 在 replay Gate 通过前，不得解释机制；也不得打开 held-out `100-149`。

## 5. 对 Goal51 的最小记录清单

| 目标状态 | 最小记录 | 为什么需要 |
|---|---|---|
| filter consistency | normalized USBL innovation、gain norm、posterior covariance trace/diag | Goal47 只有 raw innovation，Goal50 的 mean 特征无法覆盖 |
| DVL calibration | bottom-track residual、scale/misalignment（若 estimator 可导出） | Goal46 只检查 outage，未检查 calibration |
| DVL history | past-window velocity/acceleration summary、late outage fraction | Goal50 的 actuator std 不是 DVL history |
| actuator lag | command/actual cross-correlation、rate-clip count、per-channel slew | Goal49/50 只有 mean/std/saturation 摘要 |

## 6. 引用

```text
Kedia, P., Makam, R., Hamann, H., Sundaram, S. (2026). BIND-USBL: Bounding IMU
Navigation Drift using USBL in Heterogeneous ASV-AUV Teams. arXiv:2604.11861.
DOI:10.1109/oceans66983.2026.11616954.

Nitsch, M., Stenger, D., Abel, D. (2023). Automated Tuning of Nonlinear Kalman
Filters for Optimal Trajectory Tracking Performance of AUVs. arXiv:2304.03565.

Zhang, B., Liu, S., Zhou, S., Ji, D., Wang, T., Xia, T., Xu, W. (2024).
Sensor Misalignment-tolerant AUV Navigation with Passive DoA and Doppler
Measurements. arXiv:2402.07218.

Cohen, N., Levy, O., Klein, I. (2023). Leveraging Past DVL Velocity Measurements
for Acceleration-Aided AUV Navigation. arXiv:2308.11762.

Cohen, N., Yampolsky, Z., Klein, I. (2022). Set-Transformer BeamsNet for AUV
Velocity Forecasting in Complete DVL Outage Scenarios. arXiv:2212.11671.

Sahoo, A. K., Klein, I. (2026). NAViLoss: An Underwater Navigation-Aware
Dual-Residual Objective for Physics-Consistent Learning. arXiv:2610.09690.

Cui, J., Wang, M., Wu, W., Pan, X. (2025). Observability Analysis on DVL and Range
Assisted AUV Navigation in Presence of Unknown Current and Range Drift.
DOI:10.1007/978-981-96-2244-3_45.
```
