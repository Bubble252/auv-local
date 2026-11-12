# Reading note：Goal51 state-proxy null、可观测性与下一步边界

**日期**：2026-11-09<br>
**触发**：Goal51 的 13 个可评估 estimator/DVL/actuator 代理全部没有在两个
strict125 block 通过 same-feature Gate；coverage `13/14`、permutation 有效，但
三个 family 均无可支持特征。<br>
**用途**：判断这次 null 能排除什么、不能排除什么，以及是否继续做 proxy feature
sweep 或 Stage B replay。只把文献作为设计约束，不把外部结论当作本项目因果证据。

## 1. 检索方法

使用 `paper-search` skill：

```bash
/home/bubble/.cc-switch/skills/paper-search/.venv/bin/python \
  /home/bubble/.cc-switch/skills/paper-search/scripts/search_papers.py \
  --queries "AUV navigation error trajectory dependent observability DVL USBL|\
underwater navigation filter consistency innovation diagnostics|\
sensor misalignment AUV dead reckoning|\
actuator lag identification underwater vehicle|\
causal mediation state estimation" \
  --start-year 2018 --end-year 2026 --max-papers 6 \
  --sources arxiv crossref
```

arXiv 30 条、Crossref 30 条，去重后 59 条。Semantic Scholar/OpenAlex 本轮未作为
证据来源；上一轮 Goal50 note 已记录 Semantic Scholar HTTP 429 的问题。下面只列与
Goal51 null 的解释边界直接相关的 6 篇；摘要以 arXiv API 或 Crossref 返回内容为准，
没有把只读标题的条目写成机制结论。

## 2. 直接相关文献

### Kedia, Makam, Hamann, Sundaram 2026, “BIND-USBL”

- arXiv `2604.11861`；DOI `10.1109/oceans66983.2026.11616954`。
- 作者报告：没有 external position fix 时，长时 navigation failure 主要受 USBL
  fix 的 temporal sparsity 和 geometric availability 影响，而不只是单次 USBL
  测量精度。
- 对 Goal51：raw innovation 的幅值/几何代理没有通过，不等于“USBL 通道无关”。
  Goal47 trace 虽然已有 `usbl_due/accepted/rejected`，但 Goal51 冻结的是 innovation
  代理，不是 fix availability/geometry 特征。这个区别不能用于事后追加 Goal51
  特征；若要做，必须是新的预注册问题。

### Cohen, Levy, Klein 2023, “Leveraging Past DVL Velocity Measurements for Acceleration-Aided AUV Navigation”

- arXiv `2308.11762`。
- 作者从过去 DVL velocity 的 polynomial fit 构造 acceleration measurement，并报告
  相比 baseline EKF 的位置/速度/姿态精度和 covariance 改善。
- 对 Goal51：Goal51 的 DVL 特征只看 late-window residual、scale、misalignment 和
  lag-1 autocorrelation；它没有覆盖 DVL velocity history 的高阶时序。null 只能
  排除当前 4 个代理，不能写成“DVL 历史状态无关”。

### Zhang, Liu, Zhou, Ji, Wang, Xia, Xu 2024, “Sensor Misalignment-tolerant AUV Navigation with Passive DoA and Doppler Measurements”

- arXiv `2402.07218`。
- 作者把 acoustic array 与 attitude sensor 的 misalignment 当作独立状态，并用
  UKF 同时估计 navigation、beacon localization 和 sensor alignment；摘要明确指出
  misalignment 会恶化 navigation accuracy。
- 对 Goal51：trace-derived misalignment proxy 没有 two-block 支持，但没有记录
  estimator 内部的精确 alignment state。Stage B 是否值得做必须由预注册触发条件
  决定，不能因为“文献说可能”就直接 replay。

### Macatangay, Hoseinnezhad, Fowler et al. 2023, “Addressing Actuator Saturation during Fault Compensation in Model-Based Underwater Vehicle Control”

- `Electronics` 12(21):4495；DOI `10.3390/electronics12214495`。
- 作者在 hybrid AUV actuator 配置下直接处理 saturation 和 fault-tolerant
  allocation，说明 actuator saturation 会实质改变 tracking performance。
- 对 Goal51：Goal51 的 actuator tracking/rate-clip/slew/reversal 代理没有通过，
  但这不证明 actuator 物理通道无关；rate-clip flag 和 limited command 仍未在
  Goal47 trace 中显式记录。该差异属于 Stage B 的 provenance 问题，不是可以
  在 Goal51 事后补特征的理由。

### Cui, Wang, Wu, Pan 2025, “Observability Analysis on DVL and Range Assisted AUV Navigation in Presence of Unknown Current and Range Drift”

- DOI `10.1007/978-981-96-2244-3_45`；Crossref 未返回摘要。
- 标题层面的设计提示：unknown current 与 range drift 会让 navigation observability
  依赖 trajectory；因此“某个 proxy 不显著”不等于“底层状态不存在”。
- 本 note 不引用其未读取的具体数值或结论。

### Panayiotou, Simsek 2026, “Causal Discovery in Action: Learning Chain-Reaction Mechanisms from Interventions”

- arXiv `2603.22620`。
- 作者报告：在 delayed/overlapping causal effects 下 observational heuristics 会
  失败，而 blocking interventions 可以识别 chain-reaction structure。
- 对 Goal51：这支持一个保守结论——Goal51 是 observational proxy audit，不是
  intervention；13 个代理的 null 不能识别 mediator/collider，也不能证明隐藏状态
  不存在。若要把 residual 写成因果机制，需要新的预注册 intervention，而不是
  继续加 observational feature。

## 3. 综合判断

1. Goal51 的结果是**代理层的 null**：在当前 Goal47 traces 上，可评估的 13 个
   filter/DVL/actuator 代理没有一个在两个 strict125 block 同时把 partial rho
   降到 `<=0.30` 且 drop 达到 `>=0.20`。这排除了本轮冻结的低维代理，不是对
   “USBL fix timing”“DVL history”“sensor misalignment”或“actuator rate limit”
   全部机制的最终否定。
2. 由于 coverage `13/14 >= 10`、permutation valid draws `9999/10000`、family/global
   p 均为 `1.0`，Goal51 协议 §5 的 Stage B 三个触发条件都没有满足。不能把
   “文献提示还有未记录状态”当作绕过 trigger 的理由。
3. 继续在 Goal51 内换窗口、换 lag、换 proxy 或补一个特征，会变成 post-hoc
   feature sweep；继续做 Stage B 则违反预注册触发条件。正确动作是记录 null，
   `stop` 这条 retrospective proxy-explanation 主线。
4. 若要继续研究，必须新开 goal 并明确选择：controlled intervention、独立的
   outcome-relevance 问题，或显式记录某个具体 hidden state。不能把“再试一个状态”
   当作机制结论；也不能恢复 command-residual PPO、reward/gamma/seed sweep 或
   打开 held-out。

## 4. 引用

```text
Kedia, P., Makam, R., Hamann, H., Sundaram, S. (2026). BIND-USBL: Bounding IMU
Navigation Drift using USBL in Heterogeneous ASV-AUV Teams. arXiv:2604.11861.
DOI:10.1109/oceans66983.2026.11616954.

Cohen, N., Levy, O., Klein, I. (2023). Leveraging Past DVL Velocity Measurements
for Acceleration-Aided AUV Navigation. arXiv:2308.11762.

Zhang, B., Liu, S., Zhou, S., Ji, D., Wang, T., Xia, T., Xu, W. (2024).
Sensor Misalignment-tolerant AUV Navigation with Passive DoA and Doppler
Measurements. arXiv:2402.07218.

Macatangay, X., Hoseinnezhad, R., Fowler, A., et al. (2023). Addressing Actuator
Saturation during Fault Compensation in Model-Based Underwater Vehicle Control.
Electronics, 12(21), 4495. DOI:10.3390/electronics12214495.

Cui, J., Wang, M., Wu, W., Pan, X. (2025). Observability Analysis on DVL and
Range Assisted AUV Navigation in Presence of Unknown Current and Range Drift.
DOI:10.1007/978-981-96-2244-3_45.

Panayiotou, P., Simsek, O. (2026). Causal Discovery in Action: Learning
Chain-Reaction Mechanisms from Interventions. arXiv:2603.22620.
```
