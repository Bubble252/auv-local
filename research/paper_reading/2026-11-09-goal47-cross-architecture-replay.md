# Goal47 reading note：跨架构 bitwise reproducibility 与浮点误差界

日期：2026-11-09<br>
触发：server 与本地用同 checkpoint/config/seeds 得到相同 episode outcome，但旧
trace 字段在 `1e-15` 到 `1e-13` 量级不同；需要判断这是 instrumentation bug 还是
跨 CPU 架构的浮点实现差异。<br>
检索：`paper-search`，arXiv 2018–2026；摘要和 PDF 通过 arXiv 核验。

## 1. 检索问题

- bitwise reproducibility 在不同 CPU 架构/BLAS 之间是否是合理要求？
- IEEE-754 浮点加法的非结合性和不同求和顺序会产生多大误差？
- 对研究结论，应如何区分 technical replicability、statistical replicability 与
  conceptual replicability？

检索 query：

```text
reproducibility floating point deterministic simulation across CPU architectures
bitwise reproducibility floating point machine learning hardware
```

Semantic Scholar 在本次检索中返回 HTTP 429；arXiv 结果可用。未把 429 结果当作
文献证据。

## 2. 已核验论文

### Hallman & Ipsen (2021)

**Deterministic and Probabilistic Error Bounds for Floating Point Summation
Algorithms**，arXiv `2107.01604`。

- 论文分析一般求和、shifted summation 和 compensated summation 的单精度前向误差；
- 标准 IEEE 模型写为 `fl(x op y) = (x op y)(1+delta)`，其中 `|delta| <= u`；
- 一般求和的确定性误差界依赖 computational tree height，概率界也依赖求和树结构；
- 改变并行 reduction 的树高、线程数或硬件指令会改变舍入 path，即使数学表达式相同。

对本项目的含义：不同 CPU/BLAS 上出现 `~1e-15` 量级差异并不等于控制逻辑改变；
但一旦 Gate 要求 bitwise equality，就必须固定机器或改用明确的 tolerance Gate。

### McDermott et al. (2019)

**Reproducibility in Machine Learning for Health**，arXiv `1907.01463`。

- 区分 technical replicability（technical identical conditions）、statistical
  replicability（重采样）和 conceptual replicability（不同条件下的机制是否成立）；
- 仅追求 technically identical 的结果，不能替代对统计/概念可复现性的检查。

对本项目的含义：Goal47 的 formal trace 应保留同机器 technical replicability；
server 的跨机器 run 应作为 statistical/outcome corroboration，而不是伪造 bitwise
PASS。机制结论最终仍要满足 component identity 和 sample-level checks。

## 3. 综合判断

1. **跨架构 bitwise 不是 IEEE-754 的保证。** 即使所有操作都是确定性 rounding，
   非结合加法和不同 reduction tree 也会产生不同的最后几位。
2. **必须固定 machine/environment 才能使用 bitwise Gate。** Goal42 reference 由本地
   Intel 环境产生；server AMD 环境的差异在 `1e-13` 以内，不符合 Goal47 的
   `bitwise_equal` 定义。
3. **不能因为差异小而把 server trace 写成 PASS。** 只能标记
   `NOT_BITWISE_CROSS_ARCH`，作为 outcome-level corroboration。
4. **正式机制分析应使用与 reference 同架构的本地 replay。** 这保留预注册 Gate，
   也不用 tolerance 事后放宽门禁。

## 4. 对 Goal47 的约束

- formal analyzer 只读本地 40-seed instrumentation replay；
- server run 只核验 flags/row count 和数值近邻，不进入 formal trace；
- 不改 checkpoint、config、seed、late window、component share 或 verdict mapping；
- 不把跨架构舍入差异归因给 controller、DVL、OU 或 USBL；
- 不打开 held-out `100-149`。

## 5. 引用核验

| 文献 | 核验方式 | 状态 |
|---|---|---|
| Hallman & Ipsen 2021, arXiv 2107.01604 | PDF + extracted text | verified |
| McDermott et al. 2019, arXiv 1907.01463 | PDF + extracted text | verified |
| Goal46 result / Goal42 replay protocol | 本地 result + gate | verified context |
