# Reading note：small-sample rank inference、OU trajectory 与 AUV navigation drift

**日期**：2026-11-09<br>
**触发**：Goal48 的 strict125 sigma residual association 无法被 OU time shape、
seed covariates、event timing 或 small-n 解释，进入 autoresearch。

## 1. 检索

检索工具：`paper-search` skill。最终可用来源：arXiv、Crossref；DBLP 返回非 JSON
错误，Semantic Scholar 被 HTTP 429 限流，未把失败来源当作证据。

检索词：

```text
partial rank correlation small sample bias
Bayesian bootstrap correlation inference
underwater navigation drift estimator trajectory
```

共返回 30 个 unique records（arXiv 15、Crossref 15）。本 note 只记录与当前 blocker
直接相关的 5 篇；其余结果主要是 rank correlation、Bayesian bootstrap 和 AUV
navigation 的泛化文献，未作为机制证据。

## 2. 直接相关文献

### Lin & Han 2023, “On the failure of the bootstrap for Chatterjee's rank correlation”

- arXiv `2303.14088`；math.ST / econ.EM。
- 核心结论：标准 bootstrap 对 Chatterjee's rank correlation 可以不一致；论文给出
  independence 假设下的证明和一般 setting 的模拟证据，并建议使用原论文的 analytic
  variance 或 asymptotic 方法。
- 对 Goal48 的含义：不能把“bootstrap CI 很窄”直接当成 rank statistic 的严格
  inference。Goal48 同时报告了 LOO、Bayesian bootstrap 和 sigma permutation，
  permutation negative control 的 `p=0.0007/0.0063` 比单一 bootstrap CI 更可信。
- 边界：该文针对 Chatterjee's rank correlation，不是 Spearman partial
  correlation；这里只借用“rank statistic 的 bootstrap 需要额外验证”这一 caution，
  不把论文结论直接套到 Spearman。

### Pan & Wu 2015, “Underwater Doppler Navigation with Self-calibration”

- arXiv `1509.02054`；Journal of Navigation。
- 核心结论：IMU/DVL 组合在温和条件下可观测；DVL scale factor 和 misalignment
  angles 可以在线自校准，不需要外部 GPS 或 acoustic beacon；仿真验证了 analytic
  observability。
- 对 Goal48 的含义：AUV navigation error 不只是 OU mean；scale/misalignment、
  estimator feedback 和 trajectory-dependent observability 都可能把 estimator
  state 与 trajectory 绑在一起。Goal48 已排除 OU mean/slope/std/quadratic 和几个
  seed covariates，但还没有检验已记录的 controller/executor feedback。

### Terán Espinoza et al. 2023, “STERN: Simultaneous Trajectory Estimation and Relative Navigation for Autonomous Underwater Proximity Operations”

- arXiv `2309.08780`；IEEE Journal of Oceanic Engineering 2026。
- 核心结论：用 factor graph 统一建模 underwater proximity operations 的
  simultaneous trajectory estimation and relative navigation；动态假设、前端/后端
  和传感器 suite 会共同影响 estimator。
- 对 Goal48 的含义：trajectory 与 estimator state 是联合演化的；只在 seed level
  控制 OU statistics 可能不足以解释 trajectory-specific residual。下一步应优先
  检查已记录的 heading/pitch/speed error、saturation、command residual、guidance
  phase 和 path progress，而不是继续加 seed covariates。

### Azkarate-Askasua & Zerecero, “Correcting Small Sample Bias in Linear Models with Many Covariates”

- Crossref DOI `10.2139/ssrn.4322300`。
- 核心结论（标题/摘要层面）：多协变量小样本线性模型需要显式 small-sample bias
  correction；协变量越多，naive 估计越容易过拟合。
- 对 Goal48 的含义：strict125 candidate 只有 7 个 discordant seeds；Goal48 已限制
  H2 为 4 个预注册候选并逐个测试，不允许在看到结果后继续扩展协变量集合。下一步
  controller-feedback 特征族也必须先冻结、控制数量，并报告所有候选。

### Arfè & Muliere 2023, “A general Bayesian bootstrap for censored data based on the beta-Stacy process”

- Crossref DOI `10.1016/j.jspi.2022.07.001`。
- 核心结论：Bayesian bootstrap 可以用 beta-Stacy process 推广到 censored data；
  它是 nonparametric Bayesian 重加权方法，不是 row resampling。
- 对 Goal48 的含义：Goal48 的 i.i.d. row bootstrap 在 `n=7` 时产生大量 rank
  degeneracy；改用 Dirichlet Bayesian bootstrap 是标准重加权替代，保持了同一
  10,000 draws / 95% interval / 9,000 valid 门槛，且没有根据 hypothesis 结果改阈值。

## 3. 综合判断

文献与本地的共同结论：

1. strict125 sigma association 不是“再换一个 seed covariate”就能解释的问题；
   小样本多协变量容易过拟合，继续加法会变成 fishing；
2. rank statistic 的 inference 需要 permutation/analytic 交叉验证；Goal48 的
   permutation negative control 已经支持 association 真实存在且 discordant-specific；
3. AUV navigation 文献把误差与 trajectory、DVL scale/misalignment、estimator
   feedback 联系起来，下一步最合理的是 **已记录 controller/executor feedback**
   的只读审计，而不是新训练或新 replay；
4. 如果 controller-feedback 仍不能解释，才需要新的 instrumentation replay
   记录未观测的 estimator/controller 状态；这必须是新 goal，且要先冻结 schema。

## 4. 对 Goal49 的设计约束

- 复用 Goal47 已有 traces，不新增 replay、不训练；
- 预注册一个小的 controller-feedback 特征族：`material_saturated`、
  `heading_error_rad`、`pitch_error_rad`、`speed_error_mps`、`residual_action`、
  `command`/`actual_actuator` 的 late-window mean/std、
  `path_progress_m`/guidance phase；
- 每个特征族只做 block partial rho，不逐个 seed fishing；
- 同时报告 non-discordant negative control 和 sigma permutation null；
- 若 partial rho 仍 `>0.30` 且 drop `<0.20`，判 controller-feedback 不足，
  进入“新 instrumentation 或停止解释链”的 decision，而不是继续加特征。

## 5. 引用

```text
Lin, Z., Han, F. (2023). On the failure of the bootstrap for Chatterjee's rank
correlation. arXiv:2303.14088.

Pan, X., Wu, Y. (2015). Underwater Doppler Navigation with Self-calibration.
Journal of Navigation. arXiv:1509.02054.

Terán Espinoza, A., et al. (2023/2026). STERN: Simultaneous Trajectory Estimation
and Relative Navigation for Autonomous Underwater Proximity Operations.
IEEE Journal of Oceanic Engineering. arXiv:2309.08780.

Azkarate-Askasua, M., Zerecero, M. Correcting Small Sample Bias in Linear Models
with Many Covariates. SSRN. DOI:10.2139/ssrn.4322300.

Arfè, A., Muliere, P. (2023). A general Bayesian bootstrap for censored data
based on the beta-Stacy process. Journal of Statistical Planning and Inference.
DOI:10.1016/j.jspi.2022.07.001.
```
