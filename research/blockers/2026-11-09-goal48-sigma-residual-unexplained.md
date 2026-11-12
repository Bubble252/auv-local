# Goal48 blocker：strict125 sigma residual association 未被现有 trace 特征解释

**日期**：2026-11-09<br>
**Goal**：Goal48<br>
**触发**：Goal48 的四个预注册解释（OU time shape、seed covariates、event timing、
small-n fragility）全部 `FALSIFIED`，但 strict125 partial rho 仍为
`0.8336/0.8858`，且 negative controls 表明 association 不是 pipeline artifact。

## 1. 现象

Goal48 只使用 Goal47 已有 formal traces，四个 block 的 160 条 per-seed rows 全部
通过输入 hash、row count、finite 和 Goal47 partial rho 复现检查：

| scenario | mode | n | Goal47 partial rho | H1 OU-shape rho | best H2 rho | H4 event rho | LOO median |
|---|---|---:|---:|---:|---:|---:|---:|
| strict125 | baseline | 13 | `0.8336` | `0.8959` | `0.7803` | `0.8729` | `0.8204` |
| strict125 | candidate | 7 | `0.8858` | `0.8892` | `0.7228` | `0.4361` | `0.8799` |

`H1=FALSIFIED`、`H2=FALSIFIED`、`H4=FALSIFIED`，`H3=FALSIFIED`：strict125 candidate
的 LOO median 为 `0.8799`，95% Bayesian-bootstrap CI 为
`[0.5655, 0.9865]`；baseline CI 为 `[0.6260, 0.9418]`。因此 association 不是
简单由少数 seed 或 small-n 造成。

## 2. 已排除项

- **OU mean projection**：Goal47 已控制；strict125 仍有强 partial rho；
- **OU late slope**：H1 控制后 rho 仍为 `0.8959/0.8892`，drop 为负；
- **seed covariates**：`nav_bias_n`、`ou_tau_s`、`queue_age_s`、
  `initial_goal_distance_m` 没有一个在两个 strict125 block 同时把 rho 降到
  `<=0.30` 且 drop `>=0.20`；
- **event timing**：控制 OU mean + USBL + DVL 后，baseline rho `0.8729`；
  candidate 虽降到 `0.4361`，但仍高于 `0.30`，且 baseline 不通过；
- **small-n fragility**：LOO median 和 Bayesian-bootstrap lower bound 都高于
  预注册的 fragility 阈值；
- **OU dispersion / nonlinear mean**（blocker 阶段最便宜 falsification）：
  控制 `ou_mean + ou_std` 后 strict125 rho 为 `0.7816/0.9057`；
  控制 `ou_mean + ou_mean^2` 后仍为 `0.8336/0.8858`。没有把 association 解释掉。

## 3. 可证伪 hypotheses

### H-DISCORDANT-SPECIFIC-FEEDBACK：association 是 discordant trajectories 特有

若成立：non-discordant seeds 的同一 partial rho 应接近 0，且 sigma permutation
null 应很少达到 observed rho。

当前证据：**SUPPORTED**。strict125 baseline/candidate 的 non-discordant partial
rho 只有 `0.1198/0.0117`；block 内 sigma permutation 的 `p(rho >= observed)` 为
`0.0007/0.0063`（10,000 draws）。这支持“discordant 轨迹/反馈”而不是全局 sampler
artifact，但仍是 association，不是因果识别。

### H-OU-DISPERSION-NONLINEAR：association 由 OU variance 或非线性 mean 携带

若成立：控制 OU std 或 quadratic OU mean 后，strict125 partial rho 应降到
`<=0.30`。

当前证据：**FALSIFIED**。加权/线性 rank residualization 后 rho 仍为
`0.78-0.91`，没有 material drop。

### H-UNRECORDED-CONTROLLER-FEEDBACK：association 由未纳入的控制器/执行器反馈携带

若成立：加入已记录的 saturation、heading/pitch error、speed error、command
residual、path curvature/progress 或 guidance phase 后，strict125 partial rho 会
materially 降低；若这些已记录字段仍不能解释，则需要新的 estimator/controller
instrumentation。

当前证据：**OPEN**。Goal47 trace 已包含 `material_saturated`、
`heading_error_rad`、`pitch_error_rad`、`speed_error_mps`、`command`、
`residual_action`、`guidance_*`、`path_progress_m` 等字段，尚未在 Goal48 中测试。

## 4. 最便宜的证伪实验

已在 Goal48 之后、写 result 之前完成，全部只读：

1. non-discordant negative control：strict125 baseline/candidate partial rho
   `0.1198/0.0117`；
2. sigma permutation null：`p=0.0007/0.0063`；
3. OU dispersion/nonlinear controls：rho `0.7816/0.9057` 与
   `0.8336/0.8858`。

这排除了“全局 sampler artifact”“OU variance 单独解释”“纯 small-n”三种解释，
但没有排除已记录控制器反馈或未记录状态。

## 5. 裁决与下一步

```text
decision = pivot
```

停止在 Goal48 中继续增加 seed covariates（会变成 fishing）。下一步只允许预注册
Goal49：使用同一批 Goal47 traces 的 **controller/executor feedback** 字段做只读
判别，先冻结一个小的、有机制依据的特征族（saturation、heading/pitch/speed error、
command residual、path progress/curvature、guidance phase），再做 negative controls
和 block partial rho。若 Goal49 仍不能解释，下一步才考虑新的 instrumentation replay；
不得直接训练、改 controller/config/seed、参数扫描或打开 held-out。

## 6. 证据

- `doc/goals/goal48_strict125_sigma_residual_discrimination_20261109.md`
- `doc/goals/goal48_result_20261109.md`
- `artifacts/lowlevel/goal48_strict125_sigma_residual_20261109/`
- `research/paper_reading/2026-11-09-goal48-sigma-residual-inference.md`
