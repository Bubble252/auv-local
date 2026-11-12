# Goal48 decision：停止 seed-level 解释链，转 controller/executor feedback 只读审计

日期：2026-11-09<br>
Goal：Goal48<br>
诊断裁决：`SIGMA_ASSOCIATION_UNEXPLAINED`<br>
Ledger decision：`pivot`

Goal48 只使用 Goal47 已有 formal traces 和 CSV。四个冻结假设全部失败：
OU late time shape、四个 seed covariates、small-n fragility 和 USBL/DVL event
timing 都没有把 strict125 的 partial rho 降到预注册阈值。

这不能写成“association 是噪声”。strict125 baseline/candidate 的 partial rho
仍为 `0.8336/0.8858`；non-discordant seeds 只有 `0.1198/0.0117`；sigma
permutation 的 `p(rho >= observed)` 为 `0.0007/0.0063`。关联真实存在，且集中在
discordant trajectories，但不被已检查的 OU statistics、seed sampler covariates、
event timing 或简单 small-n 解释。

不再增加 seed covariates。strict125 candidate 只有 7 个 discordant seeds，
继续换协变量会变成 multiple-covariate fishing，也不能支持机制结论。

下一步只允许预注册 Goal49：复用同一批 Goal47 traces，对已记录的 controller/
executor feedback 做只读判别。特征族必须先冻结为少量、有机制依据的变量
（saturation、heading/pitch/speed error、residual action、command/actuator
late mean/std、path progress 与 guidance phase），并同时报告 non-discordant
negative control 和 sigma permutation null。

Goal49 不得训练、改 controller/config/seed/termination、参数扫描或打开 held-out
`100-149`。若 controller-feedback 仍不能解释，下一步只能预注册新的
instrumentation replay，记录未观测的 estimator/controller 状态；不能直接恢复训练。

证据：

- `doc/goals/goal48_result_20261109.md`
- `research/blockers/2026-11-09-goal48-sigma-residual-unexplained.md`
- `research/paper_reading/2026-11-09-goal48-sigma-residual-inference.md`
- `artifacts/lowlevel/goal48_strict125_sigma_residual_20261109/`
