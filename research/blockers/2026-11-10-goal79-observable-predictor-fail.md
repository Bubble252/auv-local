# Goal79：observable full-vector predictor 未过 Gate

日期：2026-11-10<br>
关联：Goal79 / Goal78 / Goal77 / Goal76 / Goal75<br>
状态：`OBSERVABLE_FULL_VECTOR_PREDICTOR_NOT_SUPPORTED`<br>
决策：`pivot`，停止 predictor 调参，转 terminal reference/reward

## 1. 现象

固定 22 维终端可见特征、seed-grouped leave-one-seed-out、ridge
`alpha=1.0`、`0.25 m` full-vector deadband 后：

| scenario | base tight true/false | pred tight true/false | true 保留率 | false 降幅 |
|---|---:|---:|---:|---:|
| default | 59/69 | 58/38 | 98.31% | 44.93% |
| strict125 | 25/33 | 25/20 | 100.00% | 39.39% |

两个 scenario 的 false 降幅都低于预注册 `>=50%`。true retention 过线，
所以不是“修正太强误杀 true”，而是 correction 不够准或不够强。

## 2. 证据

- mismatch `0`，非有限预测 `0`；
- 特征名、特征函数参数、重复特征检查均无 `eta` / `p_nav-eta` 输入；
- 预测误差：

| scenario | RMSE signed/lateral/vertical (m) | R2 signed/lateral/vertical | correction 中位 |
|---|---:|---:|---:|
| default | 0.3911/0.0655/0.0713 | -0.3792/0.9467/0.9608 | 0.0789 |
| strict125 | 0.3442/0.0625/0.0563 | 0.0032/0.9525/0.9710 | 0.1108 |

lateral/vertical 分量可预测，signed 分量不可预测。Goal78 oracle 的 correction
中位约 `0.18 m`，predictor 只有 `0.08–0.11 m`，因此只能削掉一部分 false。

## 3. 已排除

| 解释 | 状态 | 理由 |
|---|---|---|
| NoisyNet 采样主导 | 排除 | deterministic eval，`sample_noise=False`；低噪声 s240/s752/s880 仍贡献大部分剩余 false |
| residual 非零主导 | 排除 | forced zero-residual 在预测 correction 后仍剩 `7/12`、`8/10` false |
| 特征泄漏导致假失败 | 排除 | H79-B 审计通过，无 eta / p_nav-eta 输入 |
| seed 泄漏 | 排除 | seed-grouped LOOCV，同 seed 的 5 个 run 同折 |
| 修正误杀 true | 排除 | true retention `98.31%/100%`，主要问题是 false 降幅不够 |
| 需要换 SAC/REDQ/DroQ | 无证据 | 没有 optimization/exploration 或 sample-efficiency 失败证据 |

## 4. 可证伪假设与结果

### H79-A：observable predictor 能保留 true 并降低 false

预测：两个 scenario 的 tight false 降幅 `>=50%`、true retention `>=95%`。

结果：**FALSIFIED**。true retention 过线，false 降幅 `44.93%/39.39%`。

### H79-B：预测器不需要 eta 泄漏特征

预测：特征白名单和预测输入函数都不包含 `eta` / `p_nav-eta`；否则记
`INVALID`。

结果：**SUPPORTED**。审计无 forbidden token，预测输入参数无 eta。

### H79-C：失败主要由 NoiseNet 或 residual 幅度解释

预测：失败集中在高噪声 checkpoint，或 zero-residual 后明显减少。

结果：**REJECTED**。zero-residual 和低噪声 checkpoint 仍保留大部分 false。

## 5. 机制判断

可观测特征能学到 terminal `p_nav-eta` 的 lateral/vertical 部分，但学不到
signed 部分；signed 恰好是 Goal76/77 指出的 main shortfall。使用固定
`0.25 m` deadband 后，预测 correction 中位被压到 `0.08–0.11 m`，不足以把
false 降到预注册门禁以下。

这不是“换一个 alpha 就能解决”的证据。继续扫 alpha、模型族、特征组合或
deadband 会把预注册 Gate 变成事后调参，且不能回答 terminal reference 是否
才是主因。因此按失败裁决转入 terminal reference/termination 单因素实验。

## 6. 下一步

1. 预注册 terminal reference 沿 goal tangent 前移；
2. 或训练期 plane-crossing reward/termination shaping；
3. 保持 reward 其余项、gamma、seed、checkpoint 选择规则、eval split 和
   held-out `100–149` 不变；
4. 正式训练前做 `num_envs 128/256/512` 与 batch `256/512` 单卡吞吐基准；
5. 扩采样/批量另开预注册 arm；SAC/REDQ/DroQ 继续关闭。

## 7. 证据

- `doc/goals/goal79_observable_full_vector_bias_predictor_20261110.md`
- `doc/goals/goal79_result_20261110.md`
- `scripts/analyze_pc_goal79_observable_full_vector_bias_predictor.py`
- `tests/test_pc_goal79_observable_full_vector_bias_predictor.py`
- `artifacts/lowlevel/goal79_observable_bias_predictor_20261110/analysis/goal79.json`
- `artifacts/lowlevel/goal79_observable_bias_predictor_20261110/analysis/goal79.md`
- 服务器原始分析：
  `/data/yhx/underwater-robotics/src/plan_p1/goal60_action_reward_terminal_readonly_audit_20261110/artifacts/goal79_observable_bias_predictor/analysis/`
