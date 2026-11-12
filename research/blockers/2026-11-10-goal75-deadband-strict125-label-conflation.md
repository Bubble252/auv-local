# Blocker：strict125 deadband 差 3 个 true，根因是标签语义混淆

日期：2026-11-10<br>
关联：Goal75 / Goal74 `MARGIN_UNUSABLE_BIAS_SUPPORTED`<br>
状态：`PARTIAL_SUPPORT`<br>
决策：`continue`，先审计 tight vs protocol success 标签，不调 deadband 阈值

## 1. 现象

Goal75 的 deadband bias correction（只修正 `b > 0.25 m`）在两个 scenario
都把 false-success 降到 0：

| scenario | base false | deadband false | true retention |
|---|---:|---:|---:|
| default | 31 | 0 | 95.88% |
| strict125 | 14 | 0 | 93.18% |

default 通过预注册 `>=95%` 的 true retention Gate；strict125 差 3 个样本，
因此整体只能判 `PARTIAL_SUPPORT`。不能事后把阈值改成 93% 或扫新的 deadband。

## 2. 证据

| scenario | base true-success | 其中 eta 在 tight 内 | deadband 误伤 | 其中真正 tight loss |
|---|---:|---:|---:|---:|
| default | 97 | 59 | 4 | 0 |
| strict125 | 44 | 25 | 3 | 1 |

strict125 被误伤的 3 个样本：

| seed | checkpoint | bias projection (m) | p_nav distance (m) | eta distance (m) | eta tight? |
|---:|---:|---:|---:|---:|---|
| 46 | 240 | 0.5014 | 0.9567 | 1.2127 | 否 |
| 57 | 240 | 0.2802 | 0.9833 | 0.9557 | 是 |
| 53 | 1008 | 0.3309 | 0.9863 | 1.1456 | 否 |

关键观察：当前 `true_success` 由 protocol radius（`2.00/1.25 m`）定义，
而实际 success 由 tight radius（`1.75/1.00 m`）定义。一个 episode 可以在
`p_nav` tight success 的同时，true `eta` 只在 protocol radius 内。这种
“protocol-only true”在 default 有 38/97、strict125 有 19/44。

## 3. 已排除

- 不是 deadband 没效果：false-success 在两个 scenario 都降为 0；
- 不是 full correction 的替代问题：full 在两个 scenario 都误伤太多
  true-success（53/97、15/44），deadband 明显更稳；
- 不是 NoisyNet：正式评估 deterministic mean，偏差是几何的；
- 不是需要换 SAC/REDQ/DroQ：本轮没有任何 optimization/exploration 证据。

## 4. 可证伪假设

### H76-A：strict125 的 Gate 失败主要是标签语义混淆，不是修正方向错误

预测：如果把 true 标签改为 `d_e <= r_tight`，deadband 的 true retention
在 default 与 strict125 都 `>=95%`，且 false-success 仍为 0。

最便宜 falsification：只读重算现有 Goal75 records 的 tight-label success，
不新增 rollout、不训练。

### H76-B：高 `b` 的 true 与 false 可以用现有可观测 trace 特征区分

预测：在 `b > 0.25 m` 的样本中，true 与 false 在 residual/action/path/
terminal 几何等已有特征上存在稳定分离；若没有分离，则不能安全地做
learnable gating。

最便宜 falsification：复用现有 rich trace，做只读特征对比；不训练 predictor。

## 5. 下一步

预注册 Goal76：

1. 固定 `protocol_capture` 与 `perfect_tight_success` 两个标签；
2. 用 tight 标签重新评估 no-correction/full/deadband；
3. 只读检查高 `b` true vs false 的已有特征可分性；
4. 只有 H76-A 与 H76-B 至少一个成立，才允许预注册 predictor/training arm；
5. 继续关闭 held-out `100–149`，不改 checkpoint、seed、eval split。

## 6. 文献与本地记录

- `research/paper_reading/2026-11-10-goal73-terminal-capture-estimator-bias.md`
- `doc/goals/goal75_result_20261110.md`
- `artifacts/lowlevel/goal75_oracle_bias_correction_20261110/analysis/goal75.json`

## 7. 决策

`decision = continue`

不调 deadband 阈值，不换 RL 算法，不启动训练。下一步只做 label semantics
与 observable gating 的预注册只读审计。
