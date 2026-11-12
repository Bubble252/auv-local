# Goal9 S15-B 阶段报告：决策频率

更新时间：2026-11-05

状态：**完成，B10 非劣效率候选，B5 负结果。** 本报告只复用服务器已有 artifact，没有
重训、没有改 seed、没有追加训练步数、没有修改门禁。

## 1. 实验问题

在物理积分、传感器、安全终止、动作速率限制和上下文物理时长保持 50 Hz 语义的前提下，
把 actor 决策频率从 50 Hz 降到 10 Hz（B10）或 5 Hz（B5），能否保持闭环性能并降低
推理成本。

## 2. 输入与预算

- 主环境：`configs/lowlevel/odarrl_control_pc_c3_random_realistic_v1_rewardscale001.json`
- 训练 seed：`91430`
- 训练预算：`128 envs × 32 iterations`
- 评估 seed：`1000–1019`
- 评估 tags：`default`、`strict125`、`grid_hs`
- B10：`actor_decision_stride=5`，`num_steps_per_env=51`，物理总步数 `1,044,480`
- B5：`actor_decision_stride=10`，`num_steps_per_env=26`，物理总步数 `1,064,960`
- B10 配对基线：`A1_stride5`
- B5 配对基线：`A2_stride10`

## 3. 结果

| arm | 状态 | default success 变化 | strict125 success 变化 | RMSE ratio | timeout 变化 | saturation 变化 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| B10 / 10 Hz | `pass_non_inferior` | `0.0` | `0.0` | `1.0014` | `0.0` | `+0.0028` | 效率候选 |
| B5 / 5 Hz | `fail_non_inferior` | `-0.05` | `-0.05` | `1.1153` | `+0.05` | `-0.0037` | 负结果，停止分支 |

B10 的时延门禁通过：决策周期 `100 ms`，p95 推理时延约 `0.8907 ms`，compute ratio
约 `0.1186`。这只说明 10 Hz 在给定评估集上非劣且更省算力，**不说明它相对 A0 有性能
提升**。

B5 在 default 和 strict125 上都出现 success 下降、RMSE 变差和 timeout 增加，按预注册
非劣门禁失败。禁止通过增加训练步数、换 seed 或降低门禁来恢复。

## 4. 判定

- `B10_status = pass_non_inferior`
- `B10_role = efficiency_candidate`
- `B5_status = fail_non_inferior`
- `B5_action = stop_5hz_branch`
- `selected_for_s15 = false`（B10 不与 A0 混成两个同时使用的 V 基线）

## 5. 证据路径

服务器根目录：

`artifacts/lowlevel/pc_abc_temporal_20261105/`

- B10 manifest：`B10/run_manifest.json`
- B10 summary：`B10/report/summary.json`
- B10 latency：`B10/report/latency.json`
- B5 manifest：`B5/run_manifest.json`
- B5 summary：`B5/report/summary.json`
- B5 latency：`B5/report/latency.json`

## 6. 后续

B 阶段不重跑。C 阶段按协议检查触发谓词；当前 A 无历史因果归因、B10 只是效率非劣、
B5 失败，因此 C 默认记录为 `C_not_triggered`。随后冻结唯一 S15 基线并进入 S16。
