# Goal184 PPO 启动就绪审计

> **2026-11-18 后续复核**：传感器 v3、C5/C6 outage、stage-matched 7-stage demo、BC 重训、BC_ONLY、逐阶段 route-gate 阈值生产/消费以及训练产物审计接线均已完成；formal env/training config 的 warmstart 已更新到 `2292c195...`，`check_training_profile=PASS`，warmstart empirical schema 已匹配。审计器现在检查 C0–C6 provenance、20/3 gate、迭代数值和 empirical checkpoint metadata，失败非零退出。eval seeds 已按用户批准从 `90001–90003` 重冻结为 `20/21/22`，matrix/freeze checker 返回 PASS。仍未完成：实际 PE2E 标定/冻结、服务器 1-iteration smoke、用户选 B 的 75-it gate smoke。

日期：2026-11-18。范围：只读审计与本文记录；没有修改实验参数、启动训练或连接服务器核验实时状态。

依据：上一 session `01a1e778-ec61-7a11-a9b7-6e24b2fd304e` 的最后决策、当前 `goal182/infra-20261117` worktree（HEAD `3a78ffd`，含未提交规划）、Goal184、research/STATE.md 和实际代码。主仓库 main 不是本实验最新执行位置。

## 判断

还不能直接启动新协议的完整 PPO arm。最短依赖链是：v3/outage 实现 → C0–C6 stage-matched BC → 同制度逐阶段标定与阈值冻结 → 新 checkpoint 的 smoke → 首条完整 BCFT arm。已有 smoke 和旧阈值不能替代这些新协议交付。

没有证据要求重开 empirical 位置、扫描 LR/std 或等待 residual 证明有效；完整 BCFT 必做。ALOSB 的数据和阈值只在启动 ALOSB 前需要，不应阻塞第一条 PE2E。

## 启动前必须解决

- [x] **传感器与 C5/C6 outage 契约**。v3 已实现并接入 formal env；C5/C6 outage 0.05–0.3 s、无 transport delay，冻结最后有效估计；episode override、provenance、恢复、reset 和可复现测试均已覆盖。
- [x] **stage-matched BC 输入**。七阶段 demo 为 688128 samples / 594 episodes / seeds 91000+；formal env 已切到 v3，train config warmstart 已更新到 `2292c195...`，BC_ONLY 与 empirical schema 检查通过；旧 `ce4290` 只作历史证据。
- [x] **标定实现与逐阶段契约**。`scripts/calibrate_goal182_route_gate.py` 现在在每个 stage 内独立滑窗，输出 `statistics_by_stage` 和 `thresholds_by_stage`；`RouteGateThresholdTable` 按 stage 消费，缺 stage 时 fail-closed；训练入口拒绝非 `FROZEN` 表和多阶段旧单表。仍需用新 checkpoint 跑实际 7-stage 标定并冻结数值，冻结前验证 reference 自身的各阶段表现。
- [ ] **旧 reference 阈值不可复用**。旧表 cross 20.20 m / vertical 38.63 m / completion 0.80，reference 自身就不能通过；其 checkpoint、normalization 和传感器制度与新 treatment 不同。新表需记录 reference 来源、确定性/采样动作制度、profile、normalization、checkpoint/config SHA 和实际 burn-in 规则。新 BC reference 是否替代 canonical P0 作为 routine 来源，仍需明确记录，不能静默称为 P0。
- [x] **训练产物审计已接线**。`audit_goal182_artifacts.py` 新增 `--train-run`，复用现有审计器检查 completed/C6/TRAINING_ROLLOUT_20_3、C0–C6 逐 episode provenance、KL/clip/ratio/loss 有限性、final checkpoint SHA 与 empirical normalization metadata；失败非零退出。目标回归 19 项通过。

## 规划中还不够明确的地方

- [x] **重采执行规格已冻结**：7 stage × 24 envs × 4096 steps，每 stage 98304 samples，seeds `91000–91023`，共 688128 samples / 594 episodes；teacher 使用 clean/privileged 标签；逐 episode stage/outage provenance 写入 manifest。具体记录见 Goal184 §4.1。
- [x] **短 outage 的验收口径已拆分**：参数范围与单次时长由确定性测试覆盖；聚合 duty 用 C5 `0.0387%`、C6 `0.165%` 单独报告，不要求每个 episode 都触发 outage，也不因低频抽样波动改 seed。
- [x] **20 iteration 不证明晋级，已选 B**：C0 soft cap=72；按用户决定先独立跑至少 75 it 的 gate smoke，确认真实晋级后再启动 full arm。20-it 仅作 contract smoke。
- [x] **评价 seed 划分已重新冻结**：用户批准把 `90001–90003` 改为 `20/21/22`；新集合位于 formal env eval pool `0–49` 内且不进入训练流。matrix SHA `ac0dddb0...`，freeze manifest 记录了覆盖来源；held-out `100–149` 保持关闭。
- [ ] **canonical 规模的资源确认**：Goal184 已有的 smoke 是 1536 env，不能单独证明新 checkpoint 在 4096×48 下的内存和吞吐。新 checkpoint 必要 smoke 可在可用资源下兼顾 canonical 规模验证，不据此增加 seed/参数扫描。本次未查服务器空闲卡。
- [x] **执行文档歧义已清理**：STATE 顶部已指向 `2292c195` STAGE_MATCHED_BC；Goal184 和提示词已明确 formal env 相对 Goal182 允许两项差异：`normalization.mode=empirical` 与 `sensors.profile=unbiased_small_noise_v3`。旧 D4 的 full-rollout advantage 与新 Goal183 的 minibatch=true 仍是配方覆盖，报告必须注明，不能称为原 D4 完全同协议。

## 本次验证

运行环境：当前 worktree 的 `.venv/bin/python`。

```bash
.venv/bin/python -m pytest -q \
  tests/test_goal182_route_gate_calibration.py \
  tests/test_lowlevel_curriculum_trainer.py \
  tests/test_goal182_formal_configs.py \
  tests/test_goal182_eval_matrix.py \
  tests/test_goal182_eval_launcher.py
```

结果：当时为 **51 passed / 1 failed**，失败是 40 vs 100 iteration 测试漂移，后续已修正。2026-11-18 最新复核：Goal184 相关回归 **111 passed**，其中标定配置测试 **5 passed**（含逐 stage 抽样范围与 C5/C6-only outage）；范围化 `ruff`、`git diff --check`、7/7 profile、matrix、env/config load 和 Research Ledger 均通过。重冻结后的 seed-split checker 与 matrices checker 均返回 PASS。

canonical profile checker：**PASS**，Goal184 config 被识别为 `PPO-FLAT-4096x48-E5-MB4`。这不覆盖传感器、BC 有效性或 threshold 来源。

Goal184 strict ledger checker：**PASS，entries=0，提示 result/report pending**。这只证明当前文档契约可解析，不代表实验完成或就绪。

## 建议执行顺序

1. 完成传感器实现和七阶段采集规格，先冻结数据 split。
2. 重采/重训一次 PE2E BC，按阶段对照同环境 teacher，更新 SHA。
3. 逐阶段阈值接线已完成；下一步使用合法 reference 做必要标定，冻结规则与来源，旧表禁止加载。
4. 新配置经过启动检查和必要 smoke 后，执行 PE2E BCFT seed86201 完整课程；检查首次晋级，产物审计非零退出。
5. 主训练期间补齐独立评价矩阵和报告；第一条完成后再按既定顺序扩 seed/arm。

Git：本审计不 commit/push；后续若用户要求提交，只 add 本文对应路径并执行 `git diff --cached --check`。不提交实验产物。
