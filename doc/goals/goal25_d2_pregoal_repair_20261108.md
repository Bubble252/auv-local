# Goal25：D2 pre-goal 精度修复与条件式 PPO

日期：2026-11-08

状态：**已完成并 FAIL 收口：D3 pre-goal 门禁失败；stress 诊断与 PPO 均未启动**

## 1. 背景

Goal24 完成了 teacher-fixed 审计和 D0/D1/D2 数据配方对照。D2（targeted DAgger）
在 40 个 dev seeds 上取得：

- default success `0.7250`，比 D0 提升 `0.15`；
- strict125 success `0.3750`，比 D0 提升 `0.075`；
- default pre-goal RMSE `2.7883 m`，比 D0 差 `1.89%`；
- strict125 pre-goal RMSE `2.7648 m`，比 D0 差 `1.62%`。

D2 通过了 Goal24 Phase 2 的数据配方门禁，但没有达到最终 BC 建议目标：

- default pre-goal 目标 `<=2.65 m`，未达到；
- strict125 pre-goal 目标 `<=2.70 m`，未达到；
- strict125 success 目标 `>=0.40`，未达到。

Goal24 Phase 3 随后完成了 S10/S20 stress 增广，但两者都未通过预注册门禁：

- S10 nominal pre-goal RMSE 分别变差 `+2.85%/+2.98%`；
- S20 分别变差 `+5.00%/+4.94%`；
- S10 stress success 为 `-0.05/-0.125`，S20 为 `+0.025/-0.025`，均未达到收益门槛；
- safety 没有增加，但不足以补偿 nominal 精度和 stress 收益失败。

因此 Goal24 记录为 `phase3_stress_status=failed`、`phase4_ppo_status=ppo_not_started`。
本 Goal 不在未修复 BC 底座的情况下强行启动 PPO，而是先做一次受控修复；已经失败的
stress 数据增广轴关闭，不再重跑 S10/S20 或扫描其他比例。

## 2. 技术栈与固定契约

- 载体：`settled_局部规划器_反应式`；
- 动力学：`batch_torch_remus100`，不依赖 IsaacLab/IsaacSim；
- 控制：ALOS/PC teacher-fixed，`pc_control_mode=alos`；
- actor 输入：8D path，不加入 3D command；
- 网络：state `128,128`、actor `256,128`、`d_z=64`、path encoder `64`；
- 训练：PyTorch/CUDA，batch `4096`，seed `0`；
- 评估：CPU/float64，`endpoint_stop`，dev seeds `20–59`；
- 主指标：`pre_goal_3d_rmse_m`，成功口径为 `p_nav`；
- 诊断指标：完整 RMSE、success-only RMSE、plane miss、timeout、safety、
  material saturation、path-zero 退化；
- 不碰 held-out `100–149`，不碰 dev 之外的评估 seed。

## 3. 科学假设

D2 的 targeted DAgger 样本提升了失败恢复和终点捕获，但从零训练会牺牲一部分路径
跟踪精度。先使用 D0 teacher-only checkpoint 建立稳定的路径跟踪特征，再在 D2 数据上
低学习率微调，可能在不损失 success 的情况下把 pre-goal RMSE 拉回目标范围。

本 Goal 只验证这一条假设，不同时改网络、奖励、动作头、路径维数或 DADE。stress
仅复用现有 profile 做只读诊断，不进入训练配方。

## 4. Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal25 --goal-doc doc/goals/goal25_d2_pregoal_repair_20261108.md --strict
```

- **Research question**：在固定教师、网络和 8D 路径契约下，`D0 初始化 + D2 数据低学习率微调`
  是否能在不放宽 success 的前提下，把 `pre_goal_3d_rmse_m` 拉回强 BC 门禁？
- **Hypothesis H1**：D0 提供更稳定的路径跟踪特征，D2 提供失败恢复和终点捕获样本；
  D3 应同时保留 D2 的 success 收益并降低 pre-goal RMSE。
- **Falsification**：若 D3 未达到预注册门禁，或改善只来自单个 seed，则 H1 被证伪；
  停止该初始化微调轴，不扫 lr、epoch、数据比例或 seed。
- **Primary metric**：dev seeds `20–59` 上的 `pre_goal_3d_rmse_m`；success 口径为
  `p_nav`。完整 RMSE、success-only RMSE、plane miss、timeout、safety、saturation
  和 path-zero 退化只作诊断。
- **Evaluation contract**：`default/strict125 × normal/path_zero` 四格矩阵，
  CPU/float64，`endpoint_stop`，path_dim 8；不得临时更换 seed 或指标。
- **Data contract**：D0 checkpoint 和 D2 merged 数据只读；D3 的全部输出写入
  `goal25_*` 新目录；不碰 held-out `100–149`，不重跑 S10/S20。
- **Budget contract**：最多一次 D3 微调、四格评估、一次 stress 只读诊断；只有 D3
  全部 nominal 门禁通过后，才允许一次 zero-residual 检查和 1-seed PPO screen。
- **Negative-result contract**：门禁失败即写负结果并停止，不以事后门禁或额外 seed
  追结果；结论写入 `research/LEDGER.jsonl`。

## 5. 执行计划

### Phase 0：审计与资源检查

- [x] 记录 `ps`、`nvidia-smi`、`df -h /`、`df -h /dev/shm`、`git status --short`；
- [x] 复核 D2 checkpoint SHA256 和 Goal24 报告一致；
- [x] 确认 D2 的最终 BC 目标未全部达标；
- [x] 复核 `goal24_stress_gate.json` 为 `arms.s10.overall=false`、
  `arms.s20.overall=false`、顶层 `overall=false`；
- [x] 在 audit 中记录“stress 数据增广轴关闭，不重跑 S10/S20”；
- [x] 新建 `/dev/shm/pc_v1_goal25_d2_repair_20261108/`；
- [x] 记录外部 world-model 训练占用；未干扰、未重启；
- [x] 不覆盖任何旧配置、数据、checkpoint、manifest 或 summary。

### Phase 1：D2 与 D0 只读归因

- [x] 对 normal 输入的 default/strict125 做逐 seed 配对比较；
- [x] 统计 success、pre-goal RMSE、plane miss、final progress、saturation；
- [x] 找出 D2 改善最多和恶化最多的 seed；
- [x] 增加默认关闭的 trace 诊断能力，按 `0–25/25–50/50–75/75–100%` 路径进度
  分段统计三维路径误差；
- [x] 不改变旧评估默认行为；
- [x] 写 `goal25_phase1_d2_failure_attribution_20261108.md`。

### Phase 2：D3 初始化微调

- [x] 给 `scripts/train_pc_bc.py` 增加可选 `--init-checkpoint`；
- [x] 只加载模型权重，不加载旧优化器状态；
- [x] 缺 key、shape 不匹配或 path encoder alpha 未恢复时立即失败；
- [x] 以 D0 checkpoint 初始化，在 D2 merged 数据上微调；
- [x] 固定 lr `2e-4`、max epochs `120`、patience `20`，其余配方与 D2 一致；
- [x] 训练 manifest 写初始化路径、SHA256、微调 lr 和权重恢复审计；
- [x] 跑四格 40-seed dev 矩阵：default/strict125 × normal/path-zero；
- [x] 与 D0、D2 做 40-seed 主表和逐 seed 配对统计；
- [x] 写 `goal25_d3_repair_report_20261108.md`。

D3 判定：**FAIL**。default/strict125 pre-goal RMSE 分别为 `2.849170/2.829135 m`，
均超过门禁；success 和 safety 项通过但不能抵消主指标失败。

D3 门禁：

- default pre-goal `<=2.65 m`；
- default success `>=0.70`；
- strict125 pre-goal `<=2.70 m`；
- strict125 success `>=0.40`；
- 相对 D2，success 下降不超过 `0.03`；
- safety 增量不超过 `0.02`；
- path-zero 相对 normal 至少退化 `5%`；
- 无 NaN/Inf，manifest 完整。

### Phase 3：stress 只读诊断（未启动）

状态：**not_started_by_gate**。D3 nominal 门禁失败，因此按合同不执行。

Goal24 已证明在 D2 上加入 10%/20% stress 教师数据会破坏 nominal 精度，且没有带来
足够的 stress 收益。本阶段不训练，只评估 D3 在已有 `realistic_stress` profile 下的
表现。

- [ ] 复用现有 stress 评估矩阵，只替换为 D3 checkpoint；
- [ ] dev seeds 固定 `20–59`，不新增训练 seed；
- [ ] 不重新采集 stress 数据，不训练 S10/S20，不新建 stress teacher 配置；
- [ ] 记录 D2/D3 的 stress success、pre-goal RMSE、full RMSE、plane miss、timeout、
  safety 和 saturation；
- [ ] stress 结果是诊断项，不作为 PPO 的启动门禁；
- [ ] 若 stress 明显崩溃，必须在报告中列为风险，不得宣称鲁棒性已解决；
- [ ] 写 `goal25_stress_diagnostic_20261108.md`。

### Phase 4：残差 PPO（未启动）

状态：**ppo_not_started**。没有可选 BC 底座，不执行。

仅当 D3 的 nominal 门禁全部通过后执行。stress 诊断不阻塞 PPO。

- [ ] zero-residual 基线与同 harness frozen-base 完全一致；
- [ ] 检查残差初始值、BC 权重漂移和 path encoder alpha；
- [ ] 固定 `4096 env × 128 steps × 200 iterations`；
- [ ] 先跑 1 个训练 seed 的 screen；
- [ ] screen 通过后才扩到 3 个训练 seed；
- [ ] DADE 关闭，残差尺度固定，不扫参；
- [ ] 同一 40-seed dev 矩阵评估；
- [ ] 若失败，写 `ppo_not_started` 或 `ppo_failed` 并停止；
- [ ] 写 `goal25_ppo_report_20261108.md`。

PPO screen 门禁：

- 相对 D3，至少一个主场景的 pre-goal RMSE 改善 `>=0.10 m`，或 success 提升
  `>=0.05`；
- 另一主场景 success 下降不超过 `0.03`，pre-goal RMSE 恶化不超过 `2%`；
- safety 增量不超过 `0.02`；
- 完整 RMSE 只作诊断，不能替代 pre-goal RMSE 门禁；
- 若 screen 失败，写 `ppo_failed` 并停止，不扩 3 seeds。

## 6. 停止条件

- 初始化 checkpoint 不能完整恢复；
- 训练/评估出现 NaN/Inf；
- D3 未过门禁；
- 改善只来自单个 seed；
- 试图重跑已失败的 S10/S20 或扫描新的 stress 比例；
- stress 诊断出现明显崩溃时仍声称鲁棒性已解决；
- 需要 held-out 才能决策；
- 资源冲突或可能覆盖旧产物；
- 试图通过扫 lr、epoch、数据比例、seed 或降低门禁来追结果。

## 7. 产物目录

```text
/dev/shm/pc_v1_goal25_d2_repair_20261108/
  audit/
  traces_diagnostic/
  bc_d3/
  eval_matrix_d3/
  eval_stress_d3/
  ppo/
  comparison.json
  comparison.md
  run_manifest.json
```

报告和 manifest 必须包含 source、seed、sensor profile、config hash、episode id、
checkpoint 和 SHA256。

## 8. 验证命令

预期测试：

```bash
uv run pytest -q \
  tests/test_lowlevel_pc_bc.py \
  tests/test_lowlevel_pc_eval_contract.py \
  tests/test_lowlevel_pc_eval_matrix.py \
  tests/test_pc_goal24_dagger_seeds.py

uv run ruff check \
  scripts/train_pc_bc.py \
  scripts/evaluate_pc_bc.py \
  scripts/run_pc_eval_matrix.py \
  src/auv_lowlevel/batch/pc_bc.py \
  tests/test_lowlevel_pc_bc.py
```

新增初始化行为必须有单元测试，且不带 `--init-checkpoint` 的旧训练路径必须保持
原有默认行为。

## 9. Git 记录

完成后先检查：

```bash
git status --short
git diff --stat
```

只暂存本次明确修改的文件，不使用 `git add -A`：

```bash
git add \
  scripts/train_pc_bc.py \
  tests/test_lowlevel_pc_bc.py \
  doc/goals/goal25_d2_pregoal_repair_20261108.md \
  doc/goals/goal25_d2_pregoal_repair_prompt_20261108.md \
  doc/goals/goal25_phase1_d2_failure_attribution_20261108.md \
  doc/goals/goal25_d3_repair_report_20261108.md \
  doc/goals/goal25_stress_diagnostic_20261108.md \
  doc/goals/goal25_ppo_report_20261108.md

git commit -m "feat(pc): add goal25 d2 pregoal repair pipeline"
```

推送只在用户明确要求时执行。若当前不是受控 git 工作区，只记录文件清单和 SHA256。

## 10. 最终报告

用中文给出：

1. D2 失败归因；
2. D3 数据、初始化、训练和哈希；
3. D0/D2/D3 四格 40-seed 对比；
4. D3 门禁逐项判定；
5. D3 相对 D2 的 stress 只读诊断；
6. PPO 是否启动及 screen/3-seed 结果；
7. 产物路径、SHA256 和未启动项；
8. 未使用 held-out、未扩 seed、未扫超参的声明。
