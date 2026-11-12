# Goal25 执行提示词：D2 收口、pre-goal 精度修复与条件式 PPO

日期：2026-11-08

状态：**已执行并 FAIL 收口；D3 的 pre-goal 门禁失败，stress 诊断与 PPO 未启动。本文保留为执行提示词。**

## 1. 最短版提示词

> 按 `doc/goals/goal25_d2_pregoal_repair_20261108.md` 自动执行。先审计
> `/dev/shm/pc_v1_data_recipe_20261108/` 中的 D0/D1/D2 结果，并确认 Goal24 Phase 3
> 的 S10/S20 已 FAIL、该 stress 数据增广轴已经关闭。D2 是 Phase 2 胜出配方，但
> default/strict125 的 pre-goal RMSE 和 strict125 success 仍未达到最终 BC 目标。
> 先做只读失败归因，再做一个受控修复：以 D0 teacher-only checkpoint 为初始化，在
> D2 targeted 数据上低学习率微调，形成 D3；用同一套 40-seed dev 矩阵与 D2 比较。
> D3 达标后直接进入条件式残差 PPO；stress 只复用已有 profile 做评估诊断，不再训练
> S10/S20、不扫描其他 stress 比例。D3 不达标则停止并写负结果。不碰 held-out
> `100–149`，不覆盖旧配置/数据/checkpoint，不使用 `git add -A`。

## 2. 一句话目标

在不更换教师、网络和 8D 路径契约的前提下，把 D2 的 success 收益保留下来，同时将
pre-goal RMSE 拉回强 BC 目标；只有这个底座过关，才允许启动残差 PPO。已失败的
stress 数据配方不重跑。

## 3. 前置事实

Goal24 已完成：

- teacher-fixed 审计 PASS：default teacher success `0.8250`，strict125 `0.6250`；
- D0/D1/D2 均为 `393,216` samples，同网络、同 seed、同 8D path；
- D2 是 Phase 2 胜出 arm：default success `0.7250`、strict125 success `0.3750`；
- D2 pre-goal RMSE：default `2.7883 m`、strict125 `2.7648 m`；
- D2 的 Phase 2 门禁 PASS，但最终 BC 目标为 **partial_improvement**；
- Goal24 Phase 3 已完成并 FAIL：S10 nominal RMSE 变差 `+2.85%/+2.98%`，S20
  变差 `+5.00%/+4.94%`；stress success 也未达到预注册增益；
- Goal24 已按预注册停止：`phase3_stress_status=failed`、
  `phase4_ppo_status=ppo_not_started`；
- 本 Goal 关闭 stress 数据增广轴，不重跑 S10/S20，不扫描 15%/25%/30% 等比例。

详细结果见：

- `doc/goals/goal24_data_recipe_report_20261108.md`
- `doc/goals/goal24_phase0_eval_matrix_report_20261108.md`
- `doc/goals/goal24_phase1_teacher_audit_report_20261108.md`
- `doc/goals/goal24_stress_report_20261108.md`

## 4. 不可破坏的约束

- 不删除、不重命名、不覆盖旧配置、旧 checkpoint、旧 dataset、旧 manifest 和旧
  evaluation；
- 新配置、新数据、新训练和新评估统一使用 `goal25_*` 或 `d3_*` 新名称；
- 不碰 held-out seeds `100–149`；
- dev 固定 `20–59`，失败池固定 `200–239`；
- 不扩普通教师数据量，不扫超参，不根据 dev 结果事后放宽门禁；
- 不修改 D2 的原始文件；D3 只新增；
- 不使用 `git add -A`；不自动 commit/push，除非用户明确要求；
- 启动长任务前先记录 `ps`、`nvidia-smi`、`df -h`、`git status --short`；
- 若 GPU/CPU 被其他任务占用、根磁盘不足或 `/dev/shm` 不足，先等待或写
  `blocked_by_resources`，不要把 local smoke 冒充正式结果。

## 5. Phase 0：资源与结果审计

- [ ] 读取 `/dev/shm/pc_v1_data_recipe_20261108/` 的 D0/D1/D2 manifest；
- [ ] 复核 D2 checkpoint SHA256：
  `087990ec8676e66e75777326c9035953aa97159c73b5b42480bfc601cbcc4131`；
- [ ] 复核 D2 四格 `comparison.json` 与 Goal24 报告一致；
- [ ] 确认 D2 的 default/strict125 pre-goal 和 success 目标未全部达标；
- [ ] 读取 `goal24_stress_gate.json`，确认 `arms.s10.overall=false`、
  `arms.s20.overall=false`、顶层 `overall=false`，并在 audit 中记录
  “stress 数据增广轴不重跑”；
- [ ] 记录当前资源；若三个 world-model 训练仍在占用 GPU/CPU，等待它们结束；
- [ ] 新建输出根目录：`/dev/shm/pc_v1_goal25_d2_repair_20261108/`；
- [ ] 不启动任何训练，先完成只读审计。

Phase 0 产物：

```text
/dev/shm/pc_v1_goal25_d2_repair_20261108/audit/
  goal24_consistency.json
  resource_snapshot.json
  input_hashes.json
```

## 6. Phase 1：D2 与 D0 的只读失败归因

不训练，只使用 dev `20–59` 的已有结果和 D2 checkpoint。

- [ ] 对 normal 输入的 default/strict125 分别比较 D0 与 D2 的逐 seed：
  - success、pre-goal RMSE、plane miss、final progress、saturation；
  - 找出 D2 变差最大的 seed 和变好最大的 seed；
- [ ] 判断 D2 的 pre-goal 退化是否集中在少数 seed；
- [ ] 判断退化是否伴随 plane miss、终点前饱和或路径进度停滞；
- [ ] 如果现有 summary 没有按路径进度分段的误差，给评估器增加默认关闭的
  `--write-trace` 或只读分析脚本，输出 `0–25%`、`25–50%`、`50–75%`、`75–100%`
  的路径误差；不得改变旧评估默认输出和指标；
- [ ] 用最多 10 个代表性 seed 做诊断，不碰 held-out；
- [ ] 写出 `doc/goals/goal25_phase1_d2_failure_attribution_20261108.md`。

Phase 1 的判定只用于决定修复方向，不用来改写 Goal24 的 D2 结论。

## 7. Phase 2：D3 单变量修复

### 7.1 假设

D2 从零训练时，targeted DAgger 样本改变了动作分布，带来 success 收益，但也让
pre-goal 路径误差略升。先用 D0 teacher-only checkpoint 学到稳定的路径跟踪底座，
再在 D2 数据上低学习率微调，可能同时保留两者。

### 7.2 固定配方

- 初始化：D0 checkpoint
  `/dev/shm/pc_v1_data_recipe_20261108/bc_d0/pc_bc_actor.pt`；
- 微调数据：D2 merged
  `/dev/shm/pc_v1_data_recipe_20261108/d2_merged/demos.npz`，SHA256
  `fe7feeb6db0e9492212e405b2bd7780965f613188b458633644159c472577d91`；
- 网络、path_dim=8、state/actor 宽度、`d_z=64`、batch `4096`、saturation
  weight `2.0`、threshold `0.9`、seed `0` 全部与 D2 一致；
- 微调 lr：`2e-4`；
- max epochs：`120`，patience：`20`；
- 不冻结任何模块；必须记录初始化权重是否完整恢复、path encoder alpha 是否恢复、
  优化器是否重新初始化；
- 输出目录：`/dev/shm/pc_v1_goal25_d2_repair_20261108/bc_d3/`；
- 若 `train_pc_bc.py` 不支持初始化 checkpoint，只新增可选参数
  `--init-checkpoint`，默认行为必须保持不变。

### 7.3 代码要求

- [ ] 初始化 checkpoint 只加载模型权重，不加载旧优化器状态；
- [ ] 校验 state encoder、actor、path encoder、path alpha 全部加载成功；
- [ ] 缺少 key 或 shape 不匹配时立即失败，不允许静默随机初始化；
- [ ] 训练 manifest 写入 `init_checkpoint_path`、`init_checkpoint_sha256` 和
  `fine_tune_lr`；
- [ ] 旧 `train_pc_bc.py` 不带该参数时行为不变；
- [ ] 补充初始化恢复测试和 CLI 默认行为测试。

### 7.4 评估

用 Goal24 同一矩阵配置的复制版本，只把 `output_root` 改为 D3 新目录：

```text
/dev/shm/pc_v1_goal25_d2_repair_20261108/eval_matrix_d3/
```

必须跑满四格：

- default normal/path-zero；
- strict125 normal/path-zero；
- dev `20–59`，CPU/float64，`endpoint_stop`，path_dim 8。

### 7.5 D3 门禁

D3 必须同时满足：

- default pre-goal RMSE `<= 2.65 m`；
- default success `>= 0.70`；
- strict125 pre-goal RMSE `<= 2.70 m`；
- strict125 success `>= 0.40`；
- 相对 D2，任一场景 success 下降不超过 `0.03`；
- safety 增量不超过 `0.02`；
- path-zero 相对 normal 至少退化 `5%`；
- 没有 NaN/Inf，manifest 完整。

若 D3 不通过，按以下规则停止：

- 不启动 PPO；
- 不重跑 S10/S20，也不扫其他 stress 比例；
- 保留 D2 作为“成功率更好但 pre-goal 不够强”的诊断候选；
- 写负结果报告，停止该初始化微调轴，不再扫 lr/epoch/数据比例。

## 8. Phase 3：stress 只读诊断（不训练）

Goal24 已证明在 D2 上加入 10%/20% stress 教师数据会破坏 nominal 精度，且没有带来
足够的 stress 收益。本 Goal 不再重复该数据轴。

- [ ] 仅将 D3 checkpoint 带入现有 `realistic_stress` 评估矩阵，使用 dev `20–59`；
- [ ] 不重新采集 stress 数据，不训练 S10/S20，不新建 stress teacher 配置；
- [ ] 记录 D2 与 D3 的 stress success、pre-goal RMSE、full RMSE、plane miss、
  timeout、safety 和 saturation；
- [ ] stress 结果是诊断项，不作为 PPO 的启动门禁；PPO 只看 D3 的 nominal 门禁；
- [ ] 若 D3 在 stress 下明显崩溃，必须在报告中列为风险，但不得用事后新门禁替代
  预注册 nominal 判定；
- [ ] 写 `doc/goals/goal25_stress_diagnostic_20261108.md`。

## 9. Phase 4：残差 PPO（条件式，最后才做）

只有 D3 达到 Phase 2 的全部最终 BC 目标，才允许启动。D2 或部分达标的 D3 都不能
直接进入 PPO。Phase 3 的 stress 诊断不阻塞 PPO。

- DADE 关闭；
- 残差尺度固定，不扫参；
- 先 zero-residual 基线；
- 再做 1 个训练 seed 的 screen；
- screen 通过后才扩到 3 个训练 seed；
- 评估使用同一 nominal 40-seed 矩阵，另加 stress 诊断；
- 正式预算：`4096 env × 128 steps × 200 iterations`；
- 服务器优先；若只能在本地运行，必须标记 `local_only`，不能冒充服务器正式结果；
- 主门禁看 `pre_goal_3d_rmse_m` 和 success，完整 RMSE 只作诊断；
- 若 screen 失败，停止，不扫 lr、residual scale、reward 权重或 seed。

PPO screen 门禁：

- 相对 D3，至少一个主场景的 pre-goal RMSE 改善 `>=0.10 m`，或 success 提升
  `>=0.05`；
- 另一主场景 success 下降不超过 `0.03`，pre-goal RMSE 恶化不超过 `2%`；
- safety 增量不超过 `0.02`；
- 不得以完整 RMSE 的改善替代 pre-goal RMSE 门禁；
- 若 screen 失败，写 `ppo_failed` 并停止，不扩 3 seeds。

PPO 启动前还要跑一次同 harness 的 frozen-base BC 对照，确认：

- zero-residual 时策略动作与 BC 完全一致；
- 随机初始残差不超过预注册上限；
- BC 权重和 path encoder alpha 没有漂移；
- 没有 NaN/Inf。

## 10. 产物目录

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

doc/goals/goal25_phase1_d2_failure_attribution_20261108.md
doc/goals/goal25_d3_repair_report_20261108.md
doc/goals/goal25_stress_diagnostic_20261108.md
doc/goals/goal25_ppo_report_20261108.md
```

报告和 manifest 必须包含数据、seed、配置、checkpoint、sensor profile 和 SHA256。

## 11. 立即停止条件

- D3 初始化 checkpoint 不能完整恢复；
- 训练/评估出现 NaN/Inf；
- dev 改善只来自单个 seed；
- D3 门禁未通过；
- 试图重跑已失败的 S10/S20 或扫描新的 stress 比例；
- stress 诊断出现明显崩溃时仍声称鲁棒性已解决；
- 需要 held-out 才能做决策；
- 资源冲突或可能覆盖旧产物；
- 尝试通过扫 lr、epoch、数据比例或 seed 来追 dev 结果。

## 12. 最终输出

用中文给出：

1. D2 失败归因：误差在哪些 seed、哪些路径阶段、哪些动作/终点状态；
2. D3 初始化与微调 manifest；
3. D0/D2/D3 的 40-seed 四格对比表；
4. D3 门禁是否通过；
5. D3 相对 D2 的 stress 只读诊断；
6. 是否启动 PPO；若未启动，明确写 `ppo_not_started`；
7. 全部产物路径和 SHA256；
8. 未使用 held-out、未扩 seed、未扫超参的声明。

不要自动 commit/push。完成后先给出 `git status --short` 和本次新增文件清单。

## 13. 最短执行顺序

```text
资源审计
  → D2/D0 只读归因
  → D0 checkpoint 初始化 + D2 微调出 D3
  → D3 四格 dev 门禁
  → D3 现有 stress profile 只读诊断
  → 通过才做 zero-residual 与 1-seed PPO screen
  → 通过才扩 3 seeds
```
