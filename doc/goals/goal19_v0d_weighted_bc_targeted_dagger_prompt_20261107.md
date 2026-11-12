# Goal19 V-0D：Weighted BC 定向 DAgger 补数据与非破坏性复验

日期：2026-11-07

状态：**已执行并收口。判定 `FAIL_RMSE`。**

优先级：**当前最高。** 高于 Goal17 的 current estimate × crab 因子实验，也高于
action chunking / mixture-density 动作头研究。

关联文档：

- `doc/goals/goal19_v0d_weighted_bc_targeted_dagger_report_20261107.md`
- `doc/goals/goal18_v0d_sign_magnitude_bc_report_20261107.md`
- `doc/goals/goal9_v0d_r2b_saturation_bc_20261107.md`
- `doc/goals/goal16_v0d_crab_rate_limited_teacher_report_20261107.md`
- `doc/goals/goal17_v0d_current_crab_factorial_plan_20261107.md`
- `doc/goals/goal9_execution_state.md`

执行结果：default 达到 `6.6236 m / 0.70`，但 strict125 为
`10.5982 m / 0.25`，超过预注册的 `10.5 m` 上限。按第 7 节判定
`FAIL_RMSE`，未冻结新 BC，未进入残差 PPO。

## 1. 要回答的问题

在**不改变 actor 观测契约、不改变动作头、不重新扫描网络和损失超参数**的前提下，
只给当前最好的 weighted BC 补一轮针对失败轨迹的 DAgger 示范，能否把 BC 从：

- default `7.4591 m / success 0.65`；
- strict125 `9.7894 m / success 0.45`；

推进到：

- default 3D RMSE `<= 7.0 m`；
- default success `>= 0.60`；
- strict125 3D RMSE `<= 10.5 m`；
- strict125 success `>= 0.10`；
- safety 不恶化。

如果通过，才冻结新 BC 并进入残差 PPO screen。否则本 Goal 停止，不继续在同一批
结果上调权重、阈值、seed 或门禁。

## 2. 为什么做这个，而不是继续 Goal18

Goal18 已经证明 sign/magnitude 动作头不是当前修复方向：

- S1 default RMSE `10.2868 m`，比连续动作 S0 的 `8.9129 m` 恶化约 `15.4%`；
- S1 strict125 RMSE `11.7268 m`，未达到 `10.5 m`；
- sign accuracy 约 `0.95`，方向分类不是主要问题；
- 独立 sigmoid magnitude 没有恢复教师的高幅值动作；
- 动作反转率和 rate-limit 命中率明显恶化。

同时，weighted BC 的新增动作审计表明：

- 它的饱和恢复率并不高，但仍是当前闭环 RMSE 最好的可部署 BC；
- 大幅动作反转接近零，时间抖动不是它的主要瓶颈；
- 因此“必须恢复教师饱和动作才能取得好闭环效果”不成立。

本 Goal 的假设是：

> 当前 weighted BC 的剩余误差主要来自学生自身访问到、但原始示范覆盖不足的失败
> 状态分布；补这些状态上的教师动作，比继续改变动作参数化更直接。

## 3. 固定条件

以下内容在 baseline 和 retrain 中必须完全一致：

- student 起点：
  `artifacts/lowlevel/pc_v0d_r2b_saturation_bc_20261107/bc_satw/pc_bc_actor.pt`；
- 原始数据：
  `artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz`；
- 原始数据 SHA256：
  `eb21837960f89c5295af4946bd3fb914471602f6cbd15a33f40aa9ecdb2d1387`；
- actor 观测：12D base state + 8D path，`command_dim=0`；
- 动作契约：连续 `[rudder, stern, rpm]`，`action_head=continuous`；
- `d_z_total=64`、`n_in_reserved=8`、`n_z_reserved=4`；
- state encoder `[128,128]`、actor trunk `[256,128]`、path encoder `[64]`；
- 训练配方：
  `batch_size=4096`、`max_epochs=300`、`patience=30`、`lr=1e-3`、
  `weight_decay=1e-5`、`validation_fraction=0.15`、`seed=0`；
- weighted loss：`saturation_weight=2.0`、`saturation_threshold=0.9`；
- 教师：R2B；保持 `compensate_current=false` 和既有 crab/current 契约；
- 评估：CPU、float64、显式 `--path-dim 8`；
- 不改变 plant、sensors、current、reference、reward、termination、目标半径和成功定义。

本 Goal **不测试**：

- sign/magnitude、MDN、action chunking；
- current estimate / crab 因子矩阵；
- DADE、历史编码或世界模型；
- 新的奖励函数；
- 第三组教师 rate；
- PPO 算法或残差接口改造。

## 4. 防止在评估 seed 上过拟合

固定划分：

- **held-out test seeds：`0–19`**。保留为最终比较口径，不用于选择失败 seed；
- **development seeds：`20–39`**。只用于诊断当前 BC 的失败轨迹和采集新 DAgger 数据。

执行顺序：

1. 用当前代码在 test seeds `0–19` 和 dev seeds `20–39` 上分别重评 weighted BC；
2. 用 dev seeds 的失败情况选择最多 `7` 个目标 seed；
3. 新 DAgger 只在这些 dev seed 上采集；
4. 最终仍只在 `0–19` 上报告主结果。

目标 dev seed 的预注册选择规则：

- 先选 dev 中出现 timeout 或 safety 的 seed；
- 若不足 `7` 个，再按 default 3D RMSE 从高到低补足；
- 选择结果写入 `target_seeds.json` 后，不再根据训练结果改选。

不得把 test seeds `0–19` 的失败 seed 直接用于补数据，否则最终结果只能标为
`diagnostic_leaky`，不能作为泛化证据。

## 5. 数据采集设计

### 5.1 Student 和 teacher

- student：上面的 weighted BC checkpoint；
- teacher：R2B，配置沿用
  `configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json`；
- student 自己执行动作，teacher 只给访问到的状态打标签；
- `action_noise_std=0.0`，不引入额外探索噪声；
- 每个目标 dev seed 采 `4096` 步，`num_envs=1`，确保 seed 精确对应；
- 若目标 seed 少于 `5` 个，允许补充 dev 中 RMSE 最高的 seed，但总数仍不超过 `7`。

### 5.2 合并规则

新数据目录与原始数据目录必须分开保存，禁止覆盖：

```text
artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/
  baseline/
  dev_diagnosis/
  dagger_seed_<seed>/
  demos_merged_dagger2/
  bc_satw_dagger2/
  eval/
```

训练数据为：

```text
原始 demos_dagger_merged/demos.npz
+ 本轮最多 7 个 dagger_seed_*/demos.npz
```

合并后必须记录：

- 每个输入数据集路径和 SHA256；
- 总样本数、episode 数；
- 新增样本占总样本比例；
- 新旧 episode id 是否冲突；
- 输入 manifest 和输出 manifest。

## 6. 训练设计

只训练 **一组** `weighted BC + targeted DAgger`，不得并行扫描：

- 不扫 `saturation_weight`；
- 不扫 `saturation_threshold`；
- 不扫 learning rate；
- 不扫 network width；
- 不换 seed；
- 不做 early-epoch 人工挑选。

训练命令以现有
`scripts/run_pc_v0d_r2b_weighted_bc_20261107.sh` 为模板，复制成新的非破坏性
runner，或直接复用完全相同的参数：

```bash
python scripts/train_pc_bc.py \
  --demos <merged_dagger2/demos.npz> \
  --output <bc_satw_dagger2> \
  --state-encoder-hidden-sizes 128,128 \
  --actor-hidden-sizes 256,128 \
  --d-z-total 64 \
  --n-in-reserved 8 \
  --n-z-reserved 4 \
  --activation elu \
  --batch-size 4096 \
  --max-epochs 300 \
  --patience 30 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --validation-fraction 0.15 \
  --seed 0 \
  --device cuda \
  --actor-command-dim 0 \
  --path-dim 8 \
  --path-encoder-hidden-sizes 64 \
  --path-encoder-alpha 1.0 \
  --saturation-weight 2.0 \
  --saturation-threshold 0.9
```

必须保存：

- `train_command.txt`；
- `training_manifest.json`、`run_manifest.json`；
- checkpoint SHA256；
- best epoch、early-stop epoch；
- weighted/unweighted validation loss；
- 原始数据和新数据的 SHA256。

## 7. 评估与门禁

### 7.1 三组闭环评估

对新 BC 评估：

- default：`odarrl_control_pc_c3_random_realistic_v0d_learned_speed.json`；
- strict125：`odarrl_control_pc_c3_random_realistic_v0d_r0_current_strict125.json`；
- path-zero：default 配置加 `--path-zero`；
- test seeds：`0–19`；
- CPU、float64、`--path-dim 8`。

### 7.2 主门禁

必须同时满足：

- default 3D RMSE `<= 7.0 m`；
- default success `>= 0.60`；
- strict125 3D RMSE `<= 10.5 m`；
- strict125 success `>= 0.10`；
- default/strict125 safety 不高于 baseline `+0.02`；
- path-zero 相对正常 path 至少恶化 `5%`，证明路径输入仍在被使用；
- 不出现新的 NaN、非有限动作或评估器异常。

### 7.3 诊断指标

以下只记录，不作为本 Goal 的成败门禁：

- rudder/stern sign accuracy；
- 饱和率、饱和召回；
- 大幅反转率；
- rate-limit 命中率；
- predicted zero fraction；
- 每个 seed 的 RMSE、success、timeout、safety；
- 新 BC 与 weighted BC 的逐 seed 配对差值。

### 7.4 判定

- **PASS**：全部门禁满足，冻结新 checkpoint，进入残差 PPO screen；
- **FAIL_RMSE**：RMSE 或 success 未达标，本 Goal 停止；
- **FAIL_GUARDRAIL**：RMSE 达标，但 safety、path-zero 或有限性失败；
- **FAIL_DATA**：数据合并、seed 隔离或 manifest 审计失败，结果不得用于科学结论。

若为 FAIL，不得在本 Goal 内改门禁后重跑。下一步另开 Goal 研究 action chunking；
MDN 更后。

## 8. 阶段与检查项

### 阶段 0：环境与产物审计

- [x] 确认服务器没有残留 teacher/BC/PPO/eval runner；
- [x] 记录服务器仓库路径、Python 路径和当前代码状态；
- [x] 确认 `bc_satw/pc_bc_actor.pt` 存在且 SHA256 可复算；
- [x] 确认原始数据 SHA256 与本文一致；
- [x] 新建独立结果根目录，不覆盖 Goal9/16/18 的任何 artifact；
- [x] 记录当前 checkpoint 的 metadata、`path_dim` 和 `actor_command_dim`。

### 阶段 1：baseline 重评

- [x] 用当前代码在 test seeds `0–19` 重评 weighted BC 的 default/strict125/path-zero；
- [x] 用当前代码在 dev seeds `20–39` 重评 weighted BC 的 default；
- [x] 保存三份 test summary 和一份 dev summary；
- [x] 逐 seed 列出 success、timeout、safety、3D RMSE；
- [x] 写 `baseline_comparison.json`，明确当前口径与此前 `7.4591 m` 的差异；
- [x] 若 test 口径变化明显，后续只与本次 baseline 比较，不混用旧数值。

### 阶段 2：选择目标 dev seed

- [x] 按第 4 节规则从 dev seeds `20–39` 选出最多 7 个目标 seed；
- [x] 写 `target_seeds.json`，记录每个 seed 的 baseline RMSE、timeout、safety；
- [x] 记录选择时间、规则和数据分布；
- [x] 选择完成后冻结，不因新 BC 结果改选。

### 阶段 3：定向 DAgger 采集

- [x] 每个目标 seed 独立调用 `collect_pc_bc_dagger_demos.py`；
- [x] 使用 weighted BC 作为 student、R2B 作为 teacher；
- [x] 固定 `num_envs=1`、`steps_per_env=4096`、`action_noise_std=0.0`；
- [x] 每个 episode id 唯一，不覆盖原始 episode id；
- [x] 记录每个数据集的 SHA256、样本数、episode 数和 correction L2；
- [x] 检查所有新数据集均为 `pc_v1`、8D path、`command_dim=0`。

参考调用：

```bash
python scripts/collect_pc_bc_dagger_demos.py \
  --checkpoint <bc_satw/pc_bc_actor.pt> \
  --config configs/lowlevel/odarrl_control_pc_c3_random_realistic_v0d_teacher_r2b_current.json \
  --output <dagger_seed_${SEED}> \
  --num-envs 1 \
  --steps-per-env 4096 \
  --seed-start ${SEED} \
  --episode-id-offset $((1000000 + SEED * 100)) \
  --action-noise-std 0.0 \
  --device cuda \
  --dtype float64
```

### 阶段 4：合并数据

- [x] 使用 `scripts/merge_pc_bc_demos.py` 合并原始数据和新数据；
- [x] 合并前检查所有输入 manifest 的观测契约一致；
- [x] 合并后检查 episode id 唯一、样本数等于输入之和；
- [x] 记录新样本占比；若新样本占比过高或数据分布异常，先停下审计；
- [x] 不覆盖原始 `demos_dagger_merged`。

参考调用：

```bash
python scripts/merge_pc_bc_demos.py \
  --inputs \
    artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz \
    artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/dagger_seed_*/demos.npz \
  --output artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/demos_merged_dagger2
```

### 阶段 5：同配方重训

- [x] 只训练一组 weighted BC；
- [x] 参数与第 6 节逐项一致；
- [x] 记录 checkpoint SHA256 和训练 manifest；
- [x] 在 dev seeds `20–39` 先做一次冒烟评估，确认没有加载或维度错误；
- [x] 不根据 dev 结果调参；若出现实现错误，只能修复实现并记录原因。

### 阶段 6：held-out test 评估

- [x] default、strict125、path-zero 全部在 seeds `0–19` 评估；
- [x] 记录逐 seed 指标和聚合指标；
- [x] 与 baseline 做逐 seed 配对比较；
- [x] 计算 RMSE、success、timeout、safety 的差值；
- [x] 输出 `stage_decision.json` 和 `stage_decision.md`；
- [x] 只按第 7 节门禁判定，不做事后放宽。

### 阶段 7：PASS 后进入残差 PPO screen

- [x] 本阶段未触发：最终判定为 `FAIL_RMSE`，不得进入 PPO；
- [ ] 仅当 `stage_decision` 为 PASS 时执行；
- [ ] 冻结新 BC checkpoint 并记录 SHA256；
- [ ] 复用既有冻结 BC + 残差 PPO 协议；
- [ ] 先跑 zero-residual 基线，确认与 BC 闭环一致；
- [ ] 再跑小预算 residual screen；
- [ ] 不在本 Goal 内启动正式 `3×50`，是否扩大由用户确认。

## 9. 非破坏性代码要求

执行者必须遵守：

- 优先只新增独立 runner、summary 和 goal 文档；
- 若需修改公共代码，新增参数必须默认保持旧行为；
- 不得改变 `action_head=continuous` 的默认输出；
- 不得改变旧 checkpoint 的加载和推理行为；
- 不得改变 PPO/frozen BC 接口；
- 不得覆盖已有 artifact 或日志目录；
- 不得删除、移动或重写 Goal9/16/18 的结果；
- 不得用 `git add -A`，不得自动 commit/push；
- 修改后至少运行相关 pytest 和 ruff；测试失败必须先报告，不得绕过。

建议验证：

```bash
python -m pytest -q tests/test_pc_bc.py tests/test_pc_bc_merge.py
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

若现有测试文件名不同，先用 `rg --files tests | rg 'pc_bc|dagger|merge'` 定位，
不得凭猜测跳过测试。

## 10. Git 与交付

服务器 checkout 没有 `.git`，服务器只保存 artifact 和日志。本地文档与代码修改由用户
决定是否提交；执行者不得自动 commit/push。

若用户明确要求本地提交，按阶段只添加明确文件：

```bash
git status --short
git diff --check
git add doc/goals/goal19_v0d_weighted_bc_targeted_dagger_prompt_20261107.md
git add <本次新增的脚本或测试文件>
git commit -m "docs: add Goal19 targeted DAgger protocol"
```

禁止：

```bash
git add -A
git commit -am ...
git push --force
```

每个阶段完成后在 `doc/goals/goal9_execution_state.md` 追加一条简短状态，写清：

- 当前阶段和是否完成；
- 产物目录；
- 关键数值；
- 是否有进程仍在运行；
- 下一步是继续、停止还是等待人工确认。

## 11. 产物清单

服务器根目录：

```text
artifacts/lowlevel/pc_v0d_r2b_weighted_bc_dagger_20261107/
```

至少包含：

```text
baseline/default/summary.json
baseline/strict125/summary.json
baseline/pathzero/summary.json
dev_diagnosis/default/summary.json
target_seeds.json
dagger_seed_<seed>/demos.npz
dagger_seed_<seed>/demos.manifest.json
demos_merged_dagger2/demos.npz
demos_merged_dagger2/demos.manifest.json
bc_satw_dagger2/pc_bc_actor.pt
bc_satw_dagger2/training_manifest.json
bc_satw_dagger2/run_manifest.json
eval/default/summary.json
eval/strict125/summary.json
eval/pathzero/summary.json
stage_decision.json
stage_decision.md
run_manifest.json
```

日志目录：

```text
logs/pc_v0d_r2b_weighted_bc_dagger_20261107/
```

报告建议写入：

```text
doc/goals/goal19_v0d_weighted_bc_targeted_dagger_report_20261107.md
```

## 12. 最终必须回答

报告中必须用表格回答：

| 问题 | 必须给出 |
|---|---|
| baseline 当前口径是多少？ | test `0–19` 的 default/strict125/path-zero RMSE、success、timeout、safety |
| 补了哪些 dev seed？为什么？ | seed 列表、选择规则、baseline 失败类型 |
| 新数据有多少？ | 样本数、episode 数、占合并后数据比例、SHA256 |
| 新 BC 是否通过？ | PASS / FAIL_RMSE / FAIL_GUARDRAIL / FAIL_DATA |
| 相比 weighted BC 改善多少？ | RMSE、success、timeout、safety 的配对差值 |
| 路径输入是否仍有效？ | path-zero 相对正常 path 的恶化比例 |
| 是否进入 PPO？ | 是/否；若否，明确停止原因 |
| 下一步是什么？ | 若失败，进入 action chunking；若通过，进入残差 PPO screen |

在用户确认结果前，不启动正式 `3 training seeds × 50 eval seeds`，不启动 Goal17，
不开始 MDN 或 action chunking。
