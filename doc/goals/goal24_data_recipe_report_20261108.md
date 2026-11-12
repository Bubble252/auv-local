# Goal24 Phase 2 报告：D0/D1/D2 数据配方对照

日期：2026-11-08  
状态：**Phase 2 数据配方门禁 PASS；最终 BC 目标仅部分达标；Phase 3 stress 门禁 FAIL；Phase 4 未启动**  
范围：dev seeds `20–59`；未使用 held-out `100–149`

## 1. 结论

targeted DAgger 配方 D2 是三个 arm 中唯一满足全部 Phase 2 门禁的候选：

- 相对 D0，default success 从 `0.5750` 提升到 `0.7250`（`+0.15`）；
- 相对 D0，strict125 success 从 `0.3000` 提升到 `0.3750`（`+0.075`）；
- default/strict125 的 pre-goal RMSE 只分别变差 `+1.89%` 和 `+1.62%`，
  没有超过 strict125 的 `5%` 容忍线；
- safety 没有增加，path-zero 仍造成数量级退化，说明路径输入仍被策略使用。

但 D2 没有达到 Goal24 预注册的最终 BC 建议目标：

- default pre-goal RMSE `2.7883 m`，目标 `<= 2.65 m`：**FAIL**；
- strict125 pre-goal RMSE `2.7648 m`，目标 `<= 2.70 m`：**FAIL**；
- strict125 success `0.3750`，目标 `>= 0.40`：**FAIL**。

因此本次结论是：**D2 是明确胜出的数据配方，但不是可以强行送入残差 PPO 的强
BC 底座**。按 Goal24 的“部分改善不强行进入 PPO”规则，记录
`phase2_gate=PASS`、`final_bc_target=PASS_PARTIAL`、`phase4=ppo_not_started`。

## 2. 固定条件

- 教师：Phase 1 的 teacher-fixed default/strict125；
- 网络：state `128,128`、actor `256,128`、`d_z=64`、path encoder `64`；
- 训练：batch `4096`、lr `1e-3`、weight decay `1e-5`、seed `0`；
- saturation weight `2.0`、threshold `0.9`；
- 总样本：每个 arm 都是 `393,216`；
- 路径：8D；
- 评估：dev `20–59`（40 seeds）、`endpoint_stop`、CPU/float64；
- held-out `100–149`：未触碰。

## 3. 数据配方

| arm | 教师样本 | general DAgger | targeted DAgger | 总样本 | episodes | dataset SHA256 |
|---|---:|---:|---:|---:|---:|---|
| D0 | 393,216 | 0 | 0 | 393,216 | 380 | `60cd7046f94fb47bbc23bbedd20675bf4ea722adba459d250e849cf58578c5ec` |
| D1 | 352,256 | 40,960 | 0 | 393,216 | 380 | `ffaa2ec35a8a321548e7da00e24ef57e8eee363a92db047fcfc264a773bbd86e` |
| D2 | 311,296 | 0 | 81,920 | 393,216 | 377 | `fe7feeb6db0e9492212e405b2bd7780965f613188b458633644159c472577d91` |

三组都使用相同的 teacher-fixed 教师、训练 seeds、网络、优化器和样本预算；D1/D2
只替换对应比例的数据来源。

## 4. 40-seed dev 评估

### 4.1 normal 输入

| arm | scenario | success | pre-goal 3D RMSE | full 3D RMSE | plane miss | timeout | safety | saturation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | default | 0.5750 | 2.736606 | 2.690092 | 0.4250 | 0.0000 | 0.0000 | 0.199279 |
| D1 | default | 0.6750 | 2.918635 | 2.810263 | 0.3250 | 0.0000 | 0.0000 | 0.192349 |
| D2 | default | 0.7250 | 2.788330 | 2.710160 | 0.2750 | 0.0000 | 0.0000 | 0.215183 |
| D0 | strict125 | 0.3000 | 2.720555 | 2.672551 | 0.7000 | 0.0000 | 0.0000 | 0.198136 |
| D1 | strict125 | 0.4500 | 2.896128 | 2.796052 | 0.5500 | 0.0000 | 0.0000 | 0.195038 |
| D2 | strict125 | 0.3750 | 2.764764 | 2.693556 | 0.6250 | 0.0000 | 0.0000 | 0.213722 |

### 4.2 相对 D0 的变化

| arm | scenario | success delta | pre-goal RMSE delta | safety delta |
|---|---|---:|---:|---:|
| D1 | default | +0.1000 | +6.65% | +0.0000 |
| D2 | default | +0.1500 | +1.89% | +0.0000 |
| D1 | strict125 | +0.1500 | +6.45% | +0.0000 |
| D2 | strict125 | +0.0750 | +1.62% | +0.0000 |

## 5. 预注册门禁

| 门禁 | D1 | D2 |
|---|---|---|
| default pre-goal 改善 >=3% 或 success +0.05 | PASS（success +0.10） | PASS（success +0.15） |
| default success 不低于 D0 | PASS | PASS |
| strict125 pre-goal 不比 D0 差超过 5% | **FAIL（+6.45%）** | PASS（+1.62%） |
| strict125 success 不低于 D0 超过 0.05 | PASS（+0.15） | PASS（+0.075） |
| safety 增量 <=0.02 | PASS（+0.00） | PASS（+0.00） |
| path-zero 相对 normal 至少退化 5% | PASS | PASS |
| 总体 | **FAIL** | **PASS** |

D1 的失败原因很集中：strict125 的 pre-goal RMSE 超过 D0 `5%` 容忍线。D2 则以更小
的精度代价获得更高的 default/strict125 success，是唯一通过全部 Phase 2 门禁的
候选。

## 6. 最终 BC 建议目标

| scenario | metric | D2 | 目标 | 判定 |
|---|---|---:|---:|---|
| default | pre-goal RMSE | 2.7883 | <=2.65 | FAIL |
| default | success | 0.7250 | >=0.70 | PASS |
| strict125 | pre-goal RMSE | 2.7648 | <=2.70 | FAIL |
| strict125 | success | 0.3750 | >=0.40 | FAIL |

判定：**partial_improvement**。D2 的 success 收益是真实的候选收益，但 pre-goal
精度和 strict125 success 仍未达到强 BC 底座要求。

## 7. path-zero 诊断

| arm | scenario | normal pre-goal RMSE | path-zero pre-goal RMSE | delta |
|---|---|---:|---:|---:|
| D0 | default | 2.736606 | 5.799578 | +111.93% |
| D1 | default | 2.918635 | 5.184354 | +77.63% |
| D2 | default | 2.788330 | 17.738852 | +536.18% |
| D0 | strict125 | 2.720555 | 5.797585 | +113.10% |
| D1 | strict125 | 2.896128 | 5.184354 | +79.01% |
| D2 | strict125 | 2.764764 | 17.738852 | +541.60% |

D2 的 path-zero 退化最大，说明其路径分支仍被显著使用，而不是绕过路径输入。该结果
只作为路径依赖诊断，不是主指标。

## 8. 产物与哈希

- 数据与 checkpoint 根目录：`/dev/shm/pc_v1_data_recipe_20261108/`
- D0 评估：`/dev/shm/pc_v1_data_recipe_20261108/eval_matrix_d0/`
- D1 评估：`/dev/shm/pc_v1_data_recipe_20261108/eval_matrix_d1/`
- D2 评估：`/dev/shm/pc_v1_data_recipe_20261108/eval_matrix_d2/`
- 矩阵配置：`configs/lowlevel/eval/pc_v1_nominal_eval_matrix_teacher_fixed_20261108.json`
- 运行器：`scripts/run_pc_eval_matrix.py`
- D0 checkpoint SHA256：`c423b2fc2385db4c49c5cc78d95043e25bae3707cd67a28bc68584997388825b`
- D1 checkpoint SHA256：`5641cafcbefd5377a8866658467f6f6c2c734be8a25a95c55cf58a70e3bae6c5`
- D2 checkpoint SHA256：`087990ec8676e66e75777326c9035953aa97159c73b5b42480bfc601cbcc4131`

所有 summary、checkpoint 和数据哈希均已写入对应 `comparison.json`、
`training_manifest.json` 和 `demos.manifest.json`。

## 9. 验证

- `pytest -q tests/test_lowlevel_pc_eval_matrix.py tests/test_pc_goal24_dagger_seeds.py`
  为 `11 passed`；
- DAgger 定向 seed 测试为 `3 passed`；
- stress 配置、合并集和 subset 相关测试与 ruff 合计 `19 passed`，`ruff check` 通过；
- S10/S20 均为 `393,216` samples、`454` episodes、`path_dim=8`、
  `actor_command_dim=0`；
- held-out `100–149` 未运行，dev 之外的 seed 未用于本轮结论。

## 10. Phase 3/4 状态

- `phase3_stress_status = failed`；
- `phase4_ppo_status = ppo_not_started`；
- 原因一：最终 BC 目标只有部分达标，按预注册规则不得强行进入 PPO；
- 原因二：S10/S20 均未通过 Phase 3 联合门禁；详见
  `doc/goals/goal24_stress_report_20261108.md`；
- 未使用 held-out，未扩 seed，未扫超参，未覆盖旧产物。

Phase 3 在远端 8×4090 服务器完成；数据配方如下：

| arm | nominal teacher | targeted DAgger | stress | 总样本 | stress 占比 | dataset SHA256 |
|---|---:|---:|---:|---:|---:|---|
| S10 | 311,296 | 40,960 | 40,960 | 393,216 | 10.4% | `74b6afb050f592da9ca03816108959fe8f73362869e5fcc2c5cd4519629b8d71` |
| S20 | 229,376 | 81,920 | 81,920 | 393,216 | 20.8% | `f7427caf85baac5850d82c72799d561f65edb26b04ea8db984ac168120e495d0` |

- 本地数据根目录：`/dev/shm/pc_v1_stress_mix_20261108/`；远端正式运行根目录：
  `/data/yhx/underwater-robotics/src/plan_p1/goal24_stress_20261108/`；
- stress 原始数据：default SHA256 `ec06f5515806dc18a799eb412f92d077544c51caa03bc49075e916a4e292f6f7`，
  strict125 SHA256 `fd056c839b64d5b57cc5d54b3fd4db242dcdea8bc32c963eb96ce8ce79aa3812`；
- D2 stress 基线已完成：default normal success `0.4750`、pre-goal RMSE `3.122679 m`；
  strict125 normal success `0.2750`、pre-goal RMSE `3.112600 m`；

Phase 3 结果：

| arm | nominal default RMSE delta | nominal strict125 RMSE delta | stress default success gain | stress strict125 success gain | overall |
|---|---:|---:|---:|---:|---|
| S10 | +2.85% | +2.98% | -0.0500 | -0.1250 | FAIL |
| S20 | +5.00% | +4.94% | +0.0250 | -0.0250 | FAIL |

- 两个 arm 的 nominal pre-goal RMSE 均超过 D2 `2%` 容忍线；
- S20 的 success 提升不足以补偿精度代价；S10 在 stress 下反向退化；
- 正式结果位于
  `/data/yhx/underwater-robotics/src/plan_p1/goal24_stress_20261108/`；
- 因此保持 `phase3_stress_status=failed`、`phase4_ppo_status=ppo_not_started`。

## 11. 下一步

1. 保留 D2 作为 Goal24 中最佳数据配方，但记录其仍为 `partial_improvement`；
2. 停止当前 stress 比例轴，不扫描 15%/25%/30% 等未预注册比例；
3. 不启动残差 PPO；
4. 若继续，转到 Goal25 的 D2 pre-goal 修复或新的教师/动作表征假设；
5. 不得通过放宽 `2%` nominal 精度门禁来追 stress 结果。
