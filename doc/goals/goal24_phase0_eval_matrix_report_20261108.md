# Goal24 Phase 0 报告：统一评估矩阵

日期：2026-11-08  
状态：**PASS**  
范围：仅评估，不训练；未使用 held-out `100–149`

## 1. 产物

- 矩阵配置：`configs/lowlevel/eval/pc_v1_nominal_eval_matrix_20261108.json`
- 运行器：`scripts/run_pc_eval_matrix.py`
- 测试：`tests/test_lowlevel_pc_eval_matrix.py`、`tests/test_pc_goal24_dagger_seeds.py`
- 输出：`/dev/shm/pc_v1_nominal_eval_matrix_20261108/`
- 直接评估器等价性输出：
  `/dev/shm/pc_v1_matrix_direct_equiv_20261108/default_normal/`
- checkpoint：
  `/dev/shm/pc_v0d_nominal_bc_rebase_20261107/bc_N8/pc_bc_actor.pt`
- checkpoint SHA256：
  `9dd416a121f471c5b1aadafb9d591e8259b9803a8bbdf6650ce14e6b49664e36`

## 2. 固定评估口径

- dev seeds：`20–59`（40 seeds）
- 场景：`default`、`strict125`
- 路径模式：`normal`、`path_zero`
- 终止：`endpoint_stop`
- `path_dim=8`
- CPU / `float64`
- `sensor_profile=realistic_calibrated`

## 3. 矩阵结果（student）

| scenario | path | success | pre-goal 3D RMSE | full 3D RMSE | plane miss | timeout | safety | saturation |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default | normal | 0.5250 | 2.860484 | 2.847081 | 0.4750 | 0.0000 | 0.0000 | 0.1118 |
| default | path_zero | 0.0250 | 11.556313 | 9.724467 | 0.8250 | 0.1000 | 0.0500 | 0.0413 |
| strict125 | normal | 0.3250 | 2.842550 | 2.833678 | 0.6750 | 0.0000 | 0.0000 | 0.1107 |
| strict125 | path_zero | 0.0250 | 11.553919 | 9.724432 | 0.8250 | 0.1000 | 0.0500 | 0.0413 |

路径归因明显：

- default pre-goal RMSE 从 `2.860484` 退化到 `11.556313`，`+304.00%`
- strict125 pre-goal RMSE 从 `2.842550` 退化到 `11.553919`，`+306.46%`

## 4. 矩阵结果（teacher 诊断）

| scenario | normal success | normal pre-goal RMSE | path_zero success | path_zero pre-goal RMSE |
|---|---:|---:|---:|---:|
| default | 0.7500 | 2.720561 | 0.7500 | 2.720561 |
| strict125 | 0.4000 | 2.700380 | 0.4000 | 2.700380 |

注意：这里的 teacher 数字来自 Phase 0 的 N8 评估配置，只用于矩阵链路校验，不是
teacher-fixed 审计结论。strict125 teacher `0.40` 低于 Goal24 的 `0.45` 门禁，
必须由 Phase 1 的显式 teacher-fixed 配置重新判定。

## 5. 直接评估器等价性

用与矩阵 `default/normal` 完全相同的配置、checkpoint、40 seeds、`endpoint_stop`、
`path_dim=8`、CPU/float64，直接运行 `scripts/evaluate_pc_bc.py`：

- `student` 指标：逐字段完全一致
- `teacher` 指标：逐字段完全一致
- `student_episodes`：逐 episode 完全一致
- 两份 `summary.json` SHA256：完全一致

结论：矩阵包装器没有改变评估语义。

## 6. 静态检查

- `pytest -q tests/test_lowlevel_pc_eval_matrix.py tests/test_pc_goal24_dagger_seeds.py`
- 结果：`11 passed`

## 7. 判定

Phase 0 **PASS**。矩阵配置、seed 隔离、四格评估、直接评估器等价性和路径归因均已
验证。进入 Phase 1 teacher-fixed 审计；若 strict125 教师仍低于 `0.45`，按 Goal24
停止，不采集 BC 数据。
