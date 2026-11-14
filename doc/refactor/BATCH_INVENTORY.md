# auv-local Batch Package Inventory

**状态**：`G04_BATCH_B1_B4_COMPLETE / B5_DESIGN_ONLY`<br>
**仓库**：`settled_局部规划器_反应式`<br>
**基线**：`main@d97173c`<br>
**日期**：2026-11-14

## 1. 范围与基线

`META-G04-BATCH` 只做 `src/auv_lowlevel/batch/pc_*.py` 的职责归组。B1-B4 可以
迁移低/中风险模块；B5 只做只读盘点和拆分设计，不在本 Goal 修改源码。

- 拆分前全量测试：`2089 passed, 11 skipped`；
- shape：`0 new / 1189 suppressed`；
- dependency：`ok`；
- 旧 `pc_*.py` 路径必须继续可 import。

## 2. 分组成员

| 模块 | 行数 | 目标包 | 风险 | 关键公共名称 |
|---|---:|---|---|---|
| `pc_calibration.py` | 449 | `pc_support/calibration.py` | 低 | `calibrate_pc20_high_speed` |
| `pc_diagnostics.py` | 110 | `pc_support/diagnostics.py` | 低 | `parameter_snapshot`、`policy_diagnostics` |
| `pc_normalization.py` | 120 | `pc_representation/normalization.py` | 中低 | `PCNormalization` |
| `pc_history_ablation.py` | 131 | `pc_representation/history_ablation.py` | 中低 | ablation modes 和两个 `ablate_*` |
| `pc_teacher.py` | 182 | `pc_residual_env/teacher.py` | 中 | `BatchedPCTeacher` |
| `pc_controller_gain_residual.py` | 301 | `pc_residual_env/controller_gain_residual.py` | 中 | `BatchedPCControllerGainResidualEnv` |
| `pc_actuator_residual.py` | 737 | `pc_residual_env/actuator_residual.py` | 中 | `BatchedPCActuatorResidualEnv` |
| `pc_frozen_residual_vec_env.py` | 212 | `pc_residual_env/frozen_residual_vec_env.py` | 中 | `BatchTorchPCFrozenBCResidualVecEnv` |
| `pc_bc_merge.py` | 244 | `pc_bc_tools/bc_merge.py` | 中高 | `merge_pc_demonstration_datasets` |
| `pc_bc_subset.py` | 240 | `pc_bc_tools/bc_subset.py` | 中高 | `sample_pc_demonstration_dataset` 等 |
| `pc_residual_pretrain.py` | 839 | `pc_bc_tools/residual_pretrain.py` | 中高 | `PCResidualTrainingConfig` 等 |
| `pc_path.py` | 296 | B5 设计 | 高 | path schema、normalization、feature builder |
| `pc_command_residual.py` | 358 | B5 设计 | 高 | `BatchedPCCommandResidualEnv`、`smooth_terminal_gate` |
| `pc_vec_env.py` | 1001 | B5 设计 | 高 | `BatchTorchPCVecEnv` |
| `pc_policy.py` | 1403 | B5 设计 | 高 | actor-critic、layout、encoder metadata |
| `pc_bc.py` | 2049 | B5 设计 | 最高 | BC dataset/training/checkpoint/action helpers |

## 3. import 闭包

- B1：`pc_calibration` -> `batch.dynamics`；`pc_diagnostics` 无组内依赖；
- B2：`pc_normalization` 无组内依赖；`pc_history_ablation` -> `pc_path`、`pc_policy`；
- B3：teacher/actuator/controller-gain -> `batch.controllers`、`batch.geometry`、
  `pc_env`，其中两个 residual 还依赖 `pc_command_residual`；frozen vec env -> `pc_vec_env`；
- B4：merge -> `_json`、`pc_bc`、`pc_policy`；subset -> `_json`；residual pretrain ->
  `pc_bc`、`pc_normalization`、`pc_path`、`pc_policy`。

迁移后相对 import 深度增加一级，只做机械升级；不改变导入对象、定义顺序或运行时
副作用。

## 4. 兼容契约

每个旧 `pc_*.py` 在迁移后保留为模块别名：导入新实现，并在 `sys.modules` 中把旧
模块名指向同一模块对象。必须验证：

```python
import auv_lowlevel.batch.pc_X as old
from auv_lowlevel.batch.new_package.X import Thing
assert old.Thing is Thing
```

以及旧模块级 monkeypatch 对新实现可见、旧 `__all__` 名称仍可 import。旧实现不能
被复制成第二套，否则枚举、异常和 `isinstance` 身份会分叉。

## 5. B1-B4 执行顺序

| 批次 | 迁移内容 | Gate | 状态 |
|---|---|---|---|
| B1 | calibration、diagnostics -> `pc_support/` | import/type identity、全量测试、shape、dependency | `PASS` (`a4ce0c1`) |
| B2 | normalization、history ablation -> `pc_representation/` | B1 通过；旧路径和 schema 名称不变 | `PASS` (`54f3560`) |
| B3 | teacher、两种 residual wrapper、frozen vec env -> `pc_residual_env/` | B2 通过；零动作/step 语义不变 | `PASS` (`b2f665c`) |
| B4 | BC merge/subset/pretrain -> `pc_bc_tools/` | B3 通过；manifest 字段和字节不变 | `PASS` (`7fa1e1a`) |

每批单独 commit。任何一批 import 身份、全量测试、shape 或 dependency 失败，停止
该批并回滚，不继续后续批次。

## 6. B5 只读拆分设计

### 6.1 `pc_path.py`

建议拆为 `pc_path/schema.py`（常量、dim、schema hash）、`pc_path/features.py`
（feature builder、normalization）和 `pc_path/blocks.py`（block validation/mask）。
原因：schema 常量被训练、评价和 checkpoint metadata 共同读取，必须先冻结
`PC_PATH_*` 公共名称和 hash；不能把 schema 构造与 feature 数学混在一个移动中。

### 6.2 `pc_command_residual.py`

建议拆为 `pc_command_residual/gates.py`（`smooth_terminal_gate`、`_wrap_pi`）和
`pc_command_residual/env.py`（`BatchedPCCommandResidualEnv`）。先迁纯门控函数，再迁
环境 wrapper；动作缩放、terminal gate 和 step 顺序必须保持。

### 6.3 `pc_vec_env.py`

建议拆为 `pc_vec_env/history.py`（rolling buffers/history assembly）、
`pc_vec_env/observations.py`（actor/critic flatten contract）和
`pc_vec_env/environment.py`（`BatchTorchPCVecEnv` 编排）。此文件影响 RNG、auto-reset、
history 和训练观测，拆分前必须锁定固定 seed trace。

### 6.4 `pc_policy.py`

建议拆为 `pc_policy/layout.py`（常量、dim、layout、schema hash）、
`pc_policy/encoders.py`（reserved/current/path encoders）和
`pc_policy/actor_critic.py`（`PCActorCritic` 网络与 forward）。state-dict key、初始化
顺序、DADE 配置和 `__module__`/公共类型身份是硬契约。

### 6.5 `pc_bc.py`

最后拆分，建议边界为 `pc_bc/actions.py`、`pc_bc/dataset.py`、
`pc_bc/checkpoints.py`、`pc_bc/training.py` 和稳定 `pc_bc/__init__.py`。它是最高风险：
数据字段、checkpoint schema、action envelope、训练循环、RNG 和历史 warm-start 全部
集中。必须先做只读 golden checkpoint/dataset fixture，再按 actions -> dataset ->
checkpoints -> training 顺序拆分；每个子步都要单独批准。

## 7. B5 推荐顺序与未决项

推荐顺序：`pc_path` -> `pc_command_residual` -> `pc_vec_env` -> `pc_policy` -> `pc_bc`。
B5 暂不执行，等待用户审阅本设计；执行时需要再次确认目标目录、是否改为同名 package，
以及每一步的 golden fixture。

## 8. B0 证据

- 16 个候选文件及 16 个旧 import 路径已登记；
- B1-B4 的组内 import 闭包已确认；
- B5 五个模块的职责、依赖和推荐拆分边界已记录；
- B0 不修改任何源码、测试、reward、termination、success、seed 或 artifact。

## 9. B1-B4 完成证据

| 批次 | commit | 全量测试 | 结论 |
|---|---|---:|---|
| B0 | `898d8e9` | `2089 passed, 11 skipped` | inventory |
| B1 | `a4ce0c1` | `2091 passed, 11 skipped` | `PASS` |
| B2 | `54f3560` | `2093 passed, 11 skipped` | `PASS` |
| B3 | `b2f665c` | `2095 passed, 11 skipped` | `PASS` |
| B4 | `7fa1e1a` | `2097 passed, 11 skipped` | `PASS` |

- 每组定向契约测试均验证旧/新 import 指向同一 module object，公共对象身份和旧
  模块级 monkeypatch 语义未分裂；
- B4 最终 shape：`0 new / 1188 suppressed`；changed-file naming：
  `0 new / 16 suppressed`；dependency：`ok`；
- 触碰文件的 ruff 与 format 检查通过；
- B5 五个源码文件未修改，仍等待设计审阅和单独实施批准。
