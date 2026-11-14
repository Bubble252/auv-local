# META-G04-LAYOUT Follow-up

**状态**：`LAYOUT_MERGED / FOLLOWUP_PLANNED`<br>
**基线**：局部规划器 `main@852476e`，可执行位修复 `34b0465`<br>
**日期**：2026-11-13

## 1. 已完成

- `scripts/analyze_pc_goal*.py`：105/105 迁入 `scripts/analysis/pc_goal/`，旧路径 wrapper；
- `scripts/audit_pc_goal*.py`：19/23 迁入 `scripts/audit/pc_goal/`，旧路径 wrapper；
- audit 54-57 因逐级 SHA 契约保留原路径；
- 合并后主 checkout 全量测试：`2095 passed, 5 skipped`；
- shape：`0 new / 1189 suppressed`；dependency：`ok`。

## 2. 后续 Goal

| Goal | 范围 | 文档 | 状态 |
|---|---|---|---|
| `META-G04-BATCH` | `src/auv_lowlevel/batch/pc_*.py` 分批子包化 | `总协调/doc/goals/meta_g04_batch_pc_modules_20261113.md` | DRAFT |
| `META-G04-EXPERIMENTS` | 全局规划器 `experiments/m*.py` 归组 | `总协调/doc/goals/meta_g04_experiments_m_migration_20261113.md` | BLOCKED on BIG1 |
| `META-G04-AUDITSHA` | audit goal54-57 的 SHA 安全迁移 | `总协调/doc/goals/meta_g04_audit_sha_contract_20261113.md` | CLOSED / 方案 A：保持原样 |

## 3. BATCH 建议顺序

1. B0：只读 import graph、类型身份和测试清单；
2. B1：`pc_calibration` / `pc_diagnostics` -> `batch/pc_support/`；
3. B2：`pc_normalization` / `pc_history_ablation` -> `batch/pc_representation/`；
4. B3：residual/env 四个文件 -> `batch/pc_residual_env/`；
5. B4：BC tools 三个文件 -> `batch/pc_bc_tools/`；
6. B5：`pc_bc`/`pc_policy`/`pc_vec_env`/`pc_command_residual`/`pc_path` 最后单独设计。

库模块迁移必须额外验证：

```python
import auv_lowlevel.batch.pc_X as old
from auv_lowlevel.batch.new_package.X import Thing
assert old.Thing is Thing
```

以及旧模块级 monkeypatch 对新实现生效。

## 4. EXPERIMENTS 前置

先等 `META-G04-BIG1` 合入全局规划器 `main`。当前全局 `main` 仍是旧 `m7_1.py`，
不得与 BIG1 并发修改 `experiments/`。E1/E2/E3 的顺序和 Gate 见 Goal 文档。

## 5. AUDITSHA 默认方案

默认方案 A 已执行：保持 audit goal54-57 原样。4 个入口 `--help` 均通过，
逐级 SHA 常量与文件内容一致；定向测试 `35 passed`，全量测试
`2095 passed, 5 skipped`，shape `0 new / 1189 suppressed`，dependency `ok`。
只有未来明确选择 B/C 才迁移；任何 SHA 常量变化都必须在 result 中逐条列出
旧值、新值和原因。

## 6. 验收口径

每批必须同时通过：

```bash
export PYTHONPATH=$PWD/src
export PVS_SOURCE_DIR=/home/bubble/水下机器人/python/PythonVehicleSimulator-master/src
export MSS_SOURCE_DIR=/home/bubble/水下机器人/simulator/MSS-master
.venv/bin/python -m pytest -q tests
git diff --cached --check
/home/bubble/水下机器人/.venv/bin/python /home/bubble/水下机器人/总协调/scripts/check_code_shape.py --root . --check
/home/bubble/水下机器人/.venv/bin/python /home/bubble/水下机器人/scripts/check_repo_dependencies.py
```

不自动 push，不自动合并；每批独立 commit 和回滚点。
