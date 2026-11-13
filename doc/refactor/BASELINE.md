# auv-local Refactor Baseline

**状态**：`REF_LOCAL_COMPLETE`<br>
**仓库**：`settled_局部规划器_反应式`<br>
**基线 commit**：`8bf7e68`<br>
**日期**：2026-11-13

## 1. 运行环境

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/ref-local-20261113
export PVS_SOURCE_DIR=/home/bubble/水下机器人/python/PythonVehicleSimulator-master/src
export MSS_SOURCE_DIR=/home/bubble/水下机器人/simulator/MSS-master
```

使用仓库自己的 `.venv/bin/python`。该环境已 editable 安装 `auv-core`；不要用
`PYTHONPATH=../settled_共享基础/src` 绕过安装边界。

## 2. 测试与检查

| 检查 | 命令 | 基线 |
|---|---|---|
| pytest | `.venv/bin/python -m pytest -q tests` | `1984 passed, 11 skipped`，85.63s |
| ruff | `.venv/bin/python -m ruff check .` | 124 个存量错误；候选文件必须为 0 新增 |
| format | `.venv/bin/python -m ruff format --check src tests` | 220 个文件待格式化；候选文件必须通过 |
| mypy | `.venv/bin/python -m mypy` | 381 errors / 83 files；候选文件不得新增 |
| shape | `总协调/scripts/check_code_shape.py --root . --check` | 存量违规进入 `CODE_SHAPE_BASELINE.json` |
| dependency | `scripts/check_repo_dependencies.py --root . --check` | PASS |
| equivalence | `check_refactor_equivalence.py` | 每个候选必须提供 before/after fixture |

`.venv/bin/python` 是仓库根目录下的虚拟环境；在 linked worktree 内没有 `.venv`
目录，因此命令使用绝对路径：

```bash
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python -m pytest -q tests
```

## 3. 已知跳过

`11 skipped` 都在 clean source checkout 中缺外部 artifact 或 checkpoint：

- 2 个 smoke BC checkpoint；
- 1 个 Goal173 frozen BC checkpoint；
- 2 个 Goal137 latent-probe checkpoint / labels；
- 5 个 D3 BC checkpoint；
- 1 个 IsaacLab final evaluation artifact。

跳过不等于通过；本 pilot 不补 artifact，也不改变这些测试的 skip 条件。

## 4. 已知欠账

- 全仓 ruff 和 format 欠账保留，不在重构 candidate commit 中批量格式化；
- mypy 381 errors 是历史基线和第三方边界，不通过放宽配置或跳过类型检查修复；
- `pc_env.py`、`config.py`、`pc_bc.py`、`ppo_config.py`、`pc_policy.py` 和
  `evaluate_pc_ppo.py` 超过硬上限，已进入 ratchet baseline；本 pilot 不允许整文件
  重写；
- 历史脚本的 `sys.path` / `PYTHONPATH` 保留原状，新增库代码不得增加路径注入。

## 5. 回滚点

- 基线 commit：`8bf7e68`；
- 候选失败时使用 `git revert <candidate-commit>`，不在主干上 `reset --hard`；
- 回滚后重新运行本页全部命令，并确认没有任何新失败。
