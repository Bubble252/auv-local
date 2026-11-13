# auv-local Equivalence Evidence

**状态**：`REF_LOCAL_INVENTORY_COMPLETE`<br>
**仓库**：`settled_局部规划器_反应式`<br>
**基线 commit**：`8bf7e68`<br>
**日期**：2026-11-13

## 1. 候选 LOCAL-A1

```yaml
goal: META-G03
candidate: LOCAL-A1
repository: settled_局部规划器_反应式
base_commit: 8bf7e68
branch: refactor/local-20261113
files:
  - src/auv_local/_angles.py
  - src/auv_local/line_detour.py
  - src/auv_rl/planar.py
  - src/auv_rl/observation.py
  - src/auv_rl/bc/teacher.py
public_api_change: none
```

## 2. 行为契约

| 项目 | 重构前 | 重构后 | 证据 |
|---|---|---|---|
| 输入 | 标量 `float` 角度 | 相同 | `before.json` / `after.json` |
| 输出 | `(angle + pi) % (2*pi) - pi` | 完全相同 | golden fixture，`atol=0`、`rtol=0` |
| 单位 | rad | rad | fixture 输入覆盖 ±7 rad 和边界 |
| 非有限值 | NaN/Inf 继续按 Python 取模传播 | 保持相同传播 | `tests/test_local_angles.py` |
| 异常 | 无显式异常 | 无显式异常 | fixture `exception: null` |
| RNG | 不消耗 RNG | 不消耗 | 纯函数测试 |
| 全局状态 | 不修改 | 不修改 | 纯函数测试 |
| 公共 API | 4 个私有 `_wrap_angle` 仍可调用 | 通过别名保留 | 导入兼容测试 |

## 3. Golden fixture 入口

```bash
/home/bubble/水下机器人/.venv/bin/python \
  /home/bubble/水下机器人/总协调/scripts/check_refactor_equivalence.py \
  --expected tests/fixtures/refactor/local-angle/before.json \
  --actual tests/fixtures/refactor/local-angle/after.json \
  --atol 0 --rtol 0
```

fixture 至少包含 `inputs`、`outputs`、`exception`、`manifest` 和
`randomized: false`，并分别记录 4 个原调用方的输出。

## 4. 测试命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/ref-local-20261113
export PVS_SOURCE_DIR=/home/bubble/水下机器人/python/PythonVehicleSimulator-master/src
export MSS_SOURCE_DIR=/home/bubble/水下机器人/simulator/MSS-master
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python \
  -m pytest -q tests/test_local_angles.py
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python -m pytest -q tests
```

## 5. 结果

- 候选 commit：`5d8d0db`；
- 等价性：`equivalence: PASS`；
- 受影响测试：`123 passed`；
- 全量测试：`2032 passed, 11 skipped`；
- ruff：候选文件和 fixture 全部 PASS；
- format：候选文件和 fixture `already formatted`；
- mypy：5 个改动源码文件 `Success: no issues found`；
- shape：`0 new violations, 1189 suppressed`；
- dependency checker：`repository dependency boundaries: ok`；
- 已知失败：无新增；
- 是否新增失败：`no`；
- 回滚 commit：`8bf7e68`
- 结论：`PASS`

## 6. 人工检查

- [x] 公共 API 删除数为 0（4 个私有导入路径继续可用）；
- [x] 没有修改 reward、termination、success、seed split 或 held-out；
- [x] 没有移动或删除 artifact；
- [x] RNG/时钟/全局状态变化有等价证据（纯函数，无 RNG/状态副作用）；
- [x] 只改一个仓库、一个候选。

## 7. 候选 LOCAL-A2

```yaml
goal: META-G03
candidate: LOCAL-A2
repository: settled_局部规划器_反应式
base_commit: ae20d65
branch: refactor/local-20261113
files:
  - src/auv_lowlevel/_json.py
  - src/auv_lowlevel/config.py
  - src/auv_lowlevel/ppo_config.py
public_api_change: none
```

LOCAL-A2 只把 `load_config` 和 `load_training_config` 中重复的
`Path.read_text(encoding="utf-8") + json.loads` 收敛到内部只读 helper。它不改变
JSON schema、默认值、配置哈希、异常类型或文件编码。

```bash
/home/bubble/水下机器人/.venv/bin/python \
  /home/bubble/水下机器人/总协调/scripts/check_refactor_equivalence.py \
  --expected tests/fixtures/refactor/local-config-json/before.json \
  --actual tests/fixtures/refactor/local-config-json/after.json \
  --atol 0 --rtol 0
```

- 候选 commit：`29d24f4`；
- 等价性：`equivalence: PASS`；
- 受影响测试：`75 passed`；
- 全量测试：`2035 passed, 11 skipped`；
- ruff：改动文件和 fixture 全部 PASS；
- format：新增 `_json.py`、测试和 fixture PASS；两个存量配置文件按
  `WAIVERS.yaml` 的 REF-LOCAL-003/004 保留历史格式欠账；
- mypy：全仓 `381 errors / 83 files`，与基线数量一致，无新增类型错误；
- shape：`0 new violations, 1189 suppressed`；
- dependency checker：`repository dependency boundaries: ok`；
- 已知失败：无新增；
- 是否新增失败：`no`；
- 回滚 commit：`ae20d65`；
- 结论：`PASS`

## 8. 候选 LOCAL-A3

```yaml
goal: META-G03
candidate: LOCAL-A3
repository: settled_局部规划器_反应式
base_commit: d2b0dbd
branch: refactor/local-20261113
files:
  - src/auv_lowlevel/_json.py
  - src/auv_lowlevel/batch/pc_bc_merge.py
  - src/auv_lowlevel/batch/pc_bc_subset.py
public_api_change: none
```

LOCAL-A3 把两个 BC 数据工具的 manifest 写出统一为
`json.dumps(..., indent=2, sort_keys=True) + "\n"` 的内部 helper。它不创建父目录、
不改变文件名、字段、排序、缩进、末尾换行或异常行为。

验证命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/ref-local-20261113
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python \
  -m pytest -q tests/test_lowlevel_json_io.py tests/test_pc_bc_subset.py \
  tests/test_lowlevel_pc_bc_merge.py
```

- 候选 commit：`a05a59b`；
- 针对性测试：`16 passed`；
- 全量测试：`2037 passed, 11 skipped`；
- ruff：改动源码和测试全部 PASS；
- format：新增 helper 和测试 PASS；两个存量数据工具按 REF-LOCAL-005/006 保留
  历史格式欠账；
- mypy：全仓 `381 errors / 83 files`，与基线数量一致，无新增类型错误；
- shape：脚本函数长度缩短并已刷新 ratchet 基线，`0 new violations, 1189 suppressed`；
- dependency checker：`repository dependency boundaries: ok`；
- manifest 字节格式：sorted keys、2-space indent、末尾换行由单元测试锁定；
- 已知失败：无新增；
- 是否新增失败：`no`；
- 回滚 commit：`d2b0dbd`；
- 结论：`PASS`

## 9. 候选 LOCAL-A4

```yaml
goal: META-G03
candidate: LOCAL-A4
repository: settled_局部规划器_反应式
base_commit: 8ca3daa
branch: refactor/local-20261113
files:
  - src/auv_lowlevel/_paths.py
  - scripts/run_pc_standing_eval_suite.py
public_api_change: none
```

LOCAL-A4 只把 `run_pc_standing_eval_suite.py` 的私有 `_resolve` 路径语义抽成
显式接收 `root` 的纯函数：

```python
def resolve_from_root(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value
```

它不调用 `resolve()`、`expanduser()` 或 `exists()`，不读取 `Path.cwd()`，也不改变
相对路径的拼接结果。旧 `_resolve` 保留为薄 wrapper，避免脚本内部调用点和未来兼容
入口同时变化。

| 项目 | 重构前 | 重构后 | 证据 |
|---|---|---|---|
| 绝对路径 | 原样返回 | 原样返回 | helper 单测 |
| 相对路径 | `REPO_ROOT / value` | `resolve_from_root(REPO_ROOT, value)` | helper 单测和入口测试 |
| `str` / `Path` 输入 | 均可 | 均可 | helper 单测 |
| 文件存在性 | 不检查 | 不检查 | 纯路径运算测试 |
| 异常 | 只可能来自 `Path(...)` | 相同 | 类型测试 |
| RNG / 全局状态 | 不消耗/不修改 | 不变 | 纯函数检查 |

验证命令：

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/ref-local-20261113
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python \
  -m pytest -q tests/test_lowlevel_paths.py tests/test_pc_standing_eval_runner.py
```

Preflight 证据：

```yaml
skill_usage:
  - skill: auv-refactor-guard
    phase: preflight
    outcome: passed
    repo: /home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/ref-local-20261113
    module: scripts/run_pc_standing_eval_suite.py
```

- 结果：`PENDING`
