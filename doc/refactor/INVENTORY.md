# auv-local Refactor Inventory

**状态**：`REF_LOCAL_COMPLETE`<br>
**仓库**：`settled_局部规划器_反应式`<br>
**基线 commit**：`8bf7e68`<br>
**日期**：2026-11-13

## 1. 范围

本页只记录只读盘点，不修改训练、评价、奖励、终止、成功判定、seed split 或
held-out。盘点范围是 `src/`、`scripts/`、`tests/` 和 `configs/`；历史 artifact、
checkpoint、日志和生成文件不进入重构范围。

| 对象 | 数量 |
|---|---:|
| `src/` Python 模块 | 124 |
| `scripts/` Python 入口 | 345 |
| `tests/test_*.py` | 275 |
| `auv_local` 模块 | 11 |
| `auv_lowlevel` 模块 | 59 |
| `auv_rl` 模块 | 43 |
| `auv_isaaclab` 模块 | 11 |

入口分类：训练 11、评价 10、checker/audit 39、analyzer 113、config builder 33、
summary 33。数量本身不是重构目标；优先处理有明确等价边界、调用方稳定的重复。

## 2. 重复实现盘点

| 重复模式 | 现状 | 风险 | 处置 |
|---|---|---|---|
| 标量 `_wrap_angle` | 4 个 `src/` 模块独立实现；10 个历史 analyzer 也有本地副本 | 公式相同，但部分调用方依赖私有符号 | 先抽 `auv_local` 内部标量 helper；历史脚本暂不批量迁移 |
| NumPy 数组 `_wrap_angle` | `auv_rl/envs/parallel_planar.py` | shape 和 dtype 与标量不同 | 不合并到标量 helper |
| Torch Tensor `_wrap_angle` | `auv_isaaclab/tasks/p1_avoidance/batch.py` | 依赖 Tensor 运算和设备 | 不合并到标量 helper |
| `_positive` | `auv_local`、`auv_rl`、`auv_lowlevel` 多处；参数名和错误语义不完全一致 | 直接合并可能改变异常文本或路径参数 | 暂缓，先做调用方审计 |
| `_sha256` | 88 个脚本或模块附近存在同名 helper | 输入是文件、bytes 或目录时语义不同 | 暂缓；后续按 artifact 契约统一 |
| JSON 读取 | 59 个脚本存在 `_load_json` / `_read_json` 变体 | 错误处理、路径解析和 schema 校验不同 | 第二优先级，先定义只读 helper |
| JSON 写入 | 16 个脚本存在 `_write_json` 变体 | indent、排序、原子写入和父目录策略不同 | 第二优先级，和 manifest 统一一起做 |
| CLI parser | 330 个脚本直接构造 `ArgumentParser` | 入口多，但参数语义差异大 | 不做机械替换；只统一新入口骨架 |

### 2.1 标量角度 helper 的具体位置

- `src/auv_local/line_detour.py:19`
- `src/auv_rl/planar.py:19`
- `src/auv_rl/observation.py:24`
- `src/auv_rl/bc/teacher.py:13`

这四处公式均为 `(angle + pi) % (2*pi) - pi`，没有有限值拒绝逻辑。共享核心的
`auv_core._validation.wrap_angle` 会拒绝非有限值，因此不能直接替换；REF-LOCAL
必须保留现有的 NaN/Inf 传播行为。

### 2.2 artifact 路径 helper 的盘点

`scripts/` 中共有 66 个名为 `_resolve` 的本地路径 helper。多数实现是
“绝对路径原样返回，否则拼到 `REPO_ROOT`”，但至少 6 个 analyzer 使用
`Path.cwd()`，另有若干脚本使用自定义 `root` 或搜索多个候选根。因此不能用一个
隐式全局根替代全部实现；LOCAL-A4 只抽一个显式接收 `root` 的纯函数，并迁移
`run_pc_standing_eval_suite.py` 一个入口。其余脚本留给 G04 的候选队列逐个审计。

## 3. 超大模块

以下文件超过新代码硬上限，只能通过 ratchet 和单独职责拆分处理，不能在本 pilot
中整文件重写：

| 文件 | 行数 | 说明 |
|---|---:|---|
| `src/auv_lowlevel/batch/pc_env.py` | 2863 | 环境、记录、课程和诊断耦合最重 |
| `src/auv_lowlevel/config.py` | 2729 | schema、默认值和兼容解析集中 |
| `src/auv_lowlevel/batch/pc_bc.py` | 2049 | BC 数据、训练和 manifest |
| `src/auv_lowlevel/ppo_config.py` | 1662 | PPO 配置族和校验 |
| `src/auv_lowlevel/batch/pc_policy.py` | 1403 | 策略与编码器边界 |
| `scripts/evaluate_pc_ppo.py` | 3919 | 评价入口、聚合和 artifact 输出 |

这些文件全部进入 `CODE_SHAPE_BASELINE.json`，未被触碰时不需要一次性拆分。

## 4. 入口与契约风险

- 训练、评价和 checker 入口多数仍直接依赖 `auv_lowlevel` / `auv_rl`；本 pilot
  不改变入口语义，只处理可独立验证的 helper。
- Goal173 的 88-cell matrix、协议 checker 和 artifact auditor 是冻结契约，不能
  在重构 commit 中修改。
- 历史脚本使用 `sys.path` 或 `PYTHONPATH` 的地方保留原状；新增库代码禁止新增
  路径注入。
- 所有跨仓库依赖只能通过已安装的 `auv-core`；不得导入 `auv_planner` 或引用
  全局规划器源码。

## 5. 候选队列

| 优先级 | 候选 | 边界 | 状态 / 前置证据 |
|---|---|---|---|
| P0 | LOCAL-A1：标量角度 helper 去重 | 只改 4 个 `src/` 调用方，保留私有符号 | DONE；`before.json` / `after.json`、4 个模块测试 |
| P1 | LOCAL-A2：JSON 读取 helper | 只读、错误文本不变 | DONE；调用方 fixture 和异常测试 |
| P2 | LOCAL-A3：manifest 写出 helper | artifact 路径和字段不变 | DONE；manifest 字节格式测试 |
| P3 | LOCAL-A4：artifact 路径 helper | 显式传入 root；只迁移一个带测试的 CLI | DONE；helper 单测、入口测试和全量回归 |
| P4 | 评价入口拆分 | 不改变 CLI、matrix 和 artifact | PENDING；入口 help、短 smoke、Goal173 契约测试 |
| P5 | 超大模块拆分 | 一次一个完整职责 | PENDING；模块级等价性和全量测试 |

P0-P3 是 REF-LOCAL pilot 的闭合范围；P4/P5 中涉及公共接口、matrix 或大文件拆分的
候选留给后续子 Goal。本 Goal 不允许一次同时改多个仓库。

## 7. G03 最终验证与 G04 交接

最终分支验证结果：

| 检查 | 结果 |
|---|---|
| pytest | `2042 passed, 11 skipped` |
| 候选 Python diff 的 ruff check | PASS |
| 非 waiver 候选文件 format check | PASS；13 files already formatted |
| 全仓 ruff | 124 个存量错误，未新增到候选文件 |
| 全仓 format | 220 个存量文件待格式化 |
| mypy | `381 errors / 83 files`，与基线一致 |
| shape | `0 new / 1189 suppressed` |
| dependency | PASS |

允许自动处理的目录和行为：

| 级别 | 候选 | 证伪条件 | 边界 |
|---|---|---|---|
| L0 | 只对候选 diff 运行 ruff import 排序/format | 产生非格式 diff、候选文件 lint 失败或测试回退 | 禁止全仓 220 文件批量格式化 |
| L1 | 逐个迁移语义完全相同的 `_resolve` helper 到显式 root helper | 绝对/相对路径、CWD、symlink、存在性检查或 CLI 行为发生变化 | 每次只迁一个入口，必须有路径单测和 `--help` smoke |
| L1 | 合并具有相同 schema、编码、异常和父目录策略的 JSON writer | 字段、排序、缩进、尾换行或异常行为出现字节差异 | 先建立 golden fixture，再改一个调用方 |

禁止自动修改：`doc/goals/` 历史 Goal、Goal173 88-cell matrix、训练/评价协议、
`src/auv_lowlevel/config.py` 的语义字段、reward/termination/success、policy 接口、
checkpoint 选择、seed split、held-out、`artifacts/`、`outputs/` 和 `checkpoints/`。

允许修改目录是 `src/auv_local/`、`src/auv_lowlevel/`、`src/auv_rl/`、`scripts/`
和 `tests/`；修改 `scripts/` 必须附入口等价测试。G04 shadow 只能生成候选和风险分级，
不能在本分支上自动写入。

## 6. 只读盘点命令

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式/.worktrees/ref-local-20261113
find scripts -maxdepth 1 -type f -name '*.py' | wc -l
find src -type f -name '*.py' | wc -l
rg -l 'def _wrap_angle\(' src scripts
/home/bubble/水下机器人/.venv/bin/python \
  /home/bubble/水下机器人/总协调/scripts/check_code_shape.py --root . --check
```

## 7. META-G04-LAYOUT 存量布局

存量布局的逐文件职责、调用方、测试引用、依赖链、风险与分批计划见
`doc/refactor/LAYOUT_INVENTORY.md`。首批试点为四个独立
`scripts/audit_pc_goal*.py` -> `scripts/audit/pc_goal/`，历史路径保留薄 wrapper。

试点验收：`2089 passed, 11 skipped`；shape `0 new / 1189 suppressed`；repository
dependency boundaries PASS。试点不改变算法、reward、termination、success、seed、
评价协议或 artifact。
