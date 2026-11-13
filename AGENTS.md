# settled_局部规划器_反应式 项目规则

本文件适用于 `settled_局部规划器_反应式/`。更上层规则见
`/home/bubble/水下机器人/AGENTS.md` 和 `/home/bubble/.codex/AGENTS.md`。

## 0. 独立仓库边界

- 本目录现在对应独立 Git 仓库 `/home/bubble/水下机器人/settled_局部规划器_反应式`，分支 `main`。
- Python 包是 `auv_local` / `auv_rl` / `auv_lowlevel` / `auv_isaaclab`。
- 只允许依赖已安装的 `auv-core`；不得导入 `auv_planner`，也不得引用
  `../settled_全局规划器/src`。
- 安装顺序：先安装 `settled_共享基础` 的 `auv-core`，再安装本仓库。
- 外部 PVS/MSS 资产用 `PVS_SOURCE_DIR` / `MSS_SOURCE_DIR` 指定；`artifacts/`
  只作为本地测试 fixture，不进入 Git。

## 1. 项目定位

- PC 臂：合成传感器 + Batch Torch REMUS100，不依赖 IsaacLab/IsaacSim 才能跑主实验。
- 控制链路：参考路径 → 3D ALOS → `[psi_d, theta_d, u_d]` → frozen actor/controller → `[rudder, stern, rpm]`。
- 主指标是 3D path RMSE；success、timeout、safety、strict125 和 material saturation 是护栏。
- 新实验默认使用 26D observation；27D 只用于历史 checkpoint、数据和兼容回归。

## 2. 当前硬约束

- 不继续 DADE 主线，不新增第二个学习残差网络。
- 海流补偿放在 ALOS 制导层，actor 输入维度和 checkpoint 不因此改变。
- 终端阶段不得因为 `goal_distance` 减速；速度只能受执行器饱和保护影响。
- 新传感器实验使用 `realistic_calibrated` 作为 nominal；`realistic_calibrated_legacy`
  只用于复现 2026-11-07 之前的结果，`realistic_stress` 只做压力测试。
- 不碰 held-out seeds，不用 validation 结果回头改门禁。

## 3. Goal 与 Research Contract

- 新 goal 放在 `doc/goals/`，从 `doc/goals/GOAL_TEMPLATE.md` 复制。
- 每个新 goal 必须有 `autoresearch:` YAML 块；预注册 Gate 可写 `required: false`，
  但 Gate 失败后必须补 research 记录。
- 执行计划每一步用 `[ ]` / `[x]`，并写清运行命令、输出路径、验证方法和 git 步骤。

## 4. Research Ledger

项目 ledger 在 `research/`。触发 hard blocker 后按根目录 `AGENTS.md` 的流程写
`research/blockers/`、`research/LEDGER.jsonl`，并在 `research/STATE.md` 更新下一步。

完成 goal 前，在项目根目录运行：

```bash
python ../scripts/check_research_ledger.py \
  --goal Goal25 \
  --goal-doc doc/goals/goal25_d2_pregoal_repair_20261108.md \
  --strict
```

## 5. 评估纪律

- 先 smoke，再 screen，再 formal；smoke fail 立即停，不补 seed、不换 tag。
- 所有 A/B 必须同 checkpoint、同 seeds、同传感器 profile、同评估协议。
- 报告 success、timeout、3D RMSE、pre-goal RMSE、safety、material saturation 和
  path-zero 退化；不能只报均值。
- 一次 smoke 不写成正式结论；failure 保留，不通过放宽阈值救。
- **训练默认必须启用课程学习**。除非 goal 文档在测量前明确预注册了单阶段例外，
  否则缺少 `curriculum.enabled=true`、`stage_mix` 或逐 episode
  `curriculum_episodes.jsonl` 的运行一律标为 `INVALID_NONCURRICULUM`，不得进入 Gate。
- **评价默认必须跑完整矩阵**。至少覆盖所有 arm、规定 levels、所有 radii 和对应
  zero baseline；不能只跑 `default/strict125`、单 level 或挑选最好 cell。缺少任一
  cell 一律标为 `INVALID_INCOMPLETE_EVAL`。
- 新 goal 的启动器必须同时有启动前协议检查（`check_*_protocol.py`）和结束后产物审计
  （`audit_*_artifacts.py` 或等价脚本）；审计失败时启动器必须非零退出，不能只打印警告。
- 主指标优先看整条路线、分段、keypoint、completion/progress efficiency；success 和
  terminal capture 只作二级诊断，不能单独决定训练方向。

## 6. Git

- 禁止 `git add -A`。
- 只 add 本次代码、配置、goal/result 文档和 ledger；不提交 `artifacts/`、checkpoint、
  日志、缓存和大型数据。
- 提交前运行 `git diff --cached --check`；不自动 commit / push，除非用户明确要求。

## 7. 重构交接

代码重构的基线、允许目录、已知失败和 G04 候选以 `doc/refactor/` 为准：

- `BASELINE.md`：环境、测试命令、11 个已知 skip 和 lint/type 欠账；
- `INVENTORY.md`：重复实现、大文件、G04 自动化和人工批准点；
- `MODULE_MAP.md`：可修改模块的职责、非职责、公共 API 和测试；
- `EQUIVALENCE.md`：LOCAL-A1 至 LOCAL-A4 的 before/after 证据。

候选文件验证：

```bash
export PVS_SOURCE_DIR=/home/bubble/水下机器人/python/PythonVehicleSimulator-master/src
export MSS_SOURCE_DIR=/home/bubble/水下机器人/simulator/MSS-master
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python -m pytest -q tests
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python -m ruff check <changed-files>
/home/bubble/水下机器人/settled_局部规划器_反应式/.venv/bin/python -m ruff format --check <changed-files>
```

训练、评价、reward、termination、success、Goal173 matrix、checkpoint 选择和
held-out 不属于自动重构范围。
