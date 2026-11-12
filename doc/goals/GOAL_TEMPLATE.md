# GoalXX：<一句话目标>

**日期**：YYYY-MM-DD<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT`

## Research Contract

```yaml
autoresearch:
  required: true
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal GoalXX --goal-doc doc/goals/<this_file>.md --strict
```

- `required: true`：探索型 goal；触发 blocker 后必须留下 research 记录。
- `required: false`：冻结的预注册 Gate；Gate 记录前不改确认参数，失败后补记录。

## 1. 背景与证据

- 上游结果：
- 当前失败模式：
- 为什么现在做这个：

## 2. 假设

至少写两个可证伪 hypothesis：

1. H1：
2. H2：

每个 hypothesis 写清楚：如果成立应看到什么；如果失败应看到什么；最便宜的
falsification experiment 是什么。

## 3. 数据与隔离

- 训练数据：
- eval / held-out：
- 禁止进入训练的数据：
- oracle fallback：禁止 / 允许（写明范围）

## 4. 方法

- 方法组件：
- 与已有方法的关系：
- 外部论文/代码：先下载 PDF/TXT、查 code、写 reading note。

## 5. 执行计划

### 5.1 不可协商的运行契约

| 项目 | 强制要求 | 失败标签 |
|---|---|---|
| 训练课程 | `curriculum.enabled=true`；冻结 `stage_mix`；冻结 run 必须 `curriculum.promotion_enabled=false`；逐 episode `curriculum_episodes.jsonl` | `INVALID_NONCURRICULUM` |
| 训练速度 | 每个 stage 满足 `speed + current + variation <= 2.5 m/s` | `INVALID_SPEED_CONTRACT` |
| 正式评价 | 完整 `arms × levels × radii × zero baselines`，不得只跑 default/strict125 或 current10 | `INVALID_INCOMPLETE_EVAL` |
| 评价 seeds | 预注册 dev seeds；held-out 关闭，除非新 goal 明确解封 | `INVALID_SEED_SPLIT` |
| 启动与结束 | 先跑协议检查器，再跑实验，最后跑产物审计器；任一失败非零退出 | `INVALID_PROTOCOL` |
| 主指标 | 整条路线 3D/cross/vertical、五段、10% keypoints、completion/progress | 不能只看 success |

默认硬约束（除非本文档在测量前明确预注册单阶段例外）：

- 所有训练启用课程学习，并写逐 episode `curriculum_episodes.jsonl`；若 goal 声明冻结课程，必须设置 `promotion_enabled=false`，并在训练后审计 `curriculum_state.json` 的 stage/mix；
- 所有评价跑完整 arm × level × radius × zero-baseline 矩阵，不只跑一个 cell；
- 启动器先做协议检查，结束后做产物审计；审计失败必须非零退出；
- 主指标包含整条路线、分段、keypoint、completion/progress efficiency，success 只作二级诊断。

- [ ] Step 0：Run Contract；
- [ ] Step 1：实现/数据；
- [ ] Step 2：smoke；
- [ ] Step 3：正式实验；
- [ ] Step 4：Gate；
- [ ] Step 5：result md、ledger、commit/push。

## 6. Gate

| check | 阈值 |
|---|---:|
| 主指标 | |
| 稳定性 | 0 error / 0 NaN / 0 flew-out |
| 负控 | |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| | | |

## 7. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal GoalXX --goal-doc doc/goals/<this_file>.md --strict
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 GoalXX：<目标>。

先读：
- <goal doc>
- <上游 result>

硬约束：
1. 所有训练必须启用课程学习，并保留逐 episode 课程 provenance；
2. 所有评价必须跑完整 arm × level × radius × zero-baseline 矩阵；
3. ...

执行顺序：
- ...
```


## 9. 自主迭代契约

所有新 Goal 必须引用并执行 `research/AUTORESEARCH_LOOP.md`。Goal prompt 必须明确写：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 使用 `paper-search` / `deep-research` 查论文，找到可用代码，写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新的 Goal 文档和新的 prompt，不修改失败 Gate 的阈值；
6. 持续迭代，直到满足成功标准或触发必须向用户报告的停止条件；
7. 不自动 commit/push，不打开 held-out，不改变任务成功定义或主算法族。

成功标准、停止条件、论文/代码检索要求和自动执行边界见
`research/AUTORESEARCH_LOOP.md`。
