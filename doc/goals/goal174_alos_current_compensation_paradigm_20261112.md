# Goal174：ALOS 洋流补偿双范式冻结（COMP / NONE）

**日期**：2026-11-12<br>
**状态**：`PROTOCOL_IMPLEMENTED_NOT_RUN / HELDOUT_CLOSED`<br>
**上游**：Goal173 `ALOS-CRAB-EST` legacy 结果；Goal136/Goal153 standing eval suite<br>
**性质**：冻结 ALOS 洋流补偿这一维训练/评价范式，不重写 Goal173 的历史结论。

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal174 --goal-doc doc/goals/goal174_alos_current_compensation_paradigm_20261112.md --strict
```

## 1. 用户裁决与背景

`current` 在这里指**洋流**，不是电流。Goal173 及更早的 ALOS 基线实际使用
`compensate_current=false + current_crab_enabled=true`，即 estimated-current crab
补偿；它与完整的洋流前馈补偿不是同一个控制律。

用户裁决：

- 以后 ALOS 默认使用 `compensate_current`，不再默认使用 crab；
- “是否使用 `compensate_current`”本身是训练/评价范式的一维；
- 只保留两组，不新增 crab 对照组：

| 范式 ID | `compensate_current` | `current_crab_enabled` | 洋流来源 | 角色 |
|---|---:|---:|---|---|
| `ALOS-NONE` | `false` | `false` | 不使用 | 无补偿对照 |
| `ALOS-COMP` | `true` | `false` | 传感器估计 `estimated` | **默认主基线** |

硬约束：

1. 新训练/评价不得同时打开 `compensate_current` 和 `current_crab_enabled`；
2. `ALOS-COMP` 只能使用估计洋流，不能使用 true current；
3. `current_crab_enabled=true` 仅保留给历史 Goal173/Goal17 等复现，不进入 canonical 矩阵；
4. 不把 Goal173 的 legacy crab 结果静默改写为 COMP 结果。

## 2. 可证伪假设

### H174-A：估计洋流前馈比不补偿更适合强洋流路线

- 成立：同一 checkpoint、同一课程、同一 reward 下，`ALOS-COMP` 在
  `current10/current15` 的整条 3D/cross-track RMSE 和五段路线健康度优于
  `ALOS-NONE`，且 speed/actuator 饱和不恶化。
- 失败：nominal 持平但 current10/current15 也没有路线收益；此时不能把
  “补偿洋流”写成神经网络方法的优势，转查动作参数化、reward 与 terminal capture。
- 最便宜证伪：同一 checkpoint 的 eval-only 配对，4 个压力档 × 2 radii × dev40。

### H174-B：COMP 的收益不是 true-current 泄漏或 crab 残留

- 成立：trace/manifest 证明 `current_compensation_source=estimated`，actor 观测契约仍为
  hidden，crab angle 恒为 0，且 COMP 与 NONE 的差异只来自 ALOS 命令。
- 失败：配置或 trace 出现 true current、`current_crab_enabled=true`、actor 可见真实洋流；
  该 cell 判 `INVALID_CURRENT_LEAK`，不得进入 Gate。
- 最便宜证伪：配置审计 + 同一批 seed 的 trace 对照，不训练。

### H174-C：补偿模式不会以终点 success 换路线健康

- 成立：COMP 的 success/strict125 不下降超过 `2/40`，同时 path/cross/vertical 与
  keypoints 至少一个主方向改善。
- 失败：只有 success 波动或只有终点附近改善；按 `TERMINAL_ONLY_SIGNAL` 处理。
- 最便宜证伪：先用 Goal173 `P_RUD_075` 同 checkpoint 做 paired eval。

## 3. 实现契约

新增显式字段：

```text
guidance.current_compensation_source = none | true | estimated
```

- `ALOS-NONE`：`compensate_current=false, current_crab_enabled=false, source=none`；
- `ALOS-COMP`：`compensate_current=true, current_crab_enabled=false, source=estimated`；
- PC batch env 在 COMP 下把经过有效性与 age 检查、并做低通滤波的 estimated current
  传给 ALOS；invalid row 自动回退为零洋流；
- `observation.current_contract=hidden` 时，只有 `source=estimated` 允许 COMP；
  `source=true` 仍判 hidden-current leak。

Canonical 定义：

- `configs/lowlevel/eval/pc_standing_eval_suite_v1.json` 的 `alos_current_modes`；
- 默认模式 `ALOS-COMP`；
- `T0` 为 COMP 传统基线，`T0-NONE` 为无补偿对照；
- `scripts/build_pc_standing_eval_configs.py --alos-current-mode ALOS-NONE` 可显式生成对照配置。

## 4. 训练与评价矩阵

所有新 arm 必须登记 `alos_current_mode`，primary 为 `ALOS-COMP`；只有要做诊断/消融时才
显式登记 `ALOS-NONE`。

### 4.1 Screen

| 轴 | 设置 |
|---|---|
| 训练 | 课程学习开启，冻结 stage mix，逐 episode `curriculum_episodes.jsonl` |
| ALOS 模式 | `ALOS-COMP` vs `ALOS-NONE` 配对 |
| 压力 | nominal, current10, noise_n2, plant20 |
| radii | default, strict125 |
| seeds | dev 20–59；held-out 100–149 关闭 |
| 预算 | 1M transitions |

### 4.2 Formal

只有 screen 通过的 arm 才进入：

| 轴 | 设置 |
|---|---|
| 训练 | 3 seeds × 5M；课程与 reward 同 arm 冻结 |
| ALOS 模式 | 每个 retained arm 配对 COMP/NONE |
| 压力 | nominal, current05, current07, current10, current15, noise_n2, plant20, combined20 |
| route | legacy_sine, P0–P4 |
| length | 30/35/45 m |
| speed | 0.9/1.2/1.4/1.8/2.2/2.5 m/s |
| radii | default, strict125 |
| zero baseline | T0-COMP / T0-NONE 与各 arm 同 cell 配对 |

主指标仍是整条路线 3D/cross/vertical、五段、10% keypoints、completion/progress；
success、terminal capture、final distance 只作二级诊断。不得只报 success。

## 5. 执行计划

- [x] Step 0：冻结双范式与 Research Contract；
- [x] Step 1：实现 `current_compensation_source`、PC estimated-current 路径和泄漏校验；
- [x] Step 2：更新 canonical suite、validator、builder，并跑定向测试；
- [ ] Step 3：用 Goal173 `P_RUD_075` 同 checkpoint 做 eval-only COMP/NONE 配对 smoke；
- [ ] Step 4：按 screen 矩阵跑完整 4 压力 × 2 radii × dev40；
- [ ] Step 5：Gate 记录；失败则先归因，不换算法、不扫 seed；
- [ ] Step 6：写 result md、ledger，必要时提交相关文件。

## 6. Gate

| check | 阈值 |
|---|---:|
| 协议完整性 | 0 个 true-current leak；0 个 crab；所有 cell 有 `alos_current_mode` |
| 路线主指标 | COMP 相对 NONE 在 current10 的 3D route RMSE 至少 `-3%` |
| 五段/keypoints | 至少 3/5 段同向改善，至少 5/8 keypoints 改善 |
| 稳定性 | 0 error / 0 NaN / 0 flew-out；speed 不超过 2.5 m/s |
| 终点 | success/strict125 不下降超过 2/40；不得 terminal-only |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| COMP 在 nominal 退化、current 档改善 | `CURRENT_ONLY_BENEFIT` | 保留为压力收益，不报全面超越 |
| COMP 与 NONE 差异小于 seed 噪声 | `SEED_NOISE_AMBIGUOUS` | 进 autoresearch，不加 seed 硬救 |
| 只有 success 改善、路线不动 | `TERMINAL_ONLY_SIGNAL` | 查 terminal capture/reward，不换算法 |
| 出现 true current/crab | `INVALID_CURRENT_LEAK` | 修协议后重跑，不比较结果 |

## 7. Git

禁止 `git add -A`。只 add 本 Goal 相关文件，提交前运行：

```bash
.venv/bin/python ../scripts/check_research_ledger.py --goal Goal174 --goal-doc doc/goals/goal174_alos_current_compensation_paradigm_20261112.md --strict
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 Goal174：ALOS 洋流补偿双范式冻结与同协议对照。

先读：
- doc/goals/goal174_alos_current_compensation_paradigm_20261112.md
- doc/goals/goal173_result_20261112.md
- configs/lowlevel/eval/pc_standing_eval_suite_v1.json

硬约束：
1. ALOS 只有 ALOS-NONE 与 ALOS-COMP 两组；ALOS-COMP 默认，ALOS-NONE 对照；
2. ALOS-COMP 只允许 estimated current，禁止 true current 和 crab；
3. 所有训练开启课程学习并保留 curriculum_episodes.jsonl；
4. 评价必须覆盖完整压力/路线/长度/速度/radius 矩阵，不只报 success；
5. 不打开 held-out；不换算法；不自动 commit/push。

执行顺序：
- 先做同 checkpoint 的 COMP/NONE eval-only smoke；
- 通过协议审计后再跑 screen；
- 失败先归因并写 research/blockers，再决定 continue/pivot/stop。
```

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先归因，不直接换算法或扫 seed；
每个 blocker 至少写两个可证伪 hypothesis；先跑最便宜 falsification；只允许在
`pivot`/`stop` 记录后改主线。持续迭代直到满足 Gate 或触发必须报告的停止条件。
