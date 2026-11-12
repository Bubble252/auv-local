# Goal84：冻结 p_nav/eta outcome contract，并实现 eval-only success-frame 重评分

**日期**：2026-11-10<br>
**状态**：`RESULT_RECORDED / FRAME_ACCOUNTED`<br>
**上游**：Goal82 `CHANNEL_GAIN_WITHOUT_OUTCOME`、Goal83 `CAPTURE_FRAME_AMBIGUITY`<br>
**性质**：冻结评测口径 + 只读重评分现有 800 个 episode；不训练、不新增 rollout、不打开 held-out、不改 reward / action / controller / checkpoint / seed。

## Research Contract

```yaml
autoresearch:
  required: false
  triggers:
    - gate_fail_twice
    - no_metric_movement_after_3_fixes
    - heldout_collapse
    - seed_noise_ambiguous
  ledger: research/LEDGER.jsonl
  blockers_dir: research/blockers/
  reading_dir: research/paper_reading/
  max_outer_loops: 3
  check_command: python ../scripts/check_research_ledger.py --goal Goal84 --goal-doc doc/goals/goal84_outcome_contract_success_frame_20261110.md --strict --require-result
```

`required: false`：这是冻结的评测契约 Gate。Gate 记录前不得改半径、seed、run 集合、
success 定义或 held-out；若 Gate 失败，再按项目规则补 blocker、hypothesis 和 ledger
decision，不能直接改奖励或换算法。

## 1. 背景与证据

Goal83 已确认：

- residual 的 base timing、scale、rate limit、goal-frame 投影和 terminal replay
  均没有可解释 lost 的实现缺陷；
- done 时刻 p_nav/eta 的 capture 分歧为 `0`；
- 分歧主要出现在首次 p_nav crossing 的 plane sign；
- true eta 在 tight radius 内“曾经进入”可为四个 group 恢复 `12/29` lost，
  additive false-success 均低于 `0.05`。

当前真正需要停止反复争论的是**评测语义**：

```text
deployment success = p_nav_tight       # 训练、终止、部署报告
research primary   = eta_tight_ever    # 真实状态的离线研究主指标
secondary          = eta_tight_at_done # 终止时仍在 tight 半径内
```

`terminal_eta_capture` 当前使用的是 protocol 半径，只能作为标签诊断，不能冒充
tight-radius research success。Goal84 之后，除单因素诊断外，不再每轮重新拆分
p_nav/eta frame；所有报告同时给出 p_nav、eta 和 gap。

## 2. 冻结的 outcome contract

半径固定：

```text
r_tight    = eval_goal_radius_m
r_protocol = plant.goal_radius_m
```

逐 episode 冻结以下定义：

```text
p_nav_tight_success
  = terminal_kind == GOAL
    且成功来自 p_nav 的 tight-radius 终止

eta_tight_ever
  = 任一 active step 满足 ||eta[t] - goal|| <= r_tight + 1e-12

eta_tight_at_done
  = ||eta[t_done] - goal|| <= r_tight + 1e-12

protocol_eta_capture_at_done
  = ||eta[t_done] - goal|| <= r_protocol + 1e-12
```

优先级冻结：

1. `research_primary = eta_tight_ever`；用于研究结论和与传统方法的同协议比较；
2. `deployment_primary = p_nav_tight`；用于训练、终止、部署和真实系统报告；
3. `eta_tight_at_done` 只作次级诊断，不能替代 ever；
4. `protocol_eta_capture_at_done` 必须单独命名，不得与 tight success 合并；
5. 报告中 p_nav、eta、`eta - p_nav` gap 三者必须同时出现。

## 3. 评估开关

新增 eval-only 参数：

```text
--success-frame {p_nav,eta,both}
```

语义：

| 值 | 行为 |
|---|---|
| `p_nav` | 把 `p_nav_tight_success` 作为 selected success；仍保留 eta 诊断字段 |
| `eta` | 把 `eta_tight_ever` 作为 selected research success；rollout 仍按 p_nav 终止 |
| `both` | 同时输出两套 success；`selected_success` 为空，避免混淆 |

默认 `both`。开关只允许影响 episode/summary 的输出字段，**不得进入**：

- reset / rollout 分支；
- reward、termination 或 plane-miss guard；
- checkpoint 选择或 actor 输入；
- seed、半径、dwell 或只读 trace 内容。

向后兼容：`success` 字段继续表示 `p_nav_tight_success`，旧脚本不需要改读法。
新增字段至少包括：

```text
success_pnav_tight
success_eta_tight_ever
success_eta_tight_at_done
success_selected
success_frame
success_frame_gap_eta_minus_pnav
```

## 4. 数据与隔离

| 项目 | 固定值 |
|---|---|
| 上游索引 | Goal83 使用的 Goal82 corrected run 集合 |
| runs / episodes | `20 / 800` |
| seeds | dev `20–59`，40 seeds/run |
| roots | Goal71 baseline、Goal80 candidate |
| scenarios | default、strict125 |
| action modes | policy、zero-residual |
| 半径 | tight `1.75/1.00 m`；protocol `2.00/1.25 m` |
| held-out | `100–149` 关闭 |
| 新 rollout / 训练 | 无 |
| 算法切换 | 禁止；SAC/REDQ/DroQ 不启动 |

禁止把 `eta` 标签、true position 或成功结果反馈到 policy；`eta` 只在 episode 结束后
用于离线评分。禁止在 formal 结果中用 validation 选择 frame。

## 5. 假设

### H84-A：contract 可分离且不改闭环

用同一批 trace 重评分时，新的 `p_nav_tight_success` 应与原 `success` 逐 episode
`800/800` 一致；切换 `--success-frame` 后 `steps`、`reward`、`terminal_kind`、
checkpoint 和 trace SHA 全部不变。

成立应看到：identity 全通过、无 NaN、无 held-out。
失败应看到：任一 episode 的 p_nav success、终止类型、步数或 reward 变化。
最便宜证伪：不加载 policy，只对 800 个已记录 episode 重评分并与原字段逐项比对。

### H84-B：eta tight-ever 提供非零且有界的 frame gap

至少一个 group 的 `eta_tight_ever` 与 `p_nav_tight_success` 不同，且新增 success
主要来自原 p_nav lost episode；policy 组的 additive false-success 不超过预注册
`0.05`。

成立应看到：gap 非零、集中在 lost、false-success 有界。
失败应看到：所有 group gap 为 `0`，或 policy false-success 超过 `0.05`。
最便宜证伪：只读重评分 800 个 episode，按 seed 配对 bootstrap。

口径澄清（analyzer 运行前补充）：`gap = eta - p_nav` 不预设正负。Goal83 已表明
p_nav 估计偏差可能制造“p_nav 成功但 true eta 未进入”的假成功，因此 negative gap
同样是有效证据；Gate 只要求 gap 可解释、方向明确，不能把“eta 不更高”误判成
metric 无效。zero-residual 的 false-success 只作对照，不进入 policy 组阈值。

### H84-C：strict125 的 tracking 增益没有转化为 terminal outcome

在 contract 固定后，strict125 的 policy-vs-zero heading tracking 改善仍达到
`>=10%`，但 `eta_tight_ever` 和 terminal approach 指标没有同向改善。

成立应看到：tracking 改善、outcome gap 不改善或变差。
失败应看到：outcome 随 tracking 同向改善，或 tracking 改善不再复现。
最便宜证伪：复用 Goal82 的 paired 数据，按 `eta_tight_ever` 重算 outcome。

## 6. 方法

1. 新建共享 outcome-contract 工具，集中实现三个 success 定义和 selected-frame 逻辑；
2. 在 off-policy evaluator 和 PPO evaluator 中接入 `--success-frame`，默认 `both`；
3. 只读复用已有 800 个 episode 的 p_nav/eta 位置与 done 行重评分，不重新加载 policy；
4. 输出按 root × scenario 的 p_nav、eta、gap 表和 paired bootstrap CI；
5. 复核 strict125 terminal approach：tracking、closing、terminal authority 与 outcome；
6. 不因 `eta` 结果修改 reward、动作参数化或算法。

## 7. 执行计划

- [x] Step 0：冻结本文件；记录 20 runs 的 trace/config/episode SHA 与半径口径；
- [x] Step 1：实现共享 outcome contract 与 `--success-frame`，默认 `both`；
- [x] Step 2：补单元测试：p_nav identity、eta tight-ever、both 输出、eta 不改变 stopping/reward；
- [x] Step 3：跑 `pytest`、`ruff` 和 smoke；失败只修实现，不改 Gate；
- [x] Step 4：只读重评分现有 800 episodes，输出 `goal84.json` / `goal84.md`；
- [x] Step 5：Gate 判为 `FRAME_ACCOUNTED`；完成 strict125 tracking/outcome 次级审计，actuator/authority 留给 Goal85；
- [x] Step 6：写 `goal84_result_20261110.md`、decision、LEDGER、STATE；
- [x] Step 7：运行项目与工作区 checker；不自动 commit / push。

每一步验证命令：

```bash
# Step 1-2
pytest -q tests/test_pc_goal84_outcome_contract.py

# Step 3
pytest -q
ruff check scripts src tests

# Step 4（远端，按实际脚本名执行）
/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python \
  scripts/analyze_pc_goal84_outcome_contract.py \
  --output artifacts/lowlevel/goal84_outcome_contract_20261110/goal84.json

# Step 7
python ../scripts/check_research_ledger.py --goal Goal84 \
  --goal-doc doc/goals/goal84_outcome_contract_success_frame_20261110.md \
  --strict --require-result
git diff --cached --check
```

## 8. Gate

| check | 阈值 |
|---|---:|
| p_nav success identity | `800/800` |
| steps / reward / terminal_kind 不改 | `800/800` |
| NaN / invalid / held-out | `0` |
| both 输出完整性 | p_nav、eta-ever、eta-at-done、gap 全有 |
| eta additive false-success（policy 每组） | `<= 0.05` |
| checkpoint / seed / trace SHA | 不变 |

裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| p_nav identity 或闭环字段变化 | `INVALID_CONTRACT` | 回滚开关实现，修 evaluator，不分析结果 |
| 所有 group eta gap 为 0 | `CONTRACT_NO_GAP` | 停止反复拆 frame，转 strict125 terminal approach |
| eta gap 非零（任意方向）且 policy false-success `<=0.05` | `FRAME_ACCOUNTED` | 固定双口径，进入 terminal approach 审计 |
| eta false-success `>0.05` | `CONTRACT_REJECTED` | 保留 p_nav formal；写 blocker，进入 autoresearch |
| strict125 tracking 好但 eta outcome 差 | `TERMINAL_CHAIN_AUDIT` | 优先查 actuator/terminal authority，不换算法 |

## 9. Git

禁止 `git add -A`。只 add 本 goal 的文档、共享工具、两个 evaluator、测试、分析器、
结果和 ledger/STATE；不要提交 artifacts 之外的临时 trace 或 checkpoint。

建议提交命令（用户明确要求后才执行）：

```bash
git add doc/goals/goal84_outcome_contract_success_frame_20261110.md \
  doc/goals/goal84_result_20261110.md \
  src/auv_lowlevel/outcome_contract.py \
  scripts/evaluate_pc_command_residual_offpolicy.py \
  scripts/evaluate_pc_ppo.py \
  scripts/analyze_pc_goal84_outcome_contract.py \
  tests/test_pc_goal84_outcome_contract.py \
  research/LEDGER.jsonl research/STATE.md
git diff --cached --check
git commit -m "eval: freeze p_nav/eta outcome contract"
git push
```

## 10. Goal 提示词

```text
你现在执行 Goal84：冻结 p_nav/eta outcome contract，并实现 eval-only success-frame 重评分。

先读：
- doc/goals/goal84_outcome_contract_success_frame_20261110.md
- doc/goals/goal83_result_20261110.md
- research/STATE.md

硬约束：
1. research primary = eta_tight_ever；deployment primary = p_nav_tight；
2. eta_tight_at_done 和 protocol radius 必须单独命名，不能混用；
3. --success-frame 只影响输出，不影响 rollout/reward/termination/checkpoint；
4. 只读复评现有 800 个 episode，不训练、不新增 rollout、不打开 held-out；
5. 不启动 SAC/REDQ/DroQ；不因本 Gate 改 reward 或动作参数化。

执行顺序：
- 实现共享 outcome contract；
- 接入 off-policy 与 PPO evaluator；
- 单元测试与 smoke；
- 只读重评分 800 episodes；
- 按 Gate 裁决并写 result/ledger/STATE。
```
