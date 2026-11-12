# Goal163：闭环速度包线与 anti-windup 审计

**日期**：2026-11-12<br>
**状态**：`PREREGISTERED_BEFORE_MEASUREMENT / READ_ONLY_AUDIT / NO_RL_TRAINING / HELDOUT_CLOSED`<br>
**上游**：Goal162 `INVALID_SPEED_CONTRACT`；Goal149 current saturation ladder

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
  check_command: .venv/bin/python ../scripts/check_research_ledger.py --goal Goal163 --goal-doc doc/goals/goal163_speed_envelope_audit_20261112.md --strict
```

`required: true`：Goal163 是探索型审计；任何机制结论都必须有 trace 证据、至少两个
可证伪 hypothesis 和 ledger decision。Goal163 不启动 RL 训练。

## 1. 背景与证据

Goal162 的 H3 zero 与 gain residual 都在 dev40 上超过 REMUS100 `2.5 m/s` 地速：

| level | zero max | policy max | zero over 2.5 | policy over 2.5 |
|---|---:|---:|---:|---:|
| nominal | 2.563 | 2.562 | 0.96% | 0.96% |
| current10 | 2.879 | 2.879 | 7.20% | 7.20% |
| current15 | 3.308 | 3.308 | 14.68% | 14.70% |

训练配置固定 `1.2 m/s`，`direct_commands_active=false`，课程采样已保证
`speed + current + variation <= 2.5`。因此问题位于闭环控制/plant，而不是策略动作
或课程上界。

## 2. 假设

### H163-A：速度环在减速段存在积分或 RPM 回收滞后

- 机制：积分饱和/螺旋桨一阶动态使指令已降但 actual RPM 仍高，`nu` 和 ground
  speed 在数秒内维持超限。
- 成立应看到：超速样本的 RPM 回收明显慢于非超速样本；补记
  `speed_integral_rpm` 后在超速起点前后出现积分未及时回退。
- 失败应看到：RPM 已按指令回收而 ground speed 仍高。
- 最便宜证伪：固定 current10/current15 的 zero policy 做 deterministic replay，
  只读分箱，不训练。

### H163-B：转向/海流瞬态使开环包线低估闭环地速

- 机制：`reference + current_max + variation` 没有包含转向诱导的速度耦合、闭环
  overshoot 和 plant mismatch；zero controller 在强流转向时自然超限。
- 成立应看到：超限集中在高 turn rate/current 分箱；关闭 current 或降低 speed
  command 后同 seed 超限消失。
- 失败应看到：低 current、直线、无转向 replay 仍稳定超限。
- 最便宜证伪：同一批 zero replay 按 current、turn rate、speed command 和
  distance 分层，比较超限步比例。

### H163-C：评价速度定义或 plant backend 有系统错位

- 机制：`ground_velocity_body_mps`、`nu + current` 和 sensor ground speed 的坐标/
  符号定义不一致，导致“超限”是测量伪影。
- 成立应看到：多个 trace 字段无法在误差范围内互相对齐；零 current、低速直线也
  超限。
- 失败应看到：三个独立速度量一致，且超限只在强 current/转向出现。
- 最便宜证伪：逐 step 对齐 `ground_velocity_body_mps`、`nu`、`command_physical`、
  current 和 `eta` 差分，先不碰控制器。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| 数据 | Goal162 已有 train/eval trace；不新增训练 |
| primary replay | current10/current15 × seeds 86201–86203 × H3 zero |
| diagnostic | 同 seed 的 policy trace 只用于检查 residual 是否改变超限 |
| dev | seeds 20–59；不打开 held-out |
| 修改范围 | 只读分析脚本、trace 诊断字段、Goal163 文档 |
| 禁止 | 改 reward/terminal/success/radius、改 Gate、扩 seed、换 PPO/SAC |

## 4. 方法

1. 增加不改变控制行为的 rich-trace 字段：speed error、speed integral、commanded/
   actual RPM、`nu`、ground speed、current、turn rate、acceleration、reachability。
2. 对 Goal162 的 zero trace 做 deterministic replay；失败样本按 current、turn rate、
   distance 和 acceleration 分箱。
3. 对每个超限 episode，标记减速事件、超限起止时间、积分/RPM 回退时间和峰值
   ground speed。
4. 只在审计完成后提出最小控制器修正；修正必须另开预注册实验，不在 Goal163 内
   偷偷改算法或改 Gate。

## 5. 执行计划

- [x] Step 0：写 Goal163、Research Contract、边界和课程速度约束；
- [x] Step 1：实现只读速度包线分析器与 trace 字段检查；
- [x] Step 2：在 1 个 current10 seed 上做本地/远端 smoke；
- [x] Step 3：跑 current10/current15 × 3 seeds deterministic replay；
- [x] Step 4：裁决 H163-A/B/C，写 blocker/result/ledger；
- [x] Step 5：若根因明确，预注册最小控制器修正；否则停止并报告；
- [x] Step 6：跑 checker、更新 STATE；不自动 commit/push。

**结果**：`doc/goals/goal163_result_20261112.md`。H163-C 强支持：`nu` 是地速，
`ground_velocity_body` 和 `sensors.py` 重复加海流；真实超速仅
`0.31%/0.51%`，且全部在 terminal phase。H163-A 部分支持，H163-B 弱支持。
决策为 `pivot` 到 Goal164。

验证命令：

```bash
# Step 2
.venv/bin/python -m pytest -q tests/test_lowlevel_curriculum_sampler.py

# Step 3
.venv/bin/python scripts/audit_pc_speed_envelope.py \
  --trace-root artifacts/lowlevel/goal162_controller_gain_20261112/eval \
  --output artifacts/lowlevel/goal163_speed_audit_20261112

# Step 6
.venv/bin/python ../scripts/check_research_ledger.py \
  --goal Goal163 \
  --goal-doc doc/goals/goal163_speed_envelope_audit_20261112.md \
  --strict
```

## 6. Gate

| check | 阈值 |
|---|---:|
| replay 完整性 | 3 seeds × 2 levels × zero 全部完成，0 NaN/飞点 |
| 速度定义 | 三个速度量的误差可解释，不能出现互相矛盾的坐标 |
| 机制识别 | 至少一个 hypothesis 有分箱或时间序列证据，而不是只看峰值 |
| 课程约束 | 所有训练/评估场景满足 `speed + current + variation <= 2.5` |
| 风险 | 不新增训练；不改 reward/terminal/success/radius；held-out 关闭 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| RPM/积分回退滞后 | 速度环 anti-windup 问题 | 预注册 anti-windup/RPM 回收修正 |
| 只在强流/转向超限 | 闭环包线低估 | 预注册 ground-speed safety cap 或更保守 envelope |
| 低速直线也超限 | 速度定义/plant backend 问题 | 修评价或 plant，不动 RL |
| trace 不足以判定 | 诊断字段不足 | 只增加只读字段，扩一次 replay，不训练 |

## 7. Git

禁止 `git add -A`。只 add Goal163 相关文件。提交前运行 checker 和
`git diff --cached --check`；不自动 commit/push。

## 8. Goal 提示词

```text
你现在执行 Goal163：只读审计 Goal162 的闭环速度包线，不训练、不换算法。

先读：
- doc/goals/goal163_speed_envelope_audit_20261112.md
- doc/goals/goal162_result_20261112.md
- research/blockers/2026-11-12-goal162-speed-envelope-invalid.md
- research/paper_reading/2026-11-12-goal162-speed-envelope-and-gain-scheduling.md

硬约束：
1. REMUS100 地速硬上限 2.5 m/s；课程 C6 为 0.90–2.50 m/s，且
   speed + current + variation <= 2.5；
2. 不训练、不扩 seed、不打开 held-out、不改 reward/terminal/success/radius；
3. 不换 PPO/SAC/REDQ/DroQ；不把速度通道加回策略动作；
4. 先复用 Goal162 zero/policy trace，再补只读诊断字段；
5. 超限必须按 current、turn rate、distance 和减速事件分箱，不能只报峰值；
6. 机制不明确时只补诊断，不直接改控制器；明确后才预注册最小修正。

执行顺序：
- 检查 trace 字段和三个速度定义；
- current10/current15 × 3 seeds deterministic replay；
- 审计 H163-A/B/C；
- 写 result/blocker/ledger/STATE；
- 通过 Gate 后才讨论最小速度环修正。
```

## 9. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`。失败后先写 blocker 和至少两个可证伪
hypothesis，检索论文/代码并写 reading note，先跑最便宜 falsification；只有
`pivot` 才开新 Goal。直到速度契约恢复有效，才允许恢复 Goal162 的 gain residual
对照或扩 seed。
