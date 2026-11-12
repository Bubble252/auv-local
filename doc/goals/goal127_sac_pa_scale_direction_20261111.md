# Goal127：SAC P-A 残差是“方向错”还是“幅值过大”

**日期**：2026-11-11<br>
**状态**：`COMPLETED_EVAL_ONLY / SAC_PA_SCALE_INSUFFICIENT / HELDOUT_CLOSED`<br>
**上游**：Goal126 `SAC_PA_HARM`；Goal125 `PPO_BIAS_CEILING`<br>
**性质**：eval-only 诊断；复用 Goal126 的 8 个 1M SAC checkpoint，不训练、不换算法、不修改 reward/terminal/success/goal radius。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal127 --goal-doc doc/goals/goal127_sac_pa_scale_direction_20261111.md --strict --require-result
```

测量开始后不得修改 seed、倍率网格、success frame、goal radius、checkpoint 或判读阈值。

## 1. 背景与证据

Goal126 修正评估侧重复缩放后，16 个 deterministic dev40 评估有效：

| 指标 | 结果 |
|---|---:|
| strict mean | `19.75/40` |
| default mean | `28.25/40` |
| strict 达 `>=26` 的 seed | `1/8`（仅 86201：30） |
| default 达 `>=31` 的 seed | `1/8` |
| timeout+safety | `3`，全部来自 86204 |

评估动作已回到 `abs_max=0.985–1.000`，配置 scale、额外倍率和名义有效 scale 分别为
`2.0/1.0/2.0`。因此 Goal126 的退化不是旧的 2 倍评估放大。

同时，SAC 已执行残差是 frozen BC base 的 `0.35–0.68` L2 比；PPO 在 Goal125 的
deterministic 增量只有 `0.7%–4.0%`。失败 episode 几乎都是 `TIME_LIMIT`，不是安全终止。
这说明 SAC 没有被 NoisyNet 主导（当前 SAC 没有 NoisyNet），也没有“闭环完全不动”；
更可能是学会了大幅值、方向局部可用但不稳健的残差。

关键观察是倍率并非越大越好：

```text
旧评估额外 x2（名义有效 x4）: strict mean 14.375
修正后额外 x1（名义有效 x2）: strict mean 19.75
```

因此必须先做幅度剂量曲线。若降幅后恢复，说明主因是动作参数化/残差 gain；若降到零
附近仍不如 fixed-bias control，才说明当前 1M SAC 学到的方向或任务 credit 本身不可用。

## 2. 假设

### H127-A：残差方向可用，但幅值过大

- 成立：至少一个倍率 `<1.0` 达到 strict mean `>=26`、至少 4/8 seed `>=26`、
  每个 seed default `>=31`，且 timeout/safety 为 0。
- 失败：所有倍率的 strict mean `<=24.5`，或没有任何倍率优于 fixed-bias control。
- 最便宜证伪：8 个已有 checkpoint × 4 个倍率 × strict/default dev40。

### H127-B：幅值只是次因，当前方向仍然有害

- 成立：所有非零倍率都不高于 zero-residual 的 strict `24/40`，且最佳倍率仍丢多个
  control success。
- 失败：任一非零倍率稳定超过 `24/40`。
- 最便宜证伪：同上；zero-residual control 直接复用已完成的 `24/33`。

### H127-C：退化由单一执行器通道主导

- 成立：某通道 residual/base RMS 比显著高于其他通道，动作饱和也集中在该通道；
  降该通道 gain 的收益应大于等比降三通道。
- 失败：三通道残差比接近，或没有通道在失败 episode 中占主导。
- 最便宜证伪：新评估器记录三通道 action RMS、residual/base RMS 和动作饱和比例。

### H127-D：只有 seed 86201 是局部幸运点

- 成立：倍率改善只出现在 86201，不能达到 4/8 seed 的复现要求。
- 失败：至少 4 个 seed 在同一个倍率上达到 `>=26`。
- 最便宜证伪：同一倍率逐 seed 配对判读，不按 pooled mean 掩盖。

### H127-E：评估链仍存在缩放错误

- 成立：summary 的 `nominal_effective_actuator_residual_scale` 不等于
  `2.0 × multiplier`，或动作 `abs_max>1.001`。
- 失败：所有 summary 字段一致且动作位于 `[-1,1]`。
- 最便宜证伪：每个 dose run 的 manifest 与动作统计。

## 3. 数据与隔离

| 项目 | 值 |
|---|---|
| checkpoints | Goal126 `screen/S_s86201..86208/final.zip` |
| scenarios | strict `B15` / default `default_B15` |
| eval seeds | dev `20–59`，每 scenario 40 episodes |
| success frame | `both`，不改 |
| goal radius | 环境配置原值，不改 |
| zero control | `init_parity_eval`：strict `24/40`、default `33/40` |
| 已有 `1.0` 点 | Goal126 `eval/` |
| held-out | `100–149` 关闭 |

## 4. 方法

固定 checkpoint 与环境动作契约，只把 deterministic actor 输出在进入环境前乘以下列
额外倍率：

```text
0.125, 0.25, 0.5, 0.75
```

已有观测点：

```text
0.0  = zero-residual control（24/33）
1.0  = Goal126 primary screen（19.75/28.25）
2.0  = 旧错误评估（已归档，不作为算法结果）
```

环境仍从 config 读取 `actuator_residual_scale=2.0`。新 summary 必须记录：

```text
configured_actuator_residual_scale = 2.0
eval_residual_multiplier = m
nominal_effective_actuator_residual_scale = 2.0 * m
```

该 Goal 不把任何倍率结果写成新的训练收益；它只用于决定下一轮应该修动作参数化、
reward/terminal credit，还是保留当前 SAC 方向继续训练。

## 5. 执行计划

- [x] Step 0：修正 Goal126 评估侧重复 scale，归档旧评估；
- [x] Step 1：记录动作饱和、三通道 RMS 和 residual/base 通道比；
- [x] Step 2：64 个 dev40 dose eval（4 倍率 × 8 seed × 2 scenario）；另补 16 个 `1.0` 参照点；
- [x] Step 3：逐 seed 配对与 `0/1.0` 观测点汇总；拆分 frozen BC 与 SAC learner；
- [x] Step 4：按 Gate 判读为 `SAC_PA_SCALE_INSUFFICIENT`，写 result；
- [x] Step 5：写 blocker/reading note/LEDGER/STATE；
- [x] Step 6：运行 checker 通过；不自动 commit/push。

## 6. Gate

先做有效性检查：checkpoint 哈希存在、全部 finite、倍率 manifest 一致、动作
`abs_max<=1.001`、未打开 held-out。

| 观察 | 判读 | 下一步 |
|---|---|---|
| 任一倍率 strict mean `>=26`、至少 4/8 seed `>=26`、每 seed default `>=31`、timeout/safety=0 | `SAC_PA_SCALE_RECOVERS` | 预注册小 gain/重训练对照；保持 SAC 与任务契约 |
| 最佳倍率 strict mean `>24.0` 但不达 26，default `>=31`、安全为 0 | `SAC_PA_SCALE_PARTIAL` | 修残差 gain/通道参数化，不换算法 |
| 最佳倍率 strict mean 在 `(19.75,24.5]`，default `>=31`、安全为 0 | `SAC_PA_SCALE_INSUFFICIENT` | 幅值是次因；转 reward/terminal capture 审计 |
| 最佳倍率不高于 19.75 加 0.5，或所有非零倍率丢 control seeds | `SAC_PA_DIRECTION_NEGATIVE` | 不再调 SAC 超参；先查 reward/terminal/动作 credit |
| 任一倍率 timeout/safety>0 | 该倍率记 `HARM`，不用于选型 | 保留证据；仍按其余倍率判读 |
| manifest/数值/动作范围不成立 | `INVALID` | 修评估链后重跑；不改阈值 |

“最佳倍率”按 strict mean 降序、default mean 降序、timeout+safety 升序选择；选择结果
只用于诊断，不直接作为 formal 结论。

## 7. 停止与自治边界

- 不训练新的 SAC，不扫描 seed、lr、UTD、entropy 或网络宽度；
- 不因 dose 结果直接打开 held-out；
- 不把 dev 上的最佳倍率写成已部署增益；
- 若 `DIRECTION_NEGATIVE`，下一步必须审计 reward/terminal capture，而不是换 REDQ/DroQ；
- 只有明确证明“方向对、但样本效率不足”时，才允许预注册 REDQ/DroQ 对照。

## 8. Git

只 add 本 Goal 相关脚本、文档和 research 记录。禁止 `git add -A`；不提交
`artifacts/`、checkpoint、日志或缓存。

## 9. Goal 提示词

```text
你现在执行 Goal127：SAC P-A 残差剂量/方向诊断。

先读：
- doc/goals/goal127_sac_pa_scale_direction_20261111.md
- doc/goals/goal126_sac_pa_bias_prior_20261111.md
- research/blockers/2026-11-11-goal126-sac-pa-harm.md

硬约束：
1. 不训练、不换算法、不改 reward/terminal/success/goal radius；
2. 只复用 Goal126 的 8 个 1M checkpoint；
3. 倍率网格固定为 0.125/0.25/0.5/0.75，已有 0 和 1.0 观测点；
4. 不打开 held-out 100–149；
5. 按逐 seed 结果判读，不用 pooled mean 掩盖单 seed 幸运点；
6. 先判断是幅值、方向、单一通道还是 reward/terminal credit，再决定下一 Goal。
```

## 10. 自主迭代契约

执行 `research/AUTORESEARCH_LOOP.md`：失败先归因，至少写两个可证伪 hypothesis；
查阅论文/代码并写 reading note；先跑最便宜 falsification；在 LEDGER 记录
`continue/pivot/stop`；不改失败 Gate 的阈值；不自动 commit/push。
