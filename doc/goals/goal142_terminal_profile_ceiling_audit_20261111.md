# Goal142：终端 profile selector 的 headroom 审计

**日期**：2026-11-11<br>
**状态**：`EVAL_ONLY / FROZEN_TRACES / PROTOCOL_REUSES_GOAL141_THRESHOLDS`<br>
**上游**：Goal141 `terminal_contract_audit`、Goal117/131 `terminal geometry`、Goal123/124/125 `fixed bias ceiling`、Goal133 `Vm010`<br>
**性质**：只读审计冻结 trace；不训练，不改 reward、terminal、success、goal radius、动作参数化或 checkpoint，不打开 held-out。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal142 --goal-doc doc/goals/goal142_terminal_profile_ceiling_audit_20261111.md --strict --require-result
```

`required: false`：本 Goal 只读取已冻结的 dev40 trace，不产生新训练或新
checkpoint。terminal 包络阈值沿用 Goal141 在测量前固定的值；本轮不新增阈值，
也不根据结果调参。

## 1. 背景与问题

Goal141 用 Goal86 BC-residual proxy 得到两个结论：

- `GOAL` / `GOAL_PLANE_MISS` 与 raw safety 共存为 `0/80`；
- failure 在 2 m 入口前已经 geometry-bad，success 的松动主要是
  `|vertical_error| >= 1 m`。

但 Goal141 本机没有 Goal95 `P_D3/final.pt`，因此只能把结论标成 proxy。
远端已有 P_D3 的 dev40 正式 trace，以及 Goal123/124/125 的 B20、BP3、
stern bias、zero 对照 trace。现在可以回答两个更具体的问题：

1. P_D3 正式 trace 是否复现 BC proxy 的失败结构？
2. 如果只根据可观测的 2 m / 3 m 入口几何，在已有 profile 之间做 selector，
   strict125 的并集上限是否足以支持一个新的 selector 控制器？

## 2. 假设

### H142-A：P_D3 正式 trace 复现 BC proxy

- 成立应看到：failure 全部是 `GOAL_PLANE_MISS`；failure 的 2 m 入口
  `|vertical_error|` 明显大于 success；success 仍有部分 `|vertical_error| >= 1 m`；
  P_D3 相对 zero-residual 没有稳定 success 增益。
- 失败应看到：P_D3 的 success/failure 几何与 BC proxy 相反，或 P_D3 显著优于
  zero-residual。
- 最便宜证伪：直接读取 P_D3 strict125/default 的 dev40 trace，不重跑环境。

### H142-B：已有 profile 的 selector 并集足够大

候选 profile 固定为 B20、BP3、stern-0.25、zero，以及 Goal133 已记录的 Vm010。
如果 selector 有实际价值，应看到这些 profile 的成功 seed 明显互补，并集至少
达到 `27/40`，且可以通过 2 m / 3 m 入口几何在 seed 外做选择。

- 成立应看到：profile 并集 `>= 27/40`，且新增 seed 不是单点边界翻转。
- 失败应看到：并集 `<= 25/40`，或所有新增 seed 都来自同一个小几何翻转。
- 最便宜证伪：对冻结 trace 取 success seed 并集，不训练 selector。

### H142-C：B20 的 4 个增益是稳健的纵向捕获改善

B20 相对 zero 在 strict125 上 gained `{29,31,33,56}`、lost `[]`。

- 成立应看到：这 4 个 seed 的 2 m 入口纵向误差和 last-2s closure 都显著改善，
  且 terminal `|vertical_error|` 降到 `1 m` 以下。
- 失败应看到：只出现 `0.1–0.4 m` 的小幅 terminal 改善，仍有成功回合
  `|vertical_error| >= 1 m`；此时应判为边界翻转，而不是学会了捕获。
- 最便宜证伪：逐 seed 配对 B20 与 zero 的 2 m 入口、closure、terminal vertical。

### H142-D：pitch-rate 包络只是 proxy 伪影

Goal141 的 BC proxy 中 success 的 `|q|` 没有超过 15 deg/s。

- 成立应看到：P_D3/B20 的 success 也没有 `|q| >= 15 deg/s`。
- 失败应看到：正式 trace 的 success 有多个 `|q| >= 15 deg/s`；此时 pitch-rate
  只能作为 secondary metric，不能单独作为新的 success 条件。

## 3. 数据与隔离

| 项目 | 固定值 |
|---|---|
| P_D3 trace | `traces_dev40/P_D3_s86201_{default,strict125}_{policy,frozen-base}/trace.npz` |
| B20/BP3 trace | `goal125_bias_ppo_20261111/eval/{B20,BP3}_s86201|s86202/{strict,default}/trace.npz` |
| bias/zero trace | `goal124_bias_guardrail_20261111/{strict,default}_{stern-025,stern-030,zero}/trace.npz` |
| Vm010 | Goal133 已记录的 strict125 结果，只取 success seed 集合 |
| dev seeds | `20-59`，每 trace 40 episodes |
| scenarios | `strict125`、`default` |
| success 契约 | 现有 `GOAL`；Goal85 `unbiased_noise_v1` 与 `eta` 位置源 |
| held-out | `100-149` 关闭 |

禁止：

- 用 held-out 选 profile 或阈值；
- 在同一审计里改变 goal radius、success frame、terminal 或 checkpoint；
- 把并集上限写成已实现 selector 的收益；
- 用 selector 结果替代真实闭环评估。

## 4. 指标

每个 trace 只计算以下冻结指标：

- success / `GOAL_PLANE_MISS` / timeout / safety 计数；
- terminal `|pitch|`、`|q|`、`|w|`、`|vertical_error|`；
- 首次进入 2 m 时的 `|vertical_error|`、cross-track、`|pitch|`；
- terminal 前 2 s 的 radial closure slope；
- 每个 profile 的 success seed 集合、两两 gained/lost、并集。

`vertical_error` 在 endpoint 处取 `eta_z - goal_z`；cross-track 由
`sqrt(max(d^2 - tangent_offset^2 - dz^2, 0))` 计算。该定义与 Goal141 的
path projection 在 endpoint 处一致。

## 5. 执行计划

- [x] Step 0：固定 trace 列表、dev split、指标和阈值；
- [x] Step 1：审计 P_D3 strict125/default；
- [x] Step 2：审计 B20/BP3/stern/zero，并做逐 seed 配对；
- [x] Step 3：读取 Goal133 Vm010 的 success seed 集合，计算 selector 并集上限；
- [x] Step 4：按 H142-A/B/C/D 判读；
- [x] Step 5：写 result md、ledger 和 decision；不自动 commit/push。

## 6. Gate

| check | 阈值 |
|---|---:|
| trace 完整性 | 每个 trace 40 episodes，terminal kind 非空 |
| P_D3 复现 proxy | failure 以 `GOAL_PLANE_MISS` 为主，2 m 入口 vertical 可区分 |
| selector 并集 | `>= 27/40` 才允许新 selector Goal |
| 边界翻转 | gained seed 的 terminal vertical 改善 `< 0.4 m` 或仍 `>= 1 m` 时记为 weak |
| held-out | 必须为关闭 |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| 并集 `<= 25/40`，profile 无互补 | `SELECTOR_HEADROOM_INSUFFICIENT` | 不开 selector；转 C-line 课程 |
| P_D3 与 proxy 结论相反 | `P_D3_PROXY_CONTRADICTION` | 先查 trace 协议和 checkpoint 加载 |
| B20 增益仍是边界翻转 | `B20_GAIN_BOUNDARY_FLIPS` | 不继续扫固定 bias；转早期纵向跟踪 |

## 7. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal Goal142 \
  --goal-doc doc/goals/goal142_terminal_profile_ceiling_audit_20261111.md \
  --strict --require-result
git diff --cached --check
```

## 8. Goal 提示词

```text
你现在执行 Goal142：终端 profile selector 的 headroom 审计。

先读：
- doc/goals/goal141_result_20261111.md
- doc/goals/goal123_result_20261111.md
- doc/goals/goal125_result_20261111.md
- doc/goals/goal133_result_20261111.md

硬约束：
1. 只读冻结 trace，不训练、不改 reward/terminal/success/radius/checkpoint；
2. dev seeds 固定 20-59，held-out 100-149 关闭；
3. 不换 SAC/PPO，不引入 REDQ/DroQ，不扫固定 bias 或 residual scale；
4. 先做 seed 配对和并集，再决定是否值得开 selector Goal。

执行顺序：
- 审计 P_D3 strict125/default；
- 审计 B20/BP3/stern/zero；
- 计算 profile 并集和 2 m 入口几何；
- 写 result 与 ledger，decision 只能是 continue/pivot/stop。
```

## 9. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`。若 selector 并集不足，不在本
Goal 内改算法或加预算；写 `pivot`，把下一优先级交给 Goal136 C-line 课程，
Goal140 仍排在其后。只有明确证明“残差方向对，但探索/样本效率不足”时，才允许
预注册 SAC 对照或 REDQ/DroQ。
