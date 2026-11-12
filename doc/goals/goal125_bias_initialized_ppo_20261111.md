# Goal125：bias-initialized PPO 能否在 fixed stern bias 之上继续提升

**日期**：2026-11-11<br>
**状态**：`COMPLETED / PPO_BIAS_MIXED / HELDOUT_CLOSED`<br>
**上游**：Goal124 `BIAS_GUARDRAIL_PASS`（strict 24/40、default 33/40）<br>
**性质**：训练型 screen；不换算法、不改 success/goal radius、不打开 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal125 --goal-doc doc/goals/goal125_bias_initialized_ppo_20261111.md --strict --require-result
```

记录前不得修改训练臂、seed、eval split、success frame、checkpoint 选择规则或 Gate 阈值。

## 1. 背景

Goal120–Goal124 已经把失败链压缩成：

```text
PPO actor residual ≈ 0.007–0.016（BC base 的约 3%）
-> PPO 自身几乎没有离开零初始化
-> 放大 PPO 方向有权限但不 capture-aligned
-> fixed stern bias -0.25 在 strict125 恢复 4 个 success 且不伤 default
```

Goal124：

| profile | strict | default |
|---|---:|---:|
| zero | 20/40 | 33/40 |
| stern-025 | 24/40 | 33/40 |
| stern-030 | 24/40 | 33/40 |

固定 bias 是 eval-only 参数化发现，不是学习策略。Goal125 要回答：

```text
把 stern-025 作为 base prior 后，PPO 能否在它上面继续学出 +2 strict success，
同时不破坏 default？
```

## 2. 网络与执行链 parity（Step 0，已完成）

用户要求网络结构必须与旧代码/旧实验逐张量比对。已在正式训练前完成：

| 检查 | 结果 |
|---|---|
| Goal119 `C3_s86201/model_initial.pt` vs Goal125 bridge smoke `model_initial.pt` | 63/63 tensor keys、shape、value 完全一致 |
| Goal108 `B20_PPO_init/model_initial.pt` vs 同一 Goal125 init | 63/63 tensor 完全一致；architecture metadata 无差异 |
| actor residual 输出层 `actor.4.{weight,bias}` | 两侧全零 |
| NoisyNet / `rho` / `sigma` / `noise` 参数 | 不存在 |
| policy + `base_residual_bias=(0,-0.25,0)` strict dev40 | `24/40`，success seed 集合与 Goal124 `strict_stern-025` 完全一致 |
| gained / lost vs Goal124 stern-025 | `[] / []` |

因此 Goal125 新增的只有 **环境侧常数动作 bias**，不改：

```text
actor 131→256→128→3
critic 131→256→128→1
state encoder 20→128→128→60
path encoder 8→64→64
endpoint encoder 4→32→64
current encoder 3→64→60
d_z_total=64, n_z_reserved=4
```

参考基线：

- Goal95 P_D3 SHA：`2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616`
- Goal86 BC SHA：`80024ef5798cfbc9701b974ab14b0eb5ce082daaea9c665978431978d59ff9dd`

## 3. 假设

### H125-A：bias prior 能打破 PPO 零解锁定

- 成立：任一 bias arm strict mean `>=26/40`，且每个 seed 不低于同 seed fixed-bias control。
- 失败：所有 bias arm strict mean `<=24.5/40`，PPO 仍只做零附近扰动。
- 最便宜证伪：10M × 2 seeds 的 B20 臂；若不到 25，其他 bias 臂也只需看是否一致。

### H125-B：训练/评价 actuator scale 对齐会放大 PPO 增益

- 成立：B20（train scale 2.0）strict mean 比 B15（train scale 1.5）高 `>=1`。
- 失败：B20 与 B15 差异 `<=0.5`。
- 最便宜证伪：B15/B20 同 seed 配对，共用同一 fixed-bias control。

### H125-C：PPO 内部优化仍是次级瓶颈

- 成立：BP3（normalize advantage + reward scale 0.1）strict mean 比 B20 高 `>=1`。
- 失败：BP3 不高于 B20。
- 最便宜证伪：BP3 与 B20 的 2-seed screen。

### H125-D：fixed bias 是当前参数化上限

- 成立：所有训练 arm strict mean `<=24.5/40`，且 default 不下降。
- 失败：至少一个 arm strict mean `>=26/40`。
- 最便宜证伪：Goal125 全矩阵。

### H125-E：PPO 会破坏 bias prior

- 成立：任一 seed strict 低于 fixed-bias control 2 个以上，或 default 低于 control 2 个以上。
- 失败：所有 arm 在 control 附近或之上。
- 最便宜证伪：先看 iteration 0 与 final 的 strict/default paired 差。

## 4. 数据与隔离

| 项目 | 值 |
|---|---|
| base checkpoint | Goal86 P-A residual BC，SHA `80024e...f9dd` |
| reward | C3：terminal_progress=8.0，approach_enter=5.0；P3 另加 advantage normalization + `pc_ppo_scale=0.1` |
| strict env | Goal119 C3 `goal_radius_m=1.25`，pitch limit 15 deg/s |
| default env | 同 C3，仅 `goal_radius_m=2.0` |
| dev seeds | 20–59 |
| success frame | `both` |
| eval scale | actuator residual scale `2.00` |
| held-out | `100–149` 关闭 |
| training seeds | 86201、86202 |

## 5. 实验矩阵

所有训练臂：400 iter × 25 step × 1000 env = 10M transitions；4 PPO epochs、32 mini-batches、critic warmup=1。

| arm | bias `(rudder,stern,rpm)` | train actuator scale | optimizer | 训练 seed |
|---|---|---:|---|---|
| `Z15` | `(0,0,0)` | 1.5 | C3 原设置 | 86201/86202 |
| `B15` | `(0,-0.25,0)` | 1.5 | C3 原设置 | 86201/86202 |
| `B20` | `(0,-0.25,0)` | 2.0 | C3 原设置 | 86201/86202 |
| `BP3` | `(0,-0.25,0)` | 2.0 | normalize advantage + reward scale 0.1 | 86201/86202 |

总计 8 个训练 job，计划 8 × 4090 并行。

### 5.1 同 seed fixed-bias control

训练后每个 arm 必须与“同训练 seed 的 Goal119 C3 final + 注入 `const:0,-0.25,0`”配对：

| control | checkpoint | 场景 |
|---|---|---|
| `F86201` | Goal119 `C3_s86201/final.pt` | strict + default |
| `F86202` | Goal119 `C3_s86202/final.pt` | strict + default |

Goal124 已证明 `F86201` 应为 strict `24/40`、default `33/40`；Goal125 必须逐 seed 精确复现，否则判 `INVALID`。

### 5.2 评价

每个 checkpoint 评 strict + default 各 dev40：

| 指标 | 用途 |
|---|---|
| success（both） | 主指标 |
| gained/lost vs 同 seed fixed-bias control | 配对方向 |
| timeout / safety / NaN | 有效性 |
| PPO residual L2 / base L2 | 是否仍零锁定 |
| `final_eta_error_l2` / plane miss | 机制 |
| approx_kl / clip fraction | 优化是否真的移动 |

## 6. Gate

先做有效性检查：

| 观察 | 状态 |
|---|---|
| parity smoke 不通过，或 `F86201` 不复现 strict 24/default 33 | `INVALID` |
| 任一 run safety>0、NaN/Inf | `INVALID` |

训练臂判读：

| 观察 | 状态 | 下一步 |
|---|---|---|
| 任一 arm strict mean `>=26/40`，每个 seed `>=` 同 seed fixed-bias control，default mean `>=` control mean−2 | `PPO_BIAS_GAIN` | 预注册正式扩 seed / held-out 前审计 |
| 所有 arm strict mean `<=24.5/40`，default 不伤 | `BIAS_FIXED_IS_CEILING` | 不再堆 PPO 预算；转 action/reward 归因 |
| 所有 arm strict mean `<` control mean，或 default 明显下降，或 safety>0 | `PPO_BIAS_HARM` | 不换算法；查 prior 被覆盖/动作参数化 |
| 介于以上之间 | `PPO_BIAS_MIXED` | 按配对方向写 blocker，不直接开 SAC |

`PPO_BIAS_HARM` 不自动触发 SAC。只有同时满足以下条件才允许预注册 SAC 同协议对照：

```text
bias prior 存在且可复现
PPO 仍不能在其上提升
动作权限、reward、terminal capture 已被排除
```

SAC 必须使用 Goal86 P-A 架构：`bc_path_dim=8`、endpoint 4→32→64、加载 BC encoder；不得使用 Goal108 path12 随机初始化。

## 7. 执行计划

- [x] Step 0：网络 parity + 同 seed 初始化逐张量比对；
- [x] Step 1：policy + bias dev40 复现 Goal124 stern-025；
- [x] Step 2：生成 4 arm × 2 seed 配置；
- [x] Step 3：8 卡并行训练 10M；
- [x] Step 4：8 checkpoint × strict/default 并行 eval；
- [x] Step 5：fixed-bias control 逐 seed 复现；
- [x] Step 6：Gate、result、blocker/decision、LEDGER、STATE；
- [x] Step 7：checker；不自动 commit/push。

## 8. Git

禁止 `git add -A`。只 add 本 Goal 相关文件：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal125 \
  --goal-doc doc/goals/goal125_bias_initialized_ppo_20261111.md \
  --strict --require-result
git diff --cached --check
```

不提交 `artifacts/`、checkpoint、日志和缓存。

## 9. Goal 提示词

```text
你现在执行 Goal125：bias-initialized PPO。

先读：
- doc/goals/goal125_bias_initialized_ppo_20261111.md
- doc/goals/goal124_result_20261111.md
- doc/goals/goal123_result_20261111.md
- doc/goals/goal121_result_20261111.md

硬约束：
1. 不改 actor/critic/encoder 网络结构；Step 0 parity 不通过就停止；
2. 不改 success、goal radius、terminal capture、eval split；
3. 不打开 held-out 100–149；
4. 不换 SAC/REDQ/DroQ；
5. 8 个训练 job 用 8 张 4090 并行；
6. fixed-bias control 必须逐 seed 复现；否则判 INVALID；
7. Gate 记录前不改阈值、seed、checkpoint 选择规则。

执行顺序：
- 生成配置；
- 跑 1-iteration init parity 和 dev40 bias parity；
- 启动 8 卡 10M 训练；
- eval strict/default dev40；
- 写 result、LEDGER、STATE、checker。
```

## 10. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时使用 `paper-search` / `deep-research`，写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新 Goal，不修改失败 Gate 阈值；
6. 持续迭代，直到出现可复现的 success 提升或触发停止条件；
7. 不自动 commit/push，不打开 held-out，不改变任务成功定义或主算法族。
