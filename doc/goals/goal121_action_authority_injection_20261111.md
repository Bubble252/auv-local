# Goal121：residual action-authority injection 审计

**日期**：2026-11-11<br>
**状态**：`PROTOCOL_DRAFTED_BEFORE_MEASUREMENT / EVAL_ONLY / NO_TRAINING`<br>
**上游**：Goal120 `PPO_OPTIMIZATION_LIMITED / RESIDUAL_CLOSED_LOOP_INERT`<br>
**性质**：冻结的 eval-only falsification；不训练、不改 reward/terminal/success/
goal radius，不打开 held-out。

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
  check_command: python3 ../scripts/check_research_ledger.py --goal Goal121 --goal-doc doc/goals/goal121_action_authority_injection_20261111.md --strict --require-result
```

记录前不得修改 profile、seed、scenario、success frame、checkpoint、Gate 阈值。
本 Goal 不改网络、不改训练配置、不启动 PPO/SAC。

## 1. 问题

Goal120 观察到：

```text
residual applied mean       约 200–250 RPM
effective/applied           约 0.94–1.0
1 s 低通/原信号             约 0.89–0.91
四臂 terminal share         几乎相同
四臂 final eta              只差约 0.01–0.02 m
```

这不是“residual 没执行”，也不是“residual 全是高频噪声”。因此必须先回答：

1. 在该闭环里，注入一个已知的 residual 能否移动终态 eta / success？
2. PPO 学到的 residual 方向是否可区分？取反或放大后结果是否改变？
3. 如果已知 residual 也移不动，是否应先修动作参数化/终端几何，而不是换 SAC？

## 2. 固定输入

| 项目 | 值 |
|---|---|
| base checkpoint | `artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt` |
| checkpoint SHA256 | `2d57745a811268553041c561f2ae1ad590dced472b712efd90cdd9c011c1a616` |
| environment | Goal119 C3（strict125 + 15 deg/s pitch limit） |
| eval seeds | dev 20–59，40 episodes |
| success frame | `both` |
| eval scale / gains | 2.00 / `1,1,1` |
| action mode | `injected-residual` |
| held-out | `100–149` 关闭 |
| 网络 | 不改；与 Goal103/108 的 22/22 frozen-BC 张量 parity 记录一致 |

注入 residual 经过与 Goal120 相同的 `BatchTorchPCFrozenBCResidualVecEnv`：
`base + terminal_gate * injected_residual`。terminal gate 固定为训练配置的
`10 m -> 3 m`，不在本 Goal 内改。

## 3. 注入 profile

| profile | residual | 目的 |
|---|---|---|
| `zero` | `(0,0,0)` | B20 配对基线 |
| `ppo` | 当前 Goal119 C3 checkpoint 的 deterministic PPO residual | 参考 |
| `neg-ppo` | `-1 × ppo` | 方向是否可分 |
| `ppo-2x` | `2 × ppo` | 小幅放大 |
| `ppo-4x` | `4 × ppo` | 强权限测试 |
| `rudder+` | `(0.2,0,0)` | 水平通道常量权限 |
| `stern+` | `(0,0.2,0)` | 垂直通道常量权限（一个符号） |
| `stern-` | `(0,-0.2,0)` | 垂直通道常量权限（反符号） |
| `rpm+` | `(0,0,0.2)` | 速度通道常量权限 |
| `rpm-` | `(0,0,-0.2)` | 速度通道反符号 |

常量值 0.2 是归一化 residual action，经过与 PPO 相同的 gate、scale 和
actuator envelope。`ppo-2x/4x` 允许被 action clamp 截断，截断本身就是权限信息，
必须记录 `effective/applied`。

## 4. 预注册假设

### H121-A：动作权限存在，但 PPO 方向错误

- 成立时应看到：`ppo-4x` 或常量 profile 在 5 个 reachable failure 上的
  终态 eta 中位变化 `>=0.25 m`；`neg-ppo` 与 `ppo` 的终态 eta 明显不同。
- 失败时应看到：所有 profile 的终态 eta 变化 `<0.05 m`，success 集合不变。

### H121-B：residual 通道在该闭环里近乎惰性

- 成立时应看到：即使 `ppo-4x`、常量 rudder/stern/rpm 也不能让终态 eta
  中位变化超过 `0.05 m`，success 集合不变。
- 失败时应看到：任一 profile 达到 `>=0.25 m` 或翻转 `>=2` 个 success。

### H121-C：放大权限会改善 success

- 成立时应看到：`ppo-2x` 或 `ppo-4x` 相对 `zero` 增加 `>=2` 个 success，
  且 timeout/safety 不恶化。
- 失败时应看到：终态 eta 移动但 success 不增加或下降。

### H121-D：取反 PPO 反而更好

- 成立时应看到：`neg-ppo` 比 `ppo` 多 `>=2` 个 success。
- 失败时应看到：`neg-ppo` 与 `ppo` 的 success 差异 `<2`。

## 5. 指标

主指标：

```text
reachable failure seeds 25/27/31/50/51 的 final eta L2 变化（相对 zero）
dev40 success / timeout / safety
success 集合相对 zero 的 gained/lost
terminal plane miss
```

机制指标：

```text
injected residual applied / effective / effective-applied
frozen_residual_gate
guidance heading/pitch rate clip
final eta xyz 和 terminal_share_last2s
```

## 6. Gate

| 观察 | 状态 | 下一步 |
|---|---|---|
| 任一 profile 的 reachable-failure 终态 eta 中位变化 `>=0.25 m`，或 success 翻转 `>=2` | `AUTHORITY_PRESENT` | 若方向可分但 PPO 无增益，预注册 SAC 同协议对照 |
| 所有 profile 的终态 eta 中位变化 `<0.05 m` 且 success 翻转 `0` | `AUTHORITY_INERT` | 先修动作参数化/终端几何，不换算法 |
| `ppo-2x/4x` 有 eta 移动但 success 不增/下降 | `AUTHORITY_WITHOUT_CAPTURE` | 查 reward/terminal/direction，不直接 SAC |
| `neg-ppo` 比 `ppo` 多 `>=2` success | `NEG_PPO_BETTER` | 查 reward 符号/terminal credit，不直接 SAC |
| 缺失 trace、NaN/Inf、safety>0、注入未生效 | `INVALID` | 修复执行链后重跑；不改阈值 |

## 7. 执行计划

- [x] Step 0：写预注册并冻结 profile/seed/Gate；
- [ ] Step 1：给 `evaluate_pc_ppo.py` 增加 `injected-residual` action mode 和
  `--injected-residual-profile`；
- [ ] Step 2：smoke `zero` 与 `ppo`，确认 zero 复现 B20、ppo 复现 Goal120 C3；
- [ ] Step 3：10 profile × dev40 并行 eval；
- [ ] Step 4：汇总终态 eta、success 翻转、residual applied/effective、clip；
- [ ] Step 5：Gate 判定并写 result/blocker/decision/LEDGER/STATE；
- [ ] Step 6：checker 与 `git diff --check`；不自动 commit/push。

## 8. 运行方式

```bash
cd settled_局部规划器_反应式
PYTHONPATH=src python scripts/evaluate_pc_ppo.py \
  --training-config artifacts/lowlevel/goal120_ppo_optimization_20261111/training_configs/P0_s86201.json \
  --checkpoint artifacts/lowlevel/goal95_supervised_residual_20261111/P_D3/final.pt \
  --environment-config artifacts/lowlevel/goal120_ppo_optimization_20261111/environment_configs/P1.json \
  --output artifacts/lowlevel/goal121_action_authority_20261111/zero \
  --seed-start 20 --seed-count 40 \
  --success-frame both --device cpu \
  --action-mode injected-residual \
  --injected-residual-profile zero \
  --actuator-residual-scale-override 2.00 \
  --actuator-residual-channel-gains 1.000000,1.000000,1.000000 \
  --allow-guidance-rate-limit-override \
  --trace-output artifacts/lowlevel/goal121_action_authority_20261111/zero/trace.npz \
  --trace-reward-breakdown
```

注意：P0 的 training config 指向 Goal119 C3 checkpoint 路径；本 Goal 的
`--checkpoint` 统一覆盖为 Goal95 P_D3，base/frozen BC 契约必须通过 SHA 校验。
`environment_configs/P1.json` 与 P0/C3 reward 等价（`pc_ppo_scale=0.01`）；
注入模式不使用 reward，只用于环境/terminal 契约。

## 9. Git

禁止 `git add -A`。只 add 本次相关文件，提交前运行：

```bash
python3 ../scripts/check_research_ledger.py \
  --goal Goal121 \
  --goal-doc doc/goals/goal121_action_authority_injection_20261111.md \
  --strict --require-result
git diff --cached --check
```

## 10. 与 SAC 的关系

本 Goal 不训练 SAC。只有 `AUTHORITY_PRESENT` 且 `ppo/neg-ppo` 方向可分、
但 PPO 没有 success 增益时，才预注册 SAC 同协议对照。SAC 必须使用 Goal86
P-A 架构：

```text
bc_path_dim = 8
path endpoint = 4 -> 32 -> 64
加载 BC encoder
state/path/current -> z_t -> actor
不使用 Goal108 的 path12 随机初始化
```

若判 `AUTHORITY_INERT`，SAC 不开；先修动作参数化/终端几何。

## 11. 自主迭代契约

本 Goal 执行 `research/AUTORESEARCH_LOOP.md`：

1. 失败后先归因，不直接换算法或扫 seed；
2. 每次 blocker 至少写两个可证伪 hypothesis；
3. 需要外部方法时使用 `paper-search` / `deep-research`，写 reading note；
4. 先跑最便宜的 falsification，再决定 `continue` / `pivot` / `stop`；
5. `pivot` 时写新 Goal，不修改失败 Gate 的阈值；
6. 持续迭代，直到出现可复现的 success 提升或触发停止条件；
7. 不自动 commit/push，不打开 held-out，不改变任务成功定义或主算法族。
