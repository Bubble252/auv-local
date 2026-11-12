# Goal88：SAC 同协议对照预注册（仅草案，未启动）

**日期**：2026-11-10<br>
**状态**：`PREREGISTERED_DRAFT / NOT_STARTED / HELDOUT_CLOSED`<br>
**上游**：Goal87 `PPO_LOCKED_AT_MILLIMETER`、`TERMINAL_CAPTURE_MILLIMETER_ONLY`<br>
**性质**：算法对照预注册；不改变当前 Goal87 的 PPO Gate，不打开 held-out，不修改 success / goal radius / reward / terminal。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal88 --goal-doc doc/goals/goal88_sac_same_protocol_prereg_20261110.md --strict
```

`required: true`：本 Goal 只允许在 H87-O1/O2/O3 明确指向优化/信用分配、
且 PPO 仍未产生有效残差时启动。启动前必须把本文件状态改为
`PREREGISTERED / RUNNING`，并记录预算、seed、checkpoint 和 eval 协议。

## 1. 启动条件（必须全部满足）

1. Goal87 terminal/capture 审计已完成，且没有 reward/terminal 契约错位；
2. H87-O1（更新调度/学习率）和 H87-O2（非零输出初始化）至少一个被否定；
3. H87-O3 证明 BC-head 大 KL 不是 stale old-log-prob 的实现问题；
4. 当前 PPO 在 1M、3 seeds 下仍没有 `residual/base L2 >= 0.02` 或
   success 方向性改善；
5. 用户明确同意启动算法对照。

未满足以上条件时，本文件只作为提醒，不跑训练。

## 2. 固定协议

| 项目 | 固定值 |
|---|---|
| 架构 | 项目原生 `z_t`：state encoder + path8/endpoint4 + current encoder；actor/critic 不退回 flatten+MLP |
| 特征 | SAC `[z_t64, zero_dade64, command3] = 131`；PPO `[z_t64, command3, zero_dade64] = 131` |
| 环境 | Goal85 `unbiased_noise_v1`，`position_source=eta`，零传感器延迟，path12 |
| 算法 | on-policy PPO（对照）与 off-policy SAC（本 Goal 主臂）；不加入 REDQ/DroQ |
| 预算 | screen 1M transitions；formal 5M transitions；每个 arm 固定预算，不因结果好坏偷加 |
| seeds | screen 86201/86202/86203；formal 至少 3 seeds，seed 集合预注册后不改 |
| 评价 | dev seeds 20–59，default + strict125；同 seed 配对；success/p_nav/eta/plane-miss/timeout/path3D/terminal 桶全部记录 |
| held-out | `100–149` 在 formal 记录前关闭 |
| 成功语义 | 终止帧 `eta` 进入 goal radius；p_nav 只作观测/诊断 |

## 3. 实验矩阵

| arm | 算法 | 基座 | 目的 | 预算 |
|---|---|---|---|---:|
| `R1` | PPO | frozen D3 BC | 更新调度/学习率优化后的 residual | 1M screen |
| `R2` | PPO | frozen D3 BC | 非零小输出初始化 residual | 1M screen |
| `S1` | SAC | none（T2 actuator） | 同协议 off-policy 对照 | 1M screen |
| `S2` | SAC | frozen D3 BC | 冻结 BC + SAC residual，条件启动 | 1M screen |
| `F1` | 最优 R* | 同 R* | 5M formal | 5M |
| `F2` | 最优 S* | 同 S* | 5M formal | 5M |

`S2` 只有在 SAC frozen-base adapter 通过 parity 和 checkpoint 往返测试后才启动；
否则只跑 `S1`，并在结果中明确“不是同一基座”。

## 4. 假设与判读

### H88-S1：SAC 的 off-policy 复用能打破 PPO 的毫米级锁定

- 支持：S1/S2 的确定性 `residual/base L2 >= 0.02`，且 default 或 strict125
  success 比同 seed 对照高 `>=2.5 pp`，path3D 不退化 `>0.05 m`。
- 否定：SAC 也停在 `residual/base L2 < 0.01`，或 success 不移动。
- 注意：如果 SAC 只增加 residual 幅度但 success 不变，判 `WEAK_MOVE`，
  不得写成“SAC 有效”。

### H88-S2：SAC 的收益来自 replay，而不是动作参数化或 reward

- 支持：S1/S2 与 PPO 使用同一 action interface、同一 reward、同一 terminal、
  同一 z_t；差异只在算法和 replay。
- 否定：必须改动 action interface/reward/terminal 才能让 SAC 有效，则停止
  本 Goal，回到 Goal87 归因。

## 5. 预注册判读

| 观察 | 判读 | 下一步 |
|---|---|---|
| residual L2 `>=0.02` 且 success `>=+2.5 pp`，3 seeds 同向 | `MOVE` | 进 5M formal |
| residual L2 `>=0.02` 但 success 不动 | `WEAK_MOVE` | 查 terminal/reward 敏感性，不直接 formal |
| residual L2 `<0.01` | `LOCKED` | 停止 SAC 线，写 blocker；考虑 REDQ/DroQ 需新 Goal |
| success 下降 `>=5 pp` 或 plane-miss/timeout 恶化 | `HARMFUL` | 停止该 arm，不加预算救 |
| 与对照差异小于 seed 噪声 | `INVALID` | 不写胜负，扩大 seed 或改评价 |

## 6. 运行命令模板

远端仓库根目录：

```bash
export PYTHONPATH="src:/data/yhx/underwater-robotics/src/stable-baselines3"
PY=/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" scripts/train_pc_command_residual_sac.py \
  --training-config configs/lowlevel/goal86/<sac-config>.json \
  --output artifacts/lowlevel/goal88/<arm> \
  --device cuda --overwrite
```

评价：

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "$PY" scripts/evaluate_pc_sac.py \
  --training-config configs/lowlevel/goal86/<sac-config>.json \
  --checkpoint artifacts/lowlevel/goal88/<arm>/final.pt \
  --output artifacts/lowlevel/goal88/<arm>_eval_default_dev40 \
  --seed-start 20 --seed-count 40 --device cpu --dtype float64
```

strict125 评价加：

```text
--environment-config configs/lowlevel/pc_goal85_unbiased_strict125_candidate_20261110.json
```

具体配置路径在启动时填写；本草案不预填不存在的 SAC frozen-base 配置。

## 7. 停止条件

- 任一 screen arm 出现 `MOVE`：保留方向，补 3 seeds；不扩到无边界扫参；
- 所有 screen arm `LOCKED`：停止 SAC 线，写 `decision=stop` 或 `pivot`；
- 只有明确证明样本效率不足，才允许新开 REDQ/DroQ Goal；
- 任何阶段都不打开 held-out、不改 success/goal radius/reward/terminal、
  不换 seed 救结果、不自动 commit/push。

## 8. Git

```bash
git add doc/goals/goal88_sac_same_protocol_prereg_20261110.md
git diff --cached --check
git commit -m "goal88: preregister same-protocol SAC control"
```

禁止 `git add -A`；不提交 `artifacts/`、checkpoint、日志。
