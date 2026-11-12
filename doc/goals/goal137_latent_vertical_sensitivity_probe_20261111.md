# Goal137：`z_t` 纵向信息探针与残差头敏感度诊断

**日期**：2026-11-11<br>
**状态**：`COMPLETED_WITH_RESULT`<br>
**性质**：离线诊断 + 1-seed 监督小实验；不训练 PPO/SAC，不打开 held-out，不改 reward、terminal、success 或 goal radius。

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
  check_command: python ../scripts/check_research_ledger.py --goal Goal137 --goal-doc doc/goals/goal137_latent_vertical_sensitivity_probe_20261111.md --strict
```

## 1. 背景

Goal95/117/121/125 的一致现象是：原始观测和路径特征显式含纵向/俯仰信息，BC 先验已安装，PPO 残差也有非零幅度，但终端俯仰建立和 vertical capture 没有改善。现在要区分两条链：

```text
纵向信息是否还在 z_t？
        |
        +-- 不在：编码器/预训练目标是瓶颈
        |
        +-- 在：残差头是否真正消费它？
                 |
                 +-- 不消费：头、监督目标或信用分配是瓶颈
                 |
                 +-- 消费：动作 authority/时序/几何是瓶颈
```

本机没有 Goal95 `P_D3` final checkpoint；可用的同族资产是 Goal86 P-A residual BC：

- 数据：`artifacts/goal86/pa_residual_labels_20261110/pa_residual_labels.npz`
- checkpoint：`artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt`
- 该 checkpoint 的目标是 `normalize(D3_actuator - ALOS_actuator)`，因此它的输出就是可审计的残差动作，不是绝对舵面动作。

限制必须写清楚：本 Goal 的敏感度结论对应 **Goal86 BC residual head**；它不能直接替代 `P_D3` PPO head 的证据，但能判断“编码器里有没有信息”和“同族残差头是否使用它”。

## 2. 假设

### H137-A：`z_t` 保留纵向/俯仰信息

- 做法：按 episode 分组拆分，ridge 线性回归 `z_t -> vertical_error, pitch_error, q, w`。
- 成立：至少 3/4 个目标在验证 episode 上 `R^2 >= 0.50`，Pearson `r >= 0.70`。
- 失败：至少 3/4 个目标 `R^2 < 0.25`。
- 中间：`AMBIGUOUS`，不据此改编码器。

### H137-B：残差动作对纵向/俯仰输入不敏感

- 做法：对 `vertical_error`、`tangent_pitch_error`、`theta`、`q`、`w` 做中心差分 Jacobian；用 `cross_track`、`heading_error`、`phi`、`r` 作参照。
- 主量：`response_per_std = ||J||_2 * std(input) / mean(||residual||_2)`，即输入变化 1σ 时，残差动作变化占典型残差幅度的比例。
- 成立：5 个纵向/俯仰输入中至少 4 个 `response_per_std < 0.05`，或纵向/俯仰中位敏感度不足横向/航向参照中位数的 50%。
- 失败：纵向/俯仰输入敏感度与参照同量级，且 `response_per_std >= 0.10`。

### H137-C：冻结编码器不是主要瓶颈

- 做法：从同一 BC checkpoint 出发，同样数据、同样 seed、同样优化器，比较：
  - `head_only`：冻结 `actor_encoder`、`path_encoder`、`actor_current_encoder`，只训练 actor head；
  - `last_layer`：额外解冻 `actor_encoder.network` 和 `path_encoder` 的最后一个 Linear。
- 成立：`last_layer` 相对 `head_only` 的验证残差 RMSE 改善 `< 2%`，且纵向敏感度变化 `< 20%`。
- 失败：`last_layer` 改善 `>= 5%`，或纵向敏感度提升 `>= 50%`。
- 中间：`WEAK_LAST_LAYER_EFFECT`。

## 3. 数据与隔离

| 项目 | 规定 |
|---|---|
| 样本 | Goal86 `pa_residual_labels.npz`，25,600 条 |
| 分组 | 按 `episode_id` 分组，禁止同 episode 跨 train/val |
| 拆分 | 复刻 checkpoint 训练契约：`validation_fraction=0.2`、`split_seed=86051`，得到 26/6 episodes |
| 输入归一化 | `PCNormalization` + `normalize_path_features`，与 BC 训练一致 |
| 目标物理量 | `vertical_error=actor_path[:,1]`、`pitch_error=actor_path[:,3]`、`q=actor_state[:,7]`、`w=actor_state[:,11]` |
| held-out | 不打开正式 held-out seeds；本 Goal 只用这 32 个训练 episode 的内部验证拆分 |
| 泄漏 | probe、Jacobian、适配均为同一诊断拆分；不得把结果写成正式泛化结论 |

## 4. 方法

### 4.1 Latent ridge probe

```text
state_z = actor_encoder(normalize_actor_state(actor_state))
path_z  = path_encoder(normalize_path_features(actor_path))
z_t     = state_z + path_encoder_alpha * path_z
```

默认 `actor_current=0`，与原数据的 current 缺失语义一致；current encoder 的常数项保留在 `z_t` 中。
Ridge 使用训练集均值/方差标准化，固定 `alpha=1.0`，验证集不参与拟合或选参。
同时报告 `state_z`、`path_z`、`z_t` 三个特征集，避免把路径输入的贡献误记到状态编码器。

### 4.2 中心差分 Jacobian

物理输入扰动：

| 输入 | eps |
|---|---:|
| `vertical_error` | 0.05 m |
| `tangent_pitch_error` | 0.01 rad |
| `theta` | 0.01 rad |
| `q` | 0.01 rad/s |
| `w` | 0.01 m/s |
| `cross_track` | 0.05 m |
| `heading_error` | 0.01 rad |
| `phi` | 0.01 rad |
| `r` | 0.01 rad/s |

```text
J[:, j] = (a(x + eps_j e_j) - a(x - eps_j e_j)) / (2 eps_j)
```

报告归一化动作 Jacobian、物理动作 Jacobian（乘以 `[20°, 20°, 1525 rpm]` 全尺度）、
`response_per_std`、以及输入/动作通道排序。验证集上最多使用 2,048 个固定样本，避免结果依赖全量内存。

### 4.3 冻结编码器 vs 微调最后一层

| arm | 可训练参数 |
|---|---|
| `head_only` | `actor.*` |
| `last_layer` | `actor.*` + `actor_encoder.network` 最后一个 Linear + `path_encoder` 最后一个 Linear |

两臂都先加载同一 BC checkpoint，再用同一个 seed 重新随机初始化 actor MLP；encoder 保留 BC
预训练权重。这样才能比较“固定表征上训新 head”和“允许最后一层表征一起适配”。两臂使用相同
train/val 拆分、`batch_size=4096`、AdamW `lr=1e-3`、`weight_decay=1e-5`、最多 120 epochs、
patience 20、梯度裁剪 1.0。目标为 checkpoint 语义下的 `residual_physical / action_full_scale`，
不是 npz 中按 `0.25*full_scale` 记录的 RL 标签。报告 normalized/physical RMSE、per-channel R²、
encoder drift 和适配后 Jacobian。

## 5. 执行计划

- [x] Step 0：写入本预注册文档与 Research Contract；
- [x] Step 1：实现 `scripts/probe_pc_latent_vertical.py`；
- [x] Step 2：新增单元测试 `tests/test_pc_goal137_latent_probe.py`；
- [x] Step 3：跑 smoke，检查 checkpoint 重建、probe、Jacobian 和两臂适配管线；
- [x] Step 4：跑正式 probe + Jacobian + 两个监督适配 arm；
- [x] Step 5：写 result md，追加 `research/LEDGER.jsonl`，更新 `research/STATE.md`；
- [x] Step 6：运行 checker 和 `git diff --check`；不自动 commit/push。

## 6. 运行命令

从 `settled_局部规划器_反应式` 运行：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_pc_goal137_latent_probe.py

PYTHONPATH=src .venv/bin/python scripts/probe_pc_latent_vertical.py \
  --dataset artifacts/goal86/pa_residual_labels_20261110/pa_residual_labels.npz \
  --bc-checkpoint artifacts/lowlevel/goal86/pa_residual_bc_path12_cmd3_current_20261110/pc_bc_actor.pt \
  --output artifacts/lowlevel/goal137_latent_vertical_probe_20261111 \
  --device cpu --adapt-epochs 120 --adapt-patience 20
```

输出至少包含：

- `probe.json`：完整指标、命令、数据/checkpoint SHA、split 信息；
- `probe.md`：人类可读结论；
- `head_only.pt`、`last_layer.pt`：两个适配 checkpoint；
- 两臂都记录 `checkpoint_metrics`（重初始化前）、`reinitialized_head_metrics` 和 `adapted_metrics`；
- `adaptation.json`：两臂训练曲线和对比。

## 7. Gate

| check | 阈值 | 判读 |
|---|---:|---|
| probe retention | 3/4 targets `R²>=0.50` 且 `r>=0.70` | 否则不把问题归给 head |
| action response | 至少 4/5 纵向输入 `response_per_std<0.05` | 支持“头不用表征” |
| last-layer effect | val RMSE 改善 `<2%` 且敏感度变化 `<20%` | 支持“冻结不是主因” |
| 稳定性 | 0 NaN/Inf，0 shape/asset mismatch | 否则 INVALID |
| 负控 | 原始输入未泄漏到验证 episode | 否则 INVALID |

失败裁决：

| 观察 | 判读 | 下一步 |
|---|---|---|
| probe 差 | 编码器/预训练目标瓶颈 | 先查显式纵向 skip、DADE history 和预训练目标，不扫 seed |
| probe 好、response 低 | head 不消费表征 | 查 head 输入融合、监督标签和信用分配 |
| probe 好、response 高 | 不是表征/头问题 | 转向 authority、时序 handoff 和终端几何 |
| last-layer 显著更好 | 编码器适配有价值 | 长时程训练前先做小规模 closed-loop screen |

## 8. Git

禁止 `git add -A`。只 add 本 Goal 文档、脚本、测试、result、ledger/STATE 和 summary artifact 引用；
不提交 checkpoint、npz、大日志或 `artifacts/` 原始文件。提交前运行：

```bash
python ../scripts/check_research_ledger.py --goal Goal137 --goal-doc doc/goals/goal137_latent_vertical_sensitivity_probe_20261111.md --strict
git diff --cached --check
```

不自动 commit / push。

## 9. 自主迭代契约

遵守 `research/AUTORESEARCH_LOOP.md`。若 Gate 失败或 probe/Jacobian/适配给出互相矛盾的结果，
先写 blocker 和至少两个可证伪 hypothesis，再决定 `continue` / `pivot` / `stop`；本 Goal 不允许
通过换 seed、增加训练预算或放宽阈值来救结果。
