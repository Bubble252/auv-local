# Goal10：批量 Torch REMUS100 与 4096 环境训练报告

记录日期：2026-10-31

状态：**已完成。** P0–P4 的批量后端、26D schema、compile 验证、4096 环境长训练、50-seed 候选择优，以及 256/1024/4096 等预算环境规模对照均已完成。P4 的正式结论是：批量后端可用、4096 环境吞吐最高、26D 策略通过控制门禁，但本轮 PPO 没有超过 B0。

关联文档：

- `doc/goals/goal10_batch_torch_remus100_feasibility.md`
- `doc/goals/goal9_odarrl_control_reproduction.md`

## 1. 范围

本报告只覆盖 ODARRL/PVS 的低层执行器级控制与批量 Torch 后端，不覆盖 `goal8` 的二维声呐避障 17D 契约，也不把批量训练结果替代 CPU PVS 的正式评估。

正式 schema 为 26D。27D 只用于旧 checkpoint、历史数据、兼容回归和可选迁移 teacher；本轮没有修改 PVS、IsaacLab 或 rsl_rl 源码。

长期执行口径：

- 新的 ODARRL/PVS 低层控制实验默认直接使用 26D，不再新增 27D 配置或 27D 长训练；
- 若需要复用已有 27D 策略的能力，使用 BC/DAgger/动作蒸馏训练 26D student，并在 CPU PVS 上做同 seed 闭环评估；
- 迁移不是“无损删一列”。旧 PI 教师的积分状态被移除后，动作不再是当前 26D 观测的严格函数，必须通过闭环数据和重新训练补齐；
- 26D student 部署时不得暗中读取在线 PI 积分或其他 privileged state；需要时序信息时只能使用显式历史栈或循环网络，并记录网络输入形状。

## 2. 实现边界

- 环境 canonical 配置：`configs/lowlevel/odarrl_control_v2_26d.json`；canonical config hash `c19a5867…0f72e7`，文件 SHA256 `314622a7…046e33`；
- P4 长训练配置：`configs/lowlevel/odarrl_control_batch4096_v2_26d_p4.json`；
- 等预算规模对照配置：`odarrl_control_batch4096_v2_26d_p4_scale.json`、`odarrl_control_batch1024_v2_26d_p4_scale.json`、`odarrl_control_batch256_v2_26d_p4_scale.json`；
- `vector_env.compile_step` 默认 `false`；只有显式配置才启用 `torch.compile(mode="default")`；
- `reduce-overhead`/CUDA Graph 明确不支持，因为会触发输出张量被 CUDAGraph 覆盖的问题；
- compile 模式、训练配置 hash、环境配置 hash、checkpoint schema/dim 均写入 manifest 或 checkpoint metadata。

## 3. 后端与性能

### 3.1 已有后端门禁

P0 的 float64 动力学单步相对误差为 `4.47e-12`，500 步开环位置误差 `4.44e-16`。P1–P2 的批量环境、auto-reset、26D/27D 接口和 rsl_rl 2.3.3 适配已通过；P2.5 的 4096 环境 smoke 无 NaN/Inf、无维度错误。

### 3.2 step_core profile

证据：`artifacts/lowlevel/batch4096_smoke_v2/step_core_profile_gpu0_20261031_1640/summary.json`。

4096 env、26D、float64、GPU0：

- mean step：`23.71 ms`；
- p50：`24.14 ms`；
- p95：`28.52 ms`；
- 纯 step：`172,755 env-steps/s`；
- profiler 每 10 步约 `17,750` 次 CUDA kernel launch，即每步约 `1,775` 次；
- 结论：主要瓶颈是 CPU kernel launch 和同步，不是 GPU 算力。

### 3.3 eager/compiled 数值一致性

证据：`artifacts/lowlevel/batch4096_smoke_v2/compile_verify_gpu5_20261031/summary.json`。

GPU5、64 env、40 step、episode length 11、192 次 auto-reset：

| 字段 | 最大绝对差 |
|---|---:|
| observation | `8.56e-18` |
| reward | `0` |
| done | `0` |
| time_outs | `0` |
| actuator state | `4.26e-14` |
| eta | `5.55e-17` |
| nu | `2.22e-16` |

全部低于 `1e-10`，compile 可进入正式训练；CUDA Graph 不使用。

### 3.4 P4 长训练吞吐

证据：`artifacts/lowlevel/p4_26d_long_20261031/training_manifest.json` 和 `iteration_records.jsonl`。

4096×256、26D、float64、compile default、200 iterations：

- 总 env-steps：`209,715,200`；
- wall time：`784.62 s`；
- 含更新/日志/同步的墙钟吞吐：`267,282 env-steps/s`；
- collection 中位吞吐：`492,626 env-steps/s`；
- iteration 时间中位数：`3.74 s`；
- step latency p50 均值：`3.29 ms`，p95 均值：`4.16 ms`；
- GPU 峰值 allocated：`424.1 MiB`，reserved：`512.0 MiB`；
- `losses_finite=true`，200 轮无非有限观测或动作；
- approx KL 中位数 `0.00138`，clip fraction 中位数 `0.0127`，没有异常 KL 爆炸。

注意：`collection_env_step_rate_hz` 只统计采集阶段，不能当作端到端吞吐；本报告以 wall time 计算的 `267k env-steps/s` 作为端到端数字。

## 4. 环境规模控制质量对照

对照目标是只改变环境数，其他训练预算保持一致：

- 4096×64、1024×256、256×1024；
- 每个 iteration 都是 `262,144` transitions；
- 原设计预算为 400 iterations，总 `104,857,600` env-steps；
- 相同 seed `40000`、相同 26D 环境配置、相同 PPO 超参、compile default。

实际控制质量比较统一采用三个规模共同拥有的 `model_100.pt`，即 `100 updates × 262,144 = 26,214,400` env-steps。这样比较的是“同更新次数、同总样本量”，不会把 4096 额外训练到 400 轮后与 256 的 100 轮混在一起。4096 和 1024 随后跑完了 400 iterations；256 在 `model_100` 后主动停止，原因是在相同 env-step 预算下需要多得多的 Python/vec-env step 调用，继续跑只会放大调度开销。停止说明保存在 `artifacts/lowlevel/p4_26d_scale_256_20261031/stop_note.json`。

### 4.1 等预算吞吐

以下统计每个规模的前 100 iterations，即同一 `26,214,400` env-steps 训练预算。墙钟吞吐按 iteration 墙钟时间求和计算，`collection` 只统计环境交互阶段。

| envs | wall env-steps/s | iteration 均值 | collection 均值 | learn 均值 | collection env-steps/s | step p95 均值 |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | `115,034.7` | `2.279 s` | `0.634 s` | `1.645 s` | `483,358` | `4.592 ms` |
| 1024 | `66,599.3` | `3.936 s` | `2.325 s` | `1.611 s` | `122,151` | `4.857 ms` |
| 256 | `25,918.4` | `10.114 s` | `8.417 s` | `1.697 s` | `31,658` | `4.490 ms` |

三个规模均 `losses_finite=true`。PPO 学习阶段耗时几乎不变，约 `1.6–1.7 s/iteration`；主要差异来自 Python/vec-env step 调用次数。4096 的端到端吞吐约为 1024 的 `1.73×`、256 的 `4.44×`。

### 4.2 等预算控制质量

20-seed PVS、同一 B0 基座、共同 `model_100.pt`：

| envs | success B0 → residual | 三维 RMSE B0 → residual (m) | RMSE 差 (m) | material saturation | residual abs mean | constraint violations/ep |
|---:|---:|---:|---:|---:|---:|---:|
| 4096 | `1.00 → 1.00` | `1.565020 → 1.583057` | `+0.018037` | `0 → 0` | `0.05734` | `0 → 86.8` |
| 1024 | `1.00 → 1.00` | `1.565020 → 1.587668` | `+0.022648` | `0 → 0` | `0.08458` | `0 → 1.6` |
| 256 | `1.00 → 1.00` | `1.565020 → 1.597311` | `+0.032291` | `0 → 0` | `0.12097` | `0 → 1.15` |

三组都成功且没有材料级饱和，但都没有超过 B0。同一训练预算下，环境数越多，20-seed PVS 的三维 RMSE 差越小、残差幅值越小；这与每次更新覆盖更多并行环境和更多 reset 状态一致。该趋势只来自每个规模一个训练 seed `40000`，不能当作严格的规模因果结论。

4096 的 `constraint_violations=86.8/episode` 是超过数值容差的幅值/速率微裁剪计数，不是 1% 满量程以上的 material saturation；两者必须分开报告。

checkpoint 后移也会漂移。4096 规模对照跑到 400 轮后的 `final`，20-seed PVS 三维 RMSE 差变为 `+0.034527 m`，material saturation `0.000321`，残差幅值 `0.17070`，明显差于同一配置的 `model_100`。因此正式候选必须按 PVS 闭环结果选择，不能默认 final 最好。

### 4.3 规模对照结论

- 4096 同时给出最高端到端吞吐和与 B0 最接近的控制质量，是后续默认训练规模；
- 资源不足时 1024 可作降级，但不要只看 collection 吞吐，必须把 PPO 更新成本算进墙钟预算；
- 三种规模都没有产生正控制收益，说明瓶颈已经不在批量环境吞吐，而在 PPO 超参、奖励、探索和 checkpoint 选择；
- 后续若要追求控制精度，应另开超参、奖励或 early-stop 消融，不能把本轮“后端可行”误写成“控制已提升”。

## 5. P4 长训练与候选择优

P4 的 4096×256 长训练完成后，按同 seed PVS 结果比较 checkpoint。20-seed 筛选结果：

归臂说明：P4 配置文件中的 `experiment.arm="I"` 是历史标签，但实际 `base_policy.mode="online_pd_pi"`，因此它语义上是 **O 臂的 residual-only 预检**，不是冻结 `pi0` 的 I1，也不能当作 26D Controller-IL 成果。

| checkpoint | success | 3D RMSE gap vs base | material saturation | residual abs mean |
|---|---:|---:|---:|---:|
| `model_25` | 1.00 | +0.00800 m | 0 | 0.0161 |
| `model_50` | 1.00 | +0.0240 m | 0 | 0.0518 |
| `model_100` | 1.00 | +0.0382 m | 0 | 0.1323 |
| `final` | 1.00 | +0.0320 m | 0.00043 | 0.2244 |

结论：PPO 后期没有稳定改善控制质量，反而出现均值和饱和漂移。因此正式候选选择 **`model_25.pt`**，而不是 blind use final；这是一项负结果，不能用训练 reward 曲线掩盖。

`model_25.pt` 50-seed PVS 正式复核：

- 输出：`artifacts/lowlevel/p4_26d_long_20261031/pvs_eval_model25_50seed_20261031`；
- checkpoint SHA256：`5b6e66c6…9995f`；
- success：base `50/50`，residual `50/50`；
- 3D RMSE：`1.514748 → 1.522790 m`，差 `+0.008042 m`；
- horizontal RMSE：`1.468673 → 1.477027 m`；
- vertical RMSE：`0.323929 → 0.323614 m`，略改善；
- material saturation：两者均为 `0`；
- constraint violations：`0 → 17.12/episode`，属于小幅裁剪计数，不是材料级饱和；
- 控制环 p95：`14.84 → 13.77 ms`；
- PVS step p95：`0.327 → 0.318 ms`。

门禁判定：成功率不下降、三维 RMSE 劣化 `0.008 m < 0.05 m`、material saturation 不上升，**候选通过 P4 正式控制门禁**。但它相对 B0 没有精度收益，只能说明 batch 26D/PPO 路径可用且没有破坏基座。

## 6. 复现命令

```bash
# 数值一致性
PYTHONPATH="$PWD/src:$PWD/../settled_全局规划器/src" \
  python scripts/verify_batch_compile.py \
  --num-envs 64 --steps 40 --episode-steps 11 --seed 1234 \
  --device cuda:0 \
  --output artifacts/lowlevel/compile_verify/summary.json

# P4 长训练
PYTHONPATH="$PWD/src:$PWD/../settled_全局规划器/src" \
  python scripts/train_batch_lowlevel_ppo.py \
  --training-config configs/lowlevel/odarrl_control_batch4096_v2_26d_p4.json \
  --record-only

# 候选择优的 50-seed PVS
PYTHONPATH="$PWD/src:$PWD/../settled_全局规划器/src" \
  python scripts/evaluate_odarrl_control_residual.py \
  --training-config configs/lowlevel/odarrl_control_batch4096_v2_26d_p4.json \
  --checkpoint artifacts/lowlevel/p4_26d_long_20261031/model_25.pt \
  --environment-config configs/lowlevel/odarrl_control_v2_26d.json \
  --output artifacts/lowlevel/p4_26d_long_20261031/pvs_eval_model25_50seed_20261031 \
  --seeds 0-49 --workers 16
```

## 7. 限制

- 批量训练只用于训练和筛查，正式结论仍以 CPU PVS 评估为准；
- compile 首次调用有约几十秒的编译开销，稳态吞吐需排除首次编译；
- 训练环境、服务器负载和共享 GPU 会影响 wall-clock，不能把一次调度波动当作算法差异；
- 环境规模质量对照每个规模只训练了一个 seed `40000`；20-seed 是评估 seed，不是 20 次独立训练。因此“环境数越多越接近 B0”只能作为本轮观察，不能单独支撑论文级因果结论；
- 256 环境在共同 `model_100` 后主动停止，后续只用于等预算对照，不应把未跑满 400 轮写成失败或收敛结论；
- 当前 PPO 没有稳定超过 B0 的轨迹精度；后续若要追求收益，应单独做 learning-rate、探索噪声和 update 频率消融，不能混入本轮后端可行性结论。

## 8. 最终结论

- 批量 Torch 后端可行，4096 环境是当前单卡训练的首选规模；
- 26D 是长期统一 schema；27D 只保留兼容和可选迁移 teacher，不再新增 27D 长训练；
- 直接 26D 的 P3 门禁和 P4 `model_25` 50-seed 门禁均通过，但都没有超过 B0；
- 4096/1024/256 等预算对照中，4096 吞吐最高、控制质量最接近 B0，但三种规模都是负增益；
- 下一阶段要提高控制效果，应从 reward、PPO 超参、探索、early stopping 和 checkpoint 选择入手，而不是继续扩大环境数或把“批量后端通过”包装成“控制提升”。
