# 算力使用协议：先并行诊断，再预注册扩大采样/批量

日期：2026-11-10<br>
适用范围：`settled_局部规划器_反应式` 的训练、评估和只读诊断<br>
目的：加快实验迭代，同时不污染正式 Gate 的可比性

## 1. 当前机器与观测

- 服务器：`8 × RTX 4090`，`384` CPU，约 `503 GB` 内存；
- Goal69 只读分析：10 个 run 用 `--jobs 10`，墙钟约 `1.03 s`，CPU 约 `553%`，
  RSS 约 `40 MB`，未占 GPU；
- Goal74 只读分析：10 个 run 用 `--jobs 10`，墙钟约 `1.70 s`，CPU 约 `896%`，
  RSS 约 `36 MB`，未占 GPU；
- Goal75 只读分析：10 个 run 用 `--jobs 10`，墙钟约 `1.73 s`，CPU 约 `881%`，
  RSS 约 `39 MB`，未占 GPU；
- Goal76 只读分析：10 个 run 用 `--jobs 10`，墙钟约 `2.79 s`，CPU 约 `922%`，
  RSS 约 `36 MB`，未占 GPU；追加的按 checkpoint 残差分解同样是 10 进程，
  墙钟约 `4.03 s`，未占 GPU；
- Goal77 只读分析：10 个 run 用 `--jobs 10`，墙钟约 `1.78 s`，CPU 约 `889%`，
  RSS 约 `33 MB`，未占 GPU；
- Goal78 只读分析：10 个 run 用 `--jobs 10`，墙钟约 `1.77 s`，CPU 约 `870%`，
  RSS 约 `36 MB`，未占 GPU；
- Goal79 只读 LOOCV：10 个 run 用 `--jobs 10`，墙钟约 `2.71 s`，CPU 约
  `885%`，RSS 约 `39 MB`，未占 GPU；
- Goal80 正式闭环评估：远端 `384` CPU，10 个 run 并行、每个 run 40 envs，
  每进程单线程 BLAS；从第一个 summary 到最后一个 summary 墙钟约 `3.79 s`，
  输出约 `166 MB`，0 张 GPU，10/10 完成无重试；这是 10 个 run 的自然上限，
  没有为了追满 384 核而事后拆 seed 或改 `num_envs`；
- Goal67/68 的 rich trace 评估已经验证可以按 run 并行，GPU 主要用于训练而不是
  这类只读分析。

## 2. 只读诊断

1. 按 run/seed 分片并行，默认 `--jobs` 等于 run 数，最多先到 `16`；
2. 每个进程设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`，避免 384 CPU 上的
   线程过度订阅；
3. 短任务按 run 分片；单次预计超过 `60 s` 时再拆到 run×seed，避免为占满
   384 CPU 引入无意义的进程启动和合并开销；
4. 先跑最小 smoke，确认路径、checkpoint SHA、seed 范围和输出 schema；
5. 记录 `WALL`、`CPU%`、`RSS`，结果按 seed 层合并，不跨分片混 seed。

## 3. 训练实验

1. **先做吞吐基准**：同一 checkpoint/配置下测 `1 GPU` 与 `8 GPU` 的
   `transitions/s`、GPU 利用率、显存峰值和 CPU 负载，再决定并行度；
2. **按训练 seed 分 GPU**：一个 seed 一个进程，使用
   `CUDA_VISIBLE_DEVICES` 固定卡号；不要多个 seed 共享同一进程的全局 RNG；
3. **评估按 scenario/checkpoint 分片**：每个分片使用完全相同的 dev seeds、
   checkpoint 和评估协议，最后只在 seed 层合并；
4. **日志先写本地再同步**：避免 NFS/网络盘成为 384 CPU 并行的瓶颈。

## 4. 扩大 `num_envs` / batch 的规则

允许在探索型实验中扩大环境采样或 batch，但必须另开预注册 arm，并同时冻结：

- 总 transitions、UTD、replay 比例；
- eval seeds、eval split、held-out 关闭状态；
- checkpoint 选择规则和主指标；
- 实现、dtype/device、终止条件和 reward/observation 版本。

不能把 `num_envs`、batch、UTD 或 replay 比例的变化静默混进正式 Gate；也不能用
扩大预算或换 seed 来救失败结果。扩大采样/批量的目标只能是提高单位墙钟的
transitions 或降低方差，不是改变实验问题。

## 5. 建议的下一轮算力实验

Goal80 已经完成且判定 `TERMINAL_REFERENCE_HURTS`：goal-tangent 前移导致
default/strict125 plane-miss rate 增加 `+0.120/+0.060`，其中 zero-residual
也增加 `+0.075/+0.075`。因此当前不应再扫 terminal reference 值，也不应急着
开训练；先用已有 trace 做只读 termination/capture 审计。

如果 Goal81 之后需要训练型对照，先做不改变科学问题的吞吐基准：

Goal77 已证伪离线 terminal gate，Goal78 证明 full-vector oracle deadband
方向成立，Goal79 又证伪了可观测 predictor；Goal80 进一步证明 goal-tangent
前移会被 plane-miss guard 惩罚。当前没有正在训练的主线 arm，也不应继续扫
predictor、extension 或 residual 参数。建议先做 Goal81 只读审计；如果审计
支持 termination/capture 单因素 arm，再按下面三段使用训练算力，任何一段都不
改变 reward、gamma、seed 集、eval split 或 held-out：

1. **单卡吞吐基准（不评估）**：复用 Goal59/G12 的 off-policy 配置，固定
   `total_env_transitions=1M`、`update_to_data_ratio=0.125`、`learning_starts`、
   `buffer_size` 和网络结构，只扫：
   - `num_envs = 128 / 256 / 512`；
   - `batch_size = 256 / 512`；
   - 记录 `env_transitions/s`、`gradient_steps/s`、GPU 利用率、显存峰值、
     CPU 负载和墙钟；
   - 每个格点只跑 1 个短 seed，不用于报正式指标。
2. **8 卡并行正式 arm**：从吞吐基准选一个 `num_envs/batch_size`，冻结总
   transitions 和 UTD 后，`8 seeds × 8 GPUs`，一卡一个 seed，用
   `CUDA_VISIBLE_DEVICES` 固定卡号；禁止多个 seed 共享同一进程的全局 RNG。
   多个 batch 只表示并行跑多个 seed/arm，不表示在正式 arm 内静默改 batch。
3. **评估分片**：正式评估仍用 CPU/float64、`endpoint_stop`、固定 dev seeds，
   `--jobs` 最多先到 16，每进程单线程 BLAS；结果只在 seed 层合并。

只有在吞吐基准显示稳定收益后，才把新的 `num_envs/batch_size` 写进预注册
arm。SAC/REDQ/DroQ 仍然需要独立的失败证据和预注册 Gate，不能因为 8×4090
空闲就启动；扩大采样/批量也不能用来救已经失败的 Gate。

当前判断：Goal75/76/77/78/79 都是 CPU 只读分析，Goal80 是 10 个冻结
checkpoint 的 CPU/float64 闭环评估；增加 `num_envs` 或 batch 不会加快这些
问题，反而会改变协议。NoisyNet/residual 已不是剩余 plane-miss 的必要条件。
下一步 Goal81 先做只读 termination/capture 审计；只有进入训练型 arm 时，才
先做吞吐基准，再决定 `num_envs/batch_size` 和 8 卡 seed 并行度。目标是把 8
张 GPU 用于独立 seed/arm 的 batch 并行，而不是在正式 Gate 内静默改采样预算。

## 6. 记录要求

每次使用并行算力，在 result 或 blocker 中至少写：

- run 数、并行度、每进程线程设置；
- 墙钟、CPU%、RSS、GPU 数/利用率、显存峰值；
- 是否有 run 失败、重试或分片合并；
- 结果文件 SHA256 和本地/服务器路径。
