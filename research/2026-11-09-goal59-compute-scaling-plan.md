# Goal59 前的算力使用草案

日期：2026-11-09<br>
状态：草案；不改变 Goal58 结论，不启动新训练

## 1. 现状

Goal58 每个正式 seed 的 5M transitions 训练耗时约 `12,412–12,482 s`
（约 `3.45 h`），吞吐约 `400–403 transitions/s`、`49.9–50.2 gradient steps/s`。
每个 seed 共 `622,500` 次 gradient update。

服务器是 8×RTX 4090、384 CPU、503 GB RAM。Goal58 formal 只用 3 张卡，
另外 5 张卡空闲。

## 2. 瓶颈判断

按总量相除，每次 update 约 `19.9 ms`；每个 control step 约 `5` 次 update，
合计约 `99.5 ms`，而实测每个 control step 约 `99.3 ms`。因此当前 wall time
基本由 **gradient update** 决定，不是环境采样喂不饱 GPU。

这不代表 `num_envs` 或 batch size 没用，而是说明：

- 单纯增加 `num_envs` 不会自动缩短总时间；总 update 数仍由
  `transitions × update_to_data_ratio` 决定；
- 单纯增大 batch size 可能提高单次 update 的 GPU 利用率，但每次 update 更贵，
  也不减少 update 次数；
- 提高 `update_to_data_ratio` 会增加样本效率，但也会直接增加 wall time；
- 最确定的加速是 **多个独立 seed / 多个 screen 任务并行占满 8 张卡**。

以上是总量推断，Goal59 启动前应先用 microbenchmark 把 `env.step`、replay
sampling、H2D copy、forward/backward、optimizer 分开计时。

## 3. 三类加速，不要混在一起

### A. 纯并行，不改变学习协议

1. **8 路 seed 并行**：每个训练进程绑定一张 GPU，8 个独立 seed 同时跑；
   若单 seed 仍是 3.45 h，则 8 个 seed 的总 wall time 仍约 3.45 h，而不是
   8×3.45 h。formal 的 seed 集合、checkpoint 规则和 Gate 必须在 Goal59
   文档中预注册。
2. **screen 并行**：1M screen 约 39 min；可同时在 4–8 张卡上跑不同机制候选，
   每张卡一个独立输出目录，避免互相覆盖。
3. **eval 分片**：dev `20–59` 的 40 个 seed 切成 8 份，用 8 个 CPU/float64
   进程并行评估，再合并 summary；不改变 seed、场景、终止模式或 checkpoint。
4. **资源隔离**：每个 job 记录 GPU id、seed、config hash、代码树 hash；
   输出目录按 `screen_*` / `formal_s*` / `eval_*` 分开。

### B. 代码级吞吐优化，需 parity 测试

先 profile，再决定是否改：

- replay sampling / PER 更新是否占用大量 Python/NumPy 时间；
- 每步 NumPy→Tensor→GPU 的拷贝是否显著；
- GRU 序列前向、双 critic、target soft update 各占多少；
- `torch.compile`、CUDA graph、pinned memory 是否能保持数值一致。

任何优化都必须先过单测和短程 parity：同一 checkpoint、同一 seeds 下，
动作/损失/指标在容差内一致，才能用于正式实验。不能把“更快但结果不同”
当成纯工程优化。

### C. 改变学习协议，必须预注册

以下参数会改变数据分布或优化目标，不能在 screen 和 formal 之间静默切换：

| 参数 | Goal58 | Goal59 候选 | 影响 |
|---|---:|---:|---|
| `num_envs` | 40 | 128 / 256 | 每次 control step 的数据更多、并行 episode 更多；update 总数不变 |
| `batch_size` | 256 | 512 / 1024 | 单次 update 更贵；可能提高 GPU 利用率，但梯度噪声更低 |
| `update_to_data_ratio` | 0.125 | 0.25 | 每个样本更多更新；样本效率可能更好，但 wall time 近似翻倍 |
| `total_env_transitions` | 5M | 2–3M screen | 只用于 screen，不能替代 formal 5M |

“多个 batch”需要拆成两个问题：

- **batch size**：一次 update 用多少样本；
- **updates per rollout**：每个 control step 做多少次 update（由
  `update_to_data_ratio` 决定）。

两者都不是纯加速开关；只有并行 seed 和不改变数值的工程优化才是纯加速。

## 4. Goal59 建议执行顺序

1. **Phase 0：30–60 min microbenchmark**
   - `num_envs ∈ {40, 128, 256}`；
   - `batch_size ∈ {256, 512, 1024}`；
   - 固定短程 transitions，记录 `env.step ms`、`update ms`、`transitions/s`、
     `gradient/s`、GPU 利用率、显存、CPU 占用；
   - 不产生学习结论，不写入 Gate 结果。
2. **Phase 1：1-seed / 1M screen**
   - 每个机制候选 1 张卡；最多同时跑 8 个；
   - 通过 screen 后才进入 formal；
   - screen 配置与 formal 配置必须一致，或在 Goal59 中明确写成两个阶段。
3. **Phase 2：formal 8 seeds × 5M**
   - 一张卡一个 seed，8 张卡同时跑；
   - dev `20–59`，default/strict125，CPU/float64，`endpoint_stop`；
   - held-out `100–149` 继续关闭；
   - checkpoint 用预注册的 final，不选 best。
4. **Phase 3：并行 eval + 汇总**
   - 8 路分片评估，合并后统一算 paired mean / CI；
   - 不因为并行而改变任何 seed 或场景。

## 5. 护栏

- 不用更多 seed、更多算力或更大 batch 去救已经 FAIL 的 Gate；
- 不把不同 `num_envs` / `batch_size` / UTD 的结果当作同一协议直接比较；
- 不在 screen 和 formal 之间静默改配置；
- formal 仍然看效应量和护栏，不看“跑了多少卡”；
- 所有加速只服务于更快得到可信结论，不服务于得到想要的结论。

## 6. 下一步

Goal59 文档预注册时，把本草案中的 Phase 0 配置、formal seed 集合、
`num_envs` / `batch_size` / UTD 和 eval 分片方式写死；在正式长训练前先跑
microbenchmark。若 microbenchmark 显示 update 仍是瓶颈且没有安全的工程优化，
优先采用 8 路 seed 并行，不为了“用满 GPU”而改学习协议。

## 7. 2026-11-09 实测 microbenchmark

在服务器 GPU 0 上用 Goal59 G12 配置做 15 个 control step + 25 次 update 的短程
基准。数字只用于吞吐判断，不产生学习结论。

| num_envs | batch | env step ms | update ms | env transitions/s | gradients/s | peak allocated GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 256 | 23.40 | 21.25 | 1,710 | 47.1 | 0.31 |
| 40 | 512 | 22.10 | 15.44 | 1,810 | 64.8 | 0.32 |
| 40 | 1024 | 22.05 | 17.30 | 1,814 | 57.8 | 0.32 |
| 128 | 256 | 29.22 | 15.62 | 4,380 | 64.0 | 0.96 |
| 128 | 512 | 29.30 | 15.37 | 4,368 | 65.1 | 0.96 |
| 128 | 1024 | 29.27 | 17.78 | 4,374 | 56.2 | 0.96 |
| 256 | 256 | 39.44 | 16.80 | 6,491 | 59.5 | 1.91 |
| 256 | 512 | 39.49 | 15.62 | 6,483 | 64.0 | 1.91 |
| 256 | 1024 | 39.39 | 17.87 | 6,500 | 56.0 | 1.91 |

结论：

- `128 envs + batch 256` 的 update 约 `15.6 ms`，显存约 `0.96 GiB`；`256 envs`
  把 env 吞吐从约 `4.4k/s` 提到 `6.5k/s`，但 update-bound 下总 wall time
  几乎不变，反而增加显存和协议差异；
- `batch 512` 只比 `256` 快约 `1–2%`，不足以抵消“改变梯度噪声/学习协议”的
  代价；`batch 1024` 更慢；
- 因此 Goal59 保持预注册的 `128 envs / batch 256 / UTD 0.125`，真正的加速来自
  **多路 screen + 8 路 formal seed 并行**，以及 8 路 eval 分片，而不是偷偷改
  batch 或提高 UTD；
- 2026-11-09 已并行启动 `G8/G12/U12/N12/T12` 五个 1M screen，分别绑定空闲
  GPU 0/1/3/5/6；`N12` 和 `T12` 只作非 Gate 诊断，不进入 formal。

## 8. Goal59 formal 的实际调度

formal 启动时 GPU 2/4 被其他用户占用，因此先用 GPU 0/1/3/5/6/7 跑六个
预注册 seed（`240/368/496/624/752/880`）。这不改变 seed 集合、`128 envs`、
`batch 256`、UTD `0.125` 或 `final.pt` 规则。

剩余 `1008/1136` 使用一个只负责调度的 watcher：

- 监听前六个 seed 的 screen 会话；
- 任一 seed 结束并释放 GPU 后，立即在对应 GPU 启动下一个预注册 seed；
- 启动命令只替换 `--output`、`--train-seed-start` 和 `--seed`；
- watcher 不改配置、不选 best、不增加 seed，也不碰其他用户的 GPU 进程。

formal 完成后的并行策略：

- NoisyNet/raw saturation probe：8 个 seed 各一个 CPU/float64 进程并行；
- dev 配对评估：8 个 seed × 2 个场景 = 16 个 CPU/float64 进程并行；
- 每个进程仍只用 dev `20–59`、`endpoint_stop` 和各自的 `final.pt`；
- 汇总前先检查每个 summary 的 finite、seed 数、场景和协议字段，再跑配对脚本。

这样增加的是任务级并行度，而不是偷偷增加单次 batch、环境数或更新次数。正式
Gate 的学习协议保持不变。

## 9. 并发评估的实测教训

在 `1008/1136` 还在训练时，曾尝试同时启动 12 个 dev `CPU/float64` 评估进程。
每个进程默认会使用大量 CPU 线程，12 路并发把服务器 load average 推到
`1191`，两个训练进程的 GPU 利用率从正常约 `15–20%` 掉到 `7–14%`，相当于用
评估拖慢了正式训练。这次尝试被立即停止，未保留不完整输出，也未改变任何协议。

后续规则：

- 正式训练期间，评估并发最多 1–2 路；若训练吞吐下降，立即让训练优先；
- 正式训练全部结束后，再统一跑 probe 和 dev 配对评估；
- 即使训练结束，也建议最多 4 路并发，并在每轮检查 load、GPU 利用率和日志；
- 不通过设置不同线程数来“加速”正式评估，避免与缓存的 zero-residual 基线
  产生不必要的数值协议差异；
- 多卡加速应优先用于独立训练 seed，而不是把大量 CPU 评估塞进同一时间段。

结论：**算力利用率高不等于并发越多越好**。在当前 update-bound 的训练里，
保护训练吞吐比同时开满所有评估进程更重要。

## 10. 最终结论：怎样才算“充分利用算力”

用户希望增加环境采样、多个 batch 来加快迭代。Goal59 的实测给出的结论是：
**这些参数不是免费的加速开关，真正的加速来自独立任务并行**。

### 10.1 为什么“更多 env / 更大 batch”没有把 wall time 降下来

- 训练是 update-bound：`128 envs` 时 env step 约 `29 ms`，一次 update 约
  `15.6 ms`，每个 control step 约 5 次 update；总时间主要由 update 次数决定。
- `256 envs` 把 env 吞吐从约 `4.4k/s` 提到 `6.5k/s`，但 update 吞吐几乎不变，
  因此 5M transitions 的总 wall time 不会按 env 吞吐同比例下降，反而增加显存和
  数据分布差异。
- `batch 512` 比 `256` 只快约 `1–2%`，`batch 1024` 更慢；不足以抵消改变梯度
  噪声和学习协议的代价。
- “多个 batch”要拆成两件事：`batch_size`（一次 update 用多少样本）和
  `updates per rollout`（由 UTD 决定）。前者主要影响单次 update 成本和梯度噪声，
  后者直接增加 update 次数和 wall time；两者都不能静默改。

### 10.2 可以放心使用的加速

1. **8 路独立 seed 并行**：一张卡一个 seed，seed 集合、配置、预算和
   `final.pt` 规则在 Goal 文档中预注册。Goal59 的 8 个 formal seed 就是这样跑的。
2. **多机制 screen 并行**：每个候选一张卡、独立输出目录；screen 只做工程筛选，
   不把 screen 结果写成 formal。
3. **训练结束后的 eval 分片**：dev `20–59` 按 seed 切片并行，合并后再算 paired
   mean / CI；不改变 seed、场景、终止模式或 checkpoint。
4. **CPU 线程限额**：每个 eval 进程显式限制线程数，避免 12 路并发把 load 推到
   `1191`、反过来拖慢训练。训练期间 eval 最多 1–2 路；训练结束后最多 4 路，
   每轮检查 load 和训练/评估吞吐。

### 10.3 需要预注册才能改的参数

- `num_envs`：改变每个 control step 的数据分布和 episode 并行度；
- `batch_size`：改变梯度噪声和单次 update 成本；
- `update_to_data_ratio`：改变样本效率和 wall time，属于学习协议；
- `total_env_transitions`：screen 可以用更短预算，但不能替代 formal 预算。

如果下一 Goal 想试 `256 envs` 或 `batch 512`，必须先做短程 parity 和吞吐基准，
把选择写进预注册文档，再用于正式比较。仅当吞吐提升足够大（例如 `>10%`）且
短程学习曲线/动作统计在容差内一致时，才值得改；否则保持 `128 envs / batch 256`
并把算力用于更多独立 seed。

### 10.4 下一 Goal 的建议调度

- **只读诊断阶段**：优先用现有 trace 和 frozen checkpoint，CPU 并行最多 4 路；
  先限制线程数，再启动，监控 load；不启动新训练。
- **需要新训练时**：8 张卡各跑一个预注册 seed；不为占满 GPU 而改变
  `num_envs`、`batch_size` 或 UTD；screen 与 formal 的配置差异必须在文档中写明。
- **结果汇总**：先检查每个 summary 的 finite、seed 数、场景和协议字段，再跑
  配对脚本；并行只缩短等待时间，不改变统计口径。

一句话：**算力应该用来增加独立证据（seed、screen 候选、eval 分片），而不是
用来偷偷放大 batch 或环境数，把失败 Gate 跑得更快。**
