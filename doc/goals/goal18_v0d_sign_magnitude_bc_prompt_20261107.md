# Goal18 V-0D：Sign/Magnitude 动作头 BC-only 对照

日期：2026-11-07

状态：**已执行并收口。判定 `FAIL_RMSE`，停止，不进入残差 PPO。**

结果报告：`doc/goals/goal18_v0d_sign_magnitude_bc_report_20261107.md`

优先级：**高于 Goal17。** Goal17 的 current estimate × crab 因子实验暂缓；先回答
“动作表示是否是当前 BC 失败的主因”。在 Goal18 收口并人工确认前，不启动 Goal17。

关联文档：

- `doc/goals/goal16_v0d_crab_rate_limited_teacher_report_20261107.md`
- `doc/goals/goal17_v0d_current_crab_factorial_plan_20261107.md`
- `doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md`
- `doc/goals/goal9_execution_state.md`

## 1. 要回答的问题

在不改变教师、示范数据、观测契约、actor 编码器和训练预算的前提下，把连续动作回归头
替换为 sign/magnitude 动作头，能否同时做到：

1. 把 BC 的 default 3D RMSE 推进到 `<= 7.0 m`；
2. 恢复教师的主要饱和动作，而不是把 `+1/-1` 平均成接近零；
3. 不引入不可接受的零附近抖动或执行器速率限幅恶化；
4. 若 BC 通过，再进入冻结 BC 的残差 PPO screen。

本 Goal 是 **BC-only 动作表示实验**，不是教师调参、不是 PPO 算法改造，也不是新的
观测可观测性实验。

## 2. 当前证据

R2B 教师和已有 BC 的证据表明，问题更像“动作表示丢失饱和”，而不是简单的控制频率
过高：

- R2B 教师 rudder/stern 0.99 饱和率约 `55.27% / 19.85%`；
- 合并示范中约 `59.88% / 21.13%`；
- 实际执行器已有 `150 deg/s` rate limit，50 Hz 下每步最多约 `3 deg`；
- 现有示范中约 `18.65%` 的样本至少一个通道触发 rate limit；
- 相邻两步从大幅正值直接反转到大幅负值的次数为 0，说明大幅动作并没有以
  “一步反向”的方式发生；
- 现有 MSE BC 的预测 rudder/stern 饱和率只有约 `16–30% / 8–9%`；
- 教师动作高度饱和且方向明确，MSE 的均值化会把饱和尾部抹掉。

因此本 Goal 的机制假设是：

> 连续 MSE 回归无法稳定表示“先决定方向、再决定幅值”的饱和动作；
> 显式的 sign 分类与 magnitude 回归可以保留动作方向和非零幅值。

## 3. 固定条件

以下条件在 S0/S1 中必须完全一致：

- 数据：
  `artifacts/lowlevel/pc_v0d_r2b_bc_20261106/demos_dagger_merged/demos.npz`
  （SHA256 以 manifest 为准）；
- 教师：R2B，不重采、不增加 DAgger、不添加 action noise；
- actor 观测：现有 12D base state + 8D path，`command_dim=0`；
- `z_t` 宽度 `64`、`n_in_reserved=8`、`n_z_reserved=4`；
- state encoder `[128,128]`、actor trunk `[256,128]`、path encoder `[64]`；
- `batch_size=4096`、`max_epochs=300`、`patience=30`、`lr=1e-3`、
  `weight_decay=1e-5`、`validation_fraction=0.15`、`seed=0`；
- 训练 device 使用服务器 CUDA；正式闭环评估使用 CPU、float64、seed `0–19`；
- `saturation_weight=0`，不在本 Goal 用加权 MSE 混淆动作头效应；
- 不改变 plant、sensors、current、reference、reward、termination；
- 不改变评估 seed、目标半径和既有 BC/PPO 门禁。

## 4. 动作头设计

### 4.1 连续动作定义

现有动作仍为归一化三维向量：

```text
a_t = [rudder, stern, rpm]
```

其中 rudder/stern 取 `[-1, 1]`，RPM 保持现有连续归一化映射。

### 4.2 Sign/magnitude 参数化

只替换 actor 最后一层，前面的 `z_t` 和 actor trunk 不改变。

对 rudder 和 stern 分别输出：

```text
sign_logits_i ∈ R^3       # {negative, zero, positive}
magnitude_i ∈ [0, 1]      # sigmoid
```

最终解码：

```text
rudder = sign(rudder) × magnitude_rudder
stern  = sign(stern)  × magnitude_stern
rpm    = continuous_rpm_head
```

要求：

- deadband 预注册为 `epsilon=0.05`，用于生成 sign target：
  `a > epsilon -> positive`，`a < -epsilon -> negative`，否则 `zero`；
- sign 使用交叉熵；magnitude 只在 sign 非零的样本上使用 SmoothL1/L1；
- RPM 保持连续回归，不强行离散化；
- 预注册总损失为
  `CE(sign) + SmoothL1(magnitude, mask=nonzero) + MSE(rpm)`，三项等权，不做扫描；
- 推理时 RPM 截断到 `[-1, 1]`，保证输出仍是合法归一化三维动作；
- 输出仍解码成连续三维动作，以保持与现有评估器和未来残差 PPO 的接口兼容；
- 默认 `action_head="continuous"`，旧 checkpoint 和旧路径完全不变。

### 4.3 不混入的变量

本 Goal 不测试以下内容：

- current estimate 编码；
- DADE/history 侧通道；
- 新的世界模型；
- 新的观测课程；
- 第三组教师 rate；
- action chunking 或 diffusion；
- PPO 算法重写。

## 5. 两个必需实验格

### S0：连续 MSE 对照

- 现有数据、现有 actor trunk、现有训练配方；
- 最后一层恢复为连续 3D 输出；
- 使用当前代码重新训练一次，作为同代码、同数据、同 seed 的公平对照；
- 已存在的 v1 DAgger BC 和 v2.1 weighted BC 只作历史参考，不替代 S0。

### S1：sign/magnitude BC

- 与 S0 完全相同的 trunk、数据、优化器和评估；
- 仅替换最后一层和损失；
- sign 三分类、magnitude 回归、RPM 连续回归；
- 每个 run 只产出一个 `pc_bc_actor.pt`。

不增加 S2、不扫描 deadband、不扫描 sign 类别权重。若 S1 失败，下一步另开 Goal
研究 action chunking 或 mixture-density，不在本 Goal 继续调头。

## 6. 评估指标

### 闭环主指标

- default 3D RMSE；
- default success；
- strict125 3D RMSE；
- strict125 success；
- timeout、safety；
- path-zero 3D RMSE 与正常 path 的差值。

### 动作表示诊断

在固定 validation split 上计算：

- rudder/stern sign accuracy（只对 `|teacher action| >= 0.5` 的样本）；
- magnitude MAE；
- teacher 饱和样本的饱和召回率；
- predicted 0.99 饱和率与 teacher 0.99 饱和率的比值；
- 大幅反向次数/秒（两侧 `|a| > 0.5`）；
- rate-limit 命中率（用现有 `BatchedPureActionPipeline` 计算）；
- 零附近样本比例（`|a| < 0.05`）。

“饱和动作明显恢复”预先定义为：

```text
predicted_saturation_rate / teacher_saturation_rate >= 0.65  (rudder)
predicted_saturation_rate / teacher_saturation_rate >= 0.65  (stern)
sign_accuracy >= 0.90  (|teacher action| >= 0.5)
```

这三项是进入 PPO 的必要条件之一；不能只看平均动作 MSE。

## 7. BC 门禁

S1 必须同时满足：

- default 3D RMSE `<= 7.0 m`；
- default success `>= 0.60`；
- default RMSE `<= 0.80 × 旧 BC 12.2396 = 9.7917 m`；
- strict125 3D RMSE `<= 10.5 m`；
- strict125 success `>= 0.10`；
- safety 不高于旧 BC `+0.02`；
- path-zero 相对正常 path 明显恶化，证明路径输入仍被使用；
- 上述“饱和动作明显恢复”三项全部满足；
- 大幅反向率不高于教师 `+0.05 次/秒`；
- rate-limit 命中率不高于教师 `+0.05`。

结果分类：

- **PASS**：闭环门禁与动作表示门禁全部满足，允许进入残差 PPO screen；
- **FAIL_RMSE**：动作表示改善但 default RMSE `>7.0`，停止，不进入 PPO；
- **FAIL_GUARDRAIL**：RMSE 门禁通过，但 success、safety 或 path-zero 护栏失败；
- **FAIL_REPRESENTATION**：RMSE 达标但饱和/方向恢复不足，停止，不进入 PPO；
- **FAIL_STABILITY**：反向率或 rate-limit 命中率显著恶化，停止，不进入 PPO。

不得根据测试集结果修改 deadband、权重、seed 或门禁。

## 8. 阶段与检查项

### 阶段 0：基线与契约审计

- [x] 确认服务器没有残留 teacher/BC/PPO/eval runner。
- [x] 记录数据集 SHA256、样本数、episode 数和 teacher config SHA256。
- [x] 确认 S0/S1 使用同一 data split、同一 seed、同一 normalization。
- [x] 记录旧 BC、v2.1 weighted BC 的 checkpoint 和闭环指标，仅作参考。
- [x] 记录当前动作头 checkpoint schema 和 `PCActorCritic` 兼容契约。

### 阶段 1：非破坏性实现 sign/magnitude 分支

- [x] `PCBCTrainingConfig` 新增 `action_head: str = "continuous"`，默认保持旧行为；
- [x] 新增 `SignMagnitudeActionHead`，默认不创建、不改变旧 checkpoint；
- [x] `PCBCTrainableActor` 支持 `continuous` 与 `sign_magnitude` 两种输出契约；
- [x] `train_pc_bc.py` 新增显式 `--action-head sign_magnitude`（可接受
  `sign-magnitude` 别名）和 `--sign-deadband 0.05`；
- [x] checkpoint metadata 记录 `action_head`、`sign_deadband`、输出宽度和 joint
  decode 规则；
- [x] `PCBCPolicy` 自动按 metadata 解码，旧 checkpoint 不要求新字段；
- [x] 错误 checkpoint 必须明确拒绝，不能静默按连续动作加载；
- [x] 评估器仍输出连续三维动作，不改变下游 PPO/plant 接口。

### 阶段 2：单元与契约测试

- [x] 连续动作头默认路径逐位保持旧行为；
- [x] sign 三分类 target 的 deadband 边界正确；
- [x] magnitude 只在非零 sign 样本上训练；
- [x] 解码后动作在 `[-1,1]` 内，RPM 映射正确；
- [x] 旧 checkpoint 加载后动作输出不变；
- [x] sign/magnitude checkpoint 的 metadata round-trip；
- [x] 闭环评估器能加载并执行 S1 checkpoint；
- [x] targeted tests 在本地和服务器均通过。

### 阶段 3：S0 连续 MSE 对照训练

- [x] 用固定配置训练 S0；
- [x] 记录 best epoch、validation RMSE、checkpoint SHA256；
- [x] 记录 S0 的 default/strict125/path-zero 闭环指标；
- [x] S0 只用于同代码对照，不用于挑选门禁后参数。

### 阶段 4：S1 sign/magnitude BC 训练

- [x] 用与 S0 相同的数据、split、seed 和训练预算训练 S1；
- [x] 记录 sign accuracy、magnitude MAE、饱和召回率和饱和率；
- [x] 记录反向率、rate-limit 命中率；
- [x] 记录 checkpoint SHA256 和完整训练 manifest；
- [x] 不因 validation loss 单独挑选 epoch；按预注册验证损失和动作诊断选模型。

### 阶段 5：BC 闭环评估与决策

- [x] default、strict125、path-zero 全部评估；
- [x] 按第 7 节门禁给出 PASS/FAIL 分类（`FAIL_RMSE`）；
- [x] 若 FAIL，停止本 Goal，不进入 PPO；
- [ ] 若 PASS，冻结 S1 checkpoint 并记录 SHA256（未触发，S1 FAIL）。

### 阶段 6：残差 PPO screen（仅 PASS，未触发）

- [ ] 复用既有冻结 BC + 残差 PPO 协议，不重新定义 PPO；
- [ ] 先跑 zero-residual 基线；
- [ ] screen：`4096 env × 256 step × 40 iteration`，3 个训练 seed，
  评估 seed `0–19`；
- [ ] screen 门禁沿用既有协议：RMSE ratio `<=0.95`、success 下降 `<=0.05`、
  timeout 增加 `<=0.10`、safety 增加 `<=0.02`；
- [ ] screen 失败即停止，不跑正式 `3×50`；
- [ ] screen 通过后才考虑正式 `3×50`。

## 9. 产物与日志

根目录：

```text
artifacts/lowlevel/pc_v0d_sign_magnitude_bc_20261107/
```

建议子目录：

```text
s0_continuous/
s1_sign_magnitude/
eval/
ppo/
```

日志：

```text
logs/pc_v0d_sign_magnitude_bc_20261107/
```

每个 run 的 manifest 至少包含：

- git HEAD；服务器没有 `.git` 时记录本地 HEAD 并注明；
- git status/diff 摘要；
- 完整命令行；
- dataset/config/checkpoint SHA256；
- 训练 seed、评估 seed、device、dtype；
- 开始/结束时间；
- `action_head`、`sign_deadband`、输出维度和解码规则。

完成后新增结果报告，并更新：

- `doc/goals/goal9_execution_state.md`
- `doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md`
- `doc/goals/goal17_v0d_current_crab_factorial_plan_20261107.md` 的优先级说明

## 10. 停止条件

- S0/S1 数据、seed、split 或 normalization 不一致；
- S1 需要修改观测契约、教师数据或降低既有 BC 门禁才能继续；
- S1 为 `FAIL_RMSE`、`FAIL_REPRESENTATION` 或 `FAIL_STABILITY`；
- 服务器不可用，无法完成同口径评估；
- 需要引入 action chunking、MDN 或新的世界模型才能继续。

除上述情况外，不扩大参数扫描、不增加第三组动作头、不换 seed、不修改门禁。

## 11. 最终输出表

| 问题 | 需要回答 |
|---|---|
| S0 是否复现了现有 MSE BC 的量级？ | 是；`8.9129 m`，同配方历史 artifact `9.0944 m`，比值 `0.9800` |
| S1 的 default/strict125/path-zero 指标是多少？ | `10.2868 m / 0.50`、`11.7268 m / 0.30`、`11.9473 m / 0.00`（safety `0.20`） |
| S1 的 sign accuracy、饱和召回率和饱和率是多少？ | sign `0.9501 / 0.9548`；饱和恢复 `0.2689 / 0.1524`；饱和率 `16.10% / 3.22%`（教师 `59.88% / 21.13%`） |
| S1 的大幅反向率和 rate-limit 命中率是否可接受？ | 否；反转 `0.6241 / 0.1859 次/秒`，任一通道 `0.8095`；rate-limit `71.70%`（教师 `38.69%`） |
| S1 是 PASS、FAIL_RMSE、FAIL_REPRESENTATION 还是 FAIL_STABILITY？ | `FAIL_RMSE`，同时表示与稳定性门禁也失败 |
| 是否进入残差 PPO？screen 结果是什么？ | 否；未进入 PPO，因此无 screen 结果 |
| 是否修改过观测契约、教师数据、seed 或既有门禁？ | 否 |

## 12. Git 建议

不自动 commit/push；不使用 `git add -A`。若用户要求提交，只添加本 Goal 相关文件：

```bash
git add src/auv_lowlevel/batch/pc_bc.py \
  scripts/train_pc_bc.py \
  scripts/evaluate_pc_bc.py \
  tests/test_lowlevel_pc_bc.py \
  doc/goals/goal18_v0d_sign_magnitude_bc_prompt_20261107.md \
  doc/goals/goal9_execution_state.md \
  doc/goals/goal9_v0d_teacher_r2b_revalidation_20261106.md
git commit -m "docs(goal18): register sign-magnitude BC comparison"
git push origin <当前分支>
```

实际代码完成后，再将训练/评估脚本和测试加入同一个 commit；`artifacts/` 和
`logs/` 不提交。
