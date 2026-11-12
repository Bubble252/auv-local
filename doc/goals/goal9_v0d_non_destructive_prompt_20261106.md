# Goal9 V-0D 非破坏性实现提示词

记录日期：2026-11-06

用途：交给下一轮执行者，完成 V-0D 的代码、配置、测试和文档准备。本文是执行提示词，
不是实验结果。权威计划见 `goal9_v0d_direct_pure_path_plan_20261106.md`。

```text
你在 /home/bubble/水下机器人/settled_局部规划器_反应式 工作。

请完成 Goal9 V-0D 纯路径主臂的非破坏性实现与审计。只做代码、配置、测试和文档，
不启动正式训练，不启动长评估，不覆盖已有 artifacts/checkpoint/日志。完成后把改动、
风险和建议交给我人工审核。

## 已确认的目标与契约

1. Goal9 的主线是路径规划/路径跟踪，不是世界模型、SuperMimic 或完整策略探索。
2. V-0D 是主臂：actor 输入为基础状态 + 8D 路径观测，命令 c_t=[]。
3. 路径先经 path encoder 融合进 z_t；DADE/history 只消费编码后的 z_t，不直接吃原始
   8D 路径。当前速度状态可以保留，删除的是目标命令 [psi_d, theta_d, u_d]。
4. 动作层使用冻结 BC + 动作残差：R-BC-0D -> R-Z0-0D -> R-PPO-0D，不做完整策略替代。
5. V-3D 只是可选诊断控制；本轮不要实现或启动 V-3D，也不要把 V-2D/V-1D 当必经阶段。
6. 第一主指标是 3D RMSE；success、strict125、timeout、safety 是护栏指标，默认允许的
   绝对恶化不超过 0.02。material saturation 只作风险信息。
7. path-zero 必须使 3D RMSE 恶化至少 5%，否则不能宣称路径输入有效。
8. critic 可以先保留 3D 命令作为训练特权信息；部署 actor 不得读取命令。若实现时发现
   这会污染 actor，先停下来报告。
9. 不新增 speed head；速度先由 pc_path_track 奖励、当前速度状态和 RPM 隐式学习。
   转弯可自然减速但不加硬约束，终端不要求减速。
10. 不新增过冲惩罚。奖励优先级为路径准确与时间，其次是速度平滑，最后才是动作平滑。
11. R-BC-0D 示范优先复用现有 ALOS 教师采集中低速、正常转弯和路径恢复轨迹；示范
    契约必须与 V-0D 一致，命令维度为 0。
12. 初始状态的速度与 RPM 不要求为 0；示范和评估都要覆盖非零速度/非零 RPM。零初态
    只用于回归测试，不作为默认数据分布。

## 最高优先级：非破坏性写代码

- 不删除、不移动、不重写已有配置、历史报告、artifacts、checkpoint 和测试数据。
- 不修改旧配置的默认行为。旧 3D BC 配置不填新字段时，必须保持原来的 schema、奖励、
  动作和 checkpoint 加载语义。
- 优先新增文件、新增 mode、新增可选字段和 feature flag；不要在共享代码里做全局重构。
- 不重命名已有 public API、配置字段、schema 字段或 checkpoint 字段。
- 如果必须改共享代码，先用最小改动完成，并新增回归测试证明旧路径行为不变。
- 不做无关修复，不顺手改 DADE、历史编码、控制频率和奖励。
- 不运行正式训练、不覆盖输出目录、不生成正式 checkpoint；只跑 targeted tests、
  ruff 和 smoke/contract 检查。若确实需要运行命令，先说明目的和产物路径。
- 不 git commit、不 git push、不使用 git add -A。

## 第一阶段的只读审计

先检查并写出简短审计，不要立即大改：

1. 确认冻结 BC 的 base_action() 是否读取 command；若读取，说明为什么不能用现有
   3D BC checkpoint + command_mask 冒充 V-0D。
2. 确认当前 BC 训练器是否硬编码 3D command、示范数据集是否包含 path；列出需要扩展的
   最小字段和接口。
3. 确认 PC observation layout 中 state、path、command、history token 的边界；
   设计 0D actor 后，路径仍只通过 z_t 进入 actor，命令不进入任何隐式通道。
4. 确认现有 path_learned_speed、pc_path_track、path-zero、frozen_pc_bc_residual 和
   residual_action_mask 如何复用。列出不能直接复用的原因。
5. 列出计划新增/修改的文件、每个文件的理由、兼容性风险和测试点。若必须迁移旧
   checkpoint schema，先停下来报告，不要擅自迁移历史权重。

## 第二阶段的实现范围

在审计通过后，只实现 V-0D 所需的最小集合：

1. BC 数据增加可选的 actor_path 字段；旧数据不提供时行为不变。
2. BC actor 支持 command_dim=0，并增加 path encoder；旧 3D BC 默认路径不变。
3. 新增 R-BC-0D 配置：state + 8D path -> action，命令维度 0，复用现有归一化和
   teacher action 标签；示范采集复用 ALOS 教师，并允许非零速度/RPM 初态；不得覆盖旧 BC。
4. 新增 V-0D PPO 配置：actor_command_dim=0、path_dim=8、pc_path_track，
   base_policy.mode=frozen_pc_bc_residual，指向新 R-BC-0D checkpoint。
5. 复用现有 residual_action_mask；默认全 1，不改变旧配置行为。
6. 评估器支持 0D actor + 8D path 的 closed-loop 评估和 path-zero。
7. 增加回归测试：
   - 旧 3D 配置仍通过原有验证；
   - 旧 3D BC checkpoint 仍能按原方式加载；
   - 新 0D 配置能解析，actor command 宽度为 0；
   - 命令列不进入 encoder/DADE/history；
   - path-zero 能作用在路径块上；
   - V-0D reward 确实使用 pc_path_track 的路径项；
   - 冻结 BC 参数不进入 actor optimizer，残差零初始化。

## 交付物

1. 审计报告：现状、最小改动方案、兼容性风险。
2. 新配置文件和必要的代码改动。
3. targeted tests、ruff、配置解析和 smoke/contract 检查结果。
4. git diff --check、git status --short 和变更文件列表。
5. 明确说明哪些旧行为保持不变，哪些新行为只在 V-0D 新模式下生效。
6. 不启动正式训练；建议下一步只跑 R-BC-3D 与 R-BC-0D 的闭环对照。

## 停止条件

- 需要删除或重写旧配置/历史结果才能继续；
- 必须改变旧 checkpoint schema 或旧配置默认行为；
- 发现命令会被隐式编码、进入 DADE、history 或从 state 泄漏；
- BC-0D 无法在不改旧 3D BC 行为的前提下实现；
- 任何改动会污染旧实验输出或需要启动正式训练。

遇到以上情况，停止修改并报告，不要自行扩大范围。
```
