# Goal9 V-3D 非破坏性实现提示词

记录日期：2026-11-06

> **状态：已被取代，不要执行。** 用户已于 2026-11-06 最终确认直接以 `V-0D`
> 纯路径为主臂，`V-3D` 只作可选诊断控制。当前执行入口改为
> `doc/goals/goal9_v0d_non_destructive_prompt_20261106.md`；权威计划见
> `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`。本文仅保留为历史记录。

用途：交给下一轮执行者，完成 V-3D 的代码、配置和测试准备。本文是执行提示词，不是
实验结果。

```text
你在 /home/bubble/水下机器人/settled_局部规划器_反应式 工作。

请完成 Goal9 的 V-3D 非破坏性实现与审计。只做代码、配置、测试和文档，不启动正式训练，
不启动长评估，不覆盖已有 artifacts/checkpoint/日志。完成后把改动、风险和建议交给我人工审核。

## 已确认的目标与契约

1. Goal9 的主线是路径规划/路径跟踪，不是世界模型、SuperMimic 或完整策略探索。
2. V-3D 的输入是：
   - 基础状态；
   - 8D 路径观测；
   - 3D 命令 [psi_d, theta_d, u_d]。
3. 只有基础状态和路径观测编码进 z_t。3D 命令只做固定归一化
   [psi_d/pi, theta_d/(pi/2), u_d/2.5]，然后作为裸标量直接拼入 actor；命令不编码、
   不进入 DADE、不进入历史 token。
4. 动作层使用冻结 BC + 动作残差：R-BC -> R-Z0 -> R-PPO，不做完整策略替代。
5. 第一主指标是 3D RMSE；success、strict125、timeout、safety 是护栏指标，默认允许的
   绝对恶化不超过 0.02。material saturation 只作风险信息。
6. path-zero 必须使 3D RMSE 恶化至少 5%，否则不能宣称路径输入有效。
7. 在 V-3D 通过前，不启动 V-2D/V-1D/V-0D 命令消融；命令消融顺序固定为
   u_d -> psi_d -> theta_d。

## 最高优先级：非破坏性写代码

- 不删除、不移动、不重写已有配置、历史报告、artifacts、checkpoint 和测试数据。
- 不修改旧配置的默认行为。旧配置不填新字段时，必须保持原来的 schema、奖励、动作和
  checkpoint 加载语义。
- 优先新增文件、新增 mode、新增可选字段和 feature flag；不要在共享代码里做全局重构。
- 不重命名已有 public API、配置字段、schema 字段或 checkpoint 字段。
- 如果必须改共享代码，先用最小改动完成，并新增回归测试证明旧路径行为不变。
- 不做无关修复，不顺手改其他实验，不把 DADE、历史编码、控制频率和奖励同时改掉。
- 不运行正式训练、不覆盖输出目录、不生成新 checkpoint；只跑 targeted tests、ruff 和
  smoke/contract 检查。若确实需要运行命令，先说明目的和产物路径。
- 不 git commit、不 git push、不使用 git add -A。

## 第一阶段的只读审计

先检查并写出简短审计，不要立即大改：

1. 当前为什么 `pc_control_mode=alos` 强制 `reward.mode=pc_command_track`；设计一个
   向后兼容的新模式或可选组合，让 V-3D 能同时使用 3D ALOS 命令输入和 pc_path_track。
2. 当前 PC observation layout 中 state、path、command、history token 的边界；
   确认命令不进入 encoder/DADE/history，路径只通过 z_t 进入 actor。
3. 当前 BC checkpoint 的命令列、path encoder 零初始化、frozen residual 加载和 schema
   hash；确认新增 V-3D 配置不会破坏旧 checkpoint。
4. 是否存在 command mask 支持。若没有，设计一个默认全 1 的可选 mask，供后续
   V-2D/V-1D/V-0D 使用；不得改变旧配置未设置 mask 时的行为。
5. 列出计划新增/修改的文件、每个文件的理由、兼容性风险和测试点。若发现必须改变旧
   schema 才能完成，先停下来报告，不要擅自迁移历史 checkpoint。

## 第二阶段的实现范围

在审计通过后，只实现 V-3D 所需的最小集合：

1. 新增向后兼容的 control/reward 组合，使 3D ALOS 命令输入与 pc_path_track 奖励可以
   同时使用；旧模式路径保持不变。
2. 新增 V-3D 环境配置：3D 命令、8D 路径、路径奖励、与旧实验一致的归一化和传感器设置。
3. 新增 V-3D 训练配置：复用现有 3D 命令 BC，明确 R-PPO 残差配置；不得修改旧配置。
4. 增加 command mask 的接口设计和测试；即使本轮只跑 V-3D，也要保证接口可扩展到
   V-2D/V-1D/V-0D，且默认行为不变。
5. 增加回归测试：
   - 旧配置仍通过原有验证；
   - 新配置能解析；
   - 命令列仍不进入 encoder/DADE/history；
   - path-zero 能作用在路径块上；
   - 旧 checkpoint 仍能按原方式加载；
   - V-3D 的 reward 确实使用 pc_path_track 的路径项。

## 交付物

1. 审计报告：现状、最小改动方案、兼容性风险。
2. 新配置文件和必要的代码改动。
3. targeted tests、ruff、配置解析和 smoke/contract 检查结果。
4. `git diff --check`、`git status --short` 和变更文件列表。
5. 明确说明哪些旧行为保持不变，哪些新行为只在 V-3D 新模式下生效。

## 停止条件

- 需要删除或重写旧配置/历史结果才能继续；
- 必须改变旧 checkpoint schema 或旧配置默认行为；
- 发现 3D 命令会被隐式编码、进入 DADE 或历史；
- 无法在不改变网络宽度的前提下实现 command mask；
- 任何改动会污染旧实验输出或需要启动正式训练。

遇到以上情况，停止修改并报告，不要自行扩大范围。
```
