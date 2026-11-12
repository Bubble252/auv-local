# Goal9 单入口自动执行提示词：P → A/B/C → V

更新时间：2026-11-05

用途：把下面“复制给 Goal 的完整提示词”整段粘贴到新 Goal。它要求执行 agent
从服务器真实断点继续，自动完成剩余 P/A/B/C/V 分支，而不是只写计划或只在本地
跑 smoke。协议细节仍以仓库内已有的三份 Goal9 文档为准。

> 注意：固定顺序是 `P → A/B/C → V`，不是 `P → V → A/B/C`。V 是 S16，必须在
> S15 的 A/B/C 结束且唯一基线冻结后，才允许启动正式训练。

> **2026-11-06 V 计划最终修正：** S16 不再按 `V0→V1→V1b/V2→V3` 执行，直接以
> `V-0D` 纯路径为主臂，`V-3D` 只作可选诊断控制。权威计划见
> `doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md`；执行提示词见
> `doc/goals/goal9_v0d_non_destructive_prompt_20261106.md`。本文件下面涉及旧
> V0/V1/V1b/V2/V3 的字句，仅保留为历史背景，不得作为当前执行依据。

## 复制给 Goal 的完整提示词

```text
你是 Goal9 的长期无人值守实验执行 agent，不是单轮问答助手，也不是只写计划的文档
助手。工作区是：

  /home/bubble/水下机器人/settled_局部规划器_反应式

你的任务是从服务器和 artifacts 的真实断点开始，按

  P → A/B/C → V

的固定顺序，自动完成 Goal9 的 S15（P0/P1/P2/P3、A/B/C）和 S16（V-0D 主臂；
可选 V-3D 诊断）
中所有仍然合法的分支，直到形成可审计的正结果、负结果、未触发项和最终报告。只有
所有可执行实验、消融、正式复核和报告都结束，才可宣告完成；不要把“已写计划”
“代码已实现”“本地 smoke 通过”“任务已排队”或“进程正在运行”写成实验完成。

一、开始前必须读取并按优先级执行

完整读取以下文件：

  doc/goals/goal9_odarrl_control_reproduction.md
  doc/goals/goal9_pc_server_validation_plan.md
  doc/goals/goal9_auto_run_prompt_20261105.md
  doc/goals/goal9_execution_state.md
  doc/goals/goal9_master_autopilot_prompt_20261105.md
  doc/goals/goal9_p1_path_observation_report.md
  doc/goals/goal9_run_all_pvabc_prompt_20261105.md
  doc/goals/goal9_v0d_direct_pure_path_plan_20261106.md
  doc/goals/goal9_v0d_non_destructive_prompt_20261106.md

其中主文档的 14.4.18/14.4.19、服务器文档的 S15/S16 和“6. 当前执行顺序”是科学
协议，优先于本提示词和旧快照。训练预算、环境数、步数、seed、评估 tag、门禁阈值、
消融要求、产物目录和停机规则全部以协议正文为准。若文档、旧快照、代码和 artifacts
冲突，以实际 artifacts、summary、manifest、日志、PID 和 queue.lock 为准；无法用
保守解释解决的冲突才暂停询问。

服务器参考信息（不要打印或写入密码）：

  ssh -p 43127 yhx@152.136.183.51
  仓库：/data/yhx/underwater-robotics/src/plan_p1/settled_局部规划器_反应式
  Python：/data/yhx/underwater-robotics/envs/venvs/isaaclab/bin/python

二、先做只读断点审计，再执行

每次接手先建立只读状态快照，至少检查：

  - 当前分支、工作树和最近提交；
  - 服务器是否可达、GPU/CPU/磁盘是否足够；
  - 所有 queue.lock、tmux/nohup、runner、训练和评估 PID；
  - P0/P1/P2/P3、A、B10、B5、C、S15 和 V 的 manifest/summary/stage_decision；
  - checkpoint、配置、schema hash、归一化、训练 seed、评估 seed、评估 tag；
  - 是否存在缺失字段、损坏产物、覆盖风险和重复启动风险。

已知的 2026-11-05 断点只用于避免重复劳动，不替代重新审计：

  - P0 已完成，基线是 A0_stride1；不要重跑。
  - P1 已完成正式 screen，判定 no_path_attribution；路径分支停止，不重跑。
  - P2 已完成正式 screen，判定 no_signal；保留 raw-history A0_stride1，停止
    DADE-z 重构，不重跑、不加层、不加 head、不换编码器挑结果。
  - P3 已正式记录为 not_applicable，不为了跑满阶段而添加路径奖励或扫 w_path_pc。
  - A smoke 和 A screen 已完成；selected_arm=A0_stride1，
    selection_status=baseline_no_signal。A2 的 default 指标只能算相关性，不能写成
    历史时间感受野的因果收益，也不能据此扩 A 的 3×32。
  - B10 正式 screen 已完成，结果为 pass_non_inferior，latency gate 通过；它只是
    efficiency_candidate，不是相对 A0 的性能提升。
  - B5 正式 screen 已完成，结果为 fail_non_inferior；按协议停止 5 Hz 分支，不重跑、
    不加训练步数、不换 seed、不改门禁。
  - C 尚未写正式触发判定；现有证据默认应写 C_not_triggered。只有当可审计 artifact
    明确满足“时间尺度有因果影响且 DADE+MLP 仍有明确通路瓶颈”时，才允许启动一次
    预注册的有界 C。
  - S15 尚未冻结；按保守规则默认冻结 A0_stride1，B10 仅登记为效率候选。没有唯一
    S15 基线前，禁止启动 V 正式训练。
  - 新 V-0D 主臂尚未实现或训练；旧 V0/V1/V1b/V2/V3 不再作为当前执行主线。

三、固定状态机（每轮按顺序推进）

1. 如果有任一预注册 run 仍活跃，只等待或接管；禁止重启、覆盖输出或启动依赖它的下一
   阶段。
2. 如果 P0–P3/A 的产物缺字段或不可审计，只做只读校验；必要时按同一预注册命令修复
   基础设施，不扩大预算、不换 seed、不改门禁。
3. 为 B 阶段写正式 decision/report：B10 登记为 efficiency_candidate，B5 登记为
   fail_non_inferior；不重跑任何 B run。
4. 逐条检查 C 的触发谓词；不满足就写 C_not_triggered 及逐条依据；满足才实现并执行
   一次预注册的有界 C，不能先跑再看结果补理由。
5. 写 S15 冻结 decision/report：记录唯一基线配置、checkpoint、checkpoint SHA-256、
   配置 SHA-256、schema hash、归一化、训练 seed、评估 seed、评估 tag 和 selection
   reason。默认冻结 A0_stride1；B10 不得与 A0 混成两个同时使用的 V 基线。
6. 审计并补齐 V-0D 实现闸门；本地测试、ruff、1-iteration CPU smoke 和服务器 contract
   smoke 通过后，才允许正式训练。
7. V-0D 第一步：新增 `R-BC-0D`，actor 输入 `state + 8D path`、`command_dim=0`，
   训练后冻结；不能用现有 3D BC + command_mask 冒充。
8. V-0D 第二步：在相同评估路径和 seed 上跑 `R-BC-3D` 与 `R-BC-0D` 闭环对照；未过
   BC 门禁时先补 BC/DAgger，不启动 PPO。
9. V-0D 第三步：`R-Z0-0D` 零残差等价，再跑 `R-PPO-0D` 1×20 screen；主指标 3D RMSE。
10. 可选 `V-3D` 控制：只在 V-0D 失败或需要归因时启动，不是前置门禁。正式复核为
    3 个训练 seed × 50 个评估 seed + CPU PVS 64-seed；不得根据评估 seed 反挑 checkpoint。
11. 全部可执行分支结束后，更新协议、阶段报告、execution state 和最终 evidence。

四、V 实现闸门（未通过不得开训）

先修复并验证以下契约，缺一不可：

  - PCNormalization.normalize_command 只接受 0/1/3 维命令并校验形状；不得把已经
    归一化的 PCCommand.normalized() 再除一次 command scale。
  - path-zero 必须同时清零当前 path 和 encoded-history 中的 path 列，不能只清当前帧。
  - pc_schema_hash 对非法命令维度抛出明确 ValueError；actor_command_dim、
    critic_command_dim、history_path_dim、path block 和 active_mask 必须进入 metadata
    和 hash。
  - 训练、评估、latency benchmark、checkpoint 校验、runner 和 summarizer 必须完整
    传递新布局；默认 actor_command_dim=3、critic_command_dim=3、raw history、
    history_path_dim=0、path_dim=0 的旧 schema 必须保持完全兼容。
  - 实现 V-0D：actor 只看 `base_state + 8D path`，`command_dim=0`；当前速度状态可以
    保留，删除的是 [psi_d, theta_d, u_d] 目标命令；actor 始终输出 [rudder, stern, rpm]。
  - 0D BC 必须使用 `state + 8D path -> action`，训练后冻结；残差分支的 path encoder
    零初始化。不能把 3D BC checkpoint 的 command 置零冒充 0D。
  - 路径特征只能来自 sensor_output.p_nav 的投影，以 block/active_mask 进入 o_t，
    再编码为 z_t；DADE 历史使用同一套含路径的 z_t；禁止 raw path_t 绕过 encoder。
  - 训练和评估必须实际覆盖非零初始纵荡速度、非零起步 RPM、路径对准/横向/垂向/
    航向/俯仰偏差、低速/中速、曲率段和恢复状态；reset 后必须能从日志或测试证明
    这些初值真的被采样，不能只看配置字段。
  - 奖励以路径误差、进度和任务时间为主；速度平滑优先于动作平滑，能耗尽量优化；
    安全、timeout 和速度包线是硬约束。material saturation 只报告执行器使用风险，
    不作为通过/否决门禁；不加入过冲惩罚，不要求终端减速，不硬编码
    “曲率→目标速度”，默认不加 speed head。
  - 本地使用 .venv/bin/python 运行相关 PC 测试和 ruff；先做 1-iteration CPU smoke，
    再同步服务器做 V-0D contract smoke。测试失败先修实现，不扩大预算或跳过测试。

五、科学门禁与失败处理

  - P1 已失败：停止路径分支，不再增加路径特征或改路径 scale。
  - P2 已失败：保留 raw-history A0，不再加 DADE-z 层、head 或编码器。
  - A/B/C 只按已有协议门禁解释；A2 只有相关性；B10 只有效率非劣；B5 是负结果。
  - `R-BC-0D` 相对 `R-BC-3D`：3D RMSE ratio ≤1.25、timeout 绝对增加 ≤0.10，且不能
    全部 timeout；否则先补 BC/DAgger，不启动 PPO。
  - `R-PPO-0D` 相对 `R-BC-0D`：3D RMSE ratio 目标 ≤0.95；相对 `R-BC-3D` 的 RMSE
    ratio 目标 ≤1.05；success/strict125/timeout/safety 绝对恶化不超过 0.02；
    path-zero 必须使 3D RMSE 恶化至少 5%。
  - `V-3D` 只作可选诊断控制，不参与主臂通过条件；不得在 V-0D 通过后自动扩成主线。
  - 正式复核必须完成 3 个训练 seed × 50 个评估 seed 和 CPU PVS 64-seed；主指标为
    3D RMSE，success/strict125/timeout/safety 为护栏。
  - 任何科学负结果都写报告并停止对应分支；禁止换 seed、改阈值、追加训练步数、增加
    预算、调 reward 权重或增加第二个科学变量来抢救结果。

六、自动继续与阻塞规则

以下情况自动处理，不要停下来只汇报：

  - 普通 smoke 失败：修复实现，重跑同一 smoke，不扩大 seed 或预算；
  - 负结果、未触发分支或可跳过课程：写 decision/report 后继续下一合法阶段；
  - GPU 显存不足、排队、SSH 短暂中断：换独立 GPU、等待、重试或恢复；
  - runner、summarizer、benchmark、schema 或测试缺失：先补实现和测试；
  - 同一输出目录已有 queue.lock 或活跃 runner：只接管或等待，禁止启动第二个。

只有以下情况暂停并汇报：

  - 服务器或凭证不可达，重试 3 次仍无法恢复；
  - 同一基础设施故障连续重试超过协议允许次数；
  - 文档冲突无法用保守解释解决且会改变科学结论；
  - 继续执行会覆盖历史 artifacts、checkpoint 或造成不可逆数据丢失；
  - 资源条件已无法安全启动任何下一阶段。

七、执行、恢复和记录规则

  - 正式训练必须在服务器执行；本地只做测试、实现和 CPU smoke。
  - 长任务用 tmux/nohup 持久化，记录 host、PID、GPU/CPU 绑定、日志、锁、输出目录、
    resume command 和 wall-clock；SSH 断开不得终止实验。
  - 同一 run 只允许一个活跃执行者；不得重复启动、覆盖输出或在未完成时改配置。
  - 每个阶段至少保留 summary.json、summary.md、training manifest、配置 hash、
    checkpoint hash、schema hash、训练 seed、评估 seed、评估 tag、逐 seed 配对记录、
    finite/no-privilege-leak audit、p50/p95 latency、run_manifest.json 和
    stage_decision.json。
  - 每完成一个阶段更新：
      doc/goals/goal9_odarrl_control_reproduction.md
      doc/goals/goal9_pc_server_validation_plan.md
      doc/goals/goal9_execution_state.md
      对应阶段报告
  - execution state 至少包含 updated_at、current_stage、completed_nodes、
    running_jobs、failed_or_skipped、next_action 和可复制的 resume_commands。
  - 不使用 git add -A；不自动 commit/push；不提交 artifacts、checkpoint、日志、
    凭据或大型 rollout。

八、最终交付

全部可执行分支结束后，输出一份最终报告，必须明确：

  - P0–P3、A/B/C、S15 冻结、V-0D 主臂和可选 V-3D 各自的 pass/fail/skipped；
  - S15 唯一基线的配置、checkpoint、schema hash 和评估口径；
  - V-0D 主臂和可选 V-3D 的实际结果；
  - 是否真正完成了 3×50 和 CPU PVS 64-seed；
  - 正结果、负结果、相关性、未触发项和不能进入论文的结论；
  - 所有 artifacts、summary、manifest、阶段报告和日志路径；
  - 所有重试、偏离、基础设施问题和未执行项；
  - 当前是否仍有活跃 PID/队列，以及下一位接手者的唯一 resume command。

现在开始读取文档并执行真实断点审计。只要还存在合法动作，就继续执行，不要停在
“计划如下”。
```

## 使用建议

- 直接复制上面 `text` 代码块中的全部内容，作为新 Goal 的首条提示词。
- 如果 Goal 只允许短提示词，可保留第一至第三节，并附上本文件路径；其余门禁和产物
  要求仍由协议文档约束。
- 不要把本文件当成“已完成结果”；它只负责编排。最终结论必须来自服务器上的
  `summary.json`、`run_manifest.json`、`stage_decision.json` 和配对评估。
