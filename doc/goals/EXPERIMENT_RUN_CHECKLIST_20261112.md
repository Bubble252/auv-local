# 训练与评价运行检查表

**用途**：每次启动训练或正式评价前逐项勾选；这份表不能替代可执行检查器。

| 检查项 | 训练 | 正式评价 | 失败处理 |
|---|---|---|---|
| 课程学习 | `curriculum.enabled=true`，`stage_mix` 与 goal 文档一致 | 评价使用冻结 stage/level，不临时换课程 | `INVALID_NONCURRICULUM` |
| 课程晋升 | 冻结课程必须 `curriculum.promotion_enabled=false`；最终 state 的 stage/mix 与协议完全一致 | 评价不适用 | `INVALID_NONCURRICULUM` |
| 课程 provenance | 每个 run 必须有 `curriculum_episodes.jsonl`，stage 覆盖完整 | 不适用 | `INVALID_NONCURRICULUM` |
| 速度包线 | 每个 stage `speed + current + variation <= 2.5 m/s` | 检查 `ground_speed_max_mps` 与 overspeed | `INVALID_SPEED_CONTRACT` |
| 完整矩阵 | 所有预注册 arms × seeds | 所有 `arms × levels × radii + zero baselines` | `INVALID_INCOMPLETE_EVAL` |
| seed 隔离 | 只用训练 seeds | dev seeds 预注册；held-out 关闭 | `INVALID_SEED_SPLIT` |
| base 对齐 | PPO/SAC 必须共享同一 frozen BC base 与其 SHA256 | 评价读取同一 base 字段 | `INVALID_BASE_MISMATCH` |
| 启动协议检查 | `check_<goal>_protocol.py --phase train` | `check_<goal>_protocol.py --phase eval` | 不启动 |
| 进程互斥 | 目标 root 必须有 `flock`；不得有旧 trainer 引用同一 root | 同一 eval cell 不得并发写 | 先硬终止旧进程并保留现场 |
| 结束产物审计 | `audit_<goal>_artifacts.py --phase train` | `audit_<goal>_artifacts.py --phase eval` | 非零退出 |
| 主指标 | 训练日志仅作诊断 | route 3D/cross/vertical、五段、keypoints、completion/progress | 不能只看 success |

## Goal173 固定入口

```bash
cd /home/bubble/水下机器人/settled_局部规划器_反应式
bash scripts/run_goal173_pipeline.sh
.venv/bin/python scripts/analyze_goal173_rudder_only.py \
  --root artifacts/lowlevel/goal173_rudder_only_residual_20261112
```

`Goal173` 的正式评价固定为 `8 arms × 4 levels × 2 radii + 3 zero baselines × 4 levels ×
2 radii = 88 cells`。只跑 `default/strict125`、`current10` 或单个 arm 的结果只能标为
`DIAGNOSTIC_NOT_GATE`，不能进入 Gate 结论。

注意：只写 `stage_mix` 不等于冻结课程。训练 hook 默认仍可能因为 dev eval 连续通过而
自动晋升，改写运行时 stage 和 mix。凡是声明“冻结课程”的 run，必须同时写
`promotion_enabled=false`，并由协议检查器和训练后 state 审计共同验证。
