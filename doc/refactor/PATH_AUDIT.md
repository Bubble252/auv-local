# G04-PATH Resolver Audit

**状态**：`READ_ONLY_COMPLETE`<br>
**日期**：2026-11-13<br>
**范围**：`scripts/` 中名为 `_resolve` 的 68 个路径 helper。

## 1. 分类结果

| 组 | 语义 | 数量 | 处置 |
|---|---|---:|---|
| P1a | 绝对路径原样，否则 `REPO_ROOT / path`，且候选文件无存量 lint 阻塞 | 20 | 迁移到 `resolve_from_root(REPO_ROOT, path)` |
| P1b | 同一语义，但文件有存量 lint 阻塞 | 2 | 保留原实现，避免混入无关 lint 修复 |
| P2 | 绝对路径原样，否则 `Path.cwd() / path` | 9 | 保留；CWD 是调用契约的一部分 |
| P3 | `Path | str` 输入，否则 `REPO_ROOT` | 5 | 后续独立迁移 |
| P4 | `Path | str` 输入，使用局部 `value`，否则 `REPO_ROOT` | 2 | 后续独立迁移 |
| P5 | `Path | str | None`，None 原样返回 | 1 | 后续独立迁移 |
| P6 | 使用脚本自己的 `ROOT` | 1 | 保留 |
| P7 | 已迁移的 standing-eval wrapper | 1 | 已完成 |
| P8 | 多根搜索并检查文件存在性 | 3 | 保留；搜索顺序和异常是语义 |
| P9 | 格式化模板参数 | 1 | 保留 |

## 2. 本轮迁移组 P1a

以下脚本全部是同一实现：

```python
def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path
```

- `scripts/audit_goal86_rl_backends.py`
- `scripts/audit_pc_goal45_fixed_calibration_residual.py`
- `scripts/audit_pc_goal46_dynamic_residual_association.py`
- `scripts/audit_pc_goal47_estimator_state.py`
- `scripts/audit_pc_goal48_strict125_sigma_residual.py`
- `scripts/audit_pc_goal49_controller_feedback.py`
- `scripts/audit_pc_goal50_guidance_timing.py`
- `scripts/audit_pc_goal51_unrecorded_state.py`
- `scripts/audit_pc_goal52_sigma_outcome_relevance.py`
- `scripts/audit_pc_goal53_recorded_feature_outcome.py`
- `scripts/audit_pc_goal57_heading_error_replication.py`
- `scripts/audit_pc_goal86_pa_residual_labels.py`
- `scripts/audit_pc_terminal_contract.py`
- `scripts/build_pc_goal95_residual_supervision.py`
- `scripts/build_pc_standing_eval_configs.py`
- `scripts/diagnose_pc_dade_history.py`
- `scripts/evaluate_pc_ppo.py`
- `scripts/evaluate_pc_sac.py`
- `scripts/pretrain_odarrl_control_residual.py`
- `scripts/pretrain_pc_goal86_pa_residual.py`

## 3. 保留组 P1b

以下两个脚本的 helper 与 P1a 相同，但本轮保留原实现：

- `scripts/audit_goal86_bc_interfaces.py`：存量 F401/I001，避免把 import 清理混入路径迁移；
- `scripts/train_pc_command_residual_sac.py`：存量 lint 欠账，保持候选文件 lint 为 0 新增问题。

`evaluate_pc_bc_residual.py` 的同构 helper 也暂不迁移：该文件有存量 F841。Goal54、
Goal55 和 Goal56 不迁移：三者形成源码 SHA256 冻结链，任意改动都会让下游 analyzer
在 import 时拒绝运行。

迁移只改变 helper 的实现，不改变脚本内部 `_resolve` 名称、参数、调用点或 `REPO_ROOT`。
helper 不检查存在性、不创建父目录、不调用 `resolve()`，因此 symlink 和缺失路径行为
保持原样。

## 4. 明确不迁移

- `analyze_pc_goal128_credit.py` 至 `analyze_pc_goal134_stress.py` 以及
  `diagnose_pc_path_branch.py` 使用 `Path.cwd()`，不能悄悄改成 `REPO_ROOT`；
- `analyze_pc_goal61_terminal_alignment.py`、`analyze_pc_goal62_terminal_radius_counterfactual.py`
  和 `analyze_pc_goal64_terminal_window_geometry.py` 使用多候选根、存在性检查和自定义异常；
- `train_odarrl_control_residual.py` 对 `None` 有特殊返回契约；
- `analyze_pc_goal59_paired.py` 先格式化模板再拼 root；
- `generate_goal23_p5_configs.py` 使用自己的 `ROOT`；
- `run_pc_standing_eval_suite.py` 已在 G03 完成。

## 5. 测试策略

- 保留并扩展 `resolve_from_root` 的绝对路径、相对路径、缺失路径和 symlink fixture；
- 运行时检查 20 个 P1a wrapper 和 2 个 P1b 保留 wrapper 都保持 `REPO_ROOT` 语义；
- 对评价、训练、审计各选一个入口做运行时 wrapper smoke；
- 全量 pytest 必须无新增失败。
