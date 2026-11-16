# Decision：residual head 有效权限不足，下一步做最小 head-only pilot

**日期**：2026-11-14<br>
**决策**：`continue`<br>
**上游**：Goal176 Stage A complete

## 决策

- 不进入 encoder 预训练、显式纵向 skip 或 DADE 历史编码；
- 不把“冻结编码器/没冻结”作为主因；
- 先做最小 head-only / 最后一层 pilot，并把 residual 输出 scale 和
  `applied / total` Jacobian 作为验收指标。

## 理由

- 三条 COMP 臂的 `z_t` 对 `vertical_error / pitch_error / q / w` 的 held-out R²
  均在 `0.9967–0.99996`，纵向信息已经存在。
- residual 对纵向输入有方向一致的导数，但 applied rudder 只占总动作导数的
  `1e-6–2e-4`；方向对而幅度近乎无效。
- late KL 仅为 `desired_kl` 的 `0.86%–1.05%`，actor head 有更新但 logged reward
  slope 为负，说明 credit/更新也很弱。

## 不做什么

- 不打开 held-out；
- 不扩 seed 救指标；
- 不改 reward / termination / success / radius；
- 不先换 Transformer 或 encoder 结构。

## 下一步

写 head-only / last-layer pilot 的预注册，只比较冻结 encoder 训头与微调最后一层；
训练前固定 residual scale、KL 目标和评价 cell，训练后必须报告 applied/total
Jacobian 与 current10 的 route/vertical 闭环变化。
