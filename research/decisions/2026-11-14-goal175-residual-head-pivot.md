# Decision：Goal175 Gate FAIL 后转向 residual head / latent probe

**日期**：2026-11-14<br>
**决策**：`pivot`<br>
**上游**：Goal175 `GATE175_FAIL / VERTICAL_GUARD_FAIL / H175C_NOT_SUPPORTED`

## 决策

停止在 Goal175 内实现或训练 Transformer actor。主线仍是 ALOS + `z_t` +
residual control，但下一步先做 eval-only 的表征与动作敏感性审计：

1. `z_t` 线性 probe：`vertical_error / pitch_error / q / w`；
2. 对纵向/俯仰输入做小扰动，测 residual action 的 Jacobian；
3. 比较“冻结编码器只训头”和“只微调最后一层”。

## 理由

- cross-track 和整条 route 都过阈值，说明 COMP 的路线收益真实存在；
- 但 `P_RUD_075_COMP` 相对 `zero_bc50` 的 route 差只有 `0.37 mm`，
  residual 虽有 `0.0029` 的 mean absolute action，却没有形成有效闭环纠偏；
- `P_ALL_075` 的 residual 甚至让 route/cross 变差约 `3.5–4.1 cm`；
- `zero_alos_comp` vs `zero_alos_none` 已显示纵向代价主要来自 COMP 基座；
- 因此“换 actor 架构”不是当前最便宜的证伪点，先确认 latent 是否可用、
  action head 是否响应，再决定是否改预训练目标、显式纵向 skip 或 DADE。

## 不做什么

- 不扩 seed 救 Gate；
- 不改 reward / termination / success / goal radius；
- 不打开 held-out `100–149`；
- 不实现或训练 Transformer；
- 不把 A2 的 success 提升写成纵向问题已解决。

## 下一步

写 Goal176 预注册，先跑最便宜的 read-only probe/Jacobian；若 probe 好而
动作不响应，修 residual head / action interface；若 probe 差，再进入表征或
历史编码方案。
