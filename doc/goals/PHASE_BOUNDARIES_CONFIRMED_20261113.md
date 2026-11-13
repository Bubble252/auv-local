# Confirmed Goal Phases

> **状态：CONFIRMED（2026-11-13，用户确认六阶段边界）。** 本文件是 HIST-2 的正式
> 边界记录，供 registry 与 HIST-3 生成器消费。原始候选文件保留为 `PROPOSED.md`，
> 本文件不再自动重新生成；若要改动边界，必须新开治理决策并重新确认。

## Summary

| Phase | Goals | Count | Dates | Decisions | Themes | Boundary |
|---|---|---:|---|---|---|---|
| P1 | Goal1–Goal26 | 26 | 2026-10-28..2026-11-04 | null=25, pivot=1 | foundation, generalization | start |
| P2 | Goal27–Goal50 | 24 | 2026-11-09 | continue=11, pivot=8, stop=5 | terminal, residual | 从基础训练转入终端捕获与纵向 miss 诊断 |
| P3 | Goal51–Goal89 | 39 | 2026-11-09..2026-11-11 | continue=12, null=1, pivot=22, stop=4 | terminal, freeze | 从现象诊断转入 trace/schema、归因和动作参数化 |
| P4 | Goal90–Goal106 | 17 | 2026-11-11 | continue=5, pivot=12 | residual, foundation | 冻结 gate scope，转入 corrected residual 与 PPO 信用 |
| P5 | Goal107–Goal143 | 37 | 2026-11-11 | continue=16, null=1, pivot=19, stop=1 | residual, freeze | 转入 SAC/PPO 残差、pitch 权限、奖励与课程 |
| P6 | Goal144–Goal174 | 31 | 2026-11-11..2026-11-12 | continue=14, null=4, pivot=12, stop=1 | residual, terminal | 转入 ALOS/K4、terminal homing、route health |

## Confirmed phases

### P1: Goal1–Goal26 — 基础训练与 MVP / 多环境与 DADE

- Goals: 26（Goal1..Goal26）
- Dates: 2026-10-28..2026-11-04
- Decisions: null=25, pivot=1
- Top themes: foundation, generalization
- Boundary: 首个 Goal；建立最小闭环、多环境、BC/DADE 和早期 PPO 路线
- Representative goals: Goal1, Goal2, Goal3

### P2: Goal27–Goal50 — 终端捕获与纵向 miss / 估计器诊断

- Goals: 24（Goal27..Goal50）
- Dates: 2026-11-09
- Decisions: continue=11, pivot=8, stop=5
- Top themes: terminal, residual
- Boundary: 终端捕获成为独立问题；从动作权限转向纵向 miss、估计器和时序归因
- Representative goals: Goal28, Goal31, Goal40

### P3: Goal51–Goal89 — trace 契约与终端归因 / 命令残差

- Goals: 39（Goal51..Goal89）
- Dates: 2026-11-09..2026-11-11
- Decisions: continue=12, null=1, pivot=22, stop=4
- Top themes: terminal, freeze
- Boundary: 先冻结 trace/schema，再检验 terminal outcome、命令残差和成功契约
- Representative goals: Goal51, Goal65, Goal84

### P4: Goal90–Goal106 — corrected residual gate / PPO 信用与 B20

- Goals: 17（Goal90..Goal106）
- Dates: 2026-11-11
- Decisions: continue=5, pivot=12
- Top themes: residual, foundation
- Boundary: Goal89 暴露 gate scope 错误后，只 gate PPO 增量并重做动作参数化
- Representative goals: Goal90, Goal95, Goal103

### P5: Goal107–Goal143 — SAC/PPO 残差 / pitch 权限与奖励课程

- Goals: 37（Goal107..Goal143）
- Dates: 2026-11-11
- Decisions: continue=16, null=1, pivot=19, stop=1
- Top themes: residual, freeze
- Boundary: 从单一残差增量转入 SAC/PPO 对照、pitch/纵向权限、bias、课程和奖励
- Representative goals: Goal113, Goal128, Goal137

### P6: Goal144–Goal174 — ALOS/K4 / terminal homing 与 route health

- Goals: 31（Goal144..Goal174）
- Dates: 2026-11-11..2026-11-12
- Decisions: continue=14, null=4, pivot=12, stop=1
- Top themes: residual, terminal
- Boundary: frozen-BC 基座被证伪后，改在 ALOS/K4 上验证 terminal homing、残差权限和 route health
- Representative goals: Goal144, Goal165, Goal173

## Confirmation checklist

- [x] 阶段数量在 5–8 之间；
- [x] 每个 Goal 只属于一个阶段；
- [x] 每个阶段至少包含 5 个 Goal；
- [x] 边界对应真实研究转折，而不是只按编号切分；
- [x] 用户已确认六阶段边界。
