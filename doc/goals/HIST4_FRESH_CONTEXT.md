# HIST-4 Fresh-Context 演练

日期：2026-11-13<br>
范围：只使用 `INDEX.md`、`HISTORY.md`、`LINEAGE.md`、`phases/*`、`registry/goals.jsonl` 和 `research/ACTIVE.md`
回答以下问题；需要原始证据时才打开对应 `goal*.md` / result / artifact。

## 结果

| # | 问题 | 只看索引能否回答 | 答案与定位 |
|---:|---|---|---|
| 1 | 当前 active Goal 是什么？ | PASS | `META-G01`，见 `总协调/research/ACTIVE.md`。 |
| 2 | Goal173 为什么出现，它承接了哪些失败？ | PASS | P6 从 frozen-BC 瓶颈转到 ALOS/K4；Goal165 证明 gain residual 闭环 no-op，Goal171/172 转到 route-health 归因，Goal173 用 rudder-only PPO/SAC 验证 route 收益。见 [P6](phases/2026-11-11-p6.md) 与 [Goal173 result](goal173_result_20261112.md)。 |
| 3 | Goal143 的失败根因是什么？ | PASS | frozen-BC 基座瓶颈：policy 与 zero-residual 同为 `0.85`，纯 ALOS 为 `0.925`；PPO 增量没有改变行为。见 [P5](phases/2026-11-11-p5.md)。 |
| 4 | 哪些 Goal 被后续 Goal 取代？ | PARTIAL | registry 明确 `status_class=superseded` 的是 Goal1、Goal3；LINEAGE 另给出 Goal89→90、Goal92→93、Goal136→143/144、Goal143→144、Goal163→164、Goal164→165 等 `candidate` 边，不能当成已确认事实。见 [LINEAGE](LINEAGE.md)。 |
| 5 | terminal capture 属于哪个阶段？ | PASS | 从 P2 正式成为独立问题，并贯穿 P3–P6；P2 的 Goal28–50 负责几何/估计器/权限诊断。见 [P2](phases/2026-11-09-p2.md)。 |
| 6 | Goal156 的 result 和 artifact 在哪里？ | PASS | `doc/goals/goal156_result_20261112.md`；artifact root `artifacts/lowlevel/goal86_1m_standing_early_20261112`。见 [INDEX](INDEX.md) 的 Goal156 行。 |
| 7 | 哪个失败模式重复出现最多？ | PASS | “terminal capture / path-terminal conflict，以及 residual 不转化为有效物理动作”反复出现在 P2–P6；代表证据是 Goal31、Goal86、Goal137、Goal165、Goal173。见各阶段复盘。 |
| 8 | 当前仍然开放的纵向/terminal 问题是什么？ | PASS | strict125 terminal capture/success tradeoff、vertical p90 基本持平、动作 authority/saturation、训练 horizon 覆盖 terminal，以及 Goal173 的 formal 3-seed/3M 验证。见 [P6](phases/2026-11-11-p6.md)。 |
| 9 | 哪些 Goal 是 `invalid` 或 `unknown`？ | PASS | registry：invalid 6 个（Goal37/39/55/89/97/162），unknown 30 个（Goal6/7/9/10/11/12/13/15/16/17/20/21/23/24/26/27/28/29/30/31/32/33/34/35/36/59/60/61/62/63）。见 [INDEX](INDEX.md)。 |
| 10 | 下一阶段应该先读哪三份文档？ | PASS | `doc/goals/INDEX.md`、`doc/goals/HISTORY.md`、当前阶段的 `phases/<phase>.md`；需要机器查询时再读 `LINEAGE.md` 和 `registry/goals.jsonl`。 |

## 缺口

- 第 4 题只能确认 Goal1/Goal3 的 `superseded` 状态；其余谱系边仍是 `candidate`，需要后续人工确认。
- `unknown=30` 是历史文档状态提取的显式保留项，不应被静默改写成成功或失败。
- fresh-context 只能定位“哪里有问题”，不能替代原始 result/artifact 的数值核验。
