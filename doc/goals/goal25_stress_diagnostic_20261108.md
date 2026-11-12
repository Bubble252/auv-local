# Goal25 stress 诊断

日期：2026-11-08  
状态：**not_started_by_gate**

## 结论

D3 的 nominal pre-goal 门禁失败，因此按 Goal25 Research Contract，不执行 stress
只读诊断。没有新增 stress 数据、没有训练 S10/S20、没有扫描 stress 比例。

| scenario | D3 pre-goal RMSE | limit | result |
|---|---:|---:|---|
| default | 2.849170 | 2.65 | FAIL |
| strict125 | 2.829135 | 2.70 | FAIL |

参考：

- `doc/goals/goal25_d3_repair_report_20261108.md`
- `/dev/shm/pc_v1_goal25_d2_repair_20261108/d3_gate.json`
