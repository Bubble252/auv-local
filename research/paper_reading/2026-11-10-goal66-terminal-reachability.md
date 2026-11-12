# Reading note：terminal homing、可达性与捕获区域

日期：2026-11-10<br>
关联 Goal：Goal66<br>
用途：约束下一步执行器/动力学 reachability 审计

## 1. 本地证据链

- Goal59：deterministic residual 已饱和，scale/rate clip 为 0；
- Goal60/61：zero-residual 也复现 `p_nav/eta` terminal 偏差；
- Goal64：strict125 lost true 全部为 plane miss，横向 offset 中位数 `1.065 m`；
- Goal65：dwell=50 只延迟 plane miss，没有让载体进入 `1.00 m`；
- Goal66：lost corrected-command LOS cosine `0.977/0.982`，speed ratio 与 retained
  没有预期差异，`p_nav-only` 只有 `0.222/0.429`。

因此当前最缺的不是“command 方向”或“reward 权重”，而是：在最后的几何余量和
时间窗口内，载具是否物理上能完成所需的转向、俯仰和深度修正。

## 2. 外部检索

检索词：

```text
AUV terminal homing capture region docking control
```

范围：2016–2026。Semantic Scholar 连续返回 HTTP 429；arXiv 连接超时；Crossref
返回 10 条 metadata。以下只使用题名/venue 做范围判断，没有全文证据，不据此声称
任何论文的具体控制律或实验结论。

| 年份 | 题名 | venue / DOI |
|---|---|---|
| 2017 | LOON-DOCK: AUV homing and docking for high-bandwidth data transmission | OCEANS 2017, `10.1109/oceanse.2017.8084806` |
| 2023 | A Vision-Integrated Navigation Method in AUV Terminal Mobile Docking Based on Factor Graph Optimization | CACRE 2023, `10.1109/cacre58689.2023.10208719` |
| 2016 | AUV terminal docking experiments based on vision guidance | OCEANS 2016, `10.1109/oceans.2016.7760996` |
| 2023 | AUV Positioning Algorithms Used in Its Homing and Docking to Underwater Docking Station | Gyroscopy and Navigation, `10.1134/s2075108723010042` |
| 2024 | Review of AUV Underwater Recovery Docking Studies | `10.62953/ijamce.438527` |

检索结果只支持一个弱结论：AUV terminal homing/docking 通常被当作独立控制阶段，
会单独处理定位、末端接近和捕获条件；不能据此推出本项目应采用某个现成控制器。

## 3. 对 Goal67 的约束

1. 先做只读 reachability 审计，不训练、不扫 reward/gamma/seed；
2. 用 frozen checkpoint 的 trace 估计剩余修正量，并与车体速度、转向率、俯仰率、
   深度率包络比较；
3. 区分“command 不可达”和“command 可达但实际状态跟踪滞后”；
4. 若 reachability 充足，再检查 terminal target 与 capture-region 的 3D 几何；
5. 不把外部论文当作本地因果证据；若引入具体方法，必须补全文阅读 note；
6. 不打开 held-out `100–149`，不启动 SAC/REDQ/DroQ。

## 4. 检索失败记录

- Semantic Scholar：HTTP 429，重试 4 次后失败；
- arXiv：`export.arxiv.org` 连接超时；
- Crossref：成功返回 10 条 metadata；
- 未取得全文，因此本 note 不包含具体算法复现结论。
