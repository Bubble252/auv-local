# Goal164 decision：先修动作反馈参数化，再谈 anti-windup

- **decision**：`pivot`
- **reason**：速度契约本身已验证正确；真实超速只在 terminal，但控制器 actor
  feedback 把角速度误当线速度，速度环输入错误。
- **not doing**：不换 PPO/SAC/REDQ/DroQ，不放宽 2.5 m/s，不改 reward/terminal/
  success/radius，不扩 seed，不打开 held-out，不启动 gain residual 重训。
- **next**：Goal165 先修 `controller_nu` 排列和 water-relative surge 反馈；只用
  合成测试 + corrected zero replay 验证。若仍残留，再新开 anti-windup goal。

