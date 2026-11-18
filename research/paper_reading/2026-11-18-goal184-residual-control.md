# Goal184 reading note：传统反馈与学习分工

日期：2026-11-18。Paper：Tobias Johannink等，Residual Reinforcement Learning for Robot Control，arXiv:1812.03201，2018预印本。

来源：https://arxiv.org/abs/1812.03201 。本轮核验题名、作者、摘要。论文把传统反馈与RL残差信号叠加，并在真实装配任务验证。

结论/对应：传统控制处理可建模结构，学习补偿复杂部分，支持按能力分工的研究方向。它不是对AUV速度PI加学习舵的直接实验证据；当前BCRES基座是冻结BC，也不等同于传统PD/PI基座。

最小实验候选：冻结既定BCRES，只替换RPM为使用可部署速度反馈的PI，保持路径、流场、初态和评价口径；同时看速度与路线。舵效随流速改变，不能假定解耦。oracle真值海流不得暗中进入部署基线。

补充检索：仓库2026-11-12-goal161-controller-parameter-residual.md曾引用Ma等Neural Network Model-Based Reinforcement Learning Control for AUV 3-D Path Following（DOI 10.1109/TIV.2023.3282681）的PI/舵RL分工。本轮Crossref确认作者Dongfang Ma、Xi Chen、Weihao Ma、Huarong Zheng、Fengzhong Qu及2024年1月期刊日期，但IEEE页面未返回可读方法正文，所以具体PI分工仅作为历史待核线索，不进入本轮论文证据。
