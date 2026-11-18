# 文献检索完整记录

查询：smooth reinforcement learning control；2018–2026；每源最多2条。全部六个API源已尝试，以下为完整输出，不据检索排序直接采纳论文。最初系统Python3.8不兼容，项目venv缺requests，最终使用工作区.venv完成。网页工具返回404，已改用urllib直接访问作者页、arXiv和Crossref。

四条宽检索结果仅作为候选：第一条年份缺失且未全文核验，其余并不直接回答本项目机制问题，均不用于方法有效性论断。补充从已有领域知识提出CAPS、L2C2、Residual RL并在线核验；三篇已各有reading note，没有未核验的模型记忆引用。

OpenAlex与Semantic Scholar返回429，DBLP解析失败，OpenReview依赖缺失。检索范围有限，不宣称穷尽最新文献。

## 完整API输出

[dblp] Error on query 'smooth reinforcement learning control': Expecting value: line 1 column 1 (char 0)
[open_alex] Error on query 'smooth reinforcement learning control': 429 Client Error: Too Many Requests for url: https://api.openalex.org/works?search.semantic=smooth+reinforcement+learning+control&filter=publication_year%3A2018-2026&sort=relevance_score%3Adesc&page=1&per-page=2
[openreview] Error on query 'smooth reinforcement learning control': openreview not installed. pip install openreview-py
[semantic_scholar] Error on query 'smooth reinforcement learning control': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=smooth+reinforcement+learning+control&offset=0&limit=2&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=2018-2026

per-source hits: arxiv=2, dblp=0, open_alex=0, openreview=0, semantic_scholar=0, crossref=2
unique papers: 4 (0 cross-source duplicate records merged)
============================================================
  [1] (score 4) Adaptive Policy Regularization for Smooth Control in Reinforcement Learning
      Authors: Payam Parvizi, Abhishek Naik, Colin Bellinger...
      Year: None  Citations: 0  Venue:
      Sources: crossref  doi:10.36227/techrxiv.177004949.91897305/v1
      URL: https://doi.org/10.36227/techrxiv.177004949.91897305/v1

  [2] (score 3) Smooth Path Planning of 6-DOF Robot Based on Reinforcement Learning
      Authors: Jiawei Tian, Dazi Li
      Year: 2022  Citations: 2  Venue: 2022 4th International Conference on Control and Robotics (ICCR)
      Sources: crossref  doi:10.1109/iccr55715.2022.10053875
      URL: https://doi.org/10.1109/iccr55715.2022.10053875

  [3] (score 2) The Open Ant: A Robot Platform for Reinforcement Learning Research
      Authors: Elena Sorina Lupu, Patrick Spieler, Khurram Javed...
      Year: 2026  Citations: 0  Venue: arXiv
      Sources: arxiv  arXiv:2607.18488
      URL: http://arxiv.org/abs/2607.18488v1

  [4] (score 2) Value Bonuses using Ensemble Errors for Exploration in Reinforcement Learning
      Authors: Abdul Wahab, Raksha Kumaraswamy, Martha White
      Year: 2026  Citations: 0  Venue: arXiv
      Sources: arxiv  arXiv:2602.12375
      URL: http://arxiv.org/abs/2602.12375v1

Total: 4 unique papers (ranked; surveys sunk to the bottom).
