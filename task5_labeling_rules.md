# Jev 人工标注操作性定义（v1）

> 适用范围：`task5_annotations.jsonl` 中 S01–S12 的中文标注。
> **声明：本文定义是本次实验自行约定的操作口径，不是 TypeSafe/Jev 官方规定，也不来自任何论文原文。**
> 制定时间：在已获得 12 条 Jev 预测之后。制定过程只用于统一人工判断口径，未根据 Jev 的预测结果调整任何阈值；但该时间顺序本身构成潜在偏差，须在最终报告中如实披露。

## 1. 三个维度的定义

| 维度 | 只回答的问题 | 判 True 的条件 |
|---|---|---|
| relevant | 片段内容是否在谈**当前问题所问的对象** | 片段主题 = 问题所问主题；不要求片段包含答案 |
| self_contained | 片段**脱离相邻上下文能否被正确理解** | 无悬空指代（`These/This/It/Q10b`）、句子未被截断、不依赖片段外的图表编号 |
| sufficient | 片段**是否含回答所需的关键信息** | 答案所需的数值/结论/条目直接出现在片段中 |

注意：`self_contained` 与 `sufficient` 相互独立。"能读懂但答不出"记 `self_contained=True, sufficient=False`。

## 2. 三个动作的定义

| 动作 | 条件 |
|---|---|
| answer | relevant=True 且 self_contained=True 且 sufficient=True |
| expand | relevant=True 且 sufficient=False，且缺失信息位于**片段所在页及前后各 1 页**之内（连续不超过 3 页），并属于同一结构单元（图表、表格、图注或相邻段落） |
| search | relevant=False，**或** 缺失信息超出上述 3 页邻域 |
| uncertain | 两种定义同时成立、标注人无法判定；**排除出准确率统计** |

## 3. 邻域规则（本次核心约定）

- expand 的邻域 = `[page-1, page, page+1]`，即最多 **3 个连续页面**；
- 典型符合：S07（p5 表格 + p6 正文，共 2 页）；
- 典型超出：S08（p14–p18，共 5 页）→ 按规则记 `search`。

## 4. 标注状态

| review_status | 含义 | 是否计入评价 |
|---|---|---|
| pending | 未人工确认 | 否 |
| confirmed | 标注人已确认并填写 human_* | 是 |
| uncertain | 标注人明确无法判定 | 否 |

`human_* = null` 表示尚未人工确认；`suggestion_unverified` 仅为 AI 候选，任何时候都不得当作真值。
