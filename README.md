# JevSearch

> 面向长视觉文档问答的渐进式证据检索与轻量决策路由

长视觉文档通常包含数十至上百页，而回答一个问题往往只需要其中一两页的图表、表格或版式区域。JevSearch 将长文档问答拆解为「结构路由 → 轻量证据判断 → 按需多模态作答」的渐进式流程，目标是在不把整本 PDF 送入大模型的前提下，定位并验证足以回答问题的证据。

本仓库包含完整实验代码、500 题逐题指标、离线验收统计与实验报告。**仓库内所有指标均由本地原始 JSONL / CSV 重新计算得到，未使用模拟值或手工改写结果。**

> **公开版边界**：MMLongBench-Doc 数据卡未提供明确的数据再分发许可，因此本仓库不包含题目原文、文档目录（TOC）、证据片段、PDF 与逐题预测文本。500 题的逐题得分与执行状态见 [`outputs/final500/audit/final500_status_500.csv`](outputs/final500/audit/final500_status_500.csv)，该表不含任何题目或文档文本。完整原始预测保存在私有研究仓库，如需学术核对可通过 Issue 联系作者。

## 目录

- [1. 研究背景与问题](#1-研究背景与问题)
- [2. 方法设计](#2-方法设计)
- [3. 数据集与实验设置](#3-数据集与实验设置)
- [4. 正式实验结果（500 题）](#4-正式实验结果500-题)
- [5. 指标解释与结果分析](#5-指标解释与结果分析)
- [6. 早期小样本实验](#6-早期小样本实验)
- [7. 局限性与讨论](#7-局限性与讨论)
- [8. 代码结构、依赖与复现](#8-代码结构依赖与复现)
- [9. 数据许可与仓库边界](#9-数据许可与仓库边界)

### 主要结果速览

| 模块 | 执行 | 可评测 | 核心指标 |
|---|---:|---:|---|
| DeepSeek 多模态问答（oracle 证据页） | 500/500 | 495 | 严格 62.42%（309/495）、宽松 74.14%（367/495） |
| 文档目录路由（Question + TOC，可拒答） | 500/500 | 376 | Section Hit@3 58.78%、Page Recall@3 0.5350、FullEvidence@3 48.40% |
| Jev 证据判断（8 类片段来源） | 500/500 | 500 | `relevant=True` 193（38.6%）、`sufficient=True` 37（7.4%） |

> 三个模块的输入条件、问题性质与分母均不同，指标只能并排描述，不能合成为单一总分。Jev 目前缺少人工 `confirmed` 标签，因此只报告执行覆盖率、判断分布与拒答表现，**不报告准确率**。

---

## 1. 研究背景与问题

长文档视觉问答（Long Visual Document QA）的困难来自证据稀疏：文档页数多、版式复杂，问题相关的信息集中在少量页面甚至单个图表中。直接将整本文档渲染后送入多模态大模型会带来两个问题：

1. **成本随页数线性增长**，而绝大多数页面与问题无关；
2. **无关页面稀释上下文**，使模型更容易忽略关键表格或图注。

已有长文档 RAG 方法多依赖文本切块与向量检索，但视觉文档中的证据常以图表、扫描页、跨页表格形式存在，纯文本检索难以覆盖。JevSearch 关注的问题是：能否利用文档的低成本结构信息先缩小候选页范围，再用轻量决策模型判断证据是否足以回答问题，仅在必要时调用昂贵的多模态模型。

围绕这一问题，本工作设置三个可独立评估的研究点：

- **结构路由**：仅凭问题与文档目录，能否把候选页从数十页压缩到数页，并保持对 gold 证据页的召回；
- **证据判断**：轻量决策模型能否在真实检索片段上稳定输出相关性、自包含性、充分性与后续动作（`next_action`）；
- **oracle 问答上界**：在已知 gold 证据页的条件下，多模态模型的回答准确率能够达到什么水平。

---

## 2. 方法设计

### 2.1 系统总览

系统由五个技术模块构成，数据流如下：

```
PDF
 ├─ 文字层 / 版面解析 ──► 文档目录（TOC）
 │                         │
 │                         ▼
 └─ 缺文本页 OCR ──► 证据片段 ──► 目录路由 ──► 候选页 / 候选章节
                                   │
                                   ▼
                             Jev 证据判断
                       relevant / self_contained / sufficient
                                   │
                    answer ────────┼──────── expand / search
                       │
                       ▼
                多模态问答（gold 证据页或候选页）
                       │
                       ▼
                  统一实验评估
```

各模块关系为：文档解析为路由提供目录、为 Jev 提供证据片段；目录路由缩小候选页范围；Jev 在片段层面判断证据是否充分并决定后续动作；多模态问答在证据页上生成答案；评估模块对三者分别按各自口径计算指标，不进行跨模块合成。

### 2.2 文档结构解析与 OCR

- **文字层解析**：使用 PyMuPDF 读取 PDF 文本与页码信息，作为证据片段的首选来源。
- **目录构建**：优先使用 PDF 书签；无书签时依次尝试印刷目录页、正文字号启发式与 pseudo-TOC，生成包含 `section_id`、标题、层级与页码范围的结构化目录。
- **OCR 补充**：对文字层为空或过少的页面，使用 PaddleOCR 3.x 在本地 CPU 推理（`paddleocr==3.7.0`、`paddlepaddle==3.3.1`），并设置 `enable_mkldnn=False` 以规避 oneDNN 与 PP-OCRv6 的兼容问题，`PADDLE_PDX_CACHE_HOME` 指向可写缓存目录。
- **用途**：OCR 文本不直接作为答案，而是用于构造 Jev 的证据片段，使缺文本页也能获得真实模型判断。

### 2.3 文档目录路由

路由模块的输入仅为 **Question + TOC**，其中 TOC 每条包含 `section_id`、标题、层级与页码范围；不发送 gold 答案、gold 证据页或正文。

- 模型输出若干候选章节及置信度；
- 章节页码范围展开为候选页集合；
- 采用统一的可拒答协议：当目录信息不足以定位时，模型可输出拒答；
- 评分在预测完成后读取 gold `evidence_pages`，用于计算 Section Hit@K、Page Recall@K 与 FullEvidence@K。

该模块同时提供两类对照：章节标题词匹配 baseline，以及在相近候选页预算下的检索 baseline。

### 2.4 Jev 证据判断

Jev 是独立的轻量决策模型（`jev-1.14.0`，经 TypeSafe SDK 调用），用于在证据片段层面判断当前上下文是否足以回答问题。

输入 `state_sent` 仅包含四个字段：

```
doc_type: <文档类型>
page: <页码>
Question: <问题>
Evidence snippet:
<证据文本>
```

输出包括：

- `relevant`：片段是否与问题相关；
- `self_contained`：片段是否自包含、不依赖外部上下文；
- `sufficient`：片段是否足以回答问题；
- `next_action`：`answer` / `expand` / `search`，以及各字段置信度。

片段来源包括文档标题、路由候选页、缺口补全文本、OCR 文本与恢复批次，共 8 类。片段选择过程不读取 gold 答案、gold 证据页或 `evidence_sources`；离线复核中 `selection_uses_gold=True` 的计数为 0。

### 2.5 多模态问答

问答模块使用 `deepseek-v4.1-flash-1017`，输入为问题与 gold `evidence_pages` 渲染的页面图片（`input_mode=gold_evidence_pages`）。

- 该设置用于估计「已知证据页」条件下的问答上界，而非端到端检索问答；
- gold answer 不发送给模型，`offline_only` 字段仅用于离线评分；
- 空证据页按既有协议回退至第 1 页；
- 评分采用严格匹配与宽松匹配两种口径，并单独统计不可答题的拒答正确率。

### 2.6 实验评估

评估模块统一从原始 JSONL 重算指标，不依赖既有报告中的数字：

- **分层口径**：执行覆盖率、API 成功率、有效评测覆盖率、准确率、召回率、拒答表现分别统计；
- **离线审计**：`audit_final500.py` 生成 500 行状态表与模块矩阵；
- **独立验收**：`verify_final_acceptance_500.py` 重新统计唯一 qid、调用记录、字段完整性与公开副本的文件哈希；
- **防覆盖校验**：完整研究仓库以 `public_files_sha256.json` 为基线比对发布文件，确认与实验记录逐字节一致（被移除的 gold 字段除外）；该哈希清单与原始预测同属未公开文件，不随本公开版发布。

---

## 3. 数据集与实验设置

| 项目 | 设置 |
|---|---|
| 基准 | MMLongBench-Doc |
| 样本 | 固定 500 题清单，qid `Q0001`–`Q0500` |
| 可答 / 不可答 | 381 / 119 |
| 证据标注 | 每题提供 gold `evidence_pages` 与 `evidence_sources`，仅用于离线评分 |
| 问答模型 | `deepseek-v4.1-flash-1017`（oracle 证据页） |
| 决策模型 | `jev-1.14.0`（TypeSafe SDK） |
| OCR | PaddleOCR 3.7.0 / PaddlePaddle 3.3.1，本地 CPU |
| PDF 解析 | PyMuPDF 2.1 |
| Python | 3.12 |
| 评分 | 严格匹配、宽松匹配；路由按 Section Hit@K / Page Recall@K / FullEvidence@K |

实验使用同一份固定 qid 清单贯穿三个模块，500 题全部遍历，无缺失、无重复。数据集 PDF 与 500 题清单不进入仓库，需按 MMLongBench-Doc 的发布渠道获取后由脚本重新生成（见第 8 节）。

---

## 4. 正式实验结果（500 题）

### 4.1 模块执行与可评测范围

| 模块 | 已尝试 | API 成功 | 有效输入 | 可评测 | 失败 |
|---|---:|---:|---:|---:|---:|
| DeepSeek 问答 | 500 | 495 | 495 | 495 | 5 |
| 目录路由 | 500 | 500 | 500 | 376（Page Recall）/ 500（含拒答判定） | 0 |
| Jev 判断 | 500 | 500 | 500 | 500 | 0 |

> **已尝试**指实际送入模块的题数；**可评测**指能够计算该模块指标的样本数；**API 成功率**只描述接口稳定性，不等于模型判断准确率。

### 4.2 DeepSeek 多模态问答

| 指标 | 结果 |
|---|---|
| 执行 / API 成功 | 500 / 495（99.0%） |
| 严格准确率（495） | 309 / 495 = **62.42%** |
| 宽松准确率（495） | 367 / 495 = **74.14%** |
| 严格 / 宽松（500 分母，失败记错） | 61.80% / 73.40% |
| 可答题子集（与路由同口径的 376 题）严格 / 宽松 | 52.13% / 67.55% |
| 不可答题拒答正确率 | 113 / 119 = **94.96%** |
| tokens（入 / 出） | 863,966 / 27,647 |

5 道失败样本均保留在原始结果中并按错误计入：Q0134、Q0290、Q0373、Q0383 为 `no_images`（页码或 PDF 渲染异常），Q0272 为 `api_error`。

### 4.3 文档目录路由

| 指标 | 结果 |
|---|---|
| 执行 / API 成功 | 500 / 500 |
| 拒答 | 70 题：不可答 66 / 119（55.46%）；可答过度拒答 4 题（Q0060、Q0134、Q0324、Q0372） |
| 主口径可评测 | 376 题（可答、gold 页可用且未拒答） |
| Section Hit@1 | 186 / 376 = **49.47%** |
| Section Hit@3 | 221 / 376 = **58.78%** |
| Page Recall@1（宏平均） | **0.4207** |
| Page Recall@3（宏平均） | **0.5350** |
| FullEvidence@3 | **48.40%（182/376）** |
| 严格口径（379，含 3 道可答拒答记 0） | Hit@3 58.31%、PR@3 0.5308、Full@3 48.02% |
| tokens / 平均延迟 | 617,856 / 0.903 s |

另有 2 道可答题（Q0290、Q0372）没有可用 gold 证据页，按协议不进入 Page Recall 分母。

**FullEvidence@3 的实现口径。** Page Recall 与 FullEvidence 均以 gold `evidence_pages` 列表长度为分母。Q0305 的 gold 标注为 `[6,6,6,6,6,6]`（同一页重复 6 次），按列表口径该题 PR@3 = 1/6、Full@3 记为未覆盖；若按页码去重集合计算，则该题 PR@3 = 1.0、Full@3 记为覆盖。本报告统一采用原始实现口径（列表口径）：376 主口径 Full@3 = 182/376 = 48.40%，379 严格口径 = 182/379 = 48.02%。去重口径下对应为 48.67% / 48.28%（PR@3 为 0.5373 / 0.5330），该差异只来自 Q0305 一题，已在结果文件中逐题可查，两种口径不混用。


与 baseline 的对照（baseline 为 374 题历史口径，路由为 376 题主口径；分母不同，仅作参考）：

| 方法 | Section Hit | Page Recall | FullEvidence | 平均候选页数 |
|---|---:|---:|---:|---:|
| DeepSeek 目录路由（376 主口径） | Hit@3 58.78% | PR@3 0.5350 | 48.40% | — |
| 章节标题词匹配（374） | Hit@3 28.61% | PR@3 0.2329 | — | 3.21（@3） |
| 同候选页预算检索（374） | 42.78%（单预算，非 @k） | 0.3637 | 30.48% | 8.23 |

各方法的候选页预算并不严格相等，因此该表不构成完全公平的等预算对比，只用于说明路由相对于标题词匹配的方向性增益。

### 4.4 Jev 证据判断

| 指标 | 结果 |
|---|---|
| 执行覆盖率 | **500 / 500 = 100%** |
| 有效判断（四字段完整） | **500** |
| `relevant=True` | 193（38.6%） |
| `self_contained=True` | 283（56.6%） |
| `sufficient=True` | 37（7.4%） |
| 动作分布 | `answer` 35、`expand` 169、`search` 296 |
| 可答题（381） | `answer` 31、`expand` 135、`search` 215；`sufficient=True` 34 |
| 不可答题（119） | `search` 81、`expand` 34、`answer` 4；`sufficient=True` 3 |
| 合理拒答（不可答判证据不足） | **116 / 119 = 97.48%** |
| 判断准确率 | **不可计算**（无人工 `confirmed` 标签） |
| tokens（入 / 出） | 660,142 / 47,000 |
| 平均单次延迟 | 0.422 s |

片段来源分布：

| 来源 | title | route | round3 | gapfill | ocr | ocr_pilot | recover | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 条数 | 263 | 88 | 65 | 61 | 11 | 10 | 2 | **500** |

来源之间无重复 qid；每条记录均含 `model=jev-1.14.0`、`run_at` 与输入/输出 token。逐条 `state_sent` 与证据文本保存在完整研究仓库的片段文件中，不随本公开版发布。

### 4.5 模块交集与统计口径

| 交集 | 题数 |
|---|---:|
| Jev 500 ∩ QA 成功 495 | 495 |
| Jev 500 ∩ 路由可评 379 | 379 |
| Jev ∩ QA ∩ 路由可评 | **375** |

不同分母对应不同的排除规则，且均来自同一份 500 题清单与同一批预测文件：

- **500**：固定清单全部题目；
- **495**：QA 接口成功返回的题数（5 题失败按错误计入 500 分母）；
- **381 / 119**：数据集标注的可答 / 不可答；
- **374**：路由原生成功且可评（无拒答、gold 页在 PDF 范围内）；
- **376**：在 374 基础上加入 2 道补跑成功的可答题（Q0373、Q0383）；等价于 381 − 2 道无 gold 页（Q0290、Q0372）− 3 道可答拒答（Q0060、Q0134、Q0324）；
- **379**：在 376 基础上，把 3 道有 gold 页但被拒答的可答题按 0 分计入（374 + 5 道修正 gold 题）；
- **375**：三模块同时可评测的交集，受 QA 的 5 道失败影响。

所有口径均在结果文件中逐题可查，不存在静默剔除。

### 4.6 费用记录

| 模块 | 次数 | tokens（入 / 出） | 费用 |
|---|---:|---:|---:|
| Jev 全量 | 500 | 660,142 / 47,000 | 约 **1.50 元**（0.003 元/次历史实测均值，非官方报价） |
| 其中补跑批次 | 65 | 81,528 / 6,110 | 约 **0.195 元** |
| 目录路由 | 500 | 617,856（合计） | 接口未返回单价 |
| DeepSeek 问答 | 495 成功 | 863,966 / 27,647 | 接口未返回单价 |

三个模块的 token 按各自原始记录分别统计（见上表）。不同模块的 token 与计价方式不同，本仓库不将其折算为虚构的统一单价，也不发布逐条账单。

### 4.7 Baseline 与模型对比

**（1）多模态问答模型。** 三者均在同一 500 题清单上评分，严格/宽松匹配规则一致；问答输入均为 gold `evidence_pages`。

| 方法 | n | 严格 | 宽松 | 说明 |
|---|---:|---:|---:|---|
| 纯文本 baseline | 500 | 5.20%（26） | 5.80%（29） | 仅输入问题，无文档内容 |
| Qwen3-VL-Plus | 500 | 63.40%（317） | 75.40%（377） | 300 DPI、含不可答约束 |
| DeepSeek（失败记错） | 500 | 61.80%（309） | 73.40%（367） | 主口径 |
| DeepSeek（仅成功） | 495 | 62.42%（309） | 74.14%（367） | 参考口径 |

同一 500 题、同一判分规则下，DeepSeek 较 Qwen3-VL-Plus 低约 1.6–2.0 个百分点（严格/宽松），在可答子集（n=381，失败记错）上为 51.44%/66.67%，亦低于 Qwen 的 52.76%/68.50%（成功子集 n=376 下 DeepSeek 为 52.13%/67.55%，Qwen 无失败不受影响；见 `final500_stats.json`）。模型替换**未带来性能提升**。两组结果的实现脚本、提示词细节与模型版本并不完全相同，因此这是「同题同评分」的对比，不是严格受控实验。

**（2）目录路由。** 对照见 4.3 节：DeepSeek 路由的 PR@3 为 0.5350，显著高于标题词匹配的 0.2329，也高于同候选页预算 baseline 的 0.3637；但三者的候选页预算并不严格相等，不能等价为等预算结论。

**（3）Jev 证据判断。** 目前只能比较判断分布（4.4 节），不能与任何 baseline 比较"准确率"，因为不存在人工 `confirmed` 标签。

**（4）完整系统。** 仓库现有结果是三个模块的独立评测，**没有**把「目录路由 → 证据检索 → Jev 判断 → 多模态问答 → 最终评分」串联为一次端到端实验的结果文件。因此不能声称完整 JevSearch 系统优于 baseline，也不报告端到端准确率。

---

## 5. 指标解释与结果分析

### 5.1 指标定义

| 指标 | 定义 |
|---|---|
| Section Hit@K | 模型前 K 个候选章节中是否包含 gold 证据页所在章节 |
| Page Recall@K | 前 K 个候选章节展开后的页面集合覆盖 gold 证据页的比例（宏平均） |
| FullEvidence@K | 前 K 个候选页是否完整覆盖全部 gold 证据页 |
| 严格匹配 | 归一化后答案与 gold 完全一致 |
| 宽松匹配 | 允许包含关系、数字/单位等价等宽松判定 |
| 拒答正确率 | 不可答题中模型明确拒答的比例 |
| 过度拒答 | 可答题中模型错误拒答的比例 |

### 5.2 主要观察

1. **问答达到 oracle 上界但未超过第一轮多模态结果。** 在 gold 证据页条件下，DeepSeek 严格/宽松为 62.42%/74.14%（495 分母），以 500 为分母时为 61.80%/73.40%；Qwen3-VL-Plus 在同一 500 题上为 63.40%/75.40%，DeepSeek 略低 1.6–2.0 个百分点，模型替换未带来提升。纯文本 baseline 仅为 5.20%/5.80%，说明视觉证据是必要条件。该结果说明瓶颈更多在证据理解与答案规范化，而非模型规模。
2. **结构路由具有方向性收益，但召回仍偏中等。** Page Recall@1 为 0.4207、@3 为 0.5350，明显高于标题词匹配的 0.2329，但距离可用检索仍有差距。零文本页与退化目录是主要失败来源。
3. **Jev 的判断分布偏保守。** `sufficient=True` 仅 7.4%，`search` 占 59.2%，说明模型在多数片段上选择继续检索。不可答题中被判 `sufficient=False` 的比例为 97.48%（116/119，3 例反例见 5.3）；另有 4 道可答题在路由阶段被过度拒答（属路由行为，不是 Jev 判断）。
4. **覆盖率与准确率必须分开。** Jev 的 500/500 是执行覆盖率，不是判断正确率；`relevant=True` 与 `sufficient=True` 是模型输出分布，也不能替代人工标注得到的准确率。

### 5.3 三例 `sufficient=True` 异常

数据集将 Q0114、Q0363、Q0385 标注为不可答（`evidence_pages=[]`），但 Jev 判定 `sufficient=True`。三例均无 gold 泄漏，原因来自片段与问题主题的表面匹配：

| qid | 问题要点 | 送入片段 | 模型判断 | 原因 |
|---|---|---|---|---|
| Q0114 | 文档最后一页提到的州长姓名 | 第 1 页封面含 `Rick Scott GOVERNOR` | `sufficient=0.93`、`answer` | 片段确含州长姓名，但模型未处理「last page」页码约束 |
| Q0363 | PWC Technic 有多少办公室 | 第 15 页 `offices in 155 countries` | `sufficient=0.63`、`answer=0.46` | 将「覆盖国家数」误作「办公室数量」；实体在文档中不存在 |
| Q0385 | Ranking Prompt Example 中飞机类型 | 第 21 页示例对象为汽车 | `sufficient=0.69`、`answer=0.67`，但 `self_contained=False` | 以「示例位置匹配」替代「对象类型匹配」，模型自身亦判定上下文不完整 |

三例说明 `sufficient` 判断对限定词、实体与对象类型不敏感，是后续改进的直接依据。原始模型输出均未修改。

### 5.4 失败样本与证据字符串

- **QA 失败 5 题**：4 题页面渲染失败、1 题 API 错误，均按失败计错，未从分母中剔除。
- **可答但无 gold 页 2 题**：Q0290、Q0372，不进入路由 Page Recall 分母。
- **片段包含 gold 答案字符串 61 / 500**：其中 45 条答案长度 ≤ 2 字符（yes/no/数字），属无意义撞串；长度大于 2 的 16 条中，11 条所在页本身即 gold 页，5 条为跨页重复表述的巧合（Q0481、Q0275、Q0049、Q0082、Q0170）。所有片段的选页过程均未读取 gold。

### 5.5 结论与未验证项

**有数据支持的结论**

1. **目录路由相对标题词匹配有明显改善。** 在 376 主口径上 PR@3 = 0.5350，标题词匹配为 0.2329；同候选页预算 baseline 为 0.3637。方向性增益稳定，但因预算不等，不能表述为严格等预算优势。
2. **模型替换未提升问答准确率。** 在相同 500 题与相同判分规则下，DeepSeek 严格/宽松为 61.80%/73.40%，Qwen3-VL-Plus 为 63.40%/75.40%，替换后略有下降。
3. **视觉证据是必要条件。** 纯文本 baseline 仅 5.20%/5.80%，与两种多模态方案相差约 56–70 个百分点。
4. **Jev 的判断分布偏保守。** `sufficient=True` 仅 7.4%，`search` 占 59.2%；对不可答题的合理拒答为 97.48%，但存在 3 例 `sufficient=True` 误判，说明其对页码限定、实体缺失与对象类型不敏感。
5. **主要错误来源是证据理解而非接口。** QA 的 5 例失败中 4 例为页面渲染异常、1 例为内容审核；其余错误集中在图表视觉误读、证据页不含答案与答案规范化。

**尚未验证的结论**

1. **Jev 判断准确率**：无人工 `confirmed` 标签，`sufficient=True` 比例与 API 成功率都不能当作准确率。
2. **端到端系统增益**：没有把路由输出直接接入问答的串联实验，不能报告端到端准确率或相对 baseline 的提升。
3. **路由的等预算优势**：现有 baseline 与路由的候选页预算不严格相等。
4. **统计稳定性**：500 题为固定清单，未做多次采样或置信区间估计；早期小样本结果不具统计稳定性。

---

## 6. 早期小样本实验

以下实验用于验证流程可行性，样本量小、指标波动大，**不与第 4 节的 500 题正式结果混合比较**。

| 实验 | 样本 | 结果 | 说明 |
|---|---:|---|---|
| PaddleOCR 本地推理 | 4 页 | 231 个文字框、10,165 字符，平均置信度约 0.997 | 验证 CPU 推理链路；单页耗时 55–118 s |
| Jev 决策输出核验 | 12 条 | API 成功 12/12；7 条有复核记录且 `next_action` 一致 7/7 | 另 5 条未形成 `confirmed` 标签，未计入一致率 |
| 目录路由与证据召回 | 9 题 | Section Hit@1 66.7%、Hit@3 77.8%、Page Recall@3 0.6444、完整证据召回 4/9 | 标题词匹配 baseline：Hit@3 2/9、PR@3 0.20；同预算 baseline：4/9、PR 0.4222 |

这些结果只说明各模块可以真实运行，不能作为准确率或召回率的稳定估计。正式结论以第 4 节为准。

相关记录：[`outputs/task5/evaluation.json`](outputs/task5/evaluation.json)（Jev 小样本评估，仅含样本编号与统计量）。OCR 逐页对比与小样本路由明细含文档文本，未随公开版发布。

---

## 7. 局限性与讨论

1. **QA 是 oracle 证据页上界。** 问答输入直接使用 gold `evidence_pages` 渲染的页面图片，衡量的是「证据已知」条件下的回答能力，不代表端到端检索问答性能。
2. **Jev 判断准确率不可计算。** 仓库内只有 30 题候选清单（manifest），没有人工 `confirmed` 标签；因此只能报告执行覆盖率、判断分布与拒答表现，不能把 `sufficient=True` 比例或 API 成功率写成准确率。
3. **模块指标不可合成。** QA 输入 gold 页、路由输入 Question + TOC、Jev 输入真实片段，三者的问题性质、输入条件与分母不同，只能并排描述，不能加权为一个总分，也不能把 QA 准确率当作其他模块的同条件基线。
4. **路由主口径不含过度拒答。** 376 题主口径排除了可答但被拒答的样本；若按 379 严格口径将其记 0，Hit@3 降至 58.31%、PR@3 降至 0.5308。两个口径均已报告。
5. **路由召回仍偏低。** PR@1 0.4207、PR@3 0.5350，零文本页与退化目录是主要瓶颈；目录缺失时路由缺少可用结构信号。
6. **baseline 预算不对等。** 标题词匹配、同预算检索与 LLM 路由的候选页数并不严格相等，相关对照只能作为方向性参考。
7. **OCR 覆盖有限。** OCR 仅用于缺文本页补充，未对全量扫描件做系统验证；CPU 单页耗时 55–118 s，成本较高。
8. **Jev 对限定条件不敏感。** 第 5.3 节三例表明模型可能忽略页码限定、实体缺失与对象类型差异；该现象已保留在原始输出中，未做修正。
9. **样本与统计。** 500 题为固定清单，未做多次采样或置信区间估计；早期小样本结果不具备统计稳定性。
10. **尚无端到端串联结果。** 仓库没有把「目录路由 → 证据检索 → Jev 判断 → 多模态问答 → 最终评分」串联为一次实验的评测文件；因此只能证明各模块的独立表现，不能声称完整 JevSearch 系统优于 baseline。

---

## 8. 代码结构、依赖与复现

### 8.1 代码与结果结构

```
.
├── README.md
├── requirements.txt                    # 核心依赖
├── FINAL_ACCEPTANCE_REPORT_500.md      # 500 题最终验收报告
├── FINAL_500_EXPERIMENT_REPORT.md      # 500 题正式实验报告
├── EXPERIMENT_REPORT.md                # MMLongBench-Doc baseline 与多模态对照
│
├── 公共数据与 baseline
│   ├── prepare_data.py                 # 固定随机种子采样 500 题
│   ├── prepare_documents.py            # 下载 PDF 并抽取文字层
│   ├── run_baseline.py                 # 纯文本 baseline
│   ├── run_advanced_rag.py             # Qwen3-VL-Plus 多模态版本
│   └── analyze_baseline.py / compare_results.py / error_analysis.py / check_result.py
│
├── 文档解析与 OCR
│   ├── task6_build_toc.py              # 书签 → 印刷目录 → 正文字号 → pseudo-TOC
│   ├── task6_build_toc_500.py          # 500 题目录构建
│   ├── task6_ocr_missing_pages.py      # 缺文本页补 OCR
│   ├── ocr_route_500.py                # 路由候选页 OCR
│   └── run_paddleocr_task4.py          # OCR 单页验证脚本
│
├── 目录路由
│   ├── task6_route_and_eval.py         # 小样本路由与指标
│   ├── task6_route_and_eval_500.py     # 500 题路由与指标
│   └── fill_gaps_500.py                # 路由缺口补跑（可拒答）
│
├── Jev 证据判断
│   ├── test_jev.py                     # 接口连通性验证
│   ├── run_task5_jev.py                # 小样本 Jev 调用
│   ├── evaluate_task5.py               # 仅统计已确认样本
│   ├── task5_jev_500.py                # title 片段
│   ├── task5_jev_route_500.py          # route 片段
│   ├── task5_jev_ocr_500.py            # OCR 片段
│   ├── task5_jev_recover_500.py        # recover 批次
│   ├── task5_jev_ocr_pilot.py          # OCR 试点批次
│   └── fill_gaps_round3_500.py         # 缺口补跑（真实文本层 / OCR）
│
├── 多模态问答
│   ├── run_advanced_rag_ds.py          # 小样本 DeepSeek 问答
│   └── run_advanced_rag_ds_500.py      # 500 题问答（断点续跑）
│
├── 评估与验收
│   ├── final500_common.py              # qid 映射、PDF 解析、append-only 工具
│   ├── compare_500.py                  # 严格 / 宽松匹配与跨模块比较
│   ├── audit_final500.py               # 零成本离线审计
│   ├── verify_final_acceptance_500.py  # 独立验收（500 条记录 + 哈希）
│   ├── run_experiment_500.py           # 总控：分批、去重、重试、报告
│   └── diagnose_500_coverage.py / precheck_leakage_500.py
│
└── outputs/final500/
    ├── audit/                          # 逐题状态表、聚合统计、验收与缺口诊断
    ├── manifest.json                   # 500 题清单与统计口径
    └── run500/
        ├── manifest.json               # qid 映射
        ├── qids_500.txt                # 固定 500 题 qid 清单
        └── qa/run_summary.json         # 问答执行汇总（不含题目/答案）
```

> 逐题原始预测（`run500/qa/ds_results.jsonl`、`run500/toc/routes.jsonl`、`run500/jev/*.jsonl`）、文档目录 `toc.json` 与证据片段仅在完整研究仓库中保存，未随公开版发布。

### 8.2 关键结果文件

| 内容 | 路径 |
|---|---|
| 最终验收报告 | [`FINAL_ACCEPTANCE_REPORT_500.md`](FINAL_ACCEPTANCE_REPORT_500.md) |
| 机器可读验收摘要 | [`outputs/final500/audit/ACCEPTANCE_CHECKS_500.json`](outputs/final500/audit/ACCEPTANCE_CHECKS_500.json) |
| 500 行逐题状态 | [`outputs/final500/audit/final500_status_500.csv`](outputs/final500/audit/final500_status_500.csv) |
| 最终统计汇总 | [`outputs/final500/audit/final500_stats.json`](outputs/final500/audit/final500_stats.json) |
| 评估协议 | [`outputs/final500/audit/EVAL_PROTOCOL_UNIFIED.md`](outputs/final500/audit/EVAL_PROTOCOL_UNIFIED.md) |
| 不可答 119 题统计 | [`outputs/final500/audit/unanswerable_119_summary.json`](outputs/final500/audit/unanswerable_119_summary.json) |
| 路由补跑指标 | [`outputs/final500/audit/route_fill_metrics.json`](outputs/final500/audit/route_fill_metrics.json) |
| 固定 qid 清单 | [`outputs/final500/run500/qids_500.txt`](outputs/final500/run500/qids_500.txt) |
| 问答执行汇总 | [`outputs/final500/run500/qa/run_summary.json`](outputs/final500/run500/qa/run_summary.json) |
| 500 题清单与口径 | [`outputs/final500/manifest.json`](outputs/final500/manifest.json) |

> 逐题原始结果（`run500/qa/ds_results.jsonl`、`run500/toc/routes.jsonl`、`run500/jev/*.jsonl`）含题目与证据文本，不随公开版发布。

### 8.3 Python 依赖

| 用途 | 依赖 | 版本 / 说明 |
|---|---|---|
| 主流程 | Python | 3.12 |
| PDF 处理 | `pymupdf`（`import fitz`） | ≥2.0（OCR 环境为 2.1） |
| 模型调用 | `openai`（OpenAI 兼容 SDK） | ≥3.36，访问 DashScope、TypeSafe |
| OCR | `paddlepaddle`（CPU）+ `paddleocr` | 3.3.1 / 3.7.0，独立虚拟环境 `.venv_paddleocr` |

```bash
# 必需依赖（文本层解析、PDF 渲染与模型调用）
pip install -r requirements.txt

# 可选：OCR 建议单独建虚拟环境，避免依赖冲突
pip install paddlepaddle==3.3.1 paddleocr==3.7.0 Pillow>=13.0
```

### 8.4 环境变量

密钥仅从环境变量读取，仓库不包含任何密钥：

```powershell
$env:DASHSCOPE_API_KEY = "<DashScope 密钥>"   # 问答、目录路由
$env:TYPESAFE_API_KEY  = "<TypeSafe 密钥>"    # Jev
```

### 8.5 复现命令

```powershell
# 1. 零成本独立验收：重算 500 条记录与全部指标
py -3.12 verify_final_acceptance_500.py

# 2. 状态与缺口检查（不调用 API）
py -3.12 fill_gaps_round3_500.py --status
py -3.12 fill_gaps_round3_500.py --dry-run 5

# 3. 零成本离线审计（重算状态表与模块矩阵）
.venv_paddleocr\Scripts\python.exe audit_final500.py

# 4. 完整流程（会产生 API 费用）
py -3.12 run_experiment_500.py --preflight --build
py -3.12 run_experiment_500.py --selftest
py -3.12 run_experiment_500.py --run
```

数据准备与早期实验：

```bash
python prepare_data.py            # 生成固定 500 题清单
python prepare_documents.py       # 下载 PDF 并抽取文字层
python run_baseline.py            # 纯文本 baseline
python run_advanced_rag.py        # Qwen3-VL-Plus 多模态
python run_advanced_rag_ds.py --limit 10
python task6_build_toc.py
python task6_ocr_missing_pages.py
python task6_route_and_eval.py --reuse
```

注意事项：

- PDF 下载至系统临时目录（`%TEMP%\mmlongbench_pdfs`），不进入仓库；
- 独立验收脚本读取 `sampled_500_for_baseline.jsonl`（500 题清单与 gold 标注），该文件不随仓库发布：需先按第 9 节获取数据集，再执行 `python prepare_data.py` 重新生成。缺少该文件时，验收脚本仍会输出 Jev 覆盖率、唯一 qid、重复、失败、token 与状态分布，并明确打印 `[SKIP]`，不写 `ACCEPTANCE_CHECKS_500.json`；
- `task6_route_and_eval.py --reuse` 与 `task6_route_and_eval_500.py --reuse` 只读取已有预测，不调用 API；缺少本地 gold 时只打印记录数、唯一 qid、token、平均延迟与状态分布，并明确跳过指标计算。两者默认**不把 gold 写回**预测或评价文件，如需生成本地含 gold 的审计副本，显式追加 `--record-gold`（该副本不要提交）；
- `compare_results.py` 需要本地 `sampled_500_for_baseline.jsonl` 复核严格/宽松匹配，缺失时会报错退出，不会输出 0% 或 100%；
- OCR 需激活 `.venv_paddleocr`，并设置 `PADDLE_PDX_CACHE_HOME` 与 `enable_mkldnn=False`；
- 完整流程会消耗真实 API 额度，建议先用 `--limit 1` 或 `--dry-run` 验证；
- `run_experiment_500.py --selftest` 仅使用假数据验证去重与续跑逻辑，不作为实验结论。

---

## 9. 数据许可与仓库边界

- 本实验使用 **MMLongBench-Doc** 基准（Hugging Face：`yubo2333/MMLongBench-Doc`）。原始问答标注、PDF 与 500 题清单未上传至本仓库，请从该基准的官方发布渠道按其许可协议获取。
- 获取数据后，在项目根目录依次执行：
  1. `python prepare_data.py`：读取该基准 `train` 划分，`shuffle(seed=42).select(range(500))`，生成 `sampled_500_for_baseline.jsonl`。该脚本与本文实验使用同一随机种子与同一抽样顺序，因此重建后 qid `Q0001`–`Q0500` 与原实验逐题对应；
  2. `python prepare_documents.py`：按清单下载对应 PDF 至系统临时目录。
- 本公开仓库**不包含**：数据集 PDF 原件、`document_texts.json`、全量 OCR 文本、数据集压缩包、题目原文、文档目录（TOC）、证据片段、逐题模型输出，以及任何含 gold answer / gold evidence_pages 的数据文件。
- **逐题原始预测仅在完整研究仓库保存**：`qa/ds_results.jsonl`、`toc/routes.jsonl`、`jev/snippets_*.jsonl` 已移除 `offline_only`；`baseline_results.jsonl`、`advanced_rag_results.jsonl` 已移除 `answer`、`evidence_pages`、`evidence_sources`、`answer_format`；路由与问答的逐题指标不含 gold 页码。模型输出、调用记录、token、时间戳与全部统计量均未改动。公开版只发布由这些文件计算出的聚合指标与不含文本的逐题状态表。
- **离线复现方式**：`verify_final_acceptance_500.py` 从本地 `sampled_500_for_baseline.jsonl` 读取 gold，再复算严格/宽松匹配、Section Hit@K、Page Recall@K 与 FullEvidence@K；缺少该文件时脚本只做不依赖 gold 的核验并打印 `[SKIP]`，不会输出 0%/100%，也不会改写验收文件。公开版不含原始预测，因此完整复算需在完整研究仓库中执行。
- **无法独立复现的部分**：重新执行 API 实验需要原始 PDF（用于文本层、OCR 与版面信息）与 DashScope / TypeSafe 凭证；`EXPERIMENT_REPORT.md` 第 5 节的错误类型分布依赖未纳入仓库的 `error_analysis_report.jsonl` 与 `document_texts.json`，只能核对结论，不能凭仓库重新计算。本仓库可以核对全部已发布指标的计算口径，但不等价于端到端重跑。
- 本公开仓库**保留**：实验代码、500 题逐题状态表（qid / 文档类型 / 执行状态 / 各模块得分）、聚合统计、审计文档与实验报告。状态表不含题目或文档文本。
- 哈希口径：完整研究仓库以 `public_files_sha256.json` 记录发布文件哈希；该清单包含未公开文件路径，不随公开版发布。
- 历史报告中的 `<PROJECT_ROOT>`、`<USER_HOME>` 为本地绝对路径的脱敏占位符。完整研究仓库中的证据片段可能出现 PDF 正文自带的 `file:///C:/Users/...` 示例路径，与本项目环境无关。
- **数据许可状态**：MMLongBench-Doc 的数据卡当前未提供明确的 license 字段。为避免未经授权的再分发，本公开仓库只发布代码、聚合指标与实验报告，不发布数据集原文、PDF、TOC、证据片段或逐题预测文本；完整实验记录保存在私有仓库，如需学术核对或复现材料，请通过 Issue 联系作者。
- 补充批次仅新增 `jev_predictions_round3_500.jsonl` 与 `snippets_round3_500.jsonl`，未覆盖任何历史结果。
