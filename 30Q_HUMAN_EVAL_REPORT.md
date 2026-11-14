# Jev 30 道人工评测实验报告（初步）

- 报告日期：2026-11-14
- 对应实验：MMLongBench-Doc 固定 500 题
- 评估对象：Jev 证据判断（`relevant` / `self_contained` / `sufficient` / `next_action`）
- 人工标注样本：**30 / 500 道**
- 标注者：**1 人（单人标注）**

> **结论边界**：本报告是 30 道单人标注样本的初步质量检查，不是 500 道完整人工准确率评估，也不构成双人标注一致性结论。30 道样本的指标不能外推为 500 题总体准确率。

---

## 1. 实验状态与口径

| 项目 | 状态 |
|---|---|
| Jev 真实 API 调用 | **500 / 500 完成**，500 条有效判断，无失败、无重复、无缺失 |
| 人工标注 | **30 / 500 完成**，30 条均为 `review_status=confirmed` |
| 人工标注者数量 | 1 人 |
| 第二标注者 / Cohen's κ | **未进行**，本报告不计算、不报告 κ |
| 人工标签内部一致性 | 30 条 `next_action` 与 `relevant`/`sufficient` 推导规则 **0 冲突** |
| 500 题整体人工准确率 | **仍不可计算**（仅 30 道有 confirmed 标签） |
| 原始 500 条预测 | 未修改、未覆盖；评估前后 SHA256 校验一致 |
| 本次评估 API 成本 | 0（完全离线计算，未重新调用付费 API） |

因此本报告中的数字只回答一个问题：**在这 30 道已完成人工确认的样本上，Jev 的判断与人工判断一致到什么程度。**

---

## 2. 评估方法

### 2.1 样本构成

30 道题来自固定的分层样本清单，不随 Jev 预测结果重新抽取：

| 维度 | 分布 |
|---|---|
| 可答 / 不可答 | 24 / 6 |
| 片段来源 | title 10、route 9、OCR 4、OCR pilot 4、recover 2、pilot 1 |
| 人工动作 | expand 13、search 13、answer 4 |
| 证据类型（人工） | Pure-text 19、Table 5、None 3、Chart 2、Mixed 1 |

### 2.2 盲标流程

标注者只看到问题、候选证据片段、文档类型、页码与片段来源；**不展示** Jev 的预测标签、概率、gold answer、gold evidence_pages 与数据集 answerable 标记。标注字段为 `relevant`、`self_contained`、`sufficient`、`next_action`、`evidence_type`、`confidence`，完成后导出 CSV。

### 2.3 匹配与校验

- 匹配键：`qid` + `snippet_id`，30 条全部精确命中 500 条真实预测；
- 二次校验：重新计算发送给 Jev 的 `state_sent` 的 SHA256，与盲标材料中的输入哈希逐条一致；
- 仅使用 `review_status=confirmed` 的记录；
- 30 条 `qid` 与 `snippet_id` 均唯一，无重复、无缺失字段。

### 2.4 指标定义

- 字段准确率 = 人工值与 Jev 值相同数 / 30；
- 综合一致率 = `relevant`、`self_contained`、`sufficient`、`next_action` 四项**同时**一致 / 30；
- 动作协议归一化：`sufficient=True → answer`；`relevant=True` 且 `sufficient=False → expand`；`relevant=False → search`；
- 原始 `next_action` 与归一化动作分别报告，避免 Jev 原始输出与字段推导不一致时产生歧义。

---

## 3. 汇总指标

| 字段 | 正确 | 错误 | 准确率 |
|---|---:|---:|---:|
| `relevant` | 25 | 5 | **83.33%** |
| `self_contained` | 19 | 11 | **63.33%** |
| `sufficient` | 25 | 5 | **83.33%** |
| `next_action`（原始输出） | 17 | 13 | **56.67%** |
| `next_action`（协议归一化） | 21 | 9 | **70.00%** |
| **四项综合一致（原始动作口径）** | 12 | 18 | **40.00%** |
| **四项综合一致（归一化动作口径）** | 14 | 16 | **46.67%** |

动作分类补充指标：

| 指标 | 原始 `next_action` | 协议归一化动作 |
|---|---:|---:|
| Action Accuracy | 56.67% | 70.00% |
| Action Macro-F1 | 0.5628 | 0.6882 |
| Cohen's κ（人工 vs Jev） | 0.3194 | 0.5431 |

二分类指标（以 1 为正类）：

| 字段 | Precision | Recall | F1 | Accuracy | κ |
|---|---:|---:|---:|---:|---:|
| `relevant` | 0.8000 | 0.9412 | 0.8649 | 0.8333 | 0.6512 |
| `sufficient` | 0.4444 | 1.0000 | 0.6154 | 0.8333 | 0.5283 |
| `self_contained` | 0.9500 | 0.6552 | 0.7755 | 0.6333 | −0.0645 |

> 这里的 κ 是“人工标签 vs Jev 判断”的吻合度，**不是**两位标注者之间的一致性。

---

## 4. 混淆矩阵

### 4.1 `relevant`（行=人工，列=Jev）

| 人工 \ Jev | 0 | 1 | 合计 |
|---|---:|---:|---:|
| 0 | 9 | 4 | 13 |
| 1 | 1 | 16 | 17 |

### 4.2 `self_contained`（行=人工，列=Jev）

| 人工 \ Jev | 0 | 1 | 合计 |
|---|---:|---:|---:|
| 0 | 0 | 1 | 1 |
| 1 | 10 | 19 | 29 |

### 4.3 `sufficient`（行=人工，列=Jev）

| 人工 \ Jev | 0 | 1 | 合计 |
|---|---:|---:|---:|
| 0 | 21 | 5 | 26 |
| 1 | 0 | 4 | 4 |

### 4.4 `next_action`（行=人工，列=Jev 原始输出）

| 人工 \ Jev | answer | expand | search | 合计 |
|---|---:|---:|---:|---:|
| answer | 3 | 1 | 0 | 4 |
| expand | 3 | 8 | 2 | 13 |
| search | 1 | 6 | 6 | 13 |

---

## 5. 主要错误与原因分析

### 5.1 动作错误分布（13 次）

| 错误方向 | 次数 |
|---|---:|
| 人工 `search` → Jev `expand` | 6 |
| 人工 `expand` → Jev `answer` | 3 |
| 人工 `expand` → Jev `search` | 2 |
| 人工 `search` → Jev `answer` | 1 |
| 人工 `answer` → Jev `expand` | 1 |

### 5.2 字段错误模式

| 模式 | 次数 | 含义 |
|---|---:|---|
| `self_contained`：人工 1 → Jev 0 | 10 | Jev 对“片段是否自包含”明显偏严 |
| `self_contained`：人工 0 → Jev 1 | 1 | 反向误判较少 |
| `relevant`：人工 0 → Jev 1 | 4 | 关键词/标题命中被误判为相关 |
| `relevant`：人工 1 → Jev 0 | 1 | 真实相关信息被漏判 |
| `sufficient`：人工 0 → Jev 1 | 5 | 高估证据充分性，直接导致提前作答 |
| `sufficient`：人工 1 → Jev 0 | 0 | — |

**最突出的两个问题：**

1. **`self_contained` 漏判**：10/30 条人工认为片段可独立理解，Jev 判为不满足。这是单项错误最多、且 κ 接近 0（−0.0645）的字段，说明双方对该字段的判定口径差异最大。
2. **动作偏 `expand`、且易提前 `answer`**：6 次把人工的 `search` 判成 `expand`，说明 Jev 对无用片段仍倾向继续扩展；另有 3 次把人工的 `expand` 判成 `answer`，与 5 次 `sufficient` 高估相互印证。

### 5.3 分组观察

| 片段来源 | n | 动作准确率 |
|---|---:|---:|
| route | 9 | 77.78% |
| OCR pilot | 5 | 80.00% |
| title | 10 | 50.00% |
| OCR | 4 | 25.00% |
| recover | 2 | 0.00% |

| 人工证据类型 | n | 动作准确率 |
|---|---:|---:|
| Chart | 2 | 100.00% |
| Pure-text | 19 | 63.16% |
| Table | 5 | 40.00% |
| None | 3 | 33.33% |
| Mixed | 1 | 0.00% |

> 各组样本量很小（多数 n≤10），只能作为错误定位线索，不能作为分组准确率结论。OCR/recover 片段的低准确率可能与 OCR 文本破碎、上下文缺失有关；title 类片段容易被判为“相关但不充分”。

### 5.4 典型错误案例（不含题目与证据原文）

| qid | snippet_id | 来源 | 不一致字段 | relevant 人工/Jev | self_contained 人工/Jev | sufficient 人工/Jev | next_action 人工/Jev |
|---|---|---|---|---|---:|---:|---|
| Q0021 | Q0021__recover | recover | self_contained、sufficient、next_action | 1/1 | 0/1 | 0/1 | expand/answer |
| Q0191 | Q0191__route | route | relevant、self_contained、next_action | 0/1 | 1/0 | 0/0 | search/expand |
| Q0216 | Q0216__ocr | ocr | self_contained、sufficient、next_action | 1/1 | 1/0 | 0/1 | expand/answer |
| Q0363 | Q0363__title | title | relevant、sufficient、next_action | 0/1 | 1/1 | 0/1 | search/answer |
| Q0045 | Q0045__title | title | relevant、next_action | 0/1 | 1/1 | 0/0 | search/expand |
| Q0244 | Q0244__route | route | sufficient、next_action | 1/1 | 1/1 | 0/1 | expand/answer |
| Q0412 | Q0412__title | title | relevant、self_contained | 1/0 | 1/0 | 0/0 | expand/expand |
| Q0462 | Q0462__title | title | self_contained、next_action | 1/1 | 1/0 | 1/1 | answer/expand |

错误原因按证据对齐关系可归纳为四类：

1. **自包含口径偏严**：片段包含表格局部、图注或连续正文时，人工认为可独立理解，Jev 仍要求更多上下文。
2. **相关性口径偏宽**：标题、页眉、关键词密集但无实质信息的片段被 Jev 判为相关。
3. **充分性高估**：只覆盖部分证据即被判为足以回答，导致 `expand → answer`。
4. **检索动作保守**：对人工认为完全无用的片段，Jev 仍选择 `expand` 而非 `search`。

以上为标签层面的错误归因，不代表已定位到 Jev 内部的具体机制。

---

## 6. 局限性与讨论

1. **样本量小**：n=30，所有比例的分母都很小，未计算置信区间；单个样本即可造成 3.3 个百分点波动。
2. **单人标注**：没有第二标注者，无法计算 Cohen's κ 或标注者间一致性；本报告中的 κ 全部是“人工 vs Jev”，不能替代标注质量检验。
3. **样本不可外推**：30 道为固定分层样本（含 6 道不可答），其来源与动作分布不等于 500 题总体分布，不能据此宣称 500 题准确率。
4. **标签主观性**：`self_contained` 与 `sufficient` 的边界依赖标注规范；当前 10 次自包含分歧说明规范与模型口径存在系统性差异，后续应优先复核该字段定义。
5. **原始动作与字段推导可能不一致**：Jev 原始 `next_action` 偶尔与其 `relevant`/`sufficient` 字段不匹配，因此报告同时给出原始与归一化动作两个口径。
6. **500 题结论不变**：本报告不修改、不替代此前的 500 题执行与判断分布结果，也不改变任何历史指标。

---

## 7. 复现方式

评分脚本（离线，只读标签与预测）：

```bash
python scoring/evaluate_human_vs_jev.py \
  --labels <confirmed_labels.csv> \
  --out-dir scoring/results/quality30 \
  --scope 30 \
  --report-name QUALITY30_REPORT.md
```

脚本会校验：确认数、`qid`/`snippet_id` 唯一性、字段完整性、与 500 条预测的精确匹配、输入哈希一致性、原始预测文件评估前后 SHA256 不变。人工标注 CSV 含题目与证据原文，**不随公开版发布**。

公开版附带的机器可读汇总：

- [`quality30_field_metrics.csv`](outputs/final500/audit/quality30_field_metrics.csv)：各字段正确数、错误数、准确率与综合一致率；
- [`quality30_error_cases.csv`](outputs/final500/audit/quality30_error_cases.csv)：18 条不一致样本的 qid、片段来源、字段对比与置信度，**不含题目或证据文本**。

---

## 8. 数据许可与发布边界

- 本报告只发布聚合指标、字段对比与不含原文的错误统计；
- 不发布人工标注 CSV、题目原文、证据片段、文档目录、PDF 或逐题模型输出；
- MMLongBench-Doc 数据卡未提供明确再分发许可，原始数据请从官方渠道获取；
- 完整原始预测与标注材料保存在私有研究环境，如需学术核对可通过 Issue 联系作者。
