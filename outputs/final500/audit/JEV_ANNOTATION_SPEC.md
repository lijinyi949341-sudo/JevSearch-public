# Jev 判断人工标注规范（质量检查版 v1）

版本：v1（2026-11-10）
用途：先做 30 道小样本人工质量检查，验证标注口径与 Jev 输出是否可被可靠评估。
**本阶段不生成任何“正确标签”，只提供空白标注表与规范。**

---

## 1. 标注目标

对每个“问题 + 候选证据片段”，独立判断四件事：

1. 片段是否与问题相关（relevant）；
2. 片段是否自包含（self_contained）；
3. 片段是否足以完整回答问题（sufficient）；
4. 在此基础上的下一步动作（answer / expand / search）。

同时记录证据类型与标注置信度。

---

## 2. 盲标规则（必须遵守）

标注者只能看到：

- question
- evidence snippet
- doc_type、page、来源（title/route/OCR/recover/pilot）

标注者**不得**看到：

- gold answer、gold evidence_pages、evidence_sources；
- 数据集的 answerable 标记；
- Jev 的预测结果（relevant / sufficient / next_action / 概率）；
- 其他标注者的结果。

标注使用的盲标表不在仓库中（含数据集问题与片段原文，需从本地结果文件重新导出）；不要打开 manifest CSV 后再标注，manifest 仅供实验组织者做分层统计。

---

## 3. 字段定义与判定标准

### 3.1 `relevant`（0/1）

- 1：片段包含回答问题所需的至少一条实质信息（实体、数值、关系、步骤、结论）。
- 0：仅主题相同、仅出现关键词、仅标题/页眉/页码/目录，或完全不相关。

判定要点：**“出现了关键词”不等于 relevant**；必须能指向问题所需的信息。

### 3.2 `self_contained`（0/1）

- 1：只看 question + snippet 就能理解片段在说什么（指代明确、表头/单位齐全）。
- 0：片段依赖前文、缺少表头、代词无指代、图表缺轴/单位，必须回到原页才能理解。

### 3.3 `sufficient`（0/1）

- 1：片段包含完整回答所需的全部证据（多值、比较、单位、限定条件都齐全）。
- 0：只覆盖部分证据，或需要其他页/其他图表补充。

判定要点：**relevant=1 但 sufficient=0 是常见情况**，不要为了省事把两者合并。

### 3.4 `next_action`

| 标签 | 条件 |
|---|---|
| `answer` | sufficient=1，可直接回答 |
| `expand` | relevant=1 且 sufficient=0，需要扩大证据 |
| `search` | relevant=0，当前片段无用，需要重新检索 |

若 relevant=0，则 sufficient 必须为 0。

### 3.5 `evidence_type`

`Chart` / `Table` / `Figure` / `Pure-text` / `Generalized-text` / `Mixed` / `None`

- 以片段中最主要的证据形态为准；
- 片段为封面/目录/页眉 → `None`；
- 多种形态且都关键 → `Mixed`。

### 3.6 `confidence`（1/2/3）

- 3：很有把握；
- 2：基本确定但有轻微歧义；
- 1：难以判断，需要讨论。

### 3.7 `notes`

记录歧义点，例如“片段只有表头”“数值缺单位”“问题是多跳，片段只覆盖第一跳”。

---

## 4. 标注流程

1. 组织者从本地结果文件导出 30 条盲标表（标签字段全为 null；该文件不随仓库发布）。
2. 标注者独立完成 30 条，填写 6 个标签字段与 notes。
3. 其中 **10 条**由第二位标注者独立重复标注（建议 S01–S10，覆盖三种动作）。
4. 计算一致率：
   - relevant / self_contained / sufficient 用 Cohen's κ；
   - next_action 用 Cohen's κ 或混淆矩阵；
   - 目标 κ ≥ 0.60；低于 0.60 先修订定义，不进入正式评估。
5. 不一致条目由第三人仲裁，形成 `confirmed` 标签。
6. 只有 `review_status=confirmed` 的条目才可进入 Jev 指标计算。

---

## 5. 30 道样本构成

文件：

- 盲标表：不在仓库中（含问题与片段原文，需本地导出）
- 分层清单（仅组织者）：`outputs/final500/audit/annotation_sample_quality30_manifest.csv`

| 维度 | 分布 |
|---|---|
| 可答 / 不可答 | 24 / 6 |
| 证据来源 | title 10、route 9、OCR 4、pilot2 4、recover 2、pilot1 1 |
| Jev 归一化动作 | expand 11、search 10、answer 9 |
| 证据类型 | Figure 7、Pure-text 6、Table 5、Chart 5、Other/None 4、Generalized-text 3 |
| 文档类型 | Research report 9、Tutorial/Workshop 6、其余 5 类各 3 |

说明：

- 6 道不可答题全部来自 title 来源（路由跳过不可答题，OCR/recover 没有不可答输入），因此本次样本**不能**用来generalize Jev 对不可答题的整体行为。
- 样本按 qid 固定、可复现；不随 Jev 预测变化而重抽。

---

## 6. 标注后可计算的指标

| 指标 | 公式 | 说明 |
|---|---|---|
| Action Accuracy | 与人工 next_action 一致数 / confirmed 数 | 主指标 |
| Action Macro-F1 | 三类 F1 的宏平均 | 防止 answer 少样本被淹没 |
| Relevance F1 | 以 relevant=1 为正类 | 二分类 |
| Sufficiency F1 | 以 sufficient=1 为正类 | 二分类 |
| 混淆矩阵 | 3×3 | 看 search/expand/answer 的具体错法 |
| 标注者一致性 | Cohen's κ | 先验证标签可信 |

在 confirmed 数 < 30 前，只报告描述性统计，不宣称“Jev 准确率”。

---

## 7. 禁止事项

1. 不得根据 Jev 输出反推标签；
2. 不得使用 gold answer / gold evidence_pages 辅助判断；
3. 不得把“证据相关”直接标成“足够回答”；
4. 不得由脚本自动填充标签；
5. 不得修改原始预测文件。
