# JevSearch 500 题固定子集实验最终报告

生成时间：2026-11-12
数据集：MMLongBench-Doc 固定 500 题子集（`sampled_500_for_baseline.jsonl`，qid = Q0001–Q0500，134 个文档）
统计来源：全部由仓库内真实结果文件离线计算；本报告生成过程未调用任何付费 API，未修改任何原始预测。公开副本仅移除了数据集 gold 字段，模型输出与统计未改动（见 `public_files_sha256.json`）。
逐题状态表：`outputs/final500/audit/final500_status_500.csv`
机器可读统计：`outputs/final500/audit/final500_stats.json`
路由补跑明细：`outputs/final500/run500/toc/routes_fill_126.jsonl`

> 核心提醒：**500 题全部有处理状态，不等于 500 题全部成功评测。** 三个模块的任务定义、输入和分母都不同，不能把 QA 准确率、路由召回率和 Jev 判断分布合成为一个总分。

> **数据版本**：本报告只使用最终 500/500 口径的结果文件（`run500/{qa,toc,jev}/`、`final500_status_500.csv`、`final500_stats.json`）。
> 权威数字以 `FINAL_ACCEPTANCE_REPORT_500.md` 与 `outputs/final500/audit/ACCEPTANCE_CHECKS_500.json` 为准。

> **公开版说明**：本仓库为脱敏公开版。逐题原始预测、文档目录（TOC）与证据片段因 MMLongBench-Doc 数据再分发许可不明确，未随仓库发布；文中引用的 `run500/**` 原始文件仅保存在私有研究仓库。公开版可核对全部聚合指标与不含文本的逐题状态表。

---

## 1. 最终结果速览

| 模块 | 已尝试 | API 成功 | 可评测 | 覆盖率（可评测/500） | 关键指标 |
|---|---|---|---|---|---|
| DeepSeek 问答（gold 证据页，oracle） | 500 | 495 | 495 | 99.0% | 严格 62.42%（309/495）、宽松 74.14%（367/495） |
| 目录路由（Question + TOC，可拒答） | 500 | 500 | 500（Page Recall 376） | 100% | Hit@1 49.47%、Hit@3 58.78%、PR@1 0.4207、PR@3 0.5350、FullEvidence@3 0.4840 |
| Jev 证据判断 | 500 | 500 | 500 | 100% | `relevant=True` 193（38.6%）、`self_contained=True` 283（56.6%）、`sufficient=True` 37（7.4%）；**准确率不可计算** |

补充：

- 路由拒答 **70** 题 = 66 道不可答（正确拒答率 66/119 = 55.5%）+ 4 道可答过度拒答（Q0060、Q0134、Q0324、Q0372，过度拒答率 4/381 = 1.05%）。
- Jev 的 65 题历史缺口由 round3 批次以真实文本层 / OCR 片段补齐，最终 **500/500** 全部有真实判断，详见第 5.3 节。
- 三模块同时有真实输出：**495** 道（Jev ∩ QA）；若要求路由同时可算 Page Recall，则为 **375** 道（Jev ∩ QA ∩ 路由 379 口径）。

---

## 2. 数据与模块定义

### 2.1 数据规模

| 项目 | 数值 |
|---|---|
| 题目总数 / 唯一 qid | 500 / 500 |
| 唯一文档数 | 134 |
| 可回答题 | 381 |
| 不可回答题（gold = `Not answerable`） | 119 |
| 可回答但无 gold 证据页 | 2（Q0290 PDF 仅 105 页；Q0372 无标注） |
| gold 页码需修正 | 5（Q0060、Q0134、Q0324、Q0373、Q0383；仅离线评分使用，不进入模型输入） |

### 2.2 三个模块

| 模块 | 模型 | 输入 | 输出 | 性质 |
|---|---|---|---|---|
| QA | `deepseek-v4.1-flash-1017` | 问题 + gold `evidence_pages` 渲染图 | 自然语言答案 | **Oracle 上界**，非端到端 |
| 目录路由 | `deepseek-v4.1-flash-1017` | 问题 + 真实 PDF 目录 | 章节列表，或 `answerable=false` | 真实检索 + 自主拒答 |
| Jev | `jev-1.14.0`（typesafe_sdk 0.7.5） | 问题 + 真实检索片段 | `relevant/self_contained/sufficient/next_action` | 证据判断 |

gold 答案与 gold 证据页从未作为路由/Jev 的模型输入；Jev 片段全部来自 PDF 文本层、本地 OCR 或既有路由结果，`selection_uses_gold` 全量为 0。

### 2.3 不可回答题协议

| 模块 | 处理 |
|---|---|
| QA | 正常提问，按 `Not answerable` 评分 |
| 路由 | 统一执行可拒答提示词，由模型自行输出 `answerable=false`；不使用数据集标签选题 |
| Jev | 仅在路由给出章节且能提取真实文本时调用；无片段记 N/A，不强行判断 |

路由补跑的目标集合只由「是否有成功执行记录」决定，与 gold 可回答性无关。代码入口：`fill_gaps_500.py --run-route-fill`。

---

## 3. 执行状态（全量 500）

| 模块 | 已尝试 | API 成功 | 有效输入 | 可评测 | 不适用 / 无证据 | 失败 |
|---|---|---|---|---|---|---|
| QA | 500 | 495 | 495 | 495 | 0 | 5 |
| 目录路由 | 500 | 500 | 500 | 376（Page Recall）/ 500（含拒答） | 0 | 0 |
| Jev | 500 | 500 | 500 | 500 | 0 | 0 |

QA 失败 5 题：

| qid | 状态 | 原因 | 可修复性 |
|---|---|---|---|
| Q0134 | `no_images` | gold 页索引越界（0-based），渲染失败 | 页码修正后可重跑（低置信，需人工确认） |
| Q0272 | `api_error` | DashScope 400 `data_inspection_failed`（内容审核） | 重试不保证成功 |
| Q0290 | `no_images` | gold 页超出 PDF 总页数（文件不完整） | 不可修复（文件不完整） |
| Q0373 | `no_images` | gold 页码与物理页不一致 | 页码修正后可重跑 |
| Q0383 | `no_images` | gold 页码与物理页不一致 | 页码修正后可重跑 |

上述 QA 题未重跑，原始预测保持不变。

---

## 4. 目录路由结果

### 4.1 执行与拒答

| 项目 | 数值 |
|---|---|
| 执行 | 500 / 500 |
| 路由到章节 | 430 |
| 模型拒答（`answerable=false`） | 70 |
| 不可答题正确拒答 | 66 / 119 = 55.5% |
| 可答题过度拒答 | 4 / 381 = 1.05%（Q0060、Q0134、Q0324、Q0372） |

### 4.2 检索指标（376 道有可用 gold 页且未拒答）

| 指标 | 数值 |
|---|---|
| Section Hit@1 | 49.47% |
| Section Hit@3 | 58.78% |
| Page Recall@1（宏平均） | 0.4207 |
| Page Recall@3（宏平均） | 0.5350 |
| Full Evidence Recall@1 | 0.3564 |
| Full Evidence Recall@3 | 0.4840（182/376） |
| 严格口径 379 的 Full Evidence Recall@3 | 0.4802（182/379，3 道可答拒答记 0） |

分母说明：381 道可答题中，2 道无可用 gold 页（Q0290、Q0372），另 3 道（Q0060、Q0134、Q0324）被模型拒答，
故本表 Page Recall 分母为 376（逐题状态见 `final500_status_500.csv` 的 `route_eval` 列）。374 为路由原生口径；
379 = 374 + 5 道修正 gold 题（其中 3 道拒答记 0）。历史 374 题口径下的 baseline 数字与本口径不可直接比较。

`Full Evidence Recall@3` 与 Page Recall 使用 gold `evidence_pages` 列表长度作分母。Q0305 的 gold 为同一页重复 6 次
（`[6,6,6,6,6,6]`），列表口径下该题 PR@3 = 1/6、Full@3 未覆盖；去重口径下为 1.0 与覆盖。
因此去重口径的 Full@3 为 48.67%（376）/ 48.28%（379），PR@3 为 0.5373 / 0.5330。本文统一使用列表口径，两种口径不混用。

---

## 5. Jev 证据判断结果

### 5.1 来源（500 条，无重复 qid）

| 来源 | 条数 |
|---|---|
| title（目录标题匹配片段） | 263 |
| route（路由章节片段） | 88 |
| round3（缺口补跑，真实文本层 / OCR） | 65 |
| gapfill（按 qid 去重后的补跑） | 61 |
| ocr（页面 OCR 片段） | 11 |
| recover（恢复批次） | 2 |
| ocr_pilot / ocr_pilot2 | 2 / 8 |
| **合计** | **500** |

### 5.2 判断分布

| 指标 | 数值 |
|---|---|
| `relevant=True` | 193 / 500 = 38.6% |
| `self_contained=True` | 283 / 500 = 56.6% |
| `sufficient=True` | 37 / 500 = 7.4% |
| `next_action`：search / expand / answer | 296 / 169 / 35 |
| 可答题（381）动作 | search 215 / expand 135 / answer 31；`sufficient=True` 34 |
| 不可答题（119）动作 | search 81 / expand 34 / answer 4；`sufficient=True` 3 |
| **判断准确率** | **不可计算（人工 confirmed = 0）** |

500 条记录全部为 `jev-1.14.0`，含 `run_at`、输入/输出 token 与完整 `state_sent`；qid 与 snippet_id 均无重复，累计 token 为输入 660,142 / 输出 47,000。

### 5.3 缺口补齐说明

早期快照缺 65 题（53 道模型拒答 + 12 道候选页无可用文本）。round3 批次以真实文本层与 OCR 构造片段补跑，65 次调用全部成功，未覆盖任何历史预测；Jev 覆盖率最终为 **500/500**。补齐片段的语义质量不代表人工指定证据，相关限制见第 9 节。

---

## 6. 三模块交集与公平比较

| 交集 | 数量 |
|---|---|
| QA 成功 ∩ Jev 成功 | 495 |
| Jev 成功 ∩ 路由可评（379 口径） | 379 |
| QA ∩ 路由可评 ∩ Jev | **375** |

不得做的事：

- 不得用 495（QA）作分母宣称端到端准确率；
- 不得把 500（Jev 执行覆盖）当作 Jev 准确率分母；
- 不得把路由 Hit@3 与 QA 严格准确率加权成单一分数。

---

## 7. 未覆盖原因汇总

| 原因 | 题数 | 影响模块 | 是否可修复 |
|---|---|---|---|
| QA 页码/PDF/审核失败 | 5 | QA | Q0134/Q0373/Q0383 可通过页码修正重跑；Q0272 受内容审核；Q0290 PDF 不完整 |
| 可答过度拒答 | 4 | 路由 | 属于模型行为，保留并如实报告 |
| 可答但无 gold 页 | 2 | 路由 Page Recall 分母 | 不进入分母，不按错误计 |
| Jev 未执行 | 0 | Jev | 早期 65 题缺口已由 round3 批次补齐 |

---

## 8. 可复现命令

```powershell
# 环境
$env:DASHSCOPE_API_KEY = "<key>"     # QA / 路由
$env:TYPESAFE_API_KEY  = "<key>"     # Jev

# 路由补跑（付费；已成功 qid 自动跳过）
py -3.12 fill_gaps_500.py --run-route-fill --route-limit 25

# 离线评估与片段重建（免费）
py -3.12 fill_gaps_500.py --eval-route-fill
py -3.12 fill_gaps_500.py --build-jev
py -3.12 fill_gaps_500.py --dry-run-jev 5
py -3.12 precheck_leakage_500.py

# Jev 补跑（付费；按 qid 去重）
py -3.12 fill_gaps_500.py --run-jev --jev-limit 20

# 汇总
py -3.12 fill_gaps_500.py --report
```

---

## 9. 本报告不宣称的内容

1. 不宣称 500 题全部完成有效评测：QA 495、路由 Page Recall 376（严格 379）、Jev 500 仅表示执行覆盖。
2. 不宣称 Jev 准确率：无人工标注。
3. 不宣称端到端检索问答性能：QA 使用 gold 证据页。
4. 不宣称路由拒答能力已达上限：不可答拒答率 55.5%，仍有 44.5% 未拒答。
5. 不宣称缺口补齐片段的语义质量等同人工指定证据；其判断分布只用于描述模型行为。
