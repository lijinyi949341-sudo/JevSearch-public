# 三模块评估口径统一规范（500 题）

版本：v2（2026-11-12，Jev 缺口补齐后修订）
适用范围：`sampled_500_for_baseline.jsonl` 的 Q0001–Q0500
目的：把“覆盖率、调用成功率、拒答正确率、检索质量、判断准确率”拆成互不混用的指标。

> v1（2026-11-10）中的 374 / 315 分母对应 Jev 缺口补齐前的状态，已失效。
> 最终权威数字以 `FINAL_ACCEPTANCE_REPORT_500.md` 与 `ACCEPTANCE_CHECKS_500.json` 为准。

---

## 1. 固定统计集合

所有报告必须同时给出以下集合，且分母写清楚：

| 集合 | 定义 | 数量 |
|---|---|---:|
| `all_500` | 全部题目 | 500 |
| `qa_success_495` | QA 接口成功返回 | 495 |
| `answerable_381` | 数据集 `answer != "Not answerable"` | 381 |
| `unanswerable_119` | 数据集 `answer == "Not answerable"` | 119 |
| `route_native_374` | 路由原生成功且可评（无拒答、gold 页可用） | 374 |
| `route_main_376` | 374 + 2 道补跑成功的可答题（Q0373、Q0383） | 376 |
| `route_strict_379` | 374 + 5 道修正 gold 题，3 道可答拒答记 0 | 379 |
| `common_all_three` | QA 成功 ∩ 路由 379 ∩ Jev 有判断 | 375 |

禁止只报一个“500 题准确率”而不说明分母。

---

## 2. 五类指标，不得合并

| 指标族 | 定义 | 适用模块 | 分母 |
|---|---|---|---|
| 覆盖率 Coverage | 该模块有正式记录/输入的题目比例 | QA、路由、Jev | 对应集合的 n |
| 调用成功率 Call Success | 模型/接口返回可用结果的比例 | QA、路由、Jev | 实际尝试的题数 |
| 拒答正确率 Refusal Correctness | 不可答题输出 `Not answerable` 的比例 | **仅 QA** | 不可答题数 |
| 检索质量 Retrieval Quality | Section Hit / Page Recall / Full Evidence Recall | **仅路由** | 可评题数 |
| 判断质量 Judgment Quality | relevant / sufficient / next_action 与人工标签一致 | **仅 Jev** | 已 confirmed 标注数 |

**禁止**把 QA 回答准确率、路由召回率、Jev 动作一致率加权成一个“总分”。

---

## 3. 各模块处理不可回答题的规则

### 3.1 DeepSeek 问答

- 输入：question + 证据页（当前历史口径使用 gold evidence_pages，必须在报告中标注为 oracle-evidence）。
- 对 119 道不可答题：**全部作答**，gold 为 `Not answerable`。
- 指标：
  - 覆盖率 = 500/500；
  - 调用成功率 = 495/500（5 题失败：Q0134、Q0272、Q0290、Q0373、Q0383）；
  - 拒答正确率 = 113/119 = 94.96%；
  - 可答题回答准确率 = strict 196/376 = 52.13%、lenient 254/376 = 67.55%（376 = 381 可答 − 5 题失败）。
- 禁止把“拒答正确率”并入“回答准确率”。

### 3.2 目录路由

- 输入：question + TOC（仅标题/页码，不含正文、答案、gold 页）。
- 对 119 道不可答题：**统一执行可拒答协议**，由模型自行输出 `answerable=false`，不使用数据集标签选题。
- 指标：
  - 执行覆盖率 = 500/500（原生 374 + 补跑 126）；
  - 调用成功率 = 500/500；
  - 拒答 = 70 题：不可答正确拒答 66/119 = 55.46%，可答过度拒答 4/381 = 1.05%；
  - Page Recall 主口径 n = 376（可答、gold 页可用、未拒答）；严格 379 口径把 3 道可答拒答记 0。
  - Page Recall 与 FullEvidence 以 gold `evidence_pages` 列表长度作分母；Q0305 的 gold 为同一页重复 6 次，列表口径下 PR@3 = 1/6、Full@3 未覆盖，去重口径下为 1.0 与覆盖。正式报告统一使用列表口径，并在注释中给出去重口径数值。
- 禁止把“路由拒答 119 道中的 66 道”写成“路由失败了 119 道”。

### 3.3 Jev 上下文判断

- 输入：question + 候选片段（title / route / OCR / recover / 试点 / gapfill / round3），不含 gold。
- 对不可答题：119 道全部有真实片段与真实判断（不构造占位文本，不读取 gold）。
- 指标：
  - 覆盖率 = 500/500；
  - 判断分布：`relevant=True` 193、`self_contained=True` 283、`sufficient=True` 37；
  - `next_action`：search 296 / expand 169 / answer 35；
  - Jev 不输出答案，因此**不计算回答准确率或拒答正确率**；
  - 判断质量只在人工 confirmed 标注后计算（当前 0 条，报 N/A）。

---

## 4. 统一后的推荐报告口径

| 模块 | 覆盖率 | 调用成功率 | 任务指标 | 不可答题处理 |
|---|---:|---:|---|---|
| QA | 500/500 | 495/500 | strict/lenient 回答准确率 | 全部作答，单独报拒答正确率 113/119 |
| 路由 | 500/500 | 500/500 | Hit@1/@3、PR@1/@3、Full Recall（n=376，严格 379） | 可拒答协议，单独报拒答与过度拒答 |
| Jev | 500/500 | 500/500 | 判断分布；标注后报一致率/F1 | 119 道全部有真实判断，单独报告 |

---

## 5. 数据与产物

- 逐题状态：`outputs/final500/audit/final500_status_500.csv`（500 行，Jev 500/500）
- 机器可读验收：`outputs/final500/audit/ACCEPTANCE_CHECKS_500.json`
- 最终统计摘要：`FINAL_ACCEPTANCE_REPORT_500.md`
- 缺口判定（补跑前历史记录）：`outputs/final500/audit/JEV_GAP_CLASSIFICATION.md`
- 页码修复：5 道题的物理页修正表含 gold 页码，未随仓库发布；需要时从数据集重新核对后本地重建
- Jev 标注规范：`outputs/final500/audit/JEV_ANNOTATION_SPEC.md`

---

## 6. 禁止事项

1. 不得把 gold 答案、gold 证据页作为模型输入；gold 只可用于离线判分/评估标签。
2. 不得把“OCR 识别成功”写成“证据相关”或“证据足够”。
3. 不得把“Jev 调用成功”写成“Jev 判断准确”。
4. 不得把 500/500 执行覆盖写成“500 题全部评测成功”，也不得据此宣称 Jev 准确率。
5. 不得覆盖原始预测；所有修复与新统计写入独立文件。
