# JevSearch 500 题实验最终验收报告

生成时间：2026-11-12
核验方式：全部指标由 `verify_final_acceptance_500.py` 从原始 JSONL / CSV 独立重算，不引用既有 Markdown 结论
付费调用：0（仅离线核验）

> **公开版说明**：逐题原始预测、文档目录（TOC）与证据片段因 MMLongBench-Doc 数据再分发许可不明确，未随本仓库发布；报告中引用的原始 JSONL 仅保存在私有研究仓库。公开版保留验收方法、全部聚合指标与不含文本的逐题状态表 `outputs/final500/audit/final500_status_500.csv`。

---

## 1. 验收结论

| 验收项 | 结论 | 依据 |
|---|---|---|
| 500 道 Jev 调用能否独立验证 | **可以** | 500 条唯一 qid、500 条真实返回记录，字段与片段全部可回溯 |
| 是否真实 API 返回 | **是** | 全部 `model=jev-1.14.0`，含 `run_at`、输入/输出 token、完整 `state_sent` |
| 重复调用 / 重复计费 | **无** | 每个 qid 恰好 1 条成功记录，0 失败记录 |
| 原始预测是否被覆盖 | **否** | 8 个原始结果文件 SHA-256 与运行前基线完全一致 |
| gold 是否进入 Jev 输入 | **否** | `selection_uses_gold=True` 为 0；输入仅 4 个字段 |
| 3 道 sufficient=True 异常 | **已定位原因** | 检索片段含表面相关实体/数字，属模型判断偏差，非接口错误 |
| 最终指标是否可信 | **可信，但需带口径说明** | 覆盖率/召回可复算；Jev 准确率因无人工标签不可计算 |
| 是否可提交评审 | **可以提交评审** | 数据、代码、报告、局限均已就绪；准确率结论仍需人工标注 |

---

## 2. 独立核验方法

核验脚本：[`verify_final_acceptance_500.py`](verify_final_acceptance_500.py)

```powershell
py -3.12 verify_final_acceptance_500.py
```

执行输出（本次实际运行）：

```
[JEV] success=500 unique=500 dup=0 fail=0 untraceable=0 gold=0
[QA ] success=495/500 strict=0.6242 lenient=0.7414
[RTE] exec=500 fill_ok=56 abstain=70 eval376_hit@3=221 pr@3=0.5350
[HASH] changed=none
[OUT] 已写入 outputs/final500/audit/ACCEPTANCE_CHECKS_500.json
```

机器可读明细：`outputs/final500/audit/ACCEPTANCE_CHECKS_500.json`。

核验不依赖 `README.md` 或 `FINAL_500_EXPERIMENT_REPORT.md` 中的数字，全部从
`outputs/final500/run500/{qa,toc,jev}/` 的原始 JSONL 重算。

---

## 3. 500 条 Jev 记录的证据链

### 3.1 记录完整性

| 检查项 | 结果 |
|---|---|
| 成功记录数 | 500 |
| 唯一 qid | 500（Q0001–Q0500 全覆盖，缺失 0） |
| 重复 qid | 0（每 qid 恰好 1 次成功调用） |
| 失败记录 | 0 |
| 缺失 model / token / run_at / state_sent | 0 / 0 / 0 / 0 |
| 非法 `next_action` | 0 |
| 模型标识 | `jev-1.14.0` × 500 |
| 累计 tokens | 输入 660,142 / 输出 47,000 |
| 平均单次延迟 | 0.422 s |

### 3.2 输入可回溯

每条记录的 `state_sent` 中的证据文本，均能在已保存的片段文件中逐字找到（0 条不可回溯），
且 `page`、`doc_id`、`qid` 与片段文件一致。

| 片段来源 | 条数 |
|---|---:|
| title | 263 |
| route（正式版为 `snippets_route_500_v3.jsonl`） | 88 |
| round3（本次补跑） | 65 |
| gapfill | 61 |
| ocr | 11 |
| ocr_pilot（含 pilot1/pilot2） | 10 |
| recover | 2 |
| **合计** | **500** |

说明：路由预测在 11-08 20:14 生成，实际使用的是 `snippets_route_500_v3.jsonl`；早期片段文件为同批次的旧版本，
因此以 `state_sent` + v3 片段双重回溯，500 条全部可追溯。

### 3.3 输入字段白名单

`state_sent` 仅包含以下内容：

```
doc_type: <文档类型>
page: <页码>
Question: <问题>
Evidence snippet:
<证据文本>
```

未出现 `offline_only`、`gold_answer`、`evidence_pages`、`evidence_sources`、`human_*` 等字段。

---

## 4. 三道 sufficient=True 异常分析

三题数据集标注均为 `Not answerable`，`evidence_pages=[]`、`evidence_sources=[]`，
因此不存在 gold 泄漏；异常来自检索片段与问题主题的**表面匹配**。

### Q0114
- 问题：*What is the name of the governor as mentioned on the last page of the document?*
- 送入片段（第 1 页）：`Florida Department of Health ... Rick Scott GOVERNOR Celeste Philip, MD, MPH ...`
- 模型输出：`relevant=0.97`、`self_contained=0.85`、`sufficient=0.93`、`answer`（0.88）
- 原因：封面页确实出现 "Rick Scott / GOVERNOR"。模型只校验"片段里有没有州长名字"，
  未处理问题中的"**last page**"约束；数据集因此判为不可答。
- 性质：检索命中封面 + 模型忽略页码约束，属判断偏差。

### Q0363
- 问题：*How many offices dose PWC Technic have?*
- 送入片段（第 15 页）：`With offices in 155 countries and more than 284,000 people ...`
- 模型输出：`relevant=0.68`、`sufficient=0.63`、`answer`（0.46，但 expand 0.17 / search 0.37）
- 原因：片段含 "offices in 155 countries"，是"办公室/网点"的近似表述；
  模型把"覆盖国家数"误当成"办公室数量"，且实体 "PWC Technic" 在文档中并不存在。
- 性质：弱证据 + 实体不存在，模型在低置信度下仍选择 answer。

### Q0385
- 问题：*In the Ranking Prompt Example, what is the correct type of the aircraft provided?*
- 送入片段（第 21 页）：`RAR ... Please play the role of a classification expert ... '2012 Aston Martin V8 Vantage Convertible' ...`
- 模型输出：`relevant=0.73`、`self_contained=0.40（False）`、`sufficient=0.69`、`answer`（0.67）
- 原因：该页确实是 "Ranking Prompt Example"，但示例对象是**汽车**而非飞机；
  模型因"示例位置匹配"判为充分，同时 `self_contained=False` 也说明它意识到上下文不完整。
- 性质：位置匹配替代了实体匹配，模型自相矛盾但仍输出 answer。

三例均保留原始模型输出，未做任何改写。它们说明 Jev 的 `sufficient` 判断对
"限定词（last page）/ 实体（PWC Technic）/ 对象类型（aircraft）"不敏感，可作为后续改进点。

---

## 5. 指标重算

### 5.1 指标分层口径

| 层级 | 含义 | 是否为本报告主指标 |
|---|---|---|
| 执行覆盖率 | 实际送入模块的题数 / 500 | 是（描述执行完整性） |
| API 成功率 | 成功返回 / 已尝试 | 是（描述接口稳定性） |
| 有效评测覆盖率 | 能计算该模块指标的题数 / 500 | 是（描述可评测范围） |
| 准确率 | 与 gold 答案一致的比例 | 仅 QA 可算 |
| 召回率 | 路由命中 gold 证据页的比例 | 仅路由可算 |
| 拒答表现 | 不可答题正确拒答、可答题过度拒答 | 是（单独报告） |

### 5.2 DeepSeek 问答（oracle gold 证据页）

| 指标 | 结果 |
|---|---|
| 执行 | 500 / 500 |
| API 成功 | 495（99.0%） |
| 失败 | 5：Q0134、Q0290、Q0373、Q0383（`no_images`，页码/PDF 渲染异常）、Q0272（`api_error`） |
| 严格准确率（495） | 309 / 495 = **62.42%** |
| 宽松准确率（495） | 367 / 495 = **74.14%** |
| 严格 / 宽松（500 分母，失败记错） | 61.80% / 73.40% |
| 可答题 376 题严格 / 宽松 | 52.13% / 67.55% |
| 不可答题拒答正确率 | 113 / 119 = **94.96%** |
| 无 gold 页可答题 | 2 题（Q0290、Q0372），均未答对 |
| tokens | 863,966 入 / 27,647 出 |

**重要限制**：问答输入直接使用数据集 gold `evidence_pages` 渲染的页面图片，
属于"已知证据页的问答上界"，不是端到端检索结果；gold answer 未发送给模型
（`offline_only` 仅用于离线评分）。空证据页按既有协议回退第 1 页。

### 5.3 目录路由（Question + TOC，可拒答）

| 指标 | 结果 |
|---|---|
| 执行 | 500 / 500（原生 374 + 补跑 56 有效 + 70 拒答） |
| API 成功 | 500 / 500 |
| 拒答 | 70 题：不可答 66 / 119（55.46%）；可答过度拒答 4 题（Q0060、Q0134、Q0324、Q0372） |
| 可算 Page Recall | 376 题（可答且 gold 页可用、未拒答） |
| Section Hit@1 | 186 / 376 = **49.47%** |
| Section Hit@3 | 221 / 376 = **58.78%** |
| Page Recall@1（宏平均） | **0.4207** |
| Page Recall@3（宏平均） | **0.5350** |
| 完整证据页召回@3 | 182 / 376 = 48.40% |
| 严格口径（379，可答拒答记 0） | Hit@3 58.31%、PR@3 0.5308、Full@3 48.02% |
| tokens / 平均延迟 | 617,856 / 0.903 s |

`FullEvidence@3` 与 `Page Recall` 以 gold `evidence_pages` 列表长度为分母。Q0305 的 gold 标注为同一页重复 6 次
（`[6,6,6,6,6,6]`），按列表口径该题 PR@3 = 1/6、Full@3 未覆盖；按页码去重口径则为 1.0 与覆盖。
上文统一采用原始实现口径（列表口径）：376 主口径 182/376 = 48.40%，379 严格口径 182/379 = 48.02%。
去重口径下对应 48.67% / 48.28%（PR@3 0.5373 / 0.5330），差异仅由 Q0305 一题造成，两种口径不混用。

对照 baseline（374 题历史口径，仅参考）：
- 章节标题词匹配：Hit@3 28.61%、PR@3 0.2329；
- 同候选页预算 baseline：Hit 42.78%、PR 0.3637、完整召回 30.48%、平均 8.23 页。

两组候选页预算并不严格相等，**不构成完全公平的等预算对比**。

### 5.4 Jev 证据判断

| 指标 | 结果 |
|---|---|
| 执行覆盖率 | **500 / 500 = 100%** |
| 有效判断数（四字段完整） | **500**（`relevant`、`self_contained`、`sufficient`、`next_action`） |
| `relevant=True` | 193（38.6%） |
| `self_contained=True` | 283（56.6%） |
| `sufficient=True` | 37（7.4%） |
| 动作分布 | `answer` 35、`expand` 169、`search` 296 |
| 可答题（381） | `answer` 31、`expand` 135、`search` 215；`sufficient=True` 34 |
| 不可答题（119） | `search` 81、`expand` 34、`answer` 4；`sufficient=True` 3 |
| 合理拒答（不可答判证据不足） | **116 / 119 = 97.48%** |
| 判断准确率 | **不可计算**（无人工 confirmed 标签） |

> `action_normalized` 仅在后期批次写入，早期存在字段缺失，因此统一以原始 `next_action` 统计；
> 不把归一化结果当作 Jev 原始预测，也不据此计算"准确率"。

### 5.5 三模块交集

| 交集 | 题数 |
|---|---:|
| Jev 500 ∩ QA 成功 495 | 495 |
| Jev 500 ∩ 路由可评 379 | 379 |
| Jev ∩ QA ∩ 路由可评 | **375** |

---

## 6. gold 泄漏、重复与覆盖核查

### 6.1 gold 是否进入模型输入

| 模块 | 是否使用 gold | 说明 |
|---|---|---|
| Jev 判断 | 否 | 输入仅 4 字段；`selection_uses_gold=True` 计数 0 |
| 目录路由 | 否 | 输入 Question + TOC 文本；gold 仅在预测完成后用于评分 |
| DeepSeek 问答 | **是（oracle 证据页）** | 按既有协议使用 gold `evidence_pages` 渲染图片；gold answer 不发送 |

Jev 片段中答案文本自然出现的情况：

- 61 / 500 条片段包含 gold 答案字符串；
- 其中 45 条答案长度 ≤ 2 字符（yes/no/数字），属无意义撞串；
- 长度 > 2 的 16 条中，11 条所在页本身就是 gold 页（正确检索），
  5 条属跨页重复表述的巧合（Q0481、Q0275、Q0049、Q0082、Q0170）；
- 任何一条的选页过程都未读取 gold（`selection_uses_gold` 全为 False）。

检索页恰好命中 gold 页的比例为 95 / 384（24.7%），是检索效果指标，不是泄漏。

### 6.2 重复与覆盖

- 500 个唯一 qid，0 重复、0 缺失；
- 每个 qid 仅 1 次成功 Jev 调用，无重复计费记录；
- 路由 500 / 500 已执行，Jev 500 / 500 已执行；
- 500 条记录由 435 条历史记录与 65 条补充记录组成，两组 qid 无重叠。

### 6.3 原始文件未被覆盖

以下文件与 `pre_run_check.json` 记录的 SHA-256 完全一致（`changed=none`）：

- `ds_results.jsonl`
- `routes.jsonl`
- `jev_predictions_title/route/ocr/recover/ocr_pilot/ocr_pilot2_500.jsonl`

补充批次只写入 `jev_predictions_round3_500.jsonl` 与 `snippets_round3_500.jsonl`，未改动历史预测。

公开副本为遵守数据集再分发限制，已移除 `ds_results.jsonl`、`routes.jsonl` 与 `snippets_*.jsonl` 中的 gold 字段（模型预测与统计未改动），因此这两个文件的哈希与本节记录不同；公开副本的基线见 `public_files_sha256.json`。

---

## 7. API 费用

| 项目 | 次数 | tokens（入 / 出） | 费用 |
|---|---:|---:|---:|
| Jev 全量 | 500 | 660,142 / 47,000 | 约 **1.50 元**（0.003 元/次实测均值） |
| 其中补充批次 | 65 | 81,528 / 6,110 | 约 **0.195 元** |
| 目录路由 | 500 | 617,856（合计） | 接口未返回单价，无法精确回算 |
| DeepSeek 问答 | 495 成功 | 863,966 / 27,647 | 接口未返回单价，无法精确回算 |

说明：
- Jev 单价采用项目历史账单实测均值 0.0030 元/次，**不是官方报价**；
- 不把不同模块的 token 直接折算成一个"总费用"数字，避免虚构单价。

---

## 8. 实验限制

1. **Jev 准确率不可计算**：`outputs/final500/audit/annotation_sample_quality30_manifest.csv` 只有 30 题候选清单，
   没有 confirmed 标签。本报告只报告执行覆盖率、判断分布与拒答表现。
2. **三模块不可合成总分**：QA 输入 gold 页，路由输入 TOC，Jev 输入真实片段，
   任务性质、输入条件与分母均不同，只能并排描述。
3. **QA 是 oracle 上界**：不代表端到端检索问答能力。
4. **路由 376 口径不含可答拒答**：若把 3 道可答拒答记 0（379 口径），
   Hit@3 为 58.31%、PR@3 为 0.5308，略低于主口径。
5. **路由 baseline 预算不对等**：标题匹配、同预算 baseline 与 LLM 路由的候选页数不同。
6. **5 道 QA 失败**：4 道页面渲染失败、1 道 API 错误，已按失败计错，未静默剔除。
7. **2 道可答题无 gold 页**（Q0290、Q0372），不进入 Page Recall 分母。
8. **Jev 三例误判**：限定词/实体/对象类型不敏感，说明 `sufficient` 判断仍有改进空间。
9. **OCR 覆盖有限**：仅用于缺文本页补充，未做全量扫描件验证。

---

## 9. 复现方法

```powershell
# 1. 零成本独立验收（本报告全部数字）
py -3.12 verify_final_acceptance_500.py

# 2. 状态与缺口检查（不调用 API）
py -3.12 fill_gaps_round3_500.py --status

# 3. 片段离线自检（不调用 API）
py -3.12 fill_gaps_round3_500.py --dry-run 5

# 4. 重跑离线审计（不调用 API）
.venv_paddleocr\Scripts\python.exe audit_final500.py
```

历史全流程复现（会产生费用）：

```powershell
py -3.12 run_experiment_500.py --preflight --build
py -3.12 run_experiment_500.py --run
```

环境变量：`DASHSCOPE_API_KEY`（问答/路由）、`TYPESAFE_API_KEY`（Jev），均从环境变量读取，仓库不含密钥。

---

## 10. 提交范围与安全检查

### 10.1 提交范围

| 类别 | 内容 |
|---|---|
| 文档 | `README.md`、`FINAL_ACCEPTANCE_REPORT_500.md`、`FINAL_500_EXPERIMENT_REPORT.md`、`EXPERIMENT_REPORT.md`、`requirements.txt` |
| 实验代码 | 数据准备、TOC/OCR、路由、Jev 判断、问答、复算与验收脚本 |
| 真实调用记录 | `outputs/final500/run500/{qa,toc,jev}/` 下的预测、路由与证据片段 |
| 可发布统计 | 路由 374 题指标、`route_fill_metrics.json`、`final500_stats.json`、`ACCEPTANCE_CHECKS_500.json`、逐题状态表 |
| 审计证据 | `pre_run_check.json`、`public_files_sha256.json`、`manifest.json`、补跑统计 |

历史快照与过程性文档（`FULL_REPORT.md`、`compare/summary.*`、`final_summary.json`、`preflight.json`、`final500_module_matrix.md`、`ROUND3_COMPLETION_500.md`、`final_gap_77.csv`、`coverage_final_*`）保留在本地工作区，但已从提交清单与 `.gitignore` 中排除，不作为公开交付物。

### 10.2 安全检查

| 检查 | 结果 |
|---|---|
| API Key 硬编码 | 交付前扫描发现 `run_baseline.py`、`run_advanced_rag.py` 含历史硬编码密钥，已改为仅从 `DASHSCOPE_API_KEY` 读取；该密钥曾存在于旧仓库，**建议在控制台作废并重建** |
| `sk-` / `ts-` / `Bearer` 长串扫描 | 当前提交清单无真实密钥 |
| `.env` | 已被 `.gitignore` 排除 |
| PDF | 未纳入索引；仓库不含原始 PDF |
| 数据集原始样本 | `sampled_500_for_baseline.jsonl` 未被 Git 跟踪 |
| gold 字段 | 预测、路由、Jev 片段与逐题指标均不含 gold answer / gold evidence_pages；小样本 `outputs/task6/predictions*.jsonl` 已移除 `answer`、`answer_format` |
| 虚拟环境 / 缓存 | `.venv_paddleocr/`、`__pycache__/`、页面渲染图、中间输入文件均被 `.gitignore` 排除 |

### 10.3 提交前仍需注意

1. 公开前需确认 MMLongBench-Doc 数据许可与引用要求；当前仓库保持 Private；
2. 证据片段含文档原文与少量第三方联系信息，仅用于结论核验，不宜公开再分发；
3. 建议提交前执行一次 `git status` 人工复核。

---

## 11. 离线回归测试（0 次付费调用）

修复后重新执行四个脚本，全部只读取本地文件：

| 脚本 | 命令 | 结果 |
|---|---|---|
| 独立验收 | `py -3.12 verify_final_acceptance_500.py` | 上节输出，`[HASH] changed=none` |
| 基线对比 | `py -3.12 compare_results.py` | 严格 26/317，宽松 29/377，McNemar 352/4、χ²=338.23 |
| 9 题路由复算 | `py -3.12 task6_route_and_eval.py --reuse` | summary 与 `outputs/task6/evaluation.json` 完全一致 |
| 500 题路由复算 | `py -3.12 task6_route_and_eval_500.py --reuse` | summary 与 `outputs/final500/run500/toc/metrics.json` 完全一致 |

缺少本地 gold 时，四个脚本均明确提示并跳过相应指标（验收脚本不写文件，`compare_results.py` 报错退出），
不会输出 0%/100%，也不会改写已有结果。`--reuse` 默认不把 gold 写回预测或评价文件；如需本地审计副本，
显式加 `--record-gold`。

---

## 12. 最终结论

- 500 条 Jev 记录可以被独立验证为真实 API 返回：模型标识、时间戳、token、输入文本与片段来源全部可追溯，0 重复、0 失败、0 缺失。
- 三例 `sufficient=True` 异常均由"片段表面匹配"导致，不涉及 gold 泄漏，也未修改模型输出。
- QA、路由指标可复算且口径明确；Jev 的覆盖率可信，但**准确率在人工标注完成前不可宣称**。
- 当前材料可以提交评审；若要写入论文或对外报告"准确率"，需先完成 Jev 人工标注。
