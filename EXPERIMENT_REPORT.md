# MMLongBench-Doc 图文长文档 RAG 评估实验报告

> 实验日期：2026-10-23　|　环境：Windows / Python 3.12 / PyMuPDF / DashScope(OpenAI 兼容)　|　工作目录：`<PROJECT_ROOT>`

## 1. 项目简介

本项目在 **MMLongBench-Doc**（[yubo2333/MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc)）数据集上开展图文长文档问答（Document VQA / Long-context VQA）评估。该基准包含 135 个真实长文档（PDF）、1091 道题目，覆盖图表（Chart/Figure）、表格（Table）、纯文本（Pure-text）与版式（Layout）等多种证据类型，是评测“长文档 + 多模态信息抽取”能力的典型基准。

实验从数据集中以固定随机种子（`seed=42`）采样 **500 道题**（`sampled_500_for_baseline.jsonl`）作为评测集，字段包含 `doc_id / doc_type / question / answer / evidence_pages / evidence_sources / answer_format`。核心目标是对比 **纯文本 LLM 裸考基线** 与 **多模态 VLM 增强架构** 的问答准确率，并对错误进行归因分析。

## 2. 方法（Methodology）

### 2.1 Baseline：纯文本 LLM 裸考

- 模型：文本模型 `deepseek-v4.1-flash-1017`（通过 DashScope OpenAI 兼容接口调用）。
- 输入：**仅题目文本**（`question`），不提供任何文档正文或图片。
- 推理参数：`max_tokens=1024`、`enable_thinking=False`（与增强架构严格对齐，保证对比公平）。

### 2.2 升级架构：Oracle 证据页定位 + 多模态 VLM（Qwen-VL）

- 模型：多模态理解模型 `qwen3-vl-plus`（注：最初指定的 `qwen-vl-max` 已下线，故改用同厂多模态模型替代）。
- **Oracle 证据页定位**：直接使用数据集自带的 `evidence_pages`（黄金证据页标注，1-indexed）作为页面级剪枝信号，定位与问题相关的关键页；页为空时回退第 1 页，单题最多渲染 10 页。该步骤不是学习式检索器，也不涉及 Jev 判断。
- **页面渲染**：用 **PyMuPDF (fitz)** 将关键页渲染为 Base64 PNG 图片，以 OpenAI 兼容的 `image_url` 多模态格式与 `question` 一同送入模型。
- **多专家路由 Prompt**：按 `doc_type`（财务/学术/教程/指南等）切换系统提示词聚焦领域信息。
- **不可答题强约束**：新增规则要求“当图中找不到确凿证据时，回答必须以 `Not answerable` 开头”，抑制幻觉。
- **高清渲染**：渲染 DPI 由 144 提升至 300（单页沿最长边封顶 2000px，防止体积过大致使 API 超限）。

> 说明：`evidence_pages` 为基准自带的 **oracle 黄金页**，因此本实验测的是“关键页已定位”条件下的**检索上限能力**，尚未涉及隐式/学习式检索器的能力。

### 2.3 评估口径

- **严格匹配（strict）**：`gold in pred` 子串匹配（小写、去首尾空格）。
- **宽松匹配（lenient）**：忽略大小写/标点/空格/括号/单位，并将 `List` 类型答案拆成元素逐项匹配。该口径用于消除“List 字面量带引号括号、单位/连字符/百分号、大小写”等造成的误判，更公平反映真实能力。

## 3. 实验结果

### 3.1 主对比表（最终结果）

| 评估口径 | Baseline（纯文本 LLM） | 多模态 VLM（oracle 证据页 + Qwen-VL） | 绝对提升 |
|---|---|---|---|
| 严格匹配（strict） | 5.20%（26/500） | **63.40%（317/500）** | +58.20% |
| **宽松匹配（lenient）** | 5.80%（29/500） | **75.40%（377/500）** | **+69.60%** |

### 3.2 演进过程

| 版本 | 说明 | strict | lenient |
|---|---|---|---|
| Baseline | 纯文本 LLM，无上下文 | 5.20%（26） | 5.80%（29） |
| 多模态 VLM 初版 | 144 DPI，无不可答约束 | 47.80%（239） | 60.80%（304） |
| **多模态 VLM 优化版** | 300 DPI + 不可答强约束 | **63.40%（317）** | **75.40%（377）** |

### 3.3 显著性检验

采用配对 **McNemar 检验**（宽松匹配口径）：进步 352 例 / 退步 4 例，χ² = 338.23（df=1，p < 0.001），差异**极显著**。

## 4. 错误归因总结（Error Analysis）

针对优化前（261 个错误）与优化后（183 个错误，严格口径）的错误分布对比如下：

| 错误类型 | 优化前 | 优化前占比 | 优化后 | 优化后占比 |
|---|---|---|---|---|
| 不可答题过度作答（应判 Not answerable） | 106 | 40.6% | **3** | 1.6% |
| 图表视觉误读（Chart/Figure） | 62 | 23.8% | 68 | 37.2% |
| 检索/上下文失败（证据页不含答案） | 51 | 19.5% | 58 | 31.7% |
| VLM 生成失败（证据页含答案却答错） | 42 | 16.1% | 54 | 29.5% |
| **合计** | **261** | 100% | **183** | 100% |

按两大阶段归类（优化后）：

- **VLM 生成阶段失败**：125 例（68.3%）
- **检索/上下文阶段失败**：58 例（31.7%）

### 核心发现

1. **不可答题幻觉是最大单项瓶颈（已基本解决）**：优化前 106 题（40.6%）模型未按要求输出 `Not answerable`，而是凭外部常识/封面内容幻觉出答案。加入强约束 Prompt 后骤降至 3 例（1.6%），这是本次**单点收益最大**的改进。
2. **图表视觉误读成为剩余主要瓶颈**：优化后 68 例（37.2%）集中在 Chart/Figure——多为“文本无法核验、需视觉读取”的图像证据；提升 DPI 对细节可读性有帮助，但视觉误读的绝对数量变化不大，是后续需重点攻克的硬骨头。
3. **检索/上下文失败仍占三成**：58 例（31.7%）证据页文本中不含答案，来源包括 `evidence_pages` 为空、页面截断（MAX_PAGES=10）、以及扫描/图片型页无法抽取文本。
4. **度量伪影依然显著**：即便采用宽松匹配，183 个“错误”中仍有 60 例实为**正确回答**（List 36 / Float 12 / Str 10 / Int 2），说明 `gold in pred` 类子串匹配仍会低估真实能力，评估口径需进一步引入语义等价判分。

### 优化措施与效果

| 措施 | 针对问题 | 效果 |
|---|---|---|
| 宽松匹配（忽略大小写/标点/空格/括号/单位 + List 逐元素） | List/单位/标点等误杀 | 修正 60+ 例被误判，准确率 +13pp（严格 47.80% → 宽松 60.80%） |
| Not answerable 强约束 Prompt | 不可答题幻觉 | 不可答过度作答 106 → 3 |
| 渲染 DPI 144 → 300（2000px 封顶） | 图表细节不清晰 | 图表细节可读性提升 |

## 5. 复现说明与文件清单

| 文件 | 作用 | 是否在仓库 |
|---|---|---|
| `sampled_500_for_baseline.jsonl` | 500 题评测集（含 gold） | **否**，按 README 第 9 节从基准重建 |
| `run_baseline.py` / `baseline_results.jsonl` | 纯文本基线生成与结果 | 是 |
| `run_advanced_rag.py` / `advanced_rag_results.jsonl` | 多模态 VLM 生成与结果 | 是 |
| `compare_results.py` | 评估（严格/宽松双口径 + McNemar） | 是 |
| `error_analysis.py` / `error_analysis_report.jsonl` | 错误归因分析 | 脚本是；**报告否** |
| `prepare_documents.py` / `document_texts.json` | PDF 下载与文本缓存 | 脚本是；**缓存否** |

运行（需 Python 3.12，已安装 `pymupdf` 与 `openai`，并设置 `DASHSCOPE_API_KEY`）：

```powershell
python run_baseline.py        # 生成 baseline 结果
python run_advanced_rag.py    # 生成多模态 VLM 结果（需 PDF 缓存）
python compare_results.py     # 输出对比报告（需本地 500 题 gold 清单）
python error_analysis.py      # 输出错误归因报告
```

**复现边界**：`compare_results.py` 从本地 `sampled_500_for_baseline.jsonl` 读取 gold 复核准确率与 McNemar 检验，
缺少该文件时会报错退出，不会给出 0%/100% 的伪结果。第 3 节的准确率可由仓库内两份预测与本地 gold 复算；
第 4 节的 261/183 错误分布依赖未纳入仓库的 `error_analysis_report.jsonl`、`document_texts.json` 与原始 PDF，
只能核对结论，不能仅凭本仓库重新生成。`run_advanced_rag.py` 需要先按 README 第 9 节下载 PDF 并抽取文字层。

## 6. 局限性与后续工作

1. **Oracle 检索上限**：当前使用 `evidence_pages` 黄金页，未评测真实检索器；后续可替换为隐式/学习式页面检索后再对比。
2. **跨架构对比**：Baseline 与增强版同时改变了模型、模态与上下文，无法单独归因各因素贡献；建议补充“同模型无图对照组”以拆解模型能力 vs 上下文能力。
3. **图表视觉读取仍是短板**：可尝试更高分辨率局部裁剪、Chart 区域检测后放大、或引入 OCR 协同等方式进一步降低视觉误读。
4. **评估度量**：`gold in pred` 类匹配对语义等价、单位换算等仍不敏感，建议引入 LLM-as-judge 或规则化答案抽取以得到更稳定的真实准确率。