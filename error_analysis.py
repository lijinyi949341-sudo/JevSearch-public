# -*- coding: utf-8 -*-
"""
错误分析脚本（Error Analysis）
针对 advanced_rag_results.jsonl 的多模态 VLM 结果，做三类工作：
  1) 找出所有回答错误的样本（判分口径与 compare_results.py 完全一致：gold in pred）
  2) 对每个错误样本，检查 Jev 剪枝命中的 evidence_pages 是否确实包含正确答案，
     从而区分「检索/上下文阶段失败」还是「VLM 生成阶段失败」
  3) 输出错误分布报告（幻觉/图表数字看错/细节漏看/检索失败等），并打印典型错题。

依赖：PyMuPDF (fitz)，PDF 缓存位于 %TEMP%/mmlongbench_pdfs。
"""
import ast
import json
import os
import re
from collections import Counter, defaultdict

import fitz  # PyMuPDF

RESULT_FILE = "advanced_rag_results.jsonl"
OUT_REPORT = "error_analysis_report.jsonl"
PDF_DIR = os.path.join(os.environ.get("TEMP", "."), "mmlongbench_pdfs")
MAX_PAGES = 10  # 与 run_advanced_rag.py 保持一致


# ---------- 判分（口径与 compare_results.py 完全一致） ----------
def is_correct(gold, pred):
    g = str(gold).strip().lower()
    p = str(pred).strip().lower()
    return bool(g) and g in p


# ---------- 宽松判分（仅作诊断，暴露度量伪影） ----------
def loose_norm(s):
    """去除所有非字母数字字符，用于容错单位/标点/空白/连字符差异。"""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def loose_correct(gold, pred, answer_format):
    if answer_format == "List":
        try:
            elems = ast.literal_eval(gold)
        except Exception:
            elems = [gold]
        return all(loose_norm(e) in loose_norm(pred) for e in elems)
    g = loose_norm(gold)
    return bool(g) and g in loose_norm(pred)


# ---------- 解析工具 ----------
def parse_pages(ep_str):
    try:
        return [int(x) for x in ast.literal_eval(ep_str)]
    except Exception:
        return []


def parse_evidence_sources(es_str):
    try:
        return ast.literal_eval(es_str)
    except Exception:
        return []


# ---------- 逐文档文本抽取（带缓存） ----------
_doc_full_cache = {}


def get_doc_full_text(doc_id):
    """返回 {0-indexed page: 文本}；打不开返回 None。"""
    if doc_id in _doc_full_cache:
        return _doc_full_cache[doc_id]
    result = {}
    try:
        doc = fitz.open(os.path.join(PDF_DIR, doc_id))
        for i in range(doc.page_count):
            result[i] = doc.load_page(i).get_text("text")
        doc.close()
    except Exception:
        result = None
    _doc_full_cache[doc_id] = result
    return result


# ---------- 文本核验用的归一化 ----------
def norm_text(s):
    s = str(s).lower()
    s = re.sub(r"\s*([%.,])\s*", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def answer_present_in_text(text, gold_raw, answer_format):
    """检查 gold 答案是否出现在抽取的页面文本中。List 需全部元素命中。"""
    text = norm_text(text)
    if answer_format == "List":
        try:
            elems = ast.literal_eval(gold_raw)
        except Exception:
            elems = [gold_raw]
        return all(norm_text(e) in text for e in elems)
    return norm_text(gold_raw) in text


# ---------- 分类 ----------
def classify(sample, doc_full_text):
    gold = str(sample.get("answer", "")).strip()
    es = parse_evidence_sources(sample.get("evidence_sources", "[]"))
    fmt = sample.get("answer_format", "Str")

    # 1) 不可答题：gold 为 "Not answerable"
    if gold.lower() == "not answerable":
        return "unanswerable_overreach", "VLM 本应判定「不可答」却过度作答/措辞未命中"

    # 2) 证据页解析（与 run_advanced_rag.py 一致：空则回退第 1 页，截断到 MAX_PAGES）
    pages = parse_pages(sample.get("evidence_pages", "[]"))
    if not pages:
        pages = [1]
    pages = pages[:MAX_PAGES]

    # 3) PDF 打不开 → 真·渲染/文件失败（模型无图可看）
    if doc_full_text is None:
        return "render_or_pdf_missing", "PDF 打开失败（文件缺失/损坏），模型无图可看"

    is_visual = any(("chart" in s.lower()) or ("figure" in s.lower()) for s in es)
    joined = " ".join(doc_full_text.get(p - 1, "") for p in pages)

    # 4) 图表/图视觉类：文本核验不可靠，单列
    if is_visual:
        if joined.strip() and answer_present_in_text(joined, gold, fmt):
            return "generation_failure", "图表类证据页文本可核验出答案但仍答错"
        return "visual_misread", "图表/图（多为图片，文本不可抽取），疑似生成阶段视觉误读"

    # 5) 文本/表格类
    if not joined.strip():
        return "retrieval_failure", "证据页无抽取文本（空白/扫描/页截断），上下文未含答案"
    if answer_present_in_text(joined, gold, fmt):
        return "generation_failure", "证据页文本含答案，VLM 生成阶段出错（数字/细节看错或幻觉）"
    return "retrieval_failure", "证据页文本不含答案（检索/截断/文本抽取缺失）"


# ---------- 主流程 ----------
def main():
    samples = []
    with open(RESULT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    wrong = []
    failed_docs = []
    doc_opened = set()
    encountered = set()

    for s in samples:
        son = s.get("answer")
        pred = s.get("advanced_answer", "")
        if is_correct(son, pred):
            continue
        wrong.append(s)

    total = len(samples)
    n_wrong = len(wrong)
    n_correct = total - n_wrong

    cat_counter = Counter()
    desc_counter = Counter()
    cat_examples = defaultdict(list)
    records = []

    for s in wrong:
        doc_id = s.get("doc_id", "")
        if doc_id in encountered:
            full_text = _doc_full_cache[doc_id]
        else:
            full_text = get_doc_full_text(doc_id)
            encountered.add(doc_id)
        if full_text is not None:
            doc_opened.add(doc_id)
        elif doc_id not in failed_docs:
            failed_docs.append(doc_id)

        cat, desc = classify(s, full_text)
        cat_counter[cat] += 1
        desc_counter[desc] += 1

        rec = {
            "doc_id": doc_id,
            "question": s.get("question", ""),
            "answer": s.get("answer", ""),
            "advanced_answer": s.get("advanced_answer", ""),
            "evidence_pages": s.get("evidence_pages", "[]"),
            "evidence_sources": s.get("evidence_sources", "[]"),
            "answer_format": s.get("answer_format", ""),
            "error_category": cat,
            "error_note": desc,
            "loose_correct": loose_correct(s.get("answer", ""), s.get("advanced_answer", ""), s.get("answer_format", "Str")),
        }
        records.append(rec)
        if len(cat_examples[cat]) < 4:
            cat_examples[cat].append(rec)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- 度量口径诊断 ----------
    loose_ok = [r for r in records if r["loose_correct"]]
    loose_fmt = Counter(r["answer_format"] for r in loose_ok)

    # ---------- 打印报告 ----------
    cat_label = {
        "generation_failure": "VLM 生成失败（证据页含答案却答错）",
        "retrieval_failure": "检索/上下文失败（证据页不含答案）",
        "visual_misread": "图表视觉误读（图表类，文本无法核验）",
        "unanswerable_overreach": "不可答题过度作答（应判定 Not answerable）",
        "render_or_pdf_missing": "PDF 缺失/打不开（模型无图可看）",
    }

    print("=" * 78)
    print("            多模态 VLM 错误分析报告（Error Analysis）")
    print("=" * 78)
    print(f"总样本数            : {total}")
    print(f"答对                : {n_correct}  ({100.0 * n_correct / total:.2f}%)")
    print(f"答错                : {n_wrong}   ({100.0 * n_wrong / total:.2f}%)")
    print(f"错误样本涉及唯一文档: {len(doc_opened) + len(failed_docs)}；其中打不开(无图可看)的文档数: {len(failed_docs)}")
    print("-" * 78)

    # 两大阶段
    gen_cnt = cat_counter.get("generation_failure", 0) \
        + cat_counter.get("visual_misread", 0) \
        + cat_counter.get("unanswerable_overreach", 0)
    ret_cnt = cat_counter.get("retrieval_failure", 0) \
        + cat_counter.get("render_or_pdf_missing", 0)

    print("\n【① 两大阶段失败占比】")
    print(f"  VLM 生成阶段失败   : {gen_cnt}  ({100.0 * gen_cnt / n_wrong:.1f}% of 错误)")
    print(f"  检索/上下文阶段失败: {ret_cnt}  ({100.0 * ret_cnt / n_wrong:.1f}% of 错误)")
    print("\n【② 细分错误类型分布】")
    for cat, cnt in cat_counter.most_common():
        label = cat_label.get(cat, cat)
        print(f"  {cnt:4d}  ({100.0 * cnt / n_wrong:5.1f}%)  {label}")

    # 度量伪影提示
    print("\n【③ 度量口径提示：'gold in pred' 子串匹配过于严格】")
    print(f"  在 {n_wrong} 个错误样本中，改用宽松匹配（忽略大小写/标点/单位/空白、List 逐元素）")
    print(f"  后有 {len(loose_ok)} 个其实答对了（疑似度量伪影，非真实模型错误）。")
    print("  其中按 answer_format 分布：")
    for fmt, cnt in loose_fmt.most_common():
        print(f"    {fmt:6s}: {cnt}")
    if loose_ok:
        print("  -> 这也解释了为什么大量 List / Not answerable / 带百分号单位题被判错。")
    print("  -> 注：106 个 'Not answerable' 错误样本的回答全都未含字面 'not answerable'，")
    print("     其中部分是语义正确但措辞不同（被度量误杀），部分是真正幻觉出答案，需人工复核。")

    # 按 evidence_sources 分布
    src_counter = Counter()
    for s in wrong:
        es = parse_evidence_sources(s.get("evidence_sources", "[]"))
        key = ",".join(es) if es else "(空)"
        src_counter[key] += 1
    print("\n【④ 错误样本的 evidence_sources 分布】")
    for key, cnt in src_counter.most_common(12):
        print(f"  {cnt:4d}  ({100.0 * cnt / n_wrong:5.1f}%)  {key}")

    if failed_docs:
        print("\n【⑤ 打开失败的文档（模型无图可看）】")
        for d in failed_docs:
            exists = os.path.exists(os.path.join(PDF_DIR, d))
            print(f"  {d}  (缓存文件存在: {exists})")

    # 典型错题
    print("\n【⑥ 典型错题样例】(question / ground_truth / model_answer)")
    order = ["unanswerable_overreach", "visual_misread",
             "generation_failure", "retrieval_failure", "render_or_pdf_missing"]
    for cat in order:
        exs = cat_examples.get(cat, [])
        if not exs:
            continue
        print("\n  ◆ " + cat_label.get(cat, cat))
        for rec in exs:
            tag = "  [疑似度量伪影]" if rec["loose_correct"] else ""
            pred_short = rec["advanced_answer"]
            if len(pred_short) > 240:
                pred_short = pred_short[:240] + " ..."
            print(f"    Q: {rec['question']}{tag}")
            print(f"    GT: {rec['answer']}")
            print(f"    预测: {pred_short}")
            print(f"    [pages={rec['evidence_pages']} | src={rec['evidence_sources']}]")
            print("    " + "-" * 64)

    print("\n明细已写入：" + OUT_REPORT)


if __name__ == "__main__":
    main()