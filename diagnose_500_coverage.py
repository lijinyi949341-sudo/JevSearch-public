# -*- coding: utf-8 -*-
"""
500 题覆盖率缺口离线诊断（零 API）。

目的：
  1. 把 500 个 qid 在「路由 / Jev」两个模块上的状态逐一列出；
  2. 对未覆盖的题给出可审计的原因与可修复性（不需要调用任何付费 API）；
  3. 对可修复的 Jev 题，离线计算候选页文本层 / 已缓存 OCR 是否足以构造真实片段。

严格约束：
  - 只读：sampled_500、toc.json、routes.jsonl、snippets_*、jev_predictions_*、OCR 缓存、PDF；
  - 不写任何预测文件，不调用 API；
  - gold 答案与 gold 证据页只用于离线分母判定与报告，绝不进入模型输入。

输出：
  outputs/final500/audit/coverage_gap_500.csv
  outputs/final500/audit/coverage_gap_diagnosis.json
  outputs/final500/audit/COVERAGE_GAP_DIAGNOSIS.md
"""
import collections
import csv
import json
import os
import re
import sys

import final500_common as C

ROOT = C.ROOT
OUT = os.path.join(C.OUT_ROOT, "audit")
TOC_JSON = os.path.join(C.OUT_ROOT, "toc", "toc.json")
ROUTES = os.path.join(C.OUT_ROOT, "run500", "toc", "routes.jsonl")
JEV_DIR = os.path.join(C.OUT_ROOT, "run500", "jev")
OCR_FILES = [
    os.path.join(C.OUT_ROOT, "ocr", "ocr_pages_500.jsonl"),
    os.path.join(C.OUT_ROOT, "ocr", "ocr_pages_500_pilot2.jsonl"),
]
GOLD_FIXES = os.path.join(OUT, "gold_page_fixes_7.json")

JEV_SOURCES = {
    "title": "jev_predictions_title_500.jsonl",
    "route": "jev_predictions_route_500.jsonl",
    "ocr": "jev_predictions_ocr_500.jsonl",
    "recover": "jev_predictions_recover_500.jsonl",
    "pilot1": "jev_predictions_ocr_pilot_500.jsonl",
    "pilot2": "jev_predictions_ocr_pilot2_500.jsonl",
}
SNIPPET_FILES = {
    "title": "snippets_title_500.jsonl",
    "route": "snippets_route_500.jsonl",
    "ocr": "snippets_ocr_500.jsonl",
    "recover": "snippets_recover_500.jsonl",
    "pilot1": "snippets_ocr_pilot_500.jsonl",
    "pilot2": "snippets_ocr_pilot2_500.jsonl",
}

MAX_CANDIDATE_PAGES = 10
MIN_CHARS = 40
STOP = set("""a an the is are was were be been being of in on at to for from with by as that this these those
what which who whom whose when where why how many much does do did it its and or not or if than then there here
document page pages shown figure table following according based on please answer question""".split())


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", str(s or "").lower())
            if len(w) > 2 and w not in STOP]


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def load_tocs():
    d = C.read_json(TOC_JSON, default={"docs": []}) or {"docs": []}
    return {x["doc_id"]: x for x in d.get("docs", []) if x.get("build_status") == "ok" and x.get("sections")}


def load_ocr():
    out = {}
    for p in OCR_FILES:
        for r in C.load_jsonl(p):
            if r.get("status") != "ok":
                continue
            t = norm(r.get("text", ""))
            if len(t) < MIN_CHARS:
                continue
            out.setdefault(r["doc_id"], {})[int(r["page"])] = t
    return out


def load_jev():
    """qid -> [source, ...]（只统计 success=True 的记录）。"""
    out = collections.defaultdict(list)
    for src, name in JEV_SOURCES.items():
        for r in C.load_jsonl(os.path.join(JEV_DIR, name)):
            if r.get("success"):
                out[r["qid"]].append(src)
    return {q: sorted(set(v)) for q, v in out.items()}


def page_texts(doc_id, pages):
    """返回 {page: (text, has_images)}；失败返回 {}。"""
    try:
        import pymupdf
    except Exception:
        return {}
    pdf, _ = C.resolve_pdf(doc_id)
    if pdf is None:
        return {}
    res = {}
    try:
        doc = pymupdf.open(pdf)
        for p in pages:
            if p < 1 or p > doc.page_count:
                res[p] = ("", False)
                continue
            pg = doc.load_page(p - 1)
            res[p] = (norm(pg.get_text("text")), bool(pg.get_images()))
        doc.close()
    except Exception:
        return {}
    return res


def section_pages(sec):
    s = int(sec.get("start_page") or 0)
    e = int(sec.get("end_page") or s)
    if s < 1:
        return []
    if e < s:
        e = s
    return list(range(s, e + 1))


def pick_candidates(sample, toc, route, cap=MAX_CANDIDATE_PAGES):
    """按「真实检索结果」挑候选页：路由 top-3 章节优先，其次标题匹配 top-1 章节；截断到 cap 页。"""
    pages, origin = [], []
    if route and route.get("status") == "ok":
        for sec in (route.get("sections") or [])[:3]:
            for p in section_pages(sec):
                if p not in pages:
                    pages.append(p)
                    origin.append("route_top%d" % ((route.get("sections") or []).index(sec) + 1))
    if not pages and toc:
        import task6_route_and_eval as base6
        ranked = base6.baseline_rank(sample["question"], toc["sections"])
        for i, sec in enumerate(ranked[:2], 1):
            for p in section_pages(sec):
                if p not in pages:
                    pages.append(p)
                    origin.append("title_top%d" % i)
    return pages[:cap], origin[:cap]


def classify_route(sample, toc, route):
    gold = C.parse_pages(sample.get("evidence_pages", "[]"))
    if str(sample.get("answer", "")) == "Not answerable":
        return "unanswerable_skipped", "数据集标注 Not answerable；路由未执行"
    if toc is None:
        return "no_toc", "目录不可用"
    if not gold:
        return "missing_gold_pages_skipped", "无 gold 证据页"
    n = toc.get("num_pages") or 0
    if n and any(p < 1 or p > n for p in gold):
        return "invalid_gold_pages_skipped", "gold 证据页超出 PDF 页数"
    if route and route.get("status") == "ok":
        return "ok", ""
    return (route or {}).get("status", "not_run"), (route or {}).get("error", "") or ""


def main():
    samples = C.load_samples()
    tocs = load_tocs()
    routes = C.last_by_qid(C.load_jsonl(ROUTES))
    jev = load_jev()
    ocr = load_ocr()
    fixes_doc = C.read_json(GOLD_FIXES, default={}) or {}
    fixes = {x["qid"]: x for x in (fixes_doc.get("items", []) if isinstance(fixes_doc, dict) else fixes_doc)}

    rows = []
    for s in samples:
        q = s["qid"]
        toc = tocs.get(s["doc_id"])
        route = routes.get(q)
        rstatus, rreason = classify_route(s, toc, route)
        answerable = str(s.get("answer", "")) != "Not answerable"
        sources = jev.get(q, [])
        row = {
            "qid": q, "sample_index": s["sample_index"], "doc_id": s["doc_id"],
            "answerable": int(answerable), "route_status": rstatus, "route_reason": rreason,
            "jev_covered": int(bool(sources)), "jev_sources": ",".join(sources),
            "jev_gap_reason": "", "jev_fixability": "", "candidate_pages": "", "candidate_origin": "",
            "page_text_len": "", "ocr_cached": "", "needs_new_ocr": "",
            "best_page": "", "best_method": "", "best_kw_hits": "",
            "gold_pages": "", "gold_in_candidates": 0,
        }
        if not sources:
            if not answerable:
                row["jev_gap_reason"] = "unanswerable_no_retrieval"
                row["jev_fixability"] = "needs_route_abstention"
            elif rstatus in ("invalid_gold_pages_skipped", "missing_gold_pages_skipped"):
                fx = fixes.get(q, {})
                row["jev_gap_reason"] = rstatus
                row["jev_fixability"] = ("route_rerun_with_corrected_pages"
                                         if fx.get("status", "").startswith("fixed")
                                         else "unfixable_gold_metadata")
            else:
                pages, origin = pick_candidates(s, toc, route)
                gold_pages = C.parse_pages(s.get("evidence_pages", "[]"))
                row["gold_pages"] = ",".join(map(str, gold_pages))
                row["gold_in_candidates"] = int(bool(set(gold_pages) & set(pages)))
                texts = page_texts(s["doc_id"], pages) if pages else {}
                len_map = {p: len(texts.get(p, ("", False))[0]) for p in pages}
                qwords = set(words(s["question"]))
                ocr_doc = ocr.get(s["doc_id"], {})

                # 逐页打分：优先文本层，其次已缓存 OCR；命中问题关键词越多越优先
                scored = []
                for p in pages:
                    if len_map.get(p, 0) >= MIN_CHARS:
                        snippet, method = texts[p][0], "text_layer"
                    elif p in ocr_doc:
                        snippet, method = ocr_doc[p], "ocr_cached"
                    else:
                        continue
                    hit = len(qwords & set(words(snippet)))
                    scored.append((hit, len(snippet), p, method))
                best = max(scored, key=lambda x: (x[0], x[1])) if scored else None

                ocr_pages = sorted(p for p in pages if p in ocr_doc)
                text_pages = sorted(p for p in pages if len_map.get(p, 0) >= MIN_CHARS)
                need_ocr = sorted(p for p in pages if len_map.get(p, 0) < MIN_CHARS and p not in ocr_doc)
                row["candidate_pages"] = ",".join(map(str, pages))
                row["candidate_origin"] = ",".join(origin)
                row["page_text_len"] = ",".join("%d:%d" % (p, len_map.get(p, 0)) for p in pages)
                row["ocr_cached"] = ",".join(map(str, ocr_pages))
                row["needs_new_ocr"] = ",".join(map(str, need_ocr))
                if best and best[0] > 0:
                    row["best_page"] = best[2]
                    row["best_method"] = best[3]
                    row["best_kw_hits"] = best[0]
                    row["jev_gap_reason"] = "candidate_page_relevant_text"
                    row["jev_fixability"] = ("relevant_text_recoverable" if best[3] == "text_layer"
                                             else "relevant_ocr_cached_recoverable")
                elif best:
                    row["best_page"] = best[2]
                    row["best_method"] = best[3]
                    row["best_kw_hits"] = 0
                    row["jev_gap_reason"] = "candidate_page_text_no_keyword"
                    row["jev_fixability"] = "weak_text_recoverable"
                elif need_ocr:
                    row["jev_gap_reason"] = "pages_have_no_text"
                    row["jev_fixability"] = "needs_local_ocr"
                else:
                    row["jev_gap_reason"] = "no_candidate_pages"
                    row["jev_fixability"] = "unfixable_no_candidate"
        rows.append(row)

    # ---------------- 汇总
    route_counts = collections.Counter(r["route_status"] for r in rows)
    jev_missing = [r for r in rows if not r["jev_covered"]]
    fix_counts = collections.Counter(r["jev_fixability"] for r in jev_missing)
    gap_reasons = collections.Counter(r["jev_gap_reason"] for r in jev_missing)
    route_only_missing = [r for r in rows if r["route_status"] != "ok"]
    route_missing_ans = collections.Counter(
        ("answerable" if r["answerable"] else "unanswerable") for r in route_only_missing)

    summary = {
        "generated_at": C.now_str(),
        "n": len(rows),
        "route_status_counts": dict(route_counts),
        "route_not_ok": len(route_only_missing),
        "route_not_ok_by_answerable": dict(route_missing_ans),
        "jev_covered": sum(1 for r in rows if r["jev_covered"]),
        "jev_missing": len(jev_missing),
        "jev_missing_by_answerable": dict(collections.Counter(
            "answerable" if r["answerable"] else "unanswerable" for r in jev_missing)),
        "jev_gap_reasons": dict(gap_reasons),
        "jev_fixability": dict(fix_counts),
        "jev_fixable_total": sum(v for k, v in fix_counts.items()
                                 if k in ("relevant_text_recoverable", "relevant_ocr_cached_recoverable",
                                          "weak_text_recoverable", "needs_local_ocr",
                                          "route_rerun_with_corrected_pages")),
        "jev_relevant_recoverable": sum(v for k, v in fix_counts.items()
                                        if k in ("relevant_text_recoverable",
                                                 "relevant_ocr_cached_recoverable")),
        "gold_in_candidates_among_missing": sum(1 for r in jev_missing if r["gold_in_candidates"]),
    }

    C.ensure_dir(OUT)
    csv_path = os.path.join(OUT, "coverage_gap_500.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    C.write_json_atomic(os.path.join(OUT, "coverage_gap_diagnosis.json"), summary)

    # ---------------- Markdown
    lines = []
    A = lines.append
    A("# 500 题覆盖率缺口离线诊断")
    A("")
    A("生成时间：%s  " % summary["generated_at"])
    A("数据来源：`coverage_gap_500.csv`（逐题）、`routes.jsonl`、`jev_predictions_*_500.jsonl`、`toc.json`、PDF 文本层、OCR 缓存。  ")
    A("说明：本诊断不调用任何 API，不修改任何原始预测；gold 仅用于离线判定。")
    A("")
    A("## 一、路由模块状态")
    A("")
    A("| 状态 | 题数 |")
    A("|---|---|")
    for k, v in route_counts.most_common():
        A("| `%s` | %d |" % (k, v))
    A("")
    A("路由未执行 %d 题：可答 %d、不可答 %d。" % (
        summary["route_not_ok"], route_missing_ans.get("answerable", 0),
        route_missing_ans.get("unanswerable", 0)))
    A("")
    A("## 二、Jev 模块状态")
    A("")
    A("| 指标 | 数值 |")
    A("|---|---|")
    A("| 已有真实 Jev 判断 | %d |" % summary["jev_covered"])
    A("| 未覆盖 | %d |" % summary["jev_missing"])
    A("| 其中可答 | %d |" % summary["jev_missing_by_answerable"].get("answerable", 0))
    A("| 其中不可答 | %d |" % summary["jev_missing_by_answerable"].get("unanswerable", 0))
    A("")
    A("### 未覆盖原因")
    A("")
    A("| 原因 | 题数 |")
    A("|---|---|")
    for k, v in gap_reasons.most_common():
        A("| `%s` | %d |" % (k, v))
    A("")
    A("### 可修复性")
    A("")
    A("| 可修复性 | 题数 |")
    A("|---|---|")
    for k, v in fix_counts.most_common():
        A("| `%s` | %d |" % (k, v))
    A("")
    A("## 三、按 qid 的未覆盖清单")
    A("")
    A("| qid | 可答 | 路由状态 | Jev 缺口原因 | 可修复性 | 候选页 |")
    A("|---|---|---|---|---|---|")
    for r in jev_missing:
        A("| %s | %d | `%s` | `%s` | `%s` | %s |" % (
            r["qid"], r["answerable"], r["route_status"], r["jev_gap_reason"],
            r["jev_fixability"], r["candidate_pages"] or "-"))
    A("")
    md_path = os.path.join(OUT, "COVERAGE_GAP_DIAGNOSIS.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("输出：%s" % csv_path)
    print("输出：%s" % md_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
