# -*- coding: utf-8 -*-
"""
500 题覆盖率补全（离线构建 + 可选付费执行）。

设计原则：
  - 不修改任何已成功的原始预测（routes.jsonl / jev_predictions_*_500.jsonl）；
  - 片段页只能来自「真实检索结果」（已有路由 top-3 章节、title top-2 章节），
    绝不用 gold 证据页选页；gold 只在预测完成后用于离线评分；
  - 无法得到真实文本的题，明确记 no_snippet 及原因，绝不填充占位文本；
  - 所有新增结果写入新文件，便于单独审计。

新增产物：
  outputs/final500/run500/toc/routes_fill_126.jsonl                 （--run-route-fill 后）
  outputs/final500/audit/route_fill_metrics.json                    （--eval-route-fill 后）
  outputs/final500/run500/jev/snippets_gapfill_500.jsonl
  outputs/final500/run500/jev/jev_predictions_gapfill_500.jsonl     （--run-jev 后）
  outputs/final500/audit/coverage_final_500.csv
  outputs/final500/audit/coverage_final_stats.json

路由补跑只有一个入口 --run-route-fill：
  目标集合 = 「routes.jsonl 中没有成功记录」的全部题（按执行状态筛选），
  不使用数据集 answer / answerable 标签选题；提示词只含 question + 真实目录；
  模型自行判断 answerable true/false，false 即拒答。
  旧的 --run-route-fix（按 gold 修正页选题）与 --run-route-abstain（按 gold 标注选题）
  已删除。修正后的 gold 页只在 --eval-route-fill 中离线评分使用。

用法：
  py -3.12 fill_gaps_500.py --build-jev          # 离线构建补全片段（不调用 API）
  py -3.12 fill_gaps_500.py --dry-run-jev 5      # 打印前 5 条实际发送内容 + 泄漏检查
  py -3.12 fill_gaps_500.py --run-jev            # 付费：调用 Jev（成功后自动跳过）
  py -3.12 fill_gaps_500.py --plan-route         # 路由补跑计划（不调用 API）
  py -3.12 fill_gaps_500.py --run-route-fill     # 付费：126 道缺口题统一做可拒答路由
  py -3.12 fill_gaps_500.py --eval-route-fill    # 离线评估拒答率与修正 gold 页 Page Recall
  py -3.12 fill_gaps_500.py --report             # 汇总 500 题最终状态
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

import final500_common as C
import task6_route_and_eval as base6

ROOT = C.ROOT
OUT = os.path.join(C.OUT_ROOT, "audit")
TOC_JSON = os.path.join(C.OUT_ROOT, "toc", "toc.json")
ROUTES = os.path.join(C.OUT_ROOT, "run500", "toc", "routes.jsonl")
ROUTES_FILL = os.path.join(C.OUT_ROOT, "run500", "toc", "routes_fill_126.jsonl")
DIAG_CSV = os.path.join(OUT, "coverage_gap_500.csv")
GOLD_FIXES = os.path.join(OUT, "gold_page_fixes_7.json")

JEV_DIR = os.path.join(C.OUT_ROOT, "run500", "jev")
SNIPPETS_GAP = os.path.join(JEV_DIR, "snippets_gapfill_500.jsonl")
PRED_GAP = os.path.join(JEV_DIR, "jev_predictions_gapfill_500.jsonl")
SNIPPETS_R2 = os.path.join(JEV_DIR, "snippets_round2_500.jsonl")
PRED_R2 = os.path.join(JEV_DIR, "jev_predictions_round2_500.jsonl")
OCR_FILES = [
    os.path.join(C.OUT_ROOT, "ocr", "ocr_pages_500.jsonl"),
    os.path.join(C.OUT_ROOT, "ocr", "ocr_pages_500_pilot2.jsonl"),
    os.path.join(C.OUT_ROOT, "ocr", "ocr_pages_round2_500.jsonl"),
]

MAX_CANDIDATE_PAGES = 10
MIN_CHARS = 40
STOP = set("""a an the is are was were be been being of in on at to for from with by as that this these those
what which who whom whose when where why how many much does do did it its and or not if than then there here
document page pages shown figure table following according based on please answer question""".split())


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", str(s or "").lower())
            if len(w) > 2 and w not in STOP]


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def load_tocs():
    d = C.read_json(TOC_JSON, default={"docs": []}) or {"docs": []}
    return {x["doc_id"]: x for x in d.get("docs", [])
            if x.get("build_status") == "ok" and x.get("sections")}


def load_ocr_text():
    """(doc_id, page) -> text；只接受 >= MIN_CHARS 的 OCR 结果。"""
    out = {}
    for p in OCR_FILES:
        for r in C.load_jsonl(p):
            t = norm(r.get("text", ""))
            if r.get("status") == "ok" and len(t) >= MIN_CHARS:
                out[(r["doc_id"], int(r["page"]))] = t
    return out


def load_ocr_for_snippets():
    """task5_jev_500.extract_page_text 需要的 {doc_id: {page: {"ocr_text": ...}}}。"""
    out = {}
    for p in OCR_FILES:
        for r in C.load_jsonl(p):
            t = norm(r.get("text", ""))
            if r.get("status") == "ok" and t:
                out.setdefault(r["doc_id"], {})[int(r["page"])] = {"ocr_text": t}
    return out


def load_gold_fixes():
    d = C.read_json(GOLD_FIXES, default={}) or {}
    items = d.get("items", []) if isinstance(d, dict) else d
    return {x["qid"]: x for x in items}


def section_pages(sec):
    s = int(sec.get("start_page") or 0)
    e = int(sec.get("end_page") or s)
    if s < 1:
        return []
    return list(range(s, max(s, e) + 1))


def candidate_pages(sample, toc, route, cap=MAX_CANDIDATE_PAGES):
    """真实检索候选页：路由 top-3 章节优先，其次标题匹配 top-2 章节。"""
    pages, origin = [], []
    secs = []
    if route and route.get("status") == "ok":
        secs = (route.get("sections") or [])[:3]
        for i, sec in enumerate(secs, 1):
            for p in section_pages(sec):
                if p not in pages:
                    pages.append(p)
                    origin.append("route_top%d" % i)
    if not pages and toc:
        ranked = base6.baseline_rank(sample["question"], toc["sections"])
        for i, sec in enumerate(ranked[:2], 1):
            for p in section_pages(sec):
                if p not in pages:
                    pages.append(p)
                    origin.append("title_top%d" % i)
    return pages[:cap], origin[:cap]


def score_pages(sample, pages, ocr_text):
    """按文本层 / OCR 缓存给候选页打分，返回 [(hits, len, page, method)] 降序。"""
    import pymupdf
    pdf, _ = C.resolve_pdf(sample["doc_id"])
    if pdf is None:
        return []
    q = set(words(sample["question"]))
    scored = []
    doc = pymupdf.open(pdf)
    try:
        for p in pages:
            if p < 1 or p > doc.page_count:
                continue
            raw = norm(doc.load_page(p - 1).get_text("text"))
            if len(raw) >= MIN_CHARS:
                text, method = raw, "text_layer"
            elif (sample["doc_id"], p) in ocr_text:
                text, method = ocr_text[(sample["doc_id"], p)], "ocr"
            else:
                continue
            scored.append((len(q & set(words(text))), len(text), p, method))
    finally:
        doc.close()
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return scored


# ---------------------------------------------------------------- Jev 片段补全
def build_jev_snippets():
    import task5_jev_500 as t5

    diag = {r["qid"]: r for r in csv.DictReader(open(DIAG_CSV, encoding="utf-8-sig"))}
    tocs = load_tocs()
    routes = C.last_by_qid(C.load_jsonl(ROUTES))
    if os.path.exists(ROUTES_FILL):
        for q, r in C.last_by_qid(C.load_jsonl(ROUTES_FILL)).items():
            if r.get("success"):
                routes[q] = r
    ocr_text = load_ocr_text()
    ocr_for_snippet = load_ocr_for_snippets()
    samples = C.load_samples()

    out = []
    for s in samples:
        q = s["qid"]
        d = diag.get(q, {})
        if d.get("jev_covered") == "1":
            continue
        rec = {
            "snippet_id": "%s__gapfill" % q,
            "qid": q, "sample_index": s["sample_index"], "doc_id": s["doc_id"],
            "doc_type": s.get("doc_type", ""), "question": s["question"], "mode": "gapfill",
            "page": None, "section_id": None, "selection_uses_gold": False,
            "selection_rule": "路由补跑结果 top-3 章节 / 标题 top-2 章节范围；按问题词命中与文本长度选页；"
                              "不使用 gold 证据页",
            "status": "", "reason": "", "evidence_snippet": "", "char_len": 0,
            "extraction_method": "", "has_page_images": False,
            "candidate_pages": "", "candidate_origin": "",
            "evidence_type_primary": t5.primary_type(
                C.parse_sources(s.get("evidence_sources", "[]"))),
            "cross_page": len(C.parse_pages(s.get("evidence_pages", "[]"))) > 1,
            "created_at": C.now_str(),
            "offline_only": {
                "answer": s.get("answer", ""), "gold_answer": s.get("answer", ""),
                "evidence_pages": s.get("evidence_pages", "[]"),
                "evidence_sources": s.get("evidence_sources", ""),
            },
        }
        route = routes.get(q)
        # 状态只由「真实执行结果」决定：没有成功路由 → pending；模型拒答 → 无片段；
        # 不使用数据集 answer/answerable 标签做分支。
        # 注意：routes.jsonl 中 status=unanswerable_skipped / invalid_gold_pages_skipped
        # 表示「路由没有执行」，不是模型输出，因此必须先补跑，不能直接判 N/A。
        if route is None or (route.get("status") != "ok" and not route.get("success")):
            rec.update({"status": "pending_route",
                        "reason": "route_not_executed:%s" % (route or {}).get("status", "")})
            out.append(rec)
            continue
        if route.get("abstained") is True:
            rec.update({"status": "not_applicable",
                        "reason": "route_abstained_no_section"})
            out.append(rec)
            continue
        if not (route.get("sections") or []):
            rec.update({"status": "not_applicable",
                        "reason": "route_no_section:%s" % route.get("status", "")})
            out.append(rec)
            continue
        toc = tocs.get(s["doc_id"])
        pages, origin = candidate_pages(s, toc, route)
        rec["candidate_pages"] = ",".join(map(str, pages))
        rec["candidate_origin"] = ",".join(origin)
        if not pages:
            rec.update({"status": "no_snippet", "reason": "no_candidate_pages"})
            out.append(rec)
            continue
        scored = score_pages(s, pages, ocr_text)
        if not scored:
            rec.update({"status": "no_snippet", "reason": "candidate_pages_have_no_text"})
            out.append(rec)
            continue
        hits, _, page, method = scored[0]
        text, extract_method, has_images, truncated = t5.extract_page_text(
            s["doc_id"], page, ocr_for_snippet)
        if not text:
            rec.update({"status": "no_snippet", "reason": extract_method,
                        "has_page_images": has_images})
            out.append(rec)
            continue
        rec.update({"page": page, "section_id": None,
                    "selection_quality": "keyword_hit" if hits > 0 else "text_only",
                    "keyword_hits": hits,
                    "evidence_snippet": text, "char_len": len(text),
                    "extraction_method": extract_method, "has_page_images": has_images,
                    "truncated": truncated, "status": "ok", "reason": ""})
        out.append(rec)

    C.ensure_dir(JEV_DIR)
    tmp = SNIPPETS_GAP + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, SNIPPETS_GAP)

    cnt = collections.Counter(r["status"] for r in out)
    ok = [r for r in out if r["status"] == "ok"]
    print("[gapfill] 目标 %d 题：ok=%d, %s" % (
        len(out), cnt.get("ok", 0), dict(cnt)))
    print("[gapfill] 其中 keyword_hit=%d, text_only=%d, ocr=%d" % (
        sum(1 for r in ok if r.get("selection_quality") == "keyword_hit"),
        sum(1 for r in ok if r.get("selection_quality") == "text_only"),
        sum(1 for r in ok if r.get("extraction_method") == "ocr")))
    print("输出：%s" % SNIPPETS_GAP)
    return 0


def dry_run_jev(n, qids=None):
    import run_task5_jev as base5
    import task5_jev_500 as t5

    rows = [r for r in C.load_jsonl(SNIPPETS_GAP) if r.get("status") == "ok"]
    total = len(rows)
    covered = covered_jev_qids()
    rows = [r for r in rows if r["qid"] not in covered]
    skipped = total - len(rows)
    if qids:
        wanted = {x.strip().upper() for x in qids.split(",") if x.strip()}
        rows = [r for r in rows if r["qid"].upper() in wanted]
    rows = rows[:n or 5]
    print("[dry-run] 可送片段 %d 条；已执行按 qid 跳过 %d 条；本次展示 %d 条"
          % (total, skipped, len(rows)))
    if not rows:
        print("[提示] 当前没有待执行的新片段；补跑后需重新 --build-jev。")
        return 0
    print("【DRY-RUN】将检查 %d 条，不调用 API。" % len(rows))
    for r in rows:
        js = t5.to_jev_sample(r)
        state = base5.build_state(js)
        problems, gold_in_snippet = base5.leak_check(js, state)
        print("-" * 60)
        print("%s page=%s method=%s hits=%s" % (
            r["snippet_id"], r["page"], r.get("extraction_method"), r.get("keyword_hits")))
        print("发送内容：", json.dumps(state, ensure_ascii=False)[:600])
        print("泄漏检查：", "PASS" if not problems else "FAIL %s" % problems,
              "| gold_answer_visible_in_snippet=%s" % bool(gold_in_snippet))
    return 0


def covered_jev_qids():
    """所有历史预测文件中已成功的 qid，用于按题去重，避免同一题重复计费。"""
    out = set()
    for name in os.listdir(JEV_DIR):
        if name.startswith("jev_predictions_") and name.endswith(".jsonl"):
            for r in C.load_jsonl(os.path.join(JEV_DIR, name)):
                if r.get("success"):
                    out.add(r["qid"])
    return out


def run_jev(limit=0, qids=None):
    """付费执行；limit / qids 只用于冒烟测试，正式补跑时都不传。

    除 snippet_id 去重外，再按 qid 排除已成功的题：
    同一道题即使换了片段来源，也不会二次调用、二次计费。
    """
    import task5_jev_500 as t5
    covered = covered_jev_qids()
    print("[jev] 历史已覆盖 %d 题，本次按 qid 跳过" % len(covered))
    return t5.run_jev("gapfill", limit=limit, snippets_file=SNIPPETS_GAP,
                      out_file=PRED_GAP, qids=qids, exclude_qids=covered)


# ---------------------------------------------------------------- 第二轮补全
def _route_map():
    """路由结果合并：原有成功路由 + 补跑成功路由。"""
    routes = {}
    for q, r in C.last_by_qid(C.load_jsonl(ROUTES)).items():
        if r.get("status") == "ok":
            routes[q] = r
    if os.path.exists(ROUTES_FILL):
        for q, r in C.last_by_qid(C.load_jsonl(ROUTES_FILL)).items():
            if r.get("success"):
                routes[q] = r
    return routes


def full_scan_pages(sample, cap=3, max_pages=300):
    """无目录/标题不匹配时的兜底：全文档文本层按问题词命中排序（真实检索，不用 gold）。"""
    import pymupdf
    pdf, _ = C.resolve_pdf(sample["doc_id"])
    if pdf is None:
        return []
    q = set(words(sample["question"]))
    scored = []
    doc = pymupdf.open(pdf)
    try:
        for i in range(min(doc.page_count, max_pages)):
            raw = norm(doc.load_page(i).get_text("text"))
            if len(raw) >= MIN_CHARS:
                scored.append((len(q & set(words(raw))), len(raw), i + 1))
    finally:
        doc.close()
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [x[2] for x in scored[:cap]]


def round2_pages(sample, toc, route, cap=12):
    """第二轮候选页：路由章节优先；模型拒答时退回真实目录标题匹配。

    全部来自真实 PDF 目录/路由结果，不使用 gold 证据页。
    """
    pages, origin = [], []
    secs = (route or {}).get("sections") or []
    if secs:
        for i, sec in enumerate(secs[:3], 1):
            for p in section_pages(sec):
                if p not in pages:
                    pages.append(p)
                    origin.append("route_top%d" % i)
    if not pages and toc:
        ranked = base6.baseline_rank(sample["question"], toc["sections"])
        for i, sec in enumerate(ranked[:2], 1):
            for p in section_pages(sec):
                if p not in pages:
                    pages.append(p)
                    origin.append("title_top%d" % i)
    if not pages:
        for p in full_scan_pages(sample):
            if p not in pages:
                pages.append(p)
                origin.append("full_scan")
    return pages[:cap], origin[:cap]


def build_round2():
    """为 435 条之外的全部题重建片段：拒答题走真实目录标题检索，无文本题扩大候选页。"""
    import task5_jev_500 as t5

    tocs = load_tocs()
    routes = _route_map()
    ocr_text = load_ocr_text()
    ocr_for_snippet = load_ocr_for_snippets()
    samples = C.load_samples()
    covered = covered_jev_qids()
    out = []
    for s in samples:
        q = s["qid"]
        if q in covered:
            continue
        route = routes.get(q) or {}
        toc = tocs.get(s["doc_id"])
        rec = {
            "snippet_id": "%s__round2" % q, "qid": q, "sample_index": s["sample_index"],
            "doc_id": s["doc_id"], "doc_type": s.get("doc_type", ""),
            "question": s["question"], "mode": "round2", "round": 2,
            "route_state": route.get("status"), "route_abstained": bool(route.get("abstained")),
            "selection_uses_gold": False,
            "selection_rule": "路由 top-3 章节；模型拒答时退回目录标题匹配 top-2 章节；"
                              "按问题词命中与文本长度选页；不使用 gold 证据页",
            "page": None, "section_id": None, "status": "", "reason": "",
            "evidence_snippet": "", "char_len": 0, "extraction_method": "",
            "has_page_images": False, "candidate_pages": "", "candidate_origin": "",
            "retrieval_basis": "route_sections" if route.get("sections") else "title_fallback",
            "evidence_type_primary": t5.primary_type(
                C.parse_sources(s.get("evidence_sources", "[]"))),
            "cross_page": len(C.parse_pages(s.get("evidence_pages", "[]"))) > 1,
            "created_at": C.now_str(),
            "offline_only": {
                "answer": s.get("answer", ""), "gold_answer": s.get("answer", ""),
                "evidence_pages": s.get("evidence_pages", "[]"),
                "evidence_sources": s.get("evidence_sources", ""),
            },
        }
        pages, origin = round2_pages(s, toc, route)
        rec["candidate_pages"] = ",".join(map(str, pages))
        rec["candidate_origin"] = ",".join(origin)
        if not pages:
            rec.update({"status": "no_candidate_page",
                        "reason": "no_toc_or_no_title_match"})
            out.append(rec)
            continue
        scored = score_pages(s, pages, ocr_text)
        if not scored:
            rec.update({"status": "no_text_page", "reason": "candidate_pages_have_no_text"})
            out.append(rec)
            continue
        hits, _, page, method = scored[0]
        text, extract_method, has_images, truncated = t5.extract_page_text(
            s["doc_id"], page, ocr_for_snippet)
        if not text:
            rec.update({"status": "no_text_page", "reason": extract_method,
                        "has_page_images": has_images})
            out.append(rec)
            continue
        rec.update({"page": page, "selection_quality": "keyword_hit" if hits else "text_only",
                    "keyword_hits": hits, "evidence_snippet": text, "char_len": len(text),
                    "extraction_method": extract_method, "has_page_images": has_images,
                    "truncated": truncated, "status": "ok"})
        out.append(rec)

    tmp = SNIPPETS_R2 + ".tmp"
    C.ensure_dir(JEV_DIR)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, SNIPPETS_R2)
    import collections as _c
    print("[round2] 共 %d 题：%s" % (len(out), dict(_c.Counter(r["status"] for r in out))))
    print("输出: %s" % SNIPPETS_R2)
    return 0


def dry_run_round2(n=5):
    import task5_jev_500 as t5
    rows = [r for r in C.load_jsonl(SNIPPETS_R2) if r.get("status") == "ok"]
    covered = covered_jev_qids()
    total = len(rows)
    rows = [r for r in rows if r["qid"] not in covered]
    print("[dry-run] 可送 %d 条；已执行跳过 %d 条；本次展示 %d 条"
          % (total, total - len(rows), min(n, len(rows))))
    for r in rows[:n]:
        js = t5.to_jev_sample(r)
        problems, gold_in = base5.leak_check(js, base5.build_state(js))
        print("-" * 60)
        print("%s page=%s basis=%s hits=%s" % (
            r["snippet_id"], r["page"], r.get("retrieval_basis"), r.get("keyword_hits")))
        print("泄漏检查：", "PASS" if not problems else "FAIL %s" % problems,
              "| gold_answer_visible=%s" % bool(gold_in))
    return 0


def run_round2(limit=0):
    import task5_jev_500 as t5
    covered = covered_jev_qids()
    print("[round2] 历史已覆盖 %d 题，本次按 qid 跳过" % len(covered))
    return t5.run_jev("round2", limit=limit, snippets_file=SNIPPETS_R2,
                      out_file=PRED_R2, exclude_qids=covered)


def write_round2_ocr_plan():
    rows = [r for r in C.load_jsonl(SNIPPETS_R2) if r.get("status") == "no_text_page"]
    path = os.path.join(OUT, "ocr_page_plan_round2.csv")
    seen = set()
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "doc_id", "page", "need_ocr"])
        for r in rows:
            for p in [x for x in str(r.get("candidate_pages", "")).split(",") if x][:2]:
                key = (r["doc_id"], int(p))
                if key in seen:
                    continue
                seen.add(key)
                w.writerow([r["qid"], r["doc_id"], int(p), "True"])
    print("[round2-ocr] 计划 %d 页 / %d 题 -> %s" % (len(seen), len(rows), path))
    return 0


# ---------------------------------------------------------------- 路由补全
def build_abstain_prompt(question, toc_doc):
    lines = ["%s | 层级%s | 第%s-%s页 | %s" % (
        s["section_id"], s["level"], s["start_page"], s["end_page"], s["title"])
        for s in toc_doc["sections"]]
    user = (
        "你是文档检索助手。下面给出一个问题，以及该文档的章节目录（页码为 PDF 实际页码）。\n"
        "请判断：目录中是否存在任何章节，其内容可能包含回答该问题所需的信息。\n"
        "如果存在，给出最可能的 1-3 个章节；如果目录中没有任何章节可能包含答案，"
        "必须明确返回不可回答，不得强行选择章节。\n\n"
        "【问题】\n%s\n\n"
        "【文档目录】（共 %d 个章节，文档共 %d 页）\n%s\n\n"
        "【输出格式】只输出一个 JSON 对象，不要解释：\n"
        '{"answerable": true/false, "sections":[{"section_id":"<ID>","rank":1}]}\n'
        "answerable=false 时 sections 必须为空数组；answerable=true 时 section_id 必须真实存在。"
    ) % (question, len(toc_doc["sections"]), toc_doc["num_pages"], "\n".join(lines))
    system = ("你是一个严谨的文档章节定位器。你只能依据提供的目录判断，"
              "不得使用外部知识；目录中没有依据时必须判定为不可回答。")
    return system, user


def route_fill_targets(routes, samples):
    """路由补跑目标 = 现有路由文件中没有成功记录的全部题（按执行状态选，不看 gold 标注）。

    这样可以保证目标集合的确定只依赖「该题是否已经跑过路由」，
    而不是「该题在数据集里是否可回答」。
    """
    return [s for s in samples if routes.get(s["qid"], {}).get("status") != "ok"]


def route_plan():
    samples = C.load_samples()
    tocs = load_tocs()
    routes = C.last_by_qid(C.load_jsonl(ROUTES))
    targets = route_fill_targets(routes, samples)
    fix5 = [s for s in targets if load_gold_fixes().get(s["qid"], {})
            .get("status", "").startswith("fixed")]
    print(json.dumps({
        "route_fill_targets": len(targets),
        "with_toc": sum(1 for s in targets if s["doc_id"] in tocs),
        "without_toc": sum(1 for s in targets if s["doc_id"] not in tocs),
        "offline_eval_corrected_gold": len(fix5),
        "corrected_gold_qids": [s["qid"] for s in fix5],
        "selection_basis": "执行状态（无成功路由记录），不使用 answer/answerable 标签",
        "note": "本命令只输出计划，不调用 API。",
    }, ensure_ascii=False, indent=2))
    return 0


def _route_call(client, system, user, model=C.MODEL_DS):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0, max_tokens=400, extra_body={"enable_thinking": False})
    return (resp.choices[0].message.content or ""), resp


def _client():
    from openai import OpenAI
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        print("[错误] 未检测到 DASHSCOPE_API_KEY，未发出任何请求。")
        sys.exit(1)
    return OpenAI(api_key=key, base_url=C.BASE_URL)


def run_route_fill(limit=0, qids=None, dry_run=False):
    """对「没有成功路由记录」的题统一执行可拒答路由。

    安全约束：
      - 目标集合只按执行状态筛选（见 route_fill_targets），不读取 answer/answerable；
      - 提示词只包含 question + 目录（section_id/标题/层级/页码范围）；
      - 记录中不写入任何 gold 字段，修正后的 gold 页只在 --eval-route-fill 中离线使用。
    limit / qids 只用于冒烟测试，正式补跑时都不传。
    """
    samples = C.load_samples()
    tocs = load_tocs()
    routes = C.last_by_qid(C.load_jsonl(ROUTES))
    targets = sorted(route_fill_targets(routes, samples), key=lambda x: x["sample_index"])
    if qids:
        wanted = {x.strip().upper() for x in qids.split(",") if x.strip()}
        targets = [s for s in targets if s["qid"].upper() in wanted]
    existing = C.last_by_qid(C.load_jsonl(ROUTES_FILL))
    # 先剔除已成功的题，再截断：否则 --route-limit N 每次都会重复选中同一批已完成题。
    pending = [s for s in targets if not (existing.get(s["qid"]) or {}).get("success")]
    skipped = len(targets) - len(pending)
    if limit:
        pending = pending[:limit]
    print("[route-fill] 目标 %d 道，已完成 %d 道，本次待跑 %d 道；输出 %s"
          % (len(targets), skipped, len(pending), ROUTES_FILL))
    if dry_run:
        for s in pending:
            doc = tocs.get(s["doc_id"])
            chars = len(build_abstain_prompt(s["question"], doc)[1]) if doc else 0
            print("  %s sample_index=%d toc=%s prompt_chars=%d"
                  % (s["qid"], s["sample_index"], "yes" if doc else "no", chars))
        print("[dry-run] 不调用 API。")
        return 0
    client = _client()
    n = 0
    for s in pending:
        q = s["qid"]
        if existing.get(q, {}).get("success"):
            continue
        doc = tocs.get(s["doc_id"])
        if not doc:
            C.append_jsonl(ROUTES_FILL, {"qid": q, "doc_id": s["doc_id"],
                                         "success": False, "status": "no_toc",
                                         "sections": [], "run_at": C.now_str()})
            print("[%s] no_toc" % q)
            continue
        system, user = build_abstain_prompt(s["question"], doc)
        try:
            text, resp = _route_call(client, system, user)
            obj = base6.parse_model_json(text)
            answerable = bool(obj.get("answerable")) if isinstance(obj, dict) else None
            secs, invalid = base6.validate_sections(
                obj, {x["section_id"]: x for x in doc["sections"]}) if obj else ([], [])
            status = "ok"
            if obj is None:
                status = "invalid_json"
            elif answerable is True and not secs:
                status = "no_valid_section"
            elif answerable is False and secs:
                status = "inconsistent_abstain_with_sections"
            rec = {
                "qid": q, "sample_index": s["sample_index"], "doc_id": s["doc_id"],
                "question": s["question"], "task": "toc_routing_abstain_capable",
                "prompt_version": "route500_abstain_v1",
                "success": True, "status": status,
                "model_answerable": answerable,
                "abstained": answerable is False,
                "sections": secs, "invalid_section_ids": invalid,
                "raw_response": text, "model_returned": getattr(resp, "model", None),
                "usage": ({"prompt_tokens": resp.usage.prompt_tokens,
                           "completion_tokens": resp.usage.completion_tokens}
                          if getattr(resp, "usage", None) else None),
                "run_at": C.now_str(),
                "selection_uses_gold": False,
                "gold_fields_in_request": False,
            }
            C.append_jsonl(ROUTES_FILL, rec)
            n += 1
            print("[%s] %s answerable=%s sections=%d" % (q, status, answerable, len(secs)))
        except Exception as e:
            C.append_jsonl(ROUTES_FILL, {"qid": q, "success": False,
                                         "status": "api_error", "sections": [],
                                         "error": "%s: %s" % (type(e).__name__, str(e)[:300]),
                                         "run_at": C.now_str()})
            print("[%s] ERROR %s" % (q, type(e).__name__))
    print("新增路由记录 %d 条；目标 %d 条；输出 %s" % (n, len(targets), ROUTES_FILL))
    return 0


def eval_route_fill():
    """离线评估补跑路由：拒答率、过度拒答、5 道修正 gold 页题的 Page Recall。

    gold 只在这里（预测完成之后）读取，且不写回任何预测文件。
    """
    samples = {s["qid"]: s for s in C.load_samples()}
    records = C.last_by_qid(C.load_jsonl(ROUTES_FILL))
    fixes = load_gold_fixes()
    groups = collections.defaultdict(list)
    for q, r in records.items():
        if not r.get("success"):
            groups["api_failed"].append(q)
            continue
        s = samples[q]
        if str(s.get("answer", "")) == "Not answerable":
            groups["unanswerable"].append(q)
        elif fixes.get(q, {}).get("status", "").startswith("fixed"):
            groups["corrected_gold"].append(q)
        else:
            groups["other_answerable"].append(q)

    def abstain_rate(qs):
        if not qs:
            return None
        return round(sum(1 for q in qs if records[q].get("abstained")) / len(qs), 4)

    corrected = []
    for q in groups["corrected_gold"]:
        r = records[q]
        gold = set(C.parse_pages(fixes[q].get("corrected_physical_pages")
                                 or fixes[q].get("evidence_pages", "[]")))
        pred = []
        for sec in (r.get("sections") or [])[:3]:
            pred += section_pages(sec)
        pred = list(dict.fromkeys(pred))
        corrected.append({
            "qid": q, "abstained": r.get("abstained"),
            "n_sections": len(r.get("sections") or []), "pred_pages": pred,
            "gold_pages": sorted(gold),
            "hit@3": bool(gold & set(pred)),
            "page_recall@3": round(len(gold & set(pred)) / len(gold), 4) if gold else None,
        })

    out = {
        "generated_at": C.now_str(), "n_records": len(records),
        "groups": {k: len(v) for k, v in groups.items()},
        "abstention_rate_unanswerable": abstain_rate(groups["unanswerable"]),
        "over_abstention_rate_answerable": abstain_rate(
            groups["corrected_gold"] + groups["other_answerable"]),
        "corrected_gold_eval": corrected,
        "note": "本评估在预测完成后读取 gold；gold 未进入任何模型输入。",
    }
    C.write_json_atomic(os.path.join(OUT, "route_fill_metrics.json"), out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def write_ocr_plan():
    """导出「当前片段确实无文本、且未缓存 OCR」的页为 OCR 计划 CSV（零 API）。

    优先以最新 gapfill 片段为准，避免使用可能过期的诊断表；
    片段文件不存在时才回退到诊断表。
    """
    rows, seen = [], set()
    if os.path.exists(SNIPPETS_GAP):
        ocr_cached = load_ocr_text()
        for r in C.load_jsonl(SNIPPETS_GAP):
            if r.get("status") != "no_snippet":
                continue
            pages = [int(x) for x in str(r.get("candidate_pages", "")).split(",")
                     if x.strip().isdigit()]
            picked = 0
            for p in pages:
                if (r["doc_id"], p) in seen or (r["doc_id"], p) in ocr_cached:
                    continue
                seen.add((r["doc_id"], p))
                rows.append({"qid": r["qid"], "doc_id": r["doc_id"], "page": p,
                             "need_ocr": "True", "reason": "candidate_page_no_text"})
                picked += 1
                if picked >= 3:
                    break
    else:
        diag = list(csv.DictReader(open(DIAG_CSV, encoding="utf-8-sig")))
        samples = {s["qid"]: s for s in C.load_samples()}
        for d in diag:
            if d["jev_covered"] == "1" or not d["needs_new_ocr"]:
                continue
            s = samples[d["qid"]]
            for p in [int(x) for x in d["needs_new_ocr"].split(",") if x]:
                if (s["doc_id"], p) in seen:
                    continue
                seen.add((s["doc_id"], p))
                rows.append({"qid": d["qid"], "doc_id": s["doc_id"], "page": p,
                             "need_ocr": "True", "reason": "gapfill_candidate_page_no_text"})
    path = os.path.join(OUT, "ocr_page_plan_gapfill.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "doc_id", "page", "need_ocr", "reason"])
        w.writeheader()
        w.writerows(rows)
    print("OCR 计划 %d 页（去重后）-> %s" % (len(rows), path))
    print("试运行：.venv_paddleocr\\Scripts\\python.exe ocr_route_500.py --plan %s --dry-run" % path)
    return 0


def selftest_route():
    """零 API 自检：构造拒答提示词、解析固定返回、验证章节校验与评分函数。"""
    samples = C.load_samples()
    tocs = load_tocs()
    fixes = load_gold_fixes()
    routes = C.last_by_qid(C.load_jsonl(ROUTES))
    targets = [s for s in route_fill_targets(routes, samples) if s["doc_id"] in tocs][:2]
    for s in targets:
        system, user = build_abstain_prompt(s["question"], tocs[s["doc_id"]])
        print("[abstain] %s toc_sections=%d prompt_chars=%d" % (
            s["qid"], len(tocs[s["doc_id"]]["sections"]), len(user)))
    demo = {"answerable": False, "sections": []}
    secs, invalid = base6.validate_sections(demo, {x["section_id"]: x
                                                   for x in tocs[targets[0]["doc_id"]]["sections"]})
    print("[selftest] abstain_json -> sections=%d invalid=%d abstained=%s" % (
        len(secs), len(invalid), demo.get("answerable") is False))
    demo2 = {"answerable": True, "sections": [{"section_id": "S01", "rank": 1}]}
    secs2, invalid2 = base6.validate_sections(demo2, {x["section_id"]: x
                                                      for x in tocs[targets[0]["doc_id"]]["sections"]})
    print("[selftest] answerable_json -> sections=%d invalid=%d" % (len(secs2), len(invalid2)))
    print("[selftest] fixable_gold_qids=%s" % sorted(
        q for q, v in fixes.items() if v.get("status", "").startswith("fixed")))
    return 0


# ---------------------------------------------------------------- 最终状态汇总
def report():
    """汇总 500 题最终状态。

    状态只由「是否真实执行 / 模型真实输出」决定；gold 仅用于标注该题能否离线评分。
    """
    samples = C.load_samples()
    diag = {r["qid"]: r for r in csv.DictReader(open(DIAG_CSV, encoding="utf-8-sig"))}
    routes = C.last_by_qid(C.load_jsonl(ROUTES))
    fill = C.last_by_qid(C.load_jsonl(ROUTES_FILL))
    gap = {r["qid"]: r for r in C.load_jsonl(SNIPPETS_GAP)}
    pred = C.last_by_qid(C.load_jsonl(PRED_GAP))
    fixes = load_gold_fixes()

    rows = []
    for s in samples:
        q = s["qid"]
        answerable = str(s.get("answer", "")) != "Not answerable"
        r = routes.get(q)
        rf = fill.get(q)
        if r and r.get("status") == "ok":
            route_state, route_eval = "executed_ok", "page_recall"
        elif rf and rf.get("success") and rf.get("abstained"):
            route_state, route_eval = "executed_abstain", "abstention_only"
        elif rf and rf.get("success") and rf.get("sections"):
            route_state = "executed_fill"
            route_eval = ("page_recall_corrected_gold"
                          if fixes.get(q, {}).get("status", "").startswith("fixed")
                          else "no_gold_eval")
        elif rf and rf.get("success"):
            route_state, route_eval = "executed_fill_no_section", "none"
        elif rf and not rf.get("success"):
            route_state, route_eval = "api_failed", "none"
        elif diag[q]["route_status"] in ("invalid_gold_pages_skipped",
                                         "missing_gold_pages_skipped") \
                and not fixes.get(q, {}).get("status", "").startswith("fixed"):
            route_state, route_eval = "unfixable_gold_metadata", "none"
        elif diag[q]["route_status"] == "unanswerable_skipped":
            route_state, route_eval = "pending_abstain", "abstention_only"
        else:
            route_state, route_eval = "not_run", "none"

        g = gap.get(q)
        p = pred.get(q)
        if p and p.get("success"):
            jev_state, jev_eval = "executed_gapfill", "judgement"
        elif diag[q]["jev_covered"] == "1":
            jev_state, jev_eval = "executed_existing", "judgement"
        elif g and g.get("status") == "ok":
            jev_state, jev_eval = "built_not_run", "judgement"
        elif g and g.get("status") == "not_applicable":
            jev_state, jev_eval = "not_applicable", g.get("reason", "")
        elif g and g.get("status") == "pending_route":
            jev_state, jev_eval = "pending_route", g.get("reason", "")
        elif g:
            jev_state, jev_eval = "no_usable_snippet", g.get("reason", "")
        else:
            jev_state, jev_eval = "not_run", ""

        rows.append({
            "qid": q, "sample_index": s["sample_index"], "doc_id": s["doc_id"],
            "answerable": int(answerable),
            "route_state": route_state, "route_eval": route_eval,
            "jev_state": jev_state, "jev_eval": jev_eval,
            "jev_sources": diag[q]["jev_sources"],
            "route_status_raw": diag[q]["route_status"],
            "gapfill_page": (g or {}).get("page", ""),
            "gapfill_method": (g or {}).get("extraction_method", ""),
            "gapfill_quality": (g or {}).get("selection_quality", ""),
        })

    C.ensure_dir(OUT)
    csv_path = os.path.join(OUT, "coverage_final_500.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    rs = collections.Counter(r["route_state"] for r in rows)
    js = collections.Counter(r["jev_state"] for r in rows)
    stats = {
        "generated_at": C.now_str(), "n": len(rows),
        "route_state": dict(rs), "jev_state": dict(js),
        "route_executed": sum(v for k, v in rs.items() if k.startswith("executed")),
        "route_potential_page_recall": sum(
            1 for r in rows if r["route_eval"].startswith("page_recall")),
        "route_abstain_executed": rs.get("executed_abstain", 0),
        "route_pending_abstain": rs.get("pending_abstain", 0),
        "jev_executed": sum(v for k, v in js.items() if k.startswith("executed")),
        "jev_ready_to_run": js.get("built_not_run", 0),
        "jev_no_usable_snippet": js.get("no_usable_snippet", 0),
        "jev_not_applicable": js.get("not_applicable", 0),
        "note": "状态分层：状态覆盖 / API 成功 / 有效输入 / 有效评测 分开统计；"
                "Jev 判断准确率仍需人工 confirmed 标注，当前不可计算。",
    }
    C.write_json_atomic(os.path.join(OUT, "coverage_final_stats.json"), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("输出：%s" % csv_path)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-jev", action="store_true")
    ap.add_argument("--build-round2", action="store_true",
                    help="为剩余未执行题重建片段（拒答题走真实目录标题检索）")
    ap.add_argument("--dry-run-round2", type=int, default=0)
    ap.add_argument("--run-round2", action="store_true")
    ap.add_argument("--round2-limit", type=int, default=0,
                    help="每批最多调用条数（建议 20）")
    ap.add_argument("--write-round2-ocr-plan", action="store_true")
    ap.add_argument("--dry-run-jev", type=int, default=0)
    ap.add_argument("--run-jev", action="store_true")
    ap.add_argument("--jev-limit", type=int, default=0,
                    help="仅用于冒烟测试：本次最多跑 N 条")
    ap.add_argument("--jev-qids", default="",
                    help="仅用于冒烟测试：逗号分隔的 qid 白名单")
    ap.add_argument("--plan-route", action="store_true")
    ap.add_argument("--run-route-fill", action="store_true")
    ap.add_argument("--route-limit", type=int, default=0,
                    help="仅用于冒烟测试：本次最多跑 N 道路由")
    ap.add_argument("--route-qids", default="",
                    help="仅用于冒烟测试：逗号分隔的 qid 白名单")
    ap.add_argument("--dry-route-fill", action="store_true",
                    help="离线预览将要执行的路由题（不调用 API）")
    ap.add_argument("--eval-route-fill", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest-route", action="store_true")
    ap.add_argument("--write-ocr-plan", action="store_true")
    a = ap.parse_args()
    if a.build_jev:
        return build_jev_snippets()
    if a.build_round2:
        return build_round2()
    if a.dry_run_round2:
        return dry_run_round2(a.dry_run_round2)
    if a.run_round2:
        return run_round2(limit=a.round2_limit)
    if a.write_round2_ocr_plan:
        return write_round2_ocr_plan()
    if a.dry_run_jev:
        return dry_run_jev(a.dry_run_jev, qids=a.jev_qids or None)
    if a.run_jev:
        return run_jev(limit=a.jev_limit, qids=a.jev_qids or None)
    if a.plan_route:
        return route_plan()
    if a.dry_route_fill:
        return run_route_fill(limit=a.route_limit, qids=a.route_qids or None, dry_run=True)
    if a.run_route_fill:
        return run_route_fill(limit=a.route_limit, qids=a.route_qids or None)
    if a.eval_route_fill:
        return eval_route_fill()
    if a.report:
        return report()
    if a.selftest_route:
        return selftest_route()
    if a.write_ocr_plan:
        return write_ocr_plan()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
