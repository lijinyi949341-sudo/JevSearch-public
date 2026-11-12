# -*- coding: utf-8 -*-
"""
第三轮补全（round3）：为 500 题中尚未成功执行 Jev 的题构造真实证据片段并分批调用。

与已有轮次的关系：
  - 只读取、不修改任何已成功的原始预测（routes.jsonl / jev_predictions_*_500.jsonl）；
  - 缺口集合每次运行都从全部预测文件按唯一 qid 重新计算，已成功的题自动跳过；
  - 新结果写入独立文件 snippets_round3_500.jsonl / jev_predictions_round3_500.jsonl。

选页规则（全部来自真实检索，不使用 gold）：
  1) 已有成功路由的 top-3 章节；
  2) 题干中明确写出的页码（例如 "page 10"、"Pages 40-47"），仅当落在 PDF 页数范围内；
  3) 真实目录（toc.json）按标题词重合度全排序的 top-2 章节（允许 0 分，用于无词重合场景）；
  4) 全文文本层按问题词命中排序的 top-3 页。
  以上候选按「问题词命中数 → 文本长度 → 页码」择优，最多 12 页。

缺文本页（文本层 < 40 字符且无 OCR 缓存）不伪造文本：
  记为 no_text_page，并由 --ocr-plan 导出本地 PaddleOCR 计划（免费、不联网大模型）。

用法：
  py -3.12 fill_gaps_round3_500.py --build
  py -3.12 fill_gaps_round3_500.py --ocr-plan
  py -3.12 fill_gaps_round3_500.py --dry-run 5
  py -3.12 fill_gaps_round3_500.py --run --limit 20      # 付费，每批最多 20
  py -3.12 fill_gaps_round3_500.py --status
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
import task5_jev_500 as t5

OUT = os.path.join(C.OUT_ROOT, "audit")
TOC_JSON = os.path.join(C.OUT_ROOT, "toc", "toc.json")
ROUTES = os.path.join(C.OUT_ROOT, "run500", "toc", "routes.jsonl")
ROUTES_FILL = os.path.join(C.OUT_ROOT, "run500", "toc", "routes_fill_126.jsonl")
JEV_DIR = os.path.join(C.OUT_ROOT, "run500", "jev")
SNIPPETS_R3 = os.path.join(JEV_DIR, "snippets_round3_500.jsonl")
PRED_R3 = os.path.join(JEV_DIR, "jev_predictions_round3_500.jsonl")
OCR_PLAN = os.path.join(OUT, "ocr_page_plan_round3.csv")
OCR_FILES = [
    os.path.join(C.OUT_ROOT, "ocr", "ocr_pages_500.jsonl"),
    os.path.join(C.OUT_ROOT, "ocr", "ocr_pages_500_pilot2.jsonl"),
    os.path.join(C.OUT_ROOT, "ocr", "ocr_pages_round2_500.jsonl"),
]

MAX_PAGES = 12
MIN_CHARS = 40
UNIT_COST_CNY = 0.0030  # 项目已有账单实测均值（元/次）

STOP = set("""a an the is are was were be been being of in on at to for from with by as that this these those
what which who whom whose when where why how many much does do did it its and or not if than then there here
document page pages shown figure table following according based on please answer question""".split())


def words(s):
    return [w for w in re.findall(r"[a-z0-9]+", str(s or "").lower())
            if len(w) > 2 and w not in STOP]


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


# ---------------------------------------------------------------- 状态
def success_qids():
    out = set()
    for name in os.listdir(JEV_DIR):
        if name.startswith("jev_predictions_") and name.endswith(".jsonl"):
            for r in C.load_jsonl(os.path.join(JEV_DIR, name)):
                if r.get("success"):
                    out.add(r["qid"])
    return out


def missing_samples(samples=None):
    samples = samples or C.load_samples()
    ok = success_qids()
    return [s for s in samples if s["qid"] not in ok]


def load_ocr_text():
    """(doc_id, page) -> text（只接受 >= MIN_CHARS 的 ok 结果）。"""
    out = {}
    for p in OCR_FILES:
        for r in C.load_jsonl(p):
            t = norm(r.get("text", ""))
            if r.get("status") == "ok" and len(t) >= MIN_CHARS:
                out[(r["doc_id"], int(r["page"]))] = t
    return out


def load_ocr_for_snippets():
    out = {}
    for p in OCR_FILES:
        for r in C.load_jsonl(p):
            t = norm(r.get("text", ""))
            if r.get("status") == "ok" and t:
                out.setdefault(r["doc_id"], {})[int(r["page"])] = {"ocr_text": t}
    return out


def load_tocs():
    d = C.read_json(TOC_JSON, default={"docs": []}) or {"docs": []}
    return {x["doc_id"]: x for x in d.get("docs", [])
            if x.get("build_status") == "ok" and x.get("sections")}


def route_map():
    routes = {}
    for q, r in C.last_by_qid(C.load_jsonl(ROUTES)).items():
        if r.get("status") == "ok":
            routes[q] = r
    if os.path.exists(ROUTES_FILL):
        for q, r in C.last_by_qid(C.load_jsonl(ROUTES_FILL)).items():
            if r.get("success") and not r.get("abstained"):
                routes[q] = r
    return routes


# ---------------------------------------------------------------- 选页
def section_pages(sec):
    s = int(sec.get("start_page") or 0)
    e = int(sec.get("end_page") or s)
    if s < 1:
        return []
    return list(range(s, max(s, e) + 1))


def question_page_hints(question, num_pages):
    """题干中明确写出的页码（合法范围内的 1-based PDF 页码）。"""
    pages = []
    for a, b in re.findall(r"pages?\s*(\d{1,4})\s*(?:-|~|to)\s*(\d{1,4})", question, re.I):
        for p in range(int(a), int(b) + 1):
            if 1 <= p <= num_pages and p not in pages:
                pages.append(p)
    for p in re.findall(r"pages?\s*(\d{1,4})", question, re.I):
        if 1 <= int(p) <= num_pages and int(p) not in pages:
            pages.append(int(p))
    return pages


def text_layer_scores(doc_id, pages):
    """{page: (text, has_images)}，只读 PDF 文本层。"""
    import pymupdf
    pdf, _ = C.resolve_pdf(doc_id)
    if pdf is None:
        return {}
    out = {}
    try:
        doc = pymupdf.open(pdf)
        for p in pages:
            if p < 1 or p > doc.page_count:
                out[p] = ("", False)
                continue
            pg = doc.load_page(p - 1)
            out[p] = (norm(pg.get_text("text")), bool(pg.get_images()))
        doc.close()
    except Exception:
        return {}
    return out


def full_doc_scan(doc_id, question, cap=3, max_pages=300):
    import pymupdf
    pdf, _ = C.resolve_pdf(doc_id)
    if pdf is None:
        return []
    q = set(words(question))
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


def candidate_pages(sample, toc, route):
    """返回 (pages, origins)。四类真实检索证据按优先级合并，不用 gold。"""
    doc_id, question = sample["doc_id"], sample["question"]
    pages, origins = [], []
    num_pages = int((toc or {}).get("num_pages") or 0)
    if not num_pages:
        pdf, _ = C.resolve_pdf(doc_id)
        num_pages = C.pdf_page_count(pdf) or 0

    def add(p_list, tag):
        for p in p_list:
            if p not in pages and (not num_pages or 1 <= p <= num_pages):
                pages.append(p)
                origins.append(tag)

    secs = (route or {}).get("sections") or []
    for i, sec in enumerate(secs[:3], 1):
        add(section_pages(sec), "route_top%d" % i)
    add(question_page_hints(question, num_pages), "question_page_hint")
    if toc:
        ranked = base6.baseline_rank_all(question, toc["sections"])
        if ranked:
            add(section_pages(ranked[0]), "toc_rank1")
        if len(ranked) > 1:
            add(section_pages(ranked[1]), "toc_rank2")
    add(full_doc_scan(doc_id, question), "full_scan")
    return pages[:MAX_PAGES], origins[:MAX_PAGES]


def ocr_doc_scan(doc_id, question, ocr_text, cap=3):
    """整篇无文本层时的兜底：在本地 OCR 索引上按问题词命中排序（真实 OCR 文本，不用 gold）。"""
    q = set(words(question))
    scored = []
    for (d, p), text in ocr_text.items():
        if d != doc_id or len(text) < MIN_CHARS:
            continue
        scored.append((len(q & set(words(text))), -len(text), p))
    scored.sort(reverse=True)
    return [x[2] for x in scored[:cap]]


def build():
    samples = missing_samples()
    tocs = load_tocs()
    routes = route_map()
    ocr_text = load_ocr_text()
    ocr_for_snippet = load_ocr_for_snippets()
    out = []
    for s in samples:
        q = s["qid"]
        toc = tocs.get(s["doc_id"])
        route = routes.get(q) or {}
        pages, origins = candidate_pages(s, toc, route)
        rec = {
            "snippet_id": "%s__round3" % q, "qid": q, "sample_index": s["sample_index"],
            "doc_id": s["doc_id"], "doc_type": s.get("doc_type", ""),
            "question": s["question"], "mode": "round3", "round": 3,
            "route_state": route.get("status"),
            "selection_uses_gold": False,
            "selection_rule": "路由 top-3 章节 → 题干显式页码 → 目录全排序 top-2 章节 → 全文词命中 top-3 页；"
                              "按问题词命中/文本长度/页码择优；不使用 gold 证据页",
            "page": None, "section_id": None, "status": "", "reason": "",
            "evidence_snippet": "", "char_len": 0, "extraction_method": "",
            "has_page_images": False, "candidate_pages": ",".join(map(str, pages)),
            "candidate_origin": ",".join(origins), "retrieval_basis": ",".join(
                sorted(set(o.split("_top")[0] for o in origins))) or "none",
            "question_page_hint": ",".join(map(str, question_page_hints(
                s["question"], int((toc or {}).get("num_pages") or 0)))),
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
        if not pages:
            rec.update({"status": "no_candidate_page", "reason": "no_toc_no_hint_no_hit"})
            out.append(rec)
            continue
        layers = text_layer_scores(s["doc_id"], pages)
        qwords = set(words(s["question"]))

        def score_pages(page_list):
            out = []
            for p in page_list:
                text, method = layers.get(p, ("", False))[0], "text_layer"
                if len(text) < MIN_CHARS:
                    cached = ocr_text.get((s["doc_id"], p), "")
                    if not cached:
                        continue
                    text, method = cached, "ocr"
                out.append((len(qwords & set(words(text))), len(text), p, method))
            return out

        scored = score_pages(pages)
        if not scored:
            # 整篇无文本层：改用本地 OCR 索引检索候选页，候选页与依据一并记录
            extra = [p for p in ocr_doc_scan(s["doc_id"], s["question"], ocr_text)
                     if p not in pages]
            if extra:
                pages = (pages + extra)[:MAX_PAGES]
                origins = origins + ["ocr_doc_scan"] * len(extra)
                rec["candidate_pages"] = ",".join(map(str, pages))
                rec["candidate_origin"] = ",".join(origins[:len(pages)])
                rec["retrieval_basis"] = ",".join(sorted(set(
                    o.split("_top")[0] for o in origins[:len(pages)])))
                layers.update(text_layer_scores(s["doc_id"], extra))
                scored = score_pages(pages)
        if not scored:
            rec.update({"status": "no_text_page", "reason": "candidate_pages_have_no_text"})
            out.append(rec)
            continue
        # 全 0 命中时按候选顺序取第一个（保留检索依据），否则按命中/长度排序
        top_hit = max(x[0] for x in scored)
        if top_hit == 0:
            best = next(x for x in scored if x[2] == pages[0]) if any(
                x[2] == pages[0] for x in scored) else scored[0]
        else:
            best = sorted(scored, key=lambda x: (-x[0], -x[1], x[2]))[0]
        hits, _, page, method = best
        text, extract_method, has_images, truncated = t5.extract_page_text(
            s["doc_id"], page, ocr_for_snippet)
        if not text:
            rec.update({"status": "no_text_page", "reason": extract_method,
                        "has_page_images": has_images})
            out.append(rec)
            continue
        rec.update({"page": page, "selection_quality": "keyword_hit" if hits else "retrieval_prior",
                    "keyword_hits": hits, "evidence_snippet": text, "char_len": len(text),
                    "extraction_method": extract_method, "has_page_images": has_images,
                    "truncated": truncated, "status": "ok", "reason": ""})
        out.append(rec)

    C.ensure_dir(JEV_DIR)
    tmp = SNIPPETS_R3 + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, SNIPPETS_R3)
    cnt = collections.Counter(r["status"] for r in out)
    ok = [r for r in out if r["status"] == "ok"]
    print("[round3] 缺口 %d 题：%s" % (len(out), dict(cnt)))
    print("[round3] 可送 %d 条（keyword_hit=%d, retrieval_prior=%d, ocr=%d）" % (
        len(ok), sum(1 for r in ok if r["selection_quality"] == "keyword_hit"),
        sum(1 for r in ok if r["selection_quality"] == "retrieval_prior"),
        sum(1 for r in ok if r["extraction_method"] == "ocr")))
    print("输出：%s" % SNIPPETS_R3)
    return 0


def write_ocr_plan():
    """为「候选页全部无文本且无 OCR 缓存」的题导出 OCR 计划（每页一次，最多 3 页/题）。"""
    rows, seen = [], set()
    for r in C.load_jsonl(SNIPPETS_R3):
        if r.get("status") != "no_text_page":
            continue
        pages = [int(x) for x in str(r.get("candidate_pages", "")).split(",") if x.strip().isdigit()]
        picked = 0
        for p in pages:
            if (r["doc_id"], p) in seen:
                continue
            seen.add((r["doc_id"], p))
            rows.append({"qid": r["qid"], "doc_id": r["doc_id"], "page": p,
                         "need_ocr": "True", "reason": "round3_candidate_page_no_text"})
            picked += 1
            if picked >= 3:
                break
    with open(OCR_PLAN, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "doc_id", "page", "need_ocr", "reason"])
        w.writeheader()
        w.writerows(rows)
    print("[ocr-plan] %d 页 / %d 题 -> %s" % (
        len(rows), len({r['qid'] for r in rows}), OCR_PLAN))
    return 0


def dry_run(n):
    import run_task5_jev as base5
    rows = [r for r in C.load_jsonl(SNIPPETS_R3) if r.get("status") == "ok"]
    rows = [r for r in rows if r["qid"] not in success_qids()]
    print("[dry-run] 待执行 %d 条，本次检查 %d 条，不调用 API。" % (len(rows), min(n, len(rows))))
    bad = 0
    for r in rows[:n]:
        js = t5.to_jev_sample(r)
        state = base5.build_state(js)
        problems, gold_in = base5.leak_check(js, state)
        bad += 1 if problems else 0
        print("-" * 60)
        print("%s page=%s src=%s hits=%s basis=%s" % (
            r["snippet_id"], r["page"], r.get("extraction_method"),
            r.get("keyword_hits"), r.get("retrieval_basis")))
        print("发送内容：", json.dumps(state, ensure_ascii=False)[:400])
        print("泄漏检查：", "PASS" if not problems else "FAIL %s" % problems,
              "| gold_answer_visible_in_snippet=%s" % bool(gold_in))
    print("[dry-run] 泄漏检查失败 %d 条" % bad)
    return 0


def run(limit):
    covered = success_qids()
    print("[round3] 历史已成功 %d 题，本次按 qid 跳过" % len(covered))
    t5.run_jev("round3", limit=limit, snippets_file=SNIPPETS_R3,
               out_file=PRED_R3, exclude_qids=covered)
    return metrics()


def metrics():
    rows = C.load_jsonl(PRED_R3)
    last = C.last_by_qid(rows)
    ok = [r for r in last.values() if r.get("success")]
    fail = [r for r in last.values() if not r.get("success")]
    tin = sum(r.get("input_tokens") or 0 for r in ok)
    tout = sum(r.get("output_tokens") or 0 for r in ok)
    out = {
        "generated_at": C.now_str(),
        "file": PRED_R3, "records_total": len(rows), "unique_qid": len(last),
        "success_unique_qid": len(ok), "failed_unique_qid": len(fail),
        "failed_qids": sorted(r["qid"] for r in fail),
        "duplicate_success_qid": len(ok) - len({r["qid"] for r in ok}),
        "input_tokens": tin, "output_tokens": tout,
        "estimated_cost_cny": round(len(ok) * UNIT_COST_CNY, 4),
        "cost_basis": "0.0030 元/次（项目已有账单实测均值，非官方报价）",
        "leak_check_failures": sum(1 for r in ok if not r.get("leak_check_passed")),
        "gold_answer_visible_in_snippet": sum(
            1 for r in ok if r.get("gold_answer_visible_in_snippet")),
        "note": "gold_answer_visible 表示答案文本天然出现在真实片段中，不是通过 gold 选页。",
    }
    C.write_json_atomic(os.path.join(OUT, "round3_metrics.json"), out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def status():
    samples = C.load_samples()
    ok = success_qids()
    r3 = {r["qid"]: r for r in C.load_jsonl(SNIPPETS_R3)}
    miss = [s for s in samples if s["qid"] not in ok]
    states = collections.Counter()
    detail = []
    for s in miss:
        r = r3.get(s["qid"], {})
        st = r.get("status", "not_built")
        states[st] += 1
        detail.append({"qid": s["qid"], "doc_id": s["doc_id"],
                       "answerable": str(s.get("answer", "")) != "Not answerable",
                       "status": st, "reason": r.get("reason", ""),
                       "candidate_pages": r.get("candidate_pages", "")})
    out = {"generated_at": C.now_str(), "n": len(samples),
           "jev_success_unique_qid": len(ok), "jev_missing": len(miss),
           "missing_by_answerable": dict(collections.Counter(
               "answerable" if str(s.get("answer", "")) != "Not answerable" else "unanswerable"
               for s in miss)),
           "round3_states": dict(states)}
    C.write_json_atomic(os.path.join(OUT, "round3_status.json"), out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if len(sys.argv) > 2 and sys.argv[2] == "--detail":
        for d in detail:
            print(d)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ocr-plan", action="store_true")
    ap.add_argument("--dry-run", type=int, default=0)
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.build:
        return build()
    if a.ocr_plan:
        return write_ocr_plan()
    if a.dry_run:
        return dry_run(a.dry_run)
    if a.run:
        return run(a.limit)
    if a.metrics:
        return metrics()
    if a.status:
        return status()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
