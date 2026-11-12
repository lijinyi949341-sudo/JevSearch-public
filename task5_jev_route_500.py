# -*- coding: utf-8 -*-
"""
Jev 证据判断补全（500 题）：用已成功的目录路由结果，从真实 PDF 提取正文片段，补齐 Jev 输入。

设计约束：
  - 只读：README/samples、run500 的 routes.jsonl、snippets_title_500.jsonl、PDF 缓存；
  - 只写新文件：snippets_route_500.jsonl、qid_route_status_500.csv、
    ROUTE_COVERAGE_REPORT.md；不覆盖任何已有结果；
  - 不使用 gold_answer / gold evidence_pages / 人工标签 来选择证据；
    片段全部来自 PDF 文本层（或已有本地 OCR），选择依据只有目录路由排序；
  - 已有 title 片段的题不重复生成（状态 already_covered），保证与 263 条不重复；
  - 无真实证据可取的题保留 no_input 状态，不伪造片段。

证据选择（确定性）：
  对路由 sections 按返回顺序（即模型排序）取前 3 个章节；
  每个章节内按页序取 start_page 起 3 页（不超过 end_page）；
  第一页文本 ≥40 字符即采用；否则继续；全部不足则记 no_text_page。

用法（全部离线，不调用任何付费 API）：
  python task5_jev_route_500.py --build
  python task5_jev_route_500.py --verify
  python task5_jev_route_500.py --report
"""
import argparse
import collections
import csv
import json
import os
import re

import pymupdf

import final500_common as C
import task5_jev_500 as t5

JEVDIR = os.path.join(C.OUT_ROOT, "run500", "jev")
SNIP_TITLE = os.path.join(JEVDIR, "snippets_title_500.jsonl")
# v1 首次提取；v2 页眉/目录/句界修复；v3 在 v2 基础上增加问题词窗口与人工语义复核标记
SNIP_ROUTE_V1 = os.path.join(JEVDIR, "snippets_route_500_v1.jsonl")
SNIP_ROUTE_V2 = os.path.join(JEVDIR, "snippets_route_500_v2.jsonl")
SNIP_ROUTE = os.path.join(JEVDIR, "snippets_route_500_v3.jsonl")
ROUTES = os.path.join(C.OUT_ROOT, "run500", "toc", "routes.jsonl")
STATUS_CSV = os.path.join(C.OUT_ROOT, "audit", "qid_route_status_500_v3.csv")
REPORT_MD = os.path.join(C.OUT_ROOT, "audit", "ROUTE_COVERAGE_REPORT_v3.md")
QUALITY_CSV = os.path.join(C.OUT_ROOT, "audit", "route_snippet_quality_v3.csv")
OCR_CSV = os.path.join(C.OUT_ROOT, "audit", "ocr_page_plan_500_v3.csv")

# 人工语义复核后判定“纯文本不足以判断”的样本：保留片段供审计，但不计入 Jev 可调用输入
SEMANTIC_EXCLUDE = {
    "Q0051": "扫描签名页，文本层仅 ECF 页眉；需 OCR 或图像输入",
    "Q0158": "图表页仅标题与来源；问题为图形计数，需图像输入",
    "Q0302": "图像页仅分类标签；答案在图片中，需图像输入",
    "Q0461": "问题为流程图内设备数量；文本仅为场景说明，需图像输入",
}

MAX_SECTIONS = 3
MAX_PAGES_PER_SECTION = 3
MIN_CHARS = 40
MAX_CHARS = t5.MAX_CHARS


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def load_last(path, key):
    out = {}
    for r in C.load_jsonl(path):
        out[r[key]] = r
    return out


def page_text(doc_id, page, ocr):
    """返回 (text, method, has_images)；text=None 表示无可用文本。"""
    pdf, _ = C.resolve_pdf(doc_id)
    if pdf is None:
        return None, "pdf_missing", False
    try:
        doc = pymupdf.open(pdf)
    except Exception as e:
        return None, f"pdf_unreadable:{type(e).__name__}", False
    try:
        if page < 1 or page > doc.page_count:
            return None, "page_out_of_range", False
        p = doc.load_page(page - 1)
        raw = norm(p.get_text("text"))
        has_img = bool(p.get_images())
    finally:
        doc.close()
    if len(raw) >= MIN_CHARS:
        return raw[:MAX_CHARS], "text_layer", has_img
    item = ocr.get(doc_id, {}).get(page)
    if item and norm(item.get("ocr_text", "")):
        ocr_raw = norm(item["ocr_text"])
        return ocr_raw[:MAX_CHARS], "ocr", has_img
    return None, "empty_text_page", has_img


def build():
    samples = C.load_samples()
    title = load_last(SNIP_TITLE, "qid")
    routes = load_last(ROUTES, "qid")
    ocr = t5.load_ocr()

    counts = collections.Counter()
    rows = []
    for s in samples:
        qid = s["qid"]
        t = title.get(qid)
        t_ok = bool(t and t.get("status") == "ok")
        r = routes.get(qid)
        r_status = (r or {}).get("status", "missing")
        unus = str(s.get("answer", "")) == "Not answerable"
        rec = {
            "snippet_id": f"{qid}__route",
            "qid": qid, "sample_index": s["sample_index"],
            "doc_id": s["doc_id"], "doc_type": s.get("doc_type", ""),
            "question": s["question"], "mode": "route",
            "selection_uses_gold": False,
            "selection_rule": ("目录路由 top-1 章节优先，必要时 top-2/top-3；"
                               "章节内取 start_page 起最多 3 页的真实 PDF 文本层，不使用 gold"),
            "evidence_source": "", "route_rank": None, "route_section_id": None,
            "route_section_title": "", "route_sections": [], "pages_tried": [],
            "page": None, "section_id": None,
            "status": "", "reason": "",
            "evidence_snippet": "", "char_len": 0,
            "extraction_method": "", "has_page_images": False, "truncated": False,
            "evidence_type_primary": t5.primary_type(
                C.parse_sources(s.get("evidence_sources", "[]"))),
            "cross_page": len(C.parse_pages(s.get("evidence_pages", "[]"))) > 1,
            "duplicate_of": t["snippet_id"] if t_ok else "",
            "created_at": C.now_str(),
            "offline_only": {
                "answer": s.get("answer", ""),
                "gold_answer": s.get("answer", ""),
                "evidence_pages": s.get("evidence_pages", "[]"),
                "evidence_sources": s.get("evidence_sources", ""),
                "answerable": not unus,
            },
        }
        secs = (r or {}).get("sections") or []
        rec["route_sections"] = [{"section_id": x.get("section_id"),
                                  "title": str(x.get("title", ""))[:120],
                                  "start_page": x.get("start_page"),
                                  "end_page": x.get("end_page")} for x in secs[:MAX_SECTIONS]]

        if t_ok:
            rec["status"] = "already_covered"
            rec["reason"] = "已有 title 片段（本文件不重复生成，避免与 263 条重复）"
        elif unus:
            rec["status"] = "no_input"
            rec["reason"] = ("不可答题且无路由结果：路由按原规则未对不可答题调用，"
                             "没有真实检索片段可送入 Jev")
        elif r_status != "ok" or not secs:
            rec["status"] = "no_input"
            rec["reason"] = f"路由不可用：status={r_status}"
        else:
            found = None
            skipped_toc = []
            header = doc_header(s["doc_id"])
            for rank, sec in enumerate(secs[:MAX_SECTIONS], start=1):
                start = int(sec["start_page"])
                end = max(start, int(sec["end_page"]))
                for pg in range(start, min(end, start + MAX_PAGES_PER_SECTION - 1) + 1):
                    rec["pages_tried"].append(pg)
                    text, method, has_img = page_text(s["doc_id"], pg, ocr)
                    if not text:
                        continue
                    if is_toc_page(text):
                        skipped_toc.append(pg)
                        continue
                    if method == "ocr":
                        cleaned = text
                    else:
                        cleaned = clean_text(text, header)
                    if len(cleaned) < MIN_CHARS:
                        continue
                    snippet, truncated = cut_snippet(cleaned)
                    window_applied = False
                    if method == "text_layer" and len(cleaned) > MAX_CHARS + 600:
                        terms = question_terms(s["question"])
                        base = len(kw_hits(snippet, terms))
                        win, score = best_window(cleaned, terms)
                        if score >= base + 2:
                            snippet, truncated, window_applied = win.strip(), True, True
                    found = (rank, sec, pg, snippet, method, has_img, truncated, window_applied)
                    break
                if found:
                    break
            if found:
                rank, sec, pg, snippet, method, has_img, truncated, window_applied = found
                rec.update({
                    "status": "ok", "evidence_source": f"route_top{rank}",
                    "route_rank": rank, "page": pg, "section_id": sec.get("section_id"),
                    "route_section_id": sec.get("section_id"),
                    "route_section_title": str(sec.get("title", ""))[:120],
                    "evidence_snippet": snippet, "char_len": len(snippet),
                    "extraction_method": "text_layer_query_window" if window_applied
                    else method,
                    "has_page_images": has_img, "window_applied": window_applied,
                    "truncated": truncated, "skipped_toc_pages": skipped_toc,
                })
                if rec["qid"] in SEMANTIC_EXCLUDE:
                    rec["status"] = "visual_evidence_only"
                    rec["reason"] = SEMANTIC_EXCLUDE[rec["qid"]]
            else:
                rec["status"] = "no_text_page"
                rec["reason"] = "路由预测页在文本层与本地 OCR 中均不足 40 有效字符"
                rec["skipped_toc_pages"] = skipped_toc
        counts[rec["status"]] += 1
        rows.append(rec)

    C.ensure_dir(JEVDIR)
    tmp = SNIP_ROUTE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, SNIP_ROUTE)

    C.ensure_dir(os.path.dirname(STATUS_CSV))
    with open(STATUS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "doc_id", "answerable", "title_status", "route_status",
                    "route_sections", "route_pages", "chosen_page", "evidence_source",
                    "extraction_method", "char_len", "jev_input_status", "reason"])
        for r, s in zip(rows, samples):
            t = title.get(r["qid"])
            ro = routes.get(r["qid"])
            rp = sorted({p for sec in (ro or {}).get("sections", [])[:MAX_SECTIONS]
                         for p in range(int(sec["start_page"]),
                                        int(sec["end_page"]) + 1)})
            w.writerow([r["qid"], r["doc_id"], not r["offline_only"]["answerable"],
                        (t or {}).get("status", "missing"),
                        (ro or {}).get("status", "missing"),
                        len((ro or {}).get("sections", [])), rp,
                        r["page"] or "", r["evidence_source"], r["extraction_method"],
                        r["char_len"], r["status"], r["reason"]])
    print("[build] 状态分布:", dict(counts))
    print("[build] 新增有效输入:", counts["ok"])
    print(f"[build] 片段文件: {SNIP_ROUTE}\n[build] 状态表: {STATUS_CSV}")
    return rows


def verify():
    rows = C.load_jsonl(SNIP_ROUTE)
    title = {r["qid"] for r in C.load_jsonl(SNIP_TITLE) if r.get("status") == "ok"}
    ok = [r for r in rows if r["status"] == "ok"]
    qids = [r["qid"] for r in rows]
    problems = []
    if len(rows) != 500:
        problems.append(f"行数 {len(rows)} != 500")
    if len(set(qids)) != 500:
        problems.append(f"qid 唯一数 {len(set(qids))} != 500")
    dup = [r["qid"] for r in ok if r["qid"] in title]
    if dup:
        problems.append(f"与 title 有效片段重复: {dup}")
    if any(r["selection_uses_gold"] for r in ok):
        problems.append("存在 selection_uses_gold=true 的片段")
    mismatch = []
    for r in ok:
        text, method, _ = page_text(r["doc_id"], r["page"], {})
        if method == "text_layer":
            key = norm(r["evidence_snippet"])[:60]
            if key and key not in norm(text):
                mismatch.append(r["qid"])
    if mismatch:
        problems.append(f"片段与 PDF 文本不一致: {mismatch[:5]}(共{len(mismatch)})")
    print(f"[verify] 行数={len(rows)} 唯一qid={len(set(qids))} ok={len(ok)} "
          f"与title重复={len(dup)} 文本不一致={len(mismatch)}")
    print("[verify] 结论:", "通过" if not problems else "失败 -> " + "; ".join(problems))
    for r in ok[:10]:
        print(f"  {r['qid']} page={r['page']} {r['evidence_source']} "
              f"{r['extraction_method']} chars={r['char_len']} | "
              f"{r['evidence_snippet'][:70]}")
    return problems


END_PUNCT = tuple(".!?;:。！？；：)）]】”\"'")
TOC_PAT = re.compile(r"table\s+of\s+contents|^contents$|目\s*录", re.I)
SENT_END = re.compile(r"[.!?。！？；]\s")
DOC_HEADER = {}


def doc_header(doc_id):
    """按页首行频率判断页眉：出现在 >=25% 且 >=3 页的首行视为页眉。"""
    if doc_id in DOC_HEADER:
        return DOC_HEADER[doc_id]
    header = ""
    pdf, _ = C.resolve_pdf(doc_id)
    if pdf:
        try:
            doc = pymupdf.open(pdf)
            n = min(doc.page_count, 25)
            firsts = []
            for i in range(n):
                lines = [x.strip() for x in doc.load_page(i).get_text("text").splitlines() if x.strip()]
                if lines:
                    firsts.append(norm(lines[0])[:80])
            doc.close()
            if firsts:
                cand, cnt = collections.Counter(firsts).most_common(1)[0]
                if cnt >= max(3, int(0.25 * len(firsts))) and len(cand) >= 3:
                    header = cand
        except Exception:
            pass
    DOC_HEADER[doc_id] = header
    return header


def is_toc_page(raw):
    """真正的目录页：条目式点线/纯页码行占比高，而非仅在页眉写了 Table of Contents。"""
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    if not lines:
        return False
    dotted = sum(1 for x in lines if re.search(r"(\.{4,}|…{3,})\s*\d{1,4}\s*$", x))
    short_num = sum(1 for x in lines if re.match(r"^.{0,80}\s+\d{1,4}$", x))
    if len(lines) >= 5 and (dotted + short_num) / len(lines) >= 0.5:
        return True
    return bool(TOC_PAT.search(lines[0]) and len(lines) <= 6 and len(norm(raw)) < 400)


def clean_text(raw, header):
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    if lines and header and norm(lines[0])[:80] == header:
        lines = lines[1:]
    while lines and (re.fullmatch(r"(page\s*)?[0-9ivxlcdm]{1,6}", lines[-1], re.I)):
        lines.pop()
    while lines and (re.fullmatch(r"(page\s*)?[0-9ivxlcdm]{1,6}", lines[0], re.I)):
        lines.pop(0)
    return norm(" ".join(lines))


def cut_snippet(text, limit=MAX_CHARS):
    """优先在句子边界截断；找不到时退回硬截断。"""
    if len(text) <= limit:
        return text, False
    window = text[:limit]
    cuts = [m.end() for m in re.finditer(r"[.!?。！？；]", window)]
    if cuts and cuts[-1] >= int(limit * 0.6):
        return window[:cuts[-1]].strip(), True
    return window, True


QSTOP = set(("the a an of in on for to and or is are was were what which how many much "
             "does do did this that these those with from by as at be been it its page "
             "according report document").split())
QSTEM = {"warehouses": "warehouse", "devices": "device", "boxes": "box", "disks": "disk",
         "graphs": "graph", "charts": "chart", "pictures": "picture", "images": "image"}


def question_terms(q):
    return {QSTEM.get(w, w) for w in re.findall(r"[a-zA-Z0-9]{2,}", str(q).lower())
            if w not in QSTOP}


def kw_hits(text, terms):
    low = str(text).lower()
    return [w for w in terms if re.search(r"\b" + re.escape(w[:6]), low)]


def best_window(text, terms, size=MAX_CHARS, step=200):
    best, score, best_start = text[:size], 0, 0
    for start in range(0, max(1, len(text) - size + 1), step):
        seg = text[start:start + size]
        sc = len(kw_hits(seg, terms))
        if sc > score:
            best, score, best_start = seg, sc, start
    if best_start > 0:
        # 去掉窗口开头的半句，避免以半个单词开头
        cut = re.search(r"[.!?。！？;]\s", best[:260])
        if cut:
            best = best[cut.end():].strip()
    return best, score


def page_text(doc_id, page, ocr):
    """返回 (full_text, method, has_images)；full_text 未截断，便于后续清洗与切句。"""
    pdf, _ = C.resolve_pdf(doc_id)
    if pdf is None:
        return None, "pdf_missing", False
    try:
        doc = pymupdf.open(pdf)
    except Exception as e:
        return None, f"pdf_unreadable:{type(e).__name__}", False
    try:
        if page < 1 or page > doc.page_count:
            return None, "page_out_of_range", False
        p = doc.load_page(page - 1)
        raw = norm(p.get_text("text"))
        has_img = bool(p.get_images())
    finally:
        doc.close()
    if len(raw) >= MIN_CHARS:
        return raw, "text_layer", has_img
    item = ocr.get(doc_id, {}).get(page)
    if item and norm(item.get("ocr_text", "")):
        return norm(item["ocr_text"]), "ocr", has_img
    return None, "empty_text_page", has_img


def quality():
    """离线核验新增片段：来源真实性、目录页、页眉页码、截断、同页重复。"""
    rows = [r for r in C.load_jsonl(SNIP_ROUTE) if r["status"] == "ok"]
    checks = []
    seen = collections.Counter((r["doc_id"], norm(r["evidence_snippet"])) for r in rows)
    for r in rows:
        text = r["evidence_snippet"]
        flags = []
        pdf, _ = C.resolve_pdf(r["doc_id"])
        doc = None
        page_ok = False
        if pdf:
            try:
                doc = pymupdf.open(pdf)
                page_ok = 1 <= r["page"] <= doc.page_count
            except Exception:
                pass
        if not page_ok:
            flags.append("page_out_of_pdf")
        else:
            raw = norm(doc.load_page(r["page"] - 1).get_text("text"))
            cleaned = clean_text(raw, doc_header(r["doc_id"]))
            key = norm(text)[:60]
            if key and key not in cleaned:
                flags.append("pdf_text_mismatch")
        if is_toc_page(text) or (TOC_PAT.search(text[:40]) and len(text) < 500):
            flags.append("toc_like")
        lines = [x.strip() for x in re.split(r"[。\n]", text) if x.strip()]
        junk = sum(1 for x in lines if len(x) <= 3)
        if lines and junk / len(lines) > 0.7:
            flags.append("boilerplate_dominant")
        if r.get("truncated") and text[-1] not in END_PUNCT:
            flags.append("truncated_mid_sentence")
        if seen[(r["doc_id"], norm(text))] > 1:
            flags.append("same_page_as_other_qid")
        if r.get("has_page_images") and r["char_len"] < 300:
            flags.append("visual_page_short_text")
        if doc:
            doc.close()
        blocking = {"page_out_of_pdf", "pdf_text_mismatch", "toc_like",
                    "boilerplate_dominant"}
        checks.append({"qid": r["qid"], "page": r["page"],
                        "source": r["evidence_source"], "chars": r["char_len"],
                        "flags": flags, "pass": not (blocking & set(flags))})
    qcsv = QUALITY_CSV
    C.ensure_dir(os.path.dirname(qcsv))
    with open(qcsv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "page", "evidence_source", "char_len", "flags", "result"])
        for c in checks:
            w.writerow([c["qid"], c["page"], c["source"], c["chars"],
                        ";".join(c["flags"]), "pass" if c["pass"] else "fail"])
    passed = sum(1 for c in checks if c["pass"])
    fc = collections.Counter(flag for c in checks for flag in c["flags"])
    print(f"[quality] 新增片段 {len(checks)} 条，通过 {passed}，需关注 {len(checks)-passed}")
    print("[quality] 问题分布:", dict(fc))
    print("[quality] 明细:", qcsv)
    for c in checks:
        if not c["pass"]:
            print("  失败:", c["qid"], "p.%s" % c["page"], c["flags"])
    return checks


def ocr_prep():
    """离线确认 76 题缺文本页的页码、文本长度与是否已有 OCR，不执行 OCR。"""
    rows = C.load_jsonl(SNIP_ROUTE)
    ocr = t5.load_ocr()
    plan = []
    for r in rows:
        if r["status"] != "no_text_page":
            continue
        for pg in r["pages_tried"]:
            text = None
            pdf, _ = C.resolve_pdf(r["doc_id"])
            if pdf:
                try:
                    doc = pymupdf.open(pdf)
                    if 1 <= pg <= doc.page_count:
                        p = doc.load_page(pg - 1)
                        text = norm(p.get_text("text"))
                    doc.close()
                except Exception:
                    pass
            has_ocr = bool(ocr.get(r["doc_id"], {}).get(pg))
            plan.append({"qid": r["qid"], "doc_id": r["doc_id"], "page": pg,
                         "text_chars": len(text or ""),
                         "has_ocr": has_ocr,
                         "need_ocr": len(text or "") < MIN_CHARS and not has_ocr})
    need = [x for x in plan if x["need_ocr"]]
    uniq = {(x["doc_id"], x["page"]) for x in need}
    qcsv = OCR_CSV
    C.ensure_dir(os.path.dirname(qcsv))
    with open(qcsv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "doc_id", "page", "text_layer_chars", "has_ocr", "need_ocr"])
        for x in plan:
            w.writerow([x["qid"], x["doc_id"], x["page"], x["text_chars"],
                        x["has_ocr"], x["need_ocr"]])
    print(f"[ocr-prep] 覆盖 {len(rows)} 行中的 no_text_page；候选页 {len(plan)}，"
          f"需 OCR {len(need)} 页，去重后 {len(uniq)} 个(doc,page)")
    print(f"[ocr-prep] 计划表: {qcsv}")
    return plan


ROUTE_PRED = os.path.join(JEVDIR, "jev_predictions_route_500.jsonl")
OCR_SNIP = os.path.join(JEVDIR, "snippets_ocr_500.jsonl")
OCR_PRED = os.path.join(JEVDIR, "jev_predictions_ocr_500.jsonl")
RECOVER_SNIP = os.path.join(JEVDIR, "snippets_recover_500.jsonl")
RECOVER_PRED = os.path.join(JEVDIR, "jev_predictions_recover_500.jsonl")
PILOT1_SNIP = os.path.join(JEVDIR, "snippets_ocr_pilot_500.jsonl")
PILOT1_PRED = os.path.join(JEVDIR, "jev_predictions_ocr_pilot_500.jsonl")
PILOT2_SNIP = os.path.join(JEVDIR, "snippets_ocr_pilot2_500.jsonl")
PILOT2_PRED = os.path.join(JEVDIR, "jev_predictions_ocr_pilot2_500.jsonl")
TITLE_PRED = os.path.join(JEVDIR, "jev_predictions_title_500.jsonl")
MERGE_MD = os.path.join(C.OUT_ROOT, "audit", "JEV_COVERAGE_MERGED.md")
MERGE_JSON = os.path.join(C.OUT_ROOT, "audit", "JEV_COVERAGE_MERGED.json")


def _dist(preds):
    ok = [r for r in preds if r.get("success")]
    raw = collections.Counter(r.get("next_action") for r in ok)
    norm = collections.Counter()
    for r in ok:
        if r.get("relevant") is False:
            norm["search"] += 1
        elif r.get("relevant") is True and r.get("sufficient") is True:
            norm["answer"] += 1
        else:
            norm["expand"] += 1
    return {
        "success": len(ok),
        "relevant_true": sum(1 for r in ok if r.get("relevant") is True),
        "self_contained_true": sum(1 for r in ok if r.get("self_contained") is True),
        "sufficient_true": sum(1 for r in ok if r.get("sufficient") is True),
        "raw_action": dict(raw),
        "normalized_action": dict(norm),
        "conflicts": sum(1 for r in ok if r.get("next_action") != (
            "search" if r.get("relevant") is False else
            "answer" if (r.get("relevant") is True and r.get("sufficient") is True)
            else "expand")),
        "input_tokens": sum(int(r.get("input_tokens") or 0) for r in ok),
        "output_tokens": sum(int(r.get("output_tokens") or 0) for r in ok),
        "latency_sec": round(sum(float(r.get("latency_sec") or 0) for r in ok), 2),
    }


def merge_report():
    title_snips = [r for r in C.load_jsonl(SNIP_TITLE) if r.get("status") == "ok"]
    route_snips = [r for r in C.load_jsonl(SNIP_ROUTE) if r.get("status") == "ok"]
    ocr_snips = ([r for r in C.load_jsonl(OCR_SNIP) if r.get("status") == "ok"]
                 if os.path.exists(OCR_SNIP) else [])
    recover_snips = ([r for r in C.load_jsonl(RECOVER_SNIP) if r.get("status") == "ok"]
                     if os.path.exists(RECOVER_SNIP) else [])
    pilot1_snips = ([r for r in C.load_jsonl(PILOT1_SNIP) if r.get("status") == "ok"]
                    if os.path.exists(PILOT1_SNIP) else [])
    pilot2_snips = ([r for r in C.load_jsonl(PILOT2_SNIP) if r.get("status") == "ok"]
                    if os.path.exists(PILOT2_SNIP) else [])
    title_pred_rows = C.load_jsonl(TITLE_PRED) if os.path.exists(TITLE_PRED) else []
    route_pred_rows = C.load_jsonl(ROUTE_PRED) if os.path.exists(ROUTE_PRED) else []
    ocr_pred_rows = C.load_jsonl(OCR_PRED) if os.path.exists(OCR_PRED) else []
    recover_pred_rows = C.load_jsonl(RECOVER_PRED) if os.path.exists(RECOVER_PRED) else []
    pilot1_pred_rows = C.load_jsonl(PILOT1_PRED) if os.path.exists(PILOT1_PRED) else []
    pilot2_pred_rows = C.load_jsonl(PILOT2_PRED) if os.path.exists(PILOT2_PRED) else []
    t_pred = _dist({r["snippet_id"]: r for r in title_pred_rows}.values())
    r_pred = _dist({r["snippet_id"]: r for r in route_pred_rows}.values())
    o_pred = _dist({r["snippet_id"]: r for r in ocr_pred_rows}.values())
    v_pred = _dist({r["snippet_id"]: r for r in recover_pred_rows}.values())
    p1_pred = _dist({r["snippet_id"]: r for r in pilot1_pred_rows}.values())
    p2_pred = _dist({r["snippet_id"]: r for r in pilot2_pred_rows}.values())
    attempt = collections.Counter(r["snippet_id"] for r in route_pred_rows)
    route_failed = sum(1 for sid, c in attempt.items()
                       if c and not any(r.get("success") for r in route_pred_rows
                                        if r["snippet_id"] == sid))
    valid_inputs = (len(title_snips) + len(route_snips) + len(ocr_snips)
                    + len(recover_snips) + len(pilot1_snips) + len(pilot2_snips))
    api_success = (t_pred["success"] + r_pred["success"] + o_pred["success"]
                   + v_pred["success"] + p1_pred["success"] + p2_pred["success"])
    # 人工可评价：只有 confirmed 标注才能算
    ann_files = [os.path.join(JEVDIR, f) for f in
                 ("annotations_route.jsonl", "annotations_title.jsonl",
                  "annotation_sample_route.jsonl")]
    confirmed = 0
    for p in ann_files:
        if os.path.exists(p):
            confirmed += sum(1 for r in C.load_jsonl(p)
                             if str(r.get("review_status", "")).lower() == "confirmed")
    stats = {
        "valid_input_title": len(title_snips),
        "valid_input_route": len(route_snips),
        "valid_input_ocr": len(ocr_snips),
        "valid_input_recover": len(recover_snips),
        "valid_input_ocr_pilot": len(pilot1_snips),
        "valid_input_ocr_pilot2": len(pilot2_snips),
        "valid_inputs_total": valid_inputs,
        "api_attempt_route": len(route_pred_rows),
        "api_success_title": t_pred["success"],
        "api_success_route": r_pred["success"],
        "api_success_ocr": o_pred["success"],
        "api_success_recover": v_pred["success"],
        "api_success_ocr_pilot": p1_pred["success"],
        "api_success_ocr_pilot2": p2_pred["success"],
        "api_failed_route": route_failed,
        "judgments_total": api_success,
        "coverage_all_500": round(api_success / 500, 4),
        "coverage_valid_inputs": round(api_success / valid_inputs, 4) if valid_inputs else 0,
        "human_confirmed": confirmed,
        "accuracy": None,
        "title": t_pred, "route": r_pred, "ocr": o_pred,
        "pilot1": p1_pred, "pilot2": p2_pred,
        "route_cost_cny_busy": round(
            (r_pred["input_tokens"] / 1e6) * 0.042 * 7.1, 4),
        "ocr_cost_cny_busy": round(
            (o_pred["input_tokens"] / 1e6) * 0.042 * 7.1, 4),
        "recover_cost_cny_busy": round(
            (v_pred["input_tokens"] / 1e6) * 0.042 * 7.1, 4),
        "pilot1_cost_cny_busy": round(
            (p1_pred["input_tokens"] / 1e6) * 0.042 * 7.1, 4),
        "pilot2_cost_cny_busy": round(
            (p2_pred["input_tokens"] / 1e6) * 0.042 * 7.1, 4),
    }
    C.ensure_dir(os.path.dirname(MERGE_MD))
    with open(MERGE_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    md = [
        "# Jev 合并覆盖率报告（title / route / OCR / recover / OCR 试点）", "",
        f"生成时间：{C.now_str()}",
        "数据来源：`jev_predictions_title_500.jsonl`、`jev_predictions_route_500.jsonl`、"
        "`jev_predictions_ocr_500.jsonl`、`jev_predictions_recover_500.jsonl`、"
        "`jev_predictions_ocr_pilot_500.jsonl`、`jev_predictions_ocr_pilot2_500.jsonl`（只读）", "",
        "## 一、口径分离", "",
        "| 指标 | 数值 |", "|---|---|",
        f"| 有效输入（title） | {len(title_snips)} |",
        f"| 有效输入（route） | {len(route_snips)} |",
        f"| 有效输入（OCR） | {len(ocr_snips)} |",
        f"| 有效输入（recover） | {len(recover_snips)} |",
        f"| 有效输入（OCR 试点一） | {len(pilot1_snips)} |",
        f"| 有效输入（OCR 试点二） | {len(pilot2_snips)} |",
        f"| 有效输入合计 | {valid_inputs}/500 = {valid_inputs/500:.1%} |",
        f"| API 调用成功（title / route / OCR / recover） | {t_pred['success']} / "
        f"{r_pred['success']} / {o_pred['success']} / {v_pred['success']} |",
        f"| API 调用成功（OCR 试点一 / 试点二） | {p1_pred['success']} / "
        f"{p2_pred['success']} |",
        f"| API 调用失败（route 唯一题） | {route_failed} |",
        f"| 判断总数（成功 prediction） | {api_success}/500 = {api_success/500:.1%} |",
        f"| 判断覆盖率（除以有效输入） | {stats['coverage_valid_inputs']:.1%} |",
        f"| 人工 confirmed 标注 | {confirmed}（0 时不得计算准确率） |",
        "| 判断准确率 | 不可计算 |", "",
        "## 二、按来源分布", "",
        "| 来源 | 成功 | relevant=True | self_contained=True | sufficient=True | 原始动作 | 归一化动作 | 冲突 |",
        "|---|---|---|---|---|---|---|---|",
        f"| title | {t_pred['success']} | {t_pred['relevant_true']} | "
        f"{t_pred['self_contained_true']} | {t_pred['sufficient_true']} | "
        f"{t_pred['raw_action']} | {t_pred['normalized_action']} | {t_pred['conflicts']} |",
        f"| route | {r_pred['success']} | {r_pred['relevant_true']} | "
        f"{r_pred['self_contained_true']} | {r_pred['sufficient_true']} | "
        f"{r_pred['raw_action']} | {r_pred['normalized_action']} | {r_pred['conflicts']} |",
        f"| OCR | {o_pred['success']} | {o_pred['relevant_true']} | "
        f"{o_pred['self_contained_true']} | {o_pred['sufficient_true']} | "
        f"{o_pred['raw_action']} | {o_pred['normalized_action']} | {o_pred['conflicts']} |",
        f"| recover | {v_pred['success']} | {v_pred['relevant_true']} | "
        f"{v_pred['self_contained_true']} | {v_pred['sufficient_true']} | "
        f"{v_pred['raw_action']} | {v_pred['normalized_action']} | {v_pred['conflicts']} |",
        f"| OCR 试点一 | {p1_pred['success']} | {p1_pred['relevant_true']} | "
        f"{p1_pred['self_contained_true']} | {p1_pred['sufficient_true']} | "
        f"{p1_pred['raw_action']} | {p1_pred['normalized_action']} | {p1_pred['conflicts']} |",
        f"| OCR 试点二 | {p2_pred['success']} | {p2_pred['relevant_true']} | "
        f"{p2_pred['self_contained_true']} | {p2_pred['sufficient_true']} | "
        f"{p2_pred['raw_action']} | {p2_pred['normalized_action']} | {p2_pred['conflicts']} |",
        "", "## 三、费用", "",
        f"- route：输入 {r_pred['input_tokens']} / 输出 {r_pred['output_tokens']} token，"
        f"约 {stats['route_cost_cny_busy']} 元，耗时 {r_pred['latency_sec']} s",
        f"- OCR：输入 {o_pred['input_tokens']} / 输出 {o_pred['output_tokens']} token，"
        f"约 {stats['ocr_cost_cny_busy']} 元，耗时 {o_pred['latency_sec']} s",
        f"- recover：输入 {v_pred['input_tokens']} / 输出 {v_pred['output_tokens']} token，"
        f"约 {stats['recover_cost_cny_busy']} 元，耗时 {v_pred['latency_sec']} s",
        f"- OCR 试点一：输入 {p1_pred['input_tokens']} / 输出 {p1_pred['output_tokens']} token，"
        f"约 {stats['pilot1_cost_cny_busy']} 元，耗时 {p1_pred['latency_sec']} s",
        f"- OCR 试点二：输入 {p2_pred['input_tokens']} / 输出 {p2_pred['output_tokens']} token，"
        f"约 {stats['pilot2_cost_cny_busy']} 元，耗时 {p2_pred['latency_sec']} s", "",
        "> 未有人工确认标签时，本报告不输出任何准确率结论。",
    ]
    with open(MERGE_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[merge] 有效输入 {valid_inputs}，判断 {api_success}，route 成功 {r_pred['success']}")
    print(f"[merge] 报告: {MERGE_MD}")


def report():
    rows = C.load_jsonl(SNIP_ROUTE)
    counts = collections.Counter(r["status"] for r in rows)
    src = collections.Counter(r["evidence_source"] for r in rows if r["status"] == "ok")
    ok = [r for r in rows if r["status"] == "ok"]
    title_ok = 263
    total = len(rows)
    lines = [
        "# Jev route 补全 · 离线覆盖率报告",
        "",
        f"生成时间：{C.now_str()}",
        "数据来源：`outputs/final500/run500/toc/routes.jsonl`、"
        "`outputs/final500/run500/jev/snippets_title_500.jsonl`、本地 PDF 缓存（只读）",
        "本文件由 `task5_jev_route_500.py` 离线生成，未调用任何 API，未修改原始结果。",
        "",
        "## 一、状态分布",
        "",
        "| 状态 | 数量 | 含义 |",
        "|---|---|---|",
    ]
    meaning = {
        "ok": "新增可用片段（来自路由预测页的真实 PDF 文本）",
        "already_covered": "已有 title 片段，不重复生成",
        "no_input": "无真实检索片段（不可答且无路由 / 路由不可用）",
        "no_text_page": "路由预测页无可抽取文本（需 OCR）",
        "visual_evidence_only": "文本可读但证据在图像中，纯文本不足以判断（已标记，不调用 Jev）",
    }
    for k in ("ok", "already_covered", "no_text_page", "visual_evidence_only", "no_input"):
        lines.append(f"| {k} | {counts.get(k, 0)} | {meaning[k]} |")
    lines.append(f"| 合计 | {total} | |")
    lines += [
        "",
        "## 二、覆盖率",
        "",
        f"- 原有 title 有效输入：{title_ok}/500 = {title_ok/500:.1%}",
        f"- 本次新增 route 有效输入：{counts.get('ok',0)}",
        f"- Jev 总有效输入：{title_ok + counts.get('ok',0)}/500 = "
        f"{(title_ok + counts.get('ok',0))/500:.1%}",
        f"- 片段来源分布：{dict(src)}",
        "",
        "## 三、不可用原因",
        "",
    ]
    reasons = collections.Counter()
    for r in rows:
        if r["status"] == "no_input":
            reasons["不可答且无路由结果" if not r["offline_only"]["answerable"]
                    else "可答但路由不可用"] += 1
        elif r["status"] == "no_text_page":
            reasons["路由页无文本（需 OCR）"] += 1
    for k, v in reasons.most_common():
        lines.append(f"- {k}：{v} 题")
    lines += [
        "",
        "## 四、抽取样例（前 10 条）",
        "",
        "| qid | 页码 | 来源 | 方法 | 字符数 | 片段开头 |",
        "|---|---|---|---|---|---|",
    ]
    for r in ok[:10]:
        snippet = r["evidence_snippet"].replace("|", "/")[:60]
        lines.append(f"| {r['qid']} | p.{r['page']} | {r['evidence_source']} | "
                     f"{r['extraction_method']} | {r['char_len']} | {snippet} |")
    lines += [
        "",
        "## 五、后续所需 API（尚未执行）",
        "",
        f"- Jev 真实调用：{counts.get('ok',0)} 次（仅新增 route 片段，title 的 263 条已成功，不重跑）",
        "- 剩余需 OCR 的题目见上方 no_text_page 数量；OCR 为本地计算，无 API 费用。",
        "- Jev 准确率仍需人工确认标注，未标注前不得宣称准确率。",
        "",
    ]
    C.ensure_dir(os.path.dirname(REPORT_MD))
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] 已写入 {REPORT_MD}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--report", action="store_true")
    g.add_argument("--quality", action="store_true")
    g.add_argument("--ocr-prep", action="store_true")
    g.add_argument("--merge-report", action="store_true")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.verify:
        raise SystemExit(1 if verify() else 0)
    elif args.report:
        report()
    elif args.quality:
        quality()
    elif args.ocr_prep:
        ocr_prep()
    elif args.merge_report:
        merge_report()


if __name__ == "__main__":
    main()
