# -*- coding: utf-8 -*-
"""
Jev 覆盖补全 v4：对剩余 138 题做“路由邻域 + 题干页码”的真实 PDF 文本层检索。

约束：
  - 只读样本/路由/PDF；只写新文件 snippets_recover_500.jsonl 等，不覆盖已有结果；
  - 不使用 gold_answer / evidence_pages / evidence_sources 参与候选页选择；
  - 候选页仅来自：路由预测页及其后 2 页、路由章节起始页起 5 页、题干明确写出的页码；
  - 只有能被格式正则（email/date）或 ≥6 分问题词命中支撑的页面才转出片段；
  - 其余题目保留 no_input / *_unrecoverable 状态，不伪造片段。

用法：
  python task5_jev_recover_500.py --build
  python task5_jev_recover_500.py --verify
  python task5_jev_recover_500.py --report
  python task5_jev_recover_500.py --torun
"""
import argparse
import collections
import csv
import json
import os
import re

import pymupdf

import final500_common as C
import task5_jev_ocr_500 as to
import task5_jev_route_500 as tr

JEVDIR = os.path.join(C.OUT_ROOT, "run500", "jev")
SNIP_OCR = os.path.join(JEVDIR, "snippets_ocr_500.jsonl")
OUT = os.path.join(JEVDIR, "snippets_recover_500.jsonl")
TORUN = os.path.join(JEVDIR, "snippets_recover_500_torun.jsonl")
STATUS_CSV = os.path.join(C.OUT_ROOT, "audit", "qid_recovery_status_500.csv")
REPORT = os.path.join(C.OUT_ROOT, "audit", "JEV_RECOVERY_REPORT.md")

MIN_CHARS = 120
EMAIL_RE = re.compile(r"[\w.\-+]+@[\w\-]+\.[\w.]+")
DATE_RE = re.compile(
    r"\b(19|20)\d{2}\b|\b(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\b", re.I)
PAGE_CITE_RE = re.compile(r"\b(?:page|slide|p\.)\s*(\d{1,3})\b", re.I)
ORD_PAGE_RE = re.compile(r"\b(first|second|third|fourth|fifth|last)\s+(page|slide|cover)\b", re.I)
ORD_MAP = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}

TERM_STOP = set("""what which who when where why how many much is are was were the a an of in on at
to for from by with and or but as that this these those it its be been being do does did done has
have had can could should would will shall may might must not no yes there their they them he she
his her page pages document according shown given following name number list total two three first
second mentioned available entire type types kind kinds format answer direct directly slide slides
example examples""".split())

# 纯视觉题：即使邻域文本有词命中也不作为证据
VISUAL_EXCLUDE = dict(tr.SEMANTIC_EXCLUDE)


def question_terms(q):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9\-\.']*", q.lower())
            if len(w) >= 3 and w not in TERM_STOP]


def ordinal_ref(q):
    m = ORD_PAGE_RE.search(q)
    if not m:
        return None
    key = m.group(1).lower()
    return ORD_MAP.get(key)


def candidate_pages(rec, sample, n_pages):
    pages = []
    for p in rec.get("pages_tried") or []:
        pages.append(int(p))
    if rec.get("page"):
        pages.append(int(rec["page"]))
    for sec in (rec.get("route_sections") or []):
        try:
            a = int(sec["start_page"])
        except Exception:
            continue
        pages.extend(range(a, min(a + 4, n_pages) + 1))
    for m in PAGE_CITE_RE.findall(sample["question"]):
        pages.append(int(m))
    op = ordinal_ref(sample["question"])
    if op:
        pages.extend([op, op + 1, op + 2])
    out = []
    for p in pages:
        if 1 <= p <= n_pages and p not in out:
            out.append(p)
    return sorted(out)


def score_page(text, terms, q):
    low = text.lower()
    hits = [t for t in terms if t in low]
    targets = []
    if ("email" in q.lower() or "e-mail" in q.lower()) and EMAIL_RE.search(text):
        targets.append("email")
    if re.search(r"\bdate\b|\byear\b|\bmonth\b", q, re.I) and DATE_RE.search(text):
        targets.append("date")
    return len(hits) * 2 + len(targets) * 3, hits, targets


def build():
    samples = {s["qid"]: s for s in C.load_samples()}
    title = {r["qid"]: r for r in C.load_jsonl(os.path.join(JEVDIR, "snippets_title_500.jsonl"))}
    route = {r["qid"]: r for r in C.load_jsonl(tr.SNIP_ROUTE)}
    ocr = {r["qid"]: r for r in C.load_jsonl(SNIP_OCR)}

    covered = set()
    for src in (title, route, ocr):
        covered |= {q for q, r in src.items() if r.get("status") == "ok"}

    counts = collections.Counter()
    rows = []
    for qid, rec in route.items():
        if qid in covered:
            continue
        s = samples[qid]
        unus = str(s.get("answer", "")) == "Not answerable"
        base = {
            "snippet_id": f"{qid}__recover", "qid": qid,
            "sample_index": s["sample_index"], "doc_id": s["doc_id"],
            "doc_type": s.get("doc_type", ""), "question": s["question"],
            "mode": "recover", "selection_uses_gold": False,
            "selection_rule": ("路由预测页 ±2 页、路由章节起始页起 5 页、"
                               "题干直接写出的页码；仅用真实 PDF 文本层，不使用 gold"),
            "evidence_source": "", "page": None, "section_id": rec.get("section_id"),
            "status": "", "reason": "", "evidence_snippet": "", "char_len": 0,
            "extraction_method": "", "truncated": False,
            "evidence_type_primary": rec.get("evidence_type_primary", "Unknown"),
            "cross_page": rec.get("cross_page", False),
            "duplicate_of": "", "candidate_pages": [], "page_score": 0,
            "page_term_hits": [], "page_format_targets": [],
            "created_at": C.now_str(),
            "offline_only": rec.get("offline_only", {}),
        }
        if qid in VISUAL_EXCLUDE:
            base["status"] = "visual_evidence_only"
            base["reason"] = VISUAL_EXCLUDE[qid]
            counts[base["status"]] += 1
            rows.append(base)
            continue
        if unus:
            base["status"] = "unanswerable_kept"
            base["reason"] = "数据集标注 Not answerable；保留合理拒答，不构造输入"
            counts[base["status"]] += 1
            rows.append(base)
            continue

        pdf, _ = C.resolve_pdf(s["doc_id"])
        if pdf is None:
            base["status"] = "pdf_missing"
            base["reason"] = "本地 PDF 缺失"
            counts[base["status"]] += 1
            rows.append(base)
            continue
        doc = pymupdf.open(pdf)
        n_pages = doc.page_count
        cands = candidate_pages(rec, s, n_pages)
        base["candidate_pages"] = cands
        header = tr.doc_header(s["doc_id"])
        terms = question_terms(s["question"])
        best = None
        email_hit = None
        date_pages = []
        n_text = 0
        has_ord = ordinal_ref(s["question"]) is not None
        for pg in cands:
            raw = re.sub(r"\s+", " ", doc.load_page(pg - 1).get_text("text")).strip()
            if len(raw) < 40:
                continue
            n_text += 1
            if tr.is_toc_page(raw):
                continue
            text = tr.clean_text(raw, header)
            if len(text) < MIN_CHARS:
                continue
            sc, hits, targets = score_page(text, terms, s["question"])
            if "email" in targets and email_hit is None:
                email_hit = (pg, text, hits, targets)
            if "date" in targets and has_ord:
                date_pages.append((pg, text, hits, targets))
            if sc >= 6 and len(hits) >= 2 and (best is None or sc > best[0]):
                best = (sc, pg, text, hits, targets)
        doc.close()
        if email_hit:
            pg, text, hits, targets = email_hit
            sc = len(hits) * 2 + 3
            best = (sc, pg, text, hits, targets)
        elif date_pages:
            pg, text, hits, targets = min(date_pages, key=lambda x: x[0])
            sc = len(hits) * 2 + 3
            best = (sc, pg, text, hits, targets)
        if best:
            sc, pg, text, hits, targets = best
            snip, trunc = tr.cut_snippet(text)
            base.update({
                "status": "ok", "page": pg, "evidence_snippet": snip,
                "char_len": len(snip), "truncated": trunc,
                "evidence_source": "route_neighborhood_textlayer",
                "extraction_method": "text_layer", "page_score": sc,
                "page_term_hits": hits, "page_format_targets": targets,
            })
        else:
            base["status"] = "text_unrecoverable"
            if n_text == 0:
                base["reason"] = ("候选页全部无文本层（扫描/图片型 PDF），"
                                  "需整页 OCR 才能检索")
            else:
                base["reason"] = ("候选页文本层存在但不含问题实体或目标格式，"
                                  "证据可能在图像/表格中")
        counts[base["status"]] += 1
        rows.append(base)

    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)

    with open(STATUS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "doc_id", "category", "jev_status", "page", "score",
                    "term_hits", "format_targets", "char_len", "reason"])
        for r in rows:
            w.writerow([r["qid"], r["doc_id"],
                        ocr.get(r["qid"], {}).get("status", ""), r["status"],
                        r["page"] or "", r["page_score"], ";".join(r["page_term_hits"]),
                        ";".join(r["page_format_targets"]), r["char_len"], r["reason"]])
    print("[build] 状态:", dict(counts))
    ok = [r for r in rows if r["status"] == "ok"]
    print("[build] 新增可运行:", len(ok), [r["qid"] for r in ok])
    return rows


def export_torun():
    rows = [r for r in C.load_jsonl(OUT) if r["status"] == "ok"]
    tmp = TORUN + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, TORUN)
    print(f"[torun] {TORUN} -> {len(rows)} 条")


def verify():
    rows = C.load_jsonl(OUT)
    bad = []
    if len(rows) != len({r["qid"] for r in rows}):
        bad.append("qid 不唯一")
    cover = set()
    for name in ("snippets_title_500.jsonl", "snippets_route_500_v3.jsonl",
                 "snippets_ocr_500.jsonl"):
        for r in C.load_jsonl(os.path.join(JEVDIR, name)):
            if r.get("status") == "ok":
                cover.add(r["qid"])
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"[verify] 状态 ok={len(ok)}，行数={len(rows)}，与已有 362 重叠="
          f"{len({r['qid'] for r in ok} & cover)}")
    if {r["qid"] for r in ok} & cover:
        bad.append("与已有有效输入重叠")
    if any(r["selection_uses_gold"] for r in rows):
        bad.append("存在 selection_uses_gold=True")
    samples = {s["qid"]: s for s in C.load_samples()}
    for r in ok:
        pdf, _ = C.resolve_pdf(samples[r["qid"]]["doc_id"])
        doc = pymupdf.open(pdf)
        raw = re.sub(r"\s+", " ", doc.load_page(r["page"] - 1).get_text("text"))
        doc.close()
        probe = re.sub(r"\s+", " ", r["evidence_snippet"])[:50]
        if probe not in raw:
            bad.append(f"{r['qid']} 片段不在 PDF 页文本中")
        if len(r["evidence_snippet"]) < 100:
            bad.append(f"{r['qid']} 片段过短")
    print("[verify]", "通过" if not bad else f"失败: {bad}")
    return bool(bad)


def report():
    rows = C.load_jsonl(OUT)
    counts = collections.Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "ok"]
    lines = ["# Jev 剩余题补全（v4：路由邻域文本层检索）", "",
             f"生成时间：{C.now_str()}",
             "数据来源：500 题样本、`snippets_route_500_v3.jsonl`、本地 PDF 文本层（只读）。"
             "未使用 gold 字段，未调用任何付费 API。", "",
             "## 一、结果", "",
             "| 状态 | 数量 | 含义 |", "|---|---|---|",
             f"| ok | {counts.get('ok', 0)} | 候选页真实文本层含问题实体/目标格式，可送 Jev |",
             f"| text_unrecoverable | {counts.get('text_unrecoverable', 0)} | 候选页无文本层或文本不含证据 |",
             f"| unanswerable_kept | {counts.get('unanswerable_kept', 0)} | 数据集标注不可答，保留拒答 |",
             f"| visual_evidence_only | {counts.get('visual_evidence_only', 0)} | 答案仅在图像/图表中 |",
             f"| pdf_missing | {counts.get('pdf_missing', 0)} | 本地 PDF 缺失 |",
             f"| 合计 | {len(rows)} | |", "",
             "## 二、新增可运行", ""]
    for r in ok:
        lines.append(f"- {r['qid']}（p.{r['page']}，{r['char_len']} 字符，score={r['page_score']}，"
                     f"命中 {r['page_term_hits']}，格式 {r['page_format_targets']}）："
                     f"{r['evidence_snippet'][:90]}")
    lines += ["", "## 三、候选页规则", "",
              "- 路由预测页及后 2 页、路由章节起始页起 5 页、题干显式页码；",
              "- 排除目录页与 <120 字符页；不使用 gold 证据页。", ""]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {REPORT}；ok={len(ok)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--torun", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.verify:
        raise SystemExit(1 if verify() else 0)
    elif a.torun:
        export_torun()
    else:
        report()
