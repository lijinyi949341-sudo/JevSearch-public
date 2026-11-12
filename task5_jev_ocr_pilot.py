# -*- coding: utf-8 -*-
"""
OCR 试点（40 页）：把 OCR 结果按“页面类型 + 问题语义相关性”筛选，生成独立的 Jev 待运行片段。

约束：
  - 只读 OCR/计划/PDF/样本；只写 *_pilot 文件，不覆盖已有 364 条输入与预测；
  - 不使用 gold 答案或 gold 证据页选择页面；
  - 只有真实 OCR 文本且与问题存在词面关联的页面才转出片段；
  - 若页面本身有更完整的文本层，优先保留文本层。

用法：
  python task5_jev_ocr_pilot.py --build
  python task5_jev_ocr_pilot.py --verify
  python task5_jev_ocr_pilot.py --torun
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

AUD = os.path.join(C.OUT_ROOT, "audit")
JEVDIR = os.path.join(C.OUT_ROOT, "run500", "jev")
PLAN = os.path.join(AUD, "ocr_page_plan_500_pilot.csv")
OUT = os.path.join(JEVDIR, "snippets_ocr_pilot_500.jsonl")
TORUN = os.path.join(JEVDIR, "snippets_ocr_pilot_500_torun.jsonl")
STATUS_CSV = os.path.join(AUD, "qid_ocr_pilot_status.csv")
SUMMARY = os.path.join(AUD, "OCR_PILOT_SUMMARY.json")
SUFFIX = ""
MIN_CHARS = 120
TOC_HEAD_RE = re.compile(
    r"^(table of contents?|contents|agenda|outline|what we'?ll cover)\b", re.I)
TOC_LINE_RE = re.compile(r"^\s*\d{1,2}\.\s+\S+")

# 人工语义复核（只针对本次试点 13 条候选，保留页面文本供审计，不送入 Jev）
MANUAL_REJECT = {
    "Q0153": "About 页仅介绍 APPOTA 平台，不含问题所需的 Top10 App 榜单",
    "Q0167": "OCR 文本为目录页（TABLE OF CONTENT），按规则排除",
    "Q0241": "About 页无 Java/WP 开发者占比数据",
    "Q0341": "整体销量汇总页，不含 Country 曲风的 Album/Song 占比",
    "Q0150": "Key findings 页讨论社交平台，未涉及 Lufthansa 标识颜色",
    "Q0352": "Quick facts 页为航班基本信息，不含博客数量",
    "Q0458": "Key findings 页未呈现 Lufthansa 官网截图数量",
    "Q0381": "赞助方/关于页且 OCR 噪声大，不含酒店业 Wi-Fi 调研占比",
    "Q0467": "网页研讨会操作界面文字（Use Telephone 等），无调研数据",
    "Q0013": "引用页仅说明数据驱动公司更高效，未列出“非数据驱动”示例",
    "Q0439": "引用页仅 analytics ROI，未呈现 Analytics Value Chain 流程图",
}

# 第二批人工语义复核：sufficient=片段已含答案要素；related=主题相关但不足；reject=页面内容与问题不对应
RELEVANCE2 = {
    "Q0241": ("sufficient", "片段直接给出 Java 19.8% 与 WP 2.3%，可计算差值 17.5 个百分点"),
    "Q0029": ("related", "p.8 讲多点喷射与 ECU 控制喷油器，但未列出 7 个传感器"),
    "Q0361": ("related", "p.27 为 Sensors 章节内容，但“图片数量”需图像计数"),
    "Q0150": ("related", "p.6 时间线提到“社交媒体标识颜色未变”，未给出具体颜色"),
    "Q0352": ("related", "p.6 为 Twitter 活动时间线，无博客数量统计"),
    "Q0458": ("related", "p.6 提到官网崩溃，未给出官网截图实例数"),
    "Q0381": ("related", "p.37 表格给出 Hospitality 客户忠诚度 61%，缺员工口径数据"),
    "Q0467": ("related", "p.11 正是“门店 Wi-Fi 安装率”图，但 OCR 丢失条形图数值"),
    "Q0153": ("reject", "p.9 为全球开发者份额，不含越南 App 榜单"),
    "Q0167": ("reject", "p.9 为开发者份额，不含“新闻”条目"),
    "Q0013": ("reject", "p.41 讲“把行动与结果关联”，未列出“非数据驱动”示例"),
    "Q0439": ("reject", "p.41 无 Analytics Value Chain 流程图"),
}


def toc_like(text):
    head = text[:120]
    if TOC_HEAD_RE.search(head):
        return True
    lines = [x for x in text.splitlines() if x.strip()]
    return len(lines) >= 4 and sum(1 for x in lines if TOC_LINE_RE.match(x)) >= 3


def load_plan():
    plan = collections.defaultdict(list)
    with open(PLAN, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            plan[row["qid"]].append(int(row["page"]))
    for q in plan:
        plan[q] = sorted(set(plan[q]))
    return plan


def text_layer(doc_id, page):
    pdf, _ = C.resolve_pdf(doc_id)
    if pdf is None:
        return ""
    doc = pymupdf.open(pdf)
    try:
        if page > doc.page_count:
            return ""
        return re.sub(r"\s+", " ", doc.load_page(page - 1).get_text("text")).strip()
    finally:
        doc.close()


def apply_paths(plan_path, suffix):
    """按批次切换计划与输出路径，避免覆盖上一批试点结果。"""
    global PLAN, OUT, TORUN, STATUS_CSV, SUMMARY, SUFFIX
    if plan_path:
        PLAN = plan_path
    if suffix:
        SUFFIX = suffix
        OUT = os.path.join(JEVDIR, f"snippets_ocr_{suffix}_500.jsonl")
        TORUN = os.path.join(JEVDIR, f"snippets_ocr_{suffix}_500_torun.jsonl")
        STATUS_CSV = os.path.join(AUD, f"qid_ocr_{suffix}_status.csv")
        SUMMARY = os.path.join(AUD, f"OCR_{suffix.upper()}_SUMMARY.json")


def build():
    plan = load_plan()
    ocr = to.load_ocr()
    samples = {s["qid"]: s for s in C.load_samples()}
    covered = set()
    for name in ("snippets_title_500.jsonl", "snippets_route_500_v3.jsonl",
                 "snippets_ocr_500.jsonl", "snippets_recover_500.jsonl"):
        for r in C.load_jsonl(os.path.join(JEVDIR, name)):
            if r.get("status") == "ok":
                covered.add(r["qid"])

    counts = collections.Counter()
    rows = []
    for qid, pages in plan.items():
        s = samples[qid]
        base = {
            "snippet_id": f"{qid}__ocr_pilot", "qid": qid,
            "sample_index": s["sample_index"], "doc_id": s["doc_id"],
            "doc_type": s.get("doc_type", ""), "question": s["question"],
            "mode": "ocr_pilot", "selection_uses_gold": False,
            "selection_rule": ("试点计划页（各文档前 5 个无文本页）中，排除封面/议程/目录页，"
                               "按问题词命中数选页；仅用真实文本层或本地 OCR"),
            "evidence_source": "", "page": None,
            "status": "", "reason": "", "evidence_snippet": "", "char_len": 0,
            "extraction_method": "", "truncated": False,
            "ocr_avg_score": None, "ocr_page_status": "",
            "page_term_hits": [], "candidate_pages": pages,
            "created_at": C.now_str(),
        }
        if qid in covered:
            base["status"] = "already_covered"
            base["reason"] = "已有其它模式有效输入，不重复生成"
            counts["already_covered"] += 1
            rows.append(base)
            continue

        terms = tr.question_terms(s["question"])
        cands, best = [], None
        for pg in pages:
            tl = text_layer(s["doc_id"], pg)
            o = ocr.get((s["doc_id"], pg))
            if len(tl) >= MIN_CHARS:
                text, method, ostatus, oscore = tl, "text_layer", "text_layer", None
            elif o and o.get("status") == "ok":
                text, method = to.clean_ocr(o.get("text", "")), "ocr"
                ostatus, oscore = o.get("status"), o.get("avg_score")
            else:
                cands.append({"page": pg, "state": (o or {}).get("status", "no_record"),
                              "chars": 0, "hits": 0})
                continue
            if tr.is_toc_page(text) or toc_like(text):
                cands.append({"page": pg, "state": "toc_like", "chars": len(text), "hits": 0})
                continue
            hits = tr.kw_hits(text, terms)
            cover = to.cover_like(pg, text)
            cands.append({"page": pg, "state": "cover" if cover else "content",
                          "chars": len(text), "hits": len(hits),
                          "terms": sorted(hits), "method": method})
            if cover or len(text) < MIN_CHARS or not hits:
                continue
            if best is None or len(hits) > len(best[2]):
                best = (pg, text, hits, method, ostatus, oscore)

        base["candidate_pages_detail"] = cands
        if best:
            pg, text, hits, method, ostatus, oscore = best
            snip, trunc = tr.cut_snippet(text)
            base.update({
                "status": "ok", "page": pg, "evidence_snippet": snip,
                "char_len": len(snip), "truncated": trunc,
                "evidence_source": "ocr_pilot" if method == "ocr" else "pilot_text_layer",
                "extraction_method": method, "page_term_hits": sorted(hits),
                "ocr_page_status": ostatus, "ocr_avg_score": oscore,
            })
            if not SUFFIX and qid in MANUAL_REJECT:
                base["status"] = "semantic_reject"
                base["reason"] = MANUAL_REJECT[qid]
            if SUFFIX == "pilot2" and qid in RELEVANCE2:
                verdict, note = RELEVANCE2[qid]
                base["relevance_review"] = verdict
                base["relevance_note"] = note
                if verdict == "reject":
                    base["status"] = "semantic_reject"
                    base["reason"] = note
        else:
            states = [c["state"] for c in cands]
            base["status"] = "no_evidence"
            if not states:
                base["reason"] = "试点计划页无 OCR 记录"
            elif all(x in ("cover", "toc_like") for x in states):
                base["reason"] = "试点页均为封面/议程/目录页"
            elif all(x in ("empty", "low_quality", "no_record") for x in states):
                base["reason"] = "试点页无可识别文字"
            else:
                base["reason"] = "试点页有文字但与问题无词面关联"
        counts[base["status"]] += 1
        rows.append(base)

    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)

    with open(STATUS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "doc_id", "pages", "jev_status", "page", "method",
                    "hits", "chars", "ocr_score", "reason"])
        for r in rows:
            w.writerow([r["qid"], r["doc_id"], "|".join(map(str, r["candidate_pages"])),
                        r["status"], r["page"] or "", r["extraction_method"],
                        ";".join(r["page_term_hits"]), r["char_len"],
                        r["ocr_avg_score"] or "", r["reason"]])

    summary = {
        "plan_pages": len({(r["doc_id"], p) for r in rows for p in r["candidate_pages"]}),
        "questions": len(rows),
        "status": dict(counts),
        "ok_qids": [r["qid"] for r in rows if r["status"] == "ok"],
        "no_evidence": [{"qid": r["qid"], "reason": r["reason"]} for r in rows
                        if r["status"] != "ok"],
    }
    json.dump(summary, open(SUMMARY, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[build]", dict(counts), "ok:", summary["ok_qids"])
    return rows


def verify():
    rows = C.load_jsonl(OUT)
    ok = [r for r in rows if r["status"] == "ok"]
    covered = set()
    for name in ("snippets_title_500.jsonl", "snippets_route_500_v3.jsonl",
                 "snippets_ocr_500.jsonl", "snippets_recover_500.jsonl"):
        for r in C.load_jsonl(os.path.join(JEVDIR, name)):
            if r.get("status") == "ok":
                covered.add(r["qid"])
    bad = []
    if len({r["qid"] for r in rows}) != len(rows):
        bad.append("qid 不唯一")
    if {r["qid"] for r in ok} & covered:
        bad.append("与既有 364 条重复")
    if any(r["selection_uses_gold"] for r in rows):
        bad.append("存在 uses_gold=True")
    ocr = to.load_ocr()
    for r in ok:
        src = ocr.get((r["doc_id"], r["page"]))
        src_text = to.clean_ocr((src or {}).get("text", "")) if r["extraction_method"] == "ocr" \
            else text_layer(r["doc_id"], r["page"])
        probe = re.sub(r"\s+", " ", r["evidence_snippet"])[:50]
        if probe not in src_text:
            bad.append(f"{r['qid']} 片段不在来源文本中")
        if len(r["evidence_snippet"]) < 100:
            bad.append(f"{r['qid']} 片段过短")
    print(f"[verify] 行数={len(rows)} ok={len(ok)} 与 364 重叠="
          f"{len({r['qid'] for r in ok} & covered)}")
    print("[verify]", "通过" if not bad else f"失败: {bad}")
    return bool(bad)


def export_torun():
    rows = [r for r in C.load_jsonl(OUT) if r["status"] == "ok"]
    tmp = TORUN + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, TORUN)
    print(f"[torun] {TORUN} -> {len(rows)} 条")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--torun", action="store_true")
    ap.add_argument("--plan", default=None,
                    help="OCR 计划 CSV（默认第一批 ocr_page_plan_500_pilot.csv）")
    ap.add_argument("--suffix", default="",
                    help="批次后缀，如 pilot2；输出改名为 *_pilot2_*，不覆盖上一批")
    a = ap.parse_args()
    apply_paths(a.plan, a.suffix)
    if a.build:
        build()
    elif a.verify:
        raise SystemExit(1 if verify() else 0)
    else:
        export_torun()
