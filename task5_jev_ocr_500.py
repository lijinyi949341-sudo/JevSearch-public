# -*- coding: utf-8 -*-
"""
第二阶段：把本地 PaddleOCR 结果转成 Jev 待运行片段（离线）。

输入（只读）：
  outputs/final500/audit/ocr_page_plan_500_v3.csv
  outputs/final500/ocr/ocr_pages_500.jsonl
  outputs/final500/run500/jev/snippets_route_500_v3.jsonl
输出（新文件）：
  outputs/final500/run500/jev/snippets_ocr_500.jsonl
  outputs/final500/audit/qid_ocr_status_500.csv
  outputs/final500/audit/OCR_COVERAGE_REPORT.md

证据选择：只使用“路由预测页 + OCR 文本”，不使用 gold 答案/gold 证据页。
用法：
  py -3.12 task5_jev_ocr_500.py --build
  py -3.12 task5_jev_ocr_500.py --verify
  py -3.12 task5_jev_ocr_500.py --report
"""
import argparse
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final500_common as C
import task5_jev_route_500 as tr

JEVDIR = os.path.join(C.OUT_ROOT, "run500", "jev")
V3 = os.path.join(JEVDIR, "snippets_route_500_v3.jsonl")
PLAN = os.path.join(C.OUT_ROOT, "audit", "ocr_page_plan_500_v3.csv")
OCR = os.path.join(C.OUT_ROOT, "ocr", "ocr_pages_500.jsonl")
OUT = os.path.join(JEVDIR, "snippets_ocr_500.jsonl")
STATUS_CSV = os.path.join(C.OUT_ROOT, "audit", "qid_ocr_status_500.csv")
REPORT = os.path.join(C.OUT_ROOT, "audit", "OCR_COVERAGE_REPORT.md")
MIN_CHARS = 40
COVER_WORDS = 50
AGENDA_RE = re.compile(
    r"^(what we'?ll cover|agenda|table of contents|contents|outline|today'?s agenda|"
    r"meet the presenters|presenters|speaker bio)\b", re.I)


def cover_like(page, text):
    """封面/议程/讲师介绍页：仅标题、署名、版权，无与问题相关的正文。"""
    words = len(text.split())
    if AGENDA_RE.search(text[:100]):
        return True
    return int(page) == 1 and words < COVER_WORDS


def clean_ocr(text):
    lines = [re.sub(r"\s+", " ", x).strip() for x in str(text).splitlines()]
    lines = [x for x in lines if x and not re.fullmatch(
        r"(page\s*)?[0-9ivxlcdm.\-]{1,10}", x, re.I)]
    out = []
    for x in lines:
        if not out or x != out[-1]:
            out.append(x)
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def load_ocr():
    last = {}
    if os.path.exists(OCR):
        for r in C.load_jsonl(OCR):
            last[(r["doc_id"], r["page"])] = r
    return last


def load_plan():
    plan = collections.defaultdict(list)
    with open(PLAN, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row["need_ocr"]).lower() == "true":
                plan[row["qid"]].append(int(row["page"]))
    return plan


def build():
    v3 = {r["qid"]: r for r in C.load_jsonl(V3)}
    ocr = load_ocr()
    plan = load_plan()
    samples = {s["qid"]: s for s in C.load_samples()}
    rows = []
    for qid, base in v3.items():
        s = samples[qid]
        rec = {
            "snippet_id": f"{qid}__ocr", "qid": qid,
            "sample_index": base.get("sample_index"),
            "doc_id": base["doc_id"], "doc_type": base.get("doc_type", ""),
            "question": base["question"], "mode": "ocr",
            "selection_uses_gold": False,
            "selection_rule": "路由预测页（top-1~3）经本地 PaddleOCR 识别，不使用 gold",
            "evidence_source": "ocr_route_page", "page": None,
            "status": "not_applicable", "reason": "",
            "evidence_snippet": "", "char_len": 0,
            "extraction_method": "paddleocr", "ocr_avg_score": None,
            "ocr_alnum_ratio": None, "ocr_lines": 0, "ocr_page_status": "",
            "created_at": C.now_str(),
            "offline_only": {
                "answer": s.get("answer", ""),
                "gold_answer": s.get("answer", ""),
                "evidence_pages": s.get("evidence_pages", "[]"),
                "evidence_sources": s.get("evidence_sources", ""),
                "answerable": str(s.get("answer", "")) != "Not answerable",
            },
        }
        if base["status"] != "no_text_page":
            rec["reason"] = "该题已有其他来源的有效输入或不在 OCR 清单"
            rows.append(rec)
            continue
        tried = plan.get(qid, [])
        low = None
        terms = tr.question_terms(base["question"])
        candidates = []
        for order, pg in enumerate(tried):
            o = ocr.get((base["doc_id"], pg))
            if not o:
                continue
            if o.get("status") == "ok":
                txt = clean_ocr(o.get("text", ""))
                if len(txt) < MIN_CHARS:
                    continue
                snip, trunc = tr.cut_snippet(txt)
                candidates.append({
                    "page": pg, "ocr": o, "cleaned": txt, "snip": snip,
                    "truncated": trunc, "hits": len(tr.kw_hits(txt, terms)),
                    "words": len(txt.split()), "cover": cover_like(pg, txt),
                    "order": order,
                })
            elif o.get("status") == "low_quality" and low is None:
                low = (pg, o)
        if candidates:
            # 先在非封面页中按“问题词命中数 → 候选顺序”选页；只有封面时才判 cover_only
            usable = [c for c in candidates if not c["cover"]]
            if usable:
                pick = max(usable, key=lambda c: (c["hits"], -c["order"]))
                o = pick["ocr"]
                status = "ok"
                reason = ""
                if len(pick["snip"]) < 120:
                    status = "ocr_short_text"
                    reason = "非封面页但 OCR 文本不足 120 字符，证据不足"
                rec.update({
                    "status": status, "page": pick["page"], "reason": reason,
                    "evidence_snippet": pick["snip"], "char_len": len(pick["snip"]),
                    "ocr_avg_score": o.get("avg_score"),
                    "ocr_alnum_ratio": o.get("alnum_ratio"),
                    "ocr_lines": o.get("n_lines"),
                    "ocr_page_status": o.get("status"),
                    "ocr_content_words": pick["words"],
                    "ocr_question_hits": pick["hits"],
                    "truncated": pick["truncated"],
                })
            else:
                pick = max(candidates, key=lambda c: len(c["cleaned"]))
                o = pick["ocr"]
                rec.update({
                    "status": "ocr_cover_only", "page": pick["page"],
                    "evidence_snippet": pick["snip"], "char_len": len(pick["snip"]),
                    "ocr_avg_score": o.get("avg_score"),
                    "ocr_alnum_ratio": o.get("alnum_ratio"),
                    "ocr_lines": o.get("n_lines"),
                    "ocr_page_status": o.get("status"),
                    "ocr_content_words": pick["words"],
                    "ocr_question_hits": pick["hits"],
                    "reason": "OCR 成功但该页为封面/议程/讲师介绍（仅标题署名），"
                              "无问题相关正文，不送入 Jev",
                })
        elif low:
            pg, o = low
            rec.update({"status": "ocr_low_quality", "page": pg,
                        "reason": "OCR 文本质量低于阈值，不送入 Jev",
                        "ocr_avg_score": o.get("avg_score"),
                        "ocr_alnum_ratio": o.get("alnum_ratio")})
        elif any((base["doc_id"], p) in ocr for p in tried):
            rec["status"] = "ocr_empty"
            rec["page"] = tried[0] if tried else None
            rec["reason"] = "候选页 OCR 无有效文本"
        else:
            rec["status"] = "ocr_pending"
            rec["reason"] = "OCR 记录尚未生成"
        rows.append(rec)

    C.ensure_dir(JEVDIR)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, OUT)

    C.ensure_dir(os.path.dirname(STATUS_CSV))
    with open(STATUS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "doc_id", "page", "ocr_page_status", "ocr_avg_score",
                    "char_len", "ocr_content_words", "ocr_question_hits",
                    "jev_input_status", "reason"])
        for r in rows:
            if r["status"] == "not_applicable":
                continue
            w.writerow([r["qid"], r["doc_id"], r["page"] or "",
                        r["ocr_page_status"], r["ocr_avg_score"], r["char_len"],
                        r.get("ocr_content_words", ""), r.get("ocr_question_hits", ""),
                        r["status"], r["reason"]])
    counts = collections.Counter(r["status"] for r in rows)
    print("[build] 状态:", dict(counts), "新增 ok:", counts.get("ok", 0))


def verify():
    rows = C.load_jsonl(OUT)
    v3 = {r["qid"]: r for r in C.load_jsonl(V3) if r.get("status") == "ok"}
    ocr = load_ocr()
    ok = [r for r in rows if r["status"] == "ok"]
    problems = []
    if len(rows) != 500 or len({r["qid"] for r in rows}) != 500:
        problems.append("500 qid 不完整")
    if any(r["qid"] in v3 for r in ok):
        problems.append("与 v3 有效输入 qid 重复")
    if any(r.get("selection_uses_gold") for r in ok):
        problems.append("使用 gold 选择")
    for r in ok:
        o = ocr.get((r["doc_id"], r["page"]))
        if not o or o.get("status") != "ok":
            problems.append(f"{r['qid']} 无对应 OCR 记录")
            continue
        key = re.sub(r"\s+", "", r["evidence_snippet"])[:50]
        if key not in re.sub(r"\s+", "", clean_ocr(o.get("text", ""))):
            problems.append(f"{r['qid']} 片段与 OCR 文本不一致")
    print(f"[verify] 行数={len(rows)} ok={len(ok)} 与v3重复="
          f"{sum(1 for r in ok if r['qid'] in v3)}")
    print("[verify] 结论:", "通过" if not problems else problems)
    for r in ok[:10]:
        print(f"  {r['qid']} p.{r['page']} chars={r['char_len']} "
              f"score={r['ocr_avg_score']} | {r['evidence_snippet'][:70]}")
    return problems


def report():
    rows = C.load_jsonl(OUT)
    counts = collections.Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "ok"]
    cover = [r for r in rows if r["status"] == "ocr_cover_only"]
    pending = [r for r in rows if r["status"] in
               ("ocr_empty", "ocr_low_quality", "ocr_pending", "ocr_short_text")]
    total = 351 + len(ok)
    lines = [
        "# OCR 阶段覆盖报告", "",
        f"生成时间：{C.now_str()}",
        "数据来源：`ocr_pages_500.jsonl`、`snippets_route_500_v3.jsonl`（只读）；"
        "未调用任何付费 API。", "",
        "## 一、OCR 页面", "",
        f"- 计划页（去重）：53；OCR 记录见 `outputs/final500/ocr/ocr_pages_500.jsonl`",
        f"- 题目层面：需 OCR 76 题，语义可用并转出片段 **{len(ok)}** 题", "",
        "## 二、状态分布（76 题）", "",
        "| 状态 | 数量 | 含义 |", "|---|---|---|",
        f"| ok | {counts.get('ok', 0)} | 非封面正文页，OCR 文本可送 Jev（≥120 字符） |",
        f"| ocr_cover_only | {counts.get('ocr_cover_only', 0)} | OCR 成功但为封面/议程/讲师介绍，无问题相关正文 |",
        f"| ocr_short_text | {counts.get('ocr_short_text', 0)} | 非封面页但文本 <120 字符 |",
        f"| ocr_empty | {counts.get('ocr_empty', 0)} | 候选页 OCR 无有效文本 |",
        f"| ocr_low_quality | {counts.get('ocr_low_quality', 0)} | OCR 质量低于阈值 |",
        f"| ocr_pending | {counts.get('ocr_pending', 0)} | OCR 尚未完成 |",
        "", "## 三、选页规则（离线，不使用 gold）", "",
        "- 候选页只来自目录路由预测的 top-1~3 章节（真实 PDF 渲染 + 本地 PaddleOCR）；",
        "- 在全部候选页中排除封面/议程/讲师介绍页，再按“问题词命中数 → 候选顺序”选页；",
        "- 若某题所有候选页都是封面/议程，则记 `ocr_cover_only`，不送入 Jev，避免无意义输入。", "",
        "## 四、覆盖率", "",
        f"- 原有效输入：351/500 = 70.2%",
        f"- 新增 OCR 有效输入：{len(ok)}",
        f"- 合并后：{total}/500 = {total/500:.1%}", "",
        "## 五、仍需人工/无法补全的题目", "",
    ]
    for r in cover:
        lines.append(f"- {r['qid']}（p.{r['page']}，{r.get('ocr_content_words')} 词）：{r['reason']}")
    for r in pending:
        lines.append(f"- {r['qid']}（p.{r['page'] or '?'}）：{r['reason']}")
    lines += ["", "## 六、预计 API", "",
              f"- 新增 Jev 调用：{len(ok)} 次，预计约 {len(ok)*0.000405:.4f} 元",
              "- 仍需人工确认标注后才能计算判断准确率。", ""]
    C.ensure_dir(os.path.dirname(REPORT))
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] 已写入 {REPORT}；新增 {len(ok)}，合计 {total}/500")


def export_torun():
    rows = [r for r in C.load_jsonl(OUT) if r["status"] == "ok"]
    path = os.path.join(JEVDIR, "snippets_ocr_500_torun.jsonl")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"[torun] 已写入 {path}，共 {len(rows)} 条")
    return path


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--report", action="store_true")
    g.add_argument("--torun", action="store_true")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.verify:
        raise SystemExit(1 if verify() else 0)
    elif args.torun:
        export_torun()
    else:
        report()


if __name__ == "__main__":
    main()
