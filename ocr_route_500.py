# -*- coding: utf-8 -*-
"""
第二阶段 OCR：只识别 76 道 no_text_page 题所缺的页面（去重后 53 页）。

- 输入：outputs/final500/audit/ocr_page_plan_500_v3.csv（need_ocr=True）
- 输出：outputs/final500/ocr/ocr_pages_500.jsonl（append-only，按 (doc_id, page) 去重续跑）
- 不联网调用大模型；使用本地 PaddleOCR（.venv_paddleocr）
- 不修改任何已有 Jev/路由/问答结果

用法：
  .venv_paddleocr\\Scripts\\python.exe ocr_route_500.py --dry-run
  .venv_paddleocr\\Scripts\\python.exe ocr_route_500.py --limit 3
  .venv_paddleocr\\Scripts\\python.exe ocr_route_500.py
"""
import argparse
import csv
import json
import os
import re
import sys
import tempfile
import time

os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    os.path.join(os.path.expanduser("~"), ".cache", "paddlex"),
)

import pymupdf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import final500_common as C  # noqa: E402

PLAN = os.path.join(C.OUT_ROOT, "audit", "ocr_page_plan_500_v3.csv")
OUT_DIR = os.path.join(C.OUT_ROOT, "ocr")
OUT = os.path.join(OUT_DIR, "ocr_pages_500.jsonl")
DPI = 200
MIN_CHARS = 40


def load_plan(path=None):
    plan = []
    with open(path or PLAN, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if str(row["need_ocr"]).lower() != "true":
                continue
            plan.append({"qid": row["qid"], "doc_id": row["doc_id"],
                         "page": int(row["page"])})
    uniq = {}
    for x in plan:
        uniq.setdefault((x["doc_id"], x["page"]), []).append(x["qid"])
    return [{"doc_id": k[0], "page": k[1], "qids": v} for k, v in uniq.items()]


def done_keys():
    keys = set()
    if os.path.exists(OUT):
        for r in C.load_jsonl(OUT):
            if r.get("status") in ("ok", "low_quality"):
                keys.add((r["doc_id"], r["page"]))
    return keys


def extract_items(res):
    try:
        d = dict(res)
    except Exception:
        d = getattr(res, "res", None) or {}
    return list(d.get("rec_texts") or []), list(d.get("rec_scores") or [])


def quality(text, scores):
    t = re.sub(r"[ \t]+", " ", text).strip()
    lines = [x.strip() for x in t.splitlines() if x.strip()]
    alnum = sum(ch.isalnum() for ch in t)
    ratio = alnum / max(1, len(t))
    digit_lines = sum(1 for x in lines if re.fullmatch(r"[0-9ivxlcdm. -]{1,8}", x, re.I))
    avg = sum(scores) / len(scores) if scores else 0
    status = "ok"
    if len(t) < MIN_CHARS:
        status = "empty"
    elif ratio < 0.45 or (lines and digit_lines / len(lines) > 0.7) or avg < 0.55:
        status = "low_quality"
    return t, len(lines), round(avg, 3), round(ratio, 3), status


def run(limit, dry, plan_path=None):
    plan = load_plan(plan_path)
    done = done_keys()
    todo = [x for x in plan if (x["doc_id"], x["page"]) not in done]
    if limit:
        todo = todo[:limit]
    print(f"[ocr] 去重后待识别 {len(plan)} 页；已完成 {len(done)}；本次 {len(todo)}")
    if dry:
        for x in todo:
            print("  ", x["doc_id"], "p.%d" % x["page"], len(x["qids"]), "题")
        return

    from paddleocr import PaddleOCR  # 真实识别时才导入
    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False,
                    enable_mkldnn=False)
    C.ensure_dir(OUT_DIR)
    t_all = time.time()
    for i, x in enumerate(todo, 1):
        pdf, _ = C.resolve_pdf(x["doc_id"])
        if not pdf:
            rec = {**x, "status": "pdf_missing", "reason": "PDF 缓存缺失"}
            C.append_jsonl(OUT, rec)
            continue
        doc = pymupdf.open(pdf)
        if not (1 <= x["page"] <= doc.page_count):
            doc.close()
            C.append_jsonl(OUT, {**x, "status": "page_out_of_range",
                                 "reason": f"PDF 仅 {doc.page_count} 页"})
            continue
        pix = doc.load_page(x["page"] - 1).get_pixmap(dpi=DPI)
        doc.close()
        fd, png = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        pix.save(png)
        t0 = time.time()
        try:
            result = ocr.predict(png)
            texts, scores = [], []
            for res in result:
                a, b = extract_items(res)
                texts += [str(t) for t in a]
                scores += [float(s) for s in b if s is not None]
            text = "\n".join(texts)
            t, n_lines, avg, ratio, status = quality(text, scores)
            rec = {**x, "status": status, "text": t, "n_lines": n_lines,
                   "avg_score": avg, "alnum_ratio": ratio,
                   "elapsed_sec": round(time.time() - t0, 2), "dpi": DPI,
                   "image_w": pix.width, "image_h": pix.height,
                   "engine": "PaddleOCR 3.x local (lang=en, CPU, mkldnn=False)",
                   "created_at": C.now_str()}
        except Exception as e:
            rec = {**x, "status": "error", "reason": f"{type(e).__name__}: {e}",
                   "elapsed_sec": round(time.time() - t0, 2)}
        finally:
            try:
                os.remove(png)
            except Exception:
                pass
        C.append_jsonl(OUT, rec)
        print(f"[{i}/{len(todo)}] {x['doc_id']} p.{x['page']} {rec['status']} "
              f"chars={len(rec.get('text',''))} {rec.get('elapsed_sec')}s "
              f"累计 {round(time.time()-t_all,1)}s", flush=True)
    ok = sum(1 for r in C.load_jsonl(OUT) if r.get("status") == "ok")
    print(f"[ocr] 完成，输出 {OUT}，ok={ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan", default=None, help="OCR 计划 CSV，默认 v3 计划")
    args = ap.parse_args()
    run(args.limit, args.dry_run, args.plan)


if __name__ == "__main__":
    main()
