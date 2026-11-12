# -*- coding: utf-8 -*-
"""
文档结构解析（500 题版，第一步）：为所有有 PDF 的文档构建结构化目录（TOC）。

与 task6_build_toc.py 的关系：
  - 复用同一套目录构建逻辑（书签 → 印刷目录 → 正文字号 → pseudo 索引），
    不修改原来的 4 文档实验与 outputs/task6/toc.json；
  - 本脚本扩展到 500 题涉及的全部文档，按文档逐条保存、支持断点续跑。

严格约束：
  - 只读取 PDF 文本层（可选 OCR 补充），绝不读取问题、答案、gold evidence_pages
    或 evidence_sources；
  - 每份文档记录目录来源、页码、层级、置信度和构建状态；
  - 页码均为 PDF 实际页码（1-based）。

输出：
  outputs/final500/toc/toc.json
  outputs/final500/toc/toc_preview.md
  outputs/final500/toc/build_status.json

用法：
  python task6_build_toc_500.py --dry-run          # 只统计哪些文档可建、哪些缺失
  python task6_build_toc_500.py --limit 50         # 先建前 50 题涉及的文档
  python task6_build_toc_500.py --limit 0          # 全部（已成功的自动跳过）
  python task6_build_toc_500.py --rebuild          # 强制重建
"""
import argparse
import collections
import json
import os
import time

import task6_build_toc as base6
import final500_common as C

OUT_DIR = os.path.join(C.OUT_ROOT, "toc")
TOC_JSON = os.path.join(OUT_DIR, "toc.json")
PREVIEW_MD = os.path.join(OUT_DIR, "toc_preview.md")
STATUS_JSON = os.path.join(OUT_DIR, "build_status.json")


def load_ocr_extra():
    """合并第一轮 OCR 结果与 500 题目录专用的 OCR 结果（若存在）。"""
    maps = []
    for path in (os.path.join(C.ROOT, "outputs", "task6", "ocr_extra.json"),
                 os.path.join(OUT_DIR, "ocr_extra.json")):
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                out = {}
                for item in data.get("docs", []):
                    out[item["doc_id"]] = {p["page"]: p for p in item.get("pages", [])}
                maps.append(out)
            except Exception as e:
                print(f"[警告] OCR 补充文件读取失败 {path}: {e}")
    merged = {}
    for m in maps:
        for k, v in m.items():
            merged.setdefault(k, {}).update(v)
    return merged


def text_layer_stats(pdf_path):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    total = doc.page_count
    usable = 0
    for i in range(total):
        txt = doc.load_page(i).get_text("text")
        if len(txt.strip()) >= 40:
            usable += 1
    doc.close()
    return total, usable


def build_one(doc_id, ocr_extra):
    pdf_path, loc = C.resolve_pdf(doc_id)
    if pdf_path is None:
        return {"doc_id": doc_id, "build_status": "pdf_missing",
                "pdf_location": None, "error": "PDF 不在临时目录或项目 pdfs/"}
    try:
        total, usable = text_layer_stats(pdf_path)
    except Exception as e:
        return {"doc_id": doc_id, "build_status": "pdf_unreadable",
                "pdf_location": loc, "error": f"{type(e).__name__}: {e}"}
    if total <= 0:
        return {"doc_id": doc_id, "build_status": "pdf_unreadable",
                "pdf_location": loc, "num_pages": 0,
                "error": "PDF 页数为 0（通常是下载截断）"}
    try:
        # base6.build_doc 默认从 PDF_DIR 拼接路径；若 PDF 只在项目 pdfs/，
        # 临时指向实际目录，避免复制大文件，也不修改旧脚本。
        old_dir = base6.PDF_DIR
        base6.PDF_DIR = os.path.dirname(pdf_path)
        try:
            rec = base6.build_doc(doc_id, ocr_extra)
        finally:
            base6.PDF_DIR = old_dir
    except Exception as e:
        return {"doc_id": doc_id, "build_status": "build_failed",
                "pdf_location": loc, "num_pages": total,
                "error": f"{type(e).__name__}: {e}"}
    rec["pdf_path"] = pdf_path
    rec.update({
        "build_status": "ok" if rec.get("sections") else "empty_toc",
        "pdf_location": loc,
        "text_layer_pages": usable,
        "text_layer_ratio": round(usable / total, 4) if total else 0,
        "built_at": C.now_str(),
        "error": "",
    })
    return rec


def write_preview(docs, path=PREVIEW_MD):
    src = collections.Counter(d.get("toc_source") or d.get("build_status") for d in docs)
    lines = [
        "# 500 题目录（TOC）总览",
        "",
        "> 页码均为 PDF 实际页码（1-based）。目录只由 PDF 文本层/OCR 构建，"
        "未使用问题、答案或 gold evidence_pages。",
        "",
        "| 文档 | 页数 | 文本层可用页 | 目录来源 | 原始目录 | 章节数 | 状态 |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in docs:
        lines.append(
            f"| {d.get('doc_id')} | {d.get('num_pages') or '-'} | "
            f"{d.get('text_layer_pages', '-')} | {d.get('toc_source') or '-'} | "
            f"{'是' if d.get('is_original_toc') else '否'} | "
            f"{len(d.get('sections') or [])} | {d.get('build_status')} |")
    lines += ["", "## 目录来源统计", ""]
    for k, v in src.most_common():
        lines.append(f"- `{k}`: {v} 份文档")
    lines += ["", "## 章节明细（每份文档最多显示前 20 个章节）", ""]
    for d in docs:
        if d.get("build_status") != "ok":
            continue
        lines += [f"### {d['doc_id']}", "",
                  f"- 页数：{d.get('num_pages')}　目录来源：`{d.get('toc_source')}`　"
                  f"置信度：{d.get('confidence')}　文本层可用页：{d.get('text_layer_pages')}",
                  f"- 说明：{d.get('notes')}", ""]
        secs = d.get("sections") or []
        lines += ["| ID | 标题 | 层级 | 页码 | 起始页标题核验 |", "|---|---|---|---|---|"]
        for s in secs[:20]:
            mark = "通过" if s.get("verified_start_page") else "未匹配"
            lines.append(f"| {s['section_id']} | {s['title']} | {s['level']} | "
                         f"{s['start_page']}-{s['end_page']} | {mark} |")
        if len(secs) > 20:
            lines.append(f"| ... | 其余 {len(secs) - 20} 个章节见 toc.json | | | |")
        lines.append("")
    C.ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def save_all(docs_map, scope_docs):
    docs = [docs_map[d] for d in C.doc_ids_in_order() if d in docs_map]
    C.write_json_atomic(TOC_JSON, {
        "generated_at": C.now_str(),
        "page_definition": "PDF 实际页码，1-based",
        "input_policy": "仅 PDF 文本层；未读取问题/答案/evidence_pages。",
        "scope_docs": len(scope_docs),
        "docs": docs,
    })
    status_counts = collections.Counter(d.get("build_status") for d in docs)
    C.write_json_atomic(STATUS_JSON, {
        "generated_at": C.now_str(),
        "docs_total": len(docs),
        "status_counts": dict(status_counts),
        "docs": [{"doc_id": d["doc_id"], "build_status": d.get("build_status"),
                  "toc_source": d.get("toc_source"), "num_pages": d.get("num_pages"),
                  "sections": len(d.get("sections") or []),
                  "error": d.get("error", "")} for d in docs],
    })
    write_preview(docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 题涉及的文档；0=全部")
    ap.add_argument("--docs", default="", help="只处理指定 doc_id（逗号分隔）")
    ap.add_argument("--rebuild", action="store_true", help="忽略已有成功记录，强制重建")
    ap.add_argument("--dry-run", action="store_true", help="只统计，不构建目录")
    args = ap.parse_args()

    samples = C.load_samples()
    if args.limit:
        samples = samples[:args.limit]
    scope_docs = list(dict.fromkeys(s["doc_id"] for s in samples))
    if args.docs:
        wanted = {x.strip() for x in args.docs.split(",") if x.strip()}
        scope_docs = [d for d in scope_docs if d in wanted]

    existing = C.read_json(TOC_JSON, default={"docs": []}) or {"docs": []}
    docs_map = {d["doc_id"]: d for d in existing.get("docs", [])}

    if args.dry_run:
        ok = missing = unreadable = todo = 0
        for d in scope_docs:
            pdf, _ = C.resolve_pdf(d)
            if pdf is None:
                missing += 1
            elif C.pdf_page_count(pdf) is None:
                unreadable += 1
            elif docs_map.get(d, {}).get("build_status") == "ok" \
                    and docs_map.get(d, {}).get("sections") and not args.rebuild:
                ok += 1
            else:
                todo += 1
        print("=== TOC DRY-RUN（不构建） ===")
        print(f"范围内文档 {len(scope_docs)}：可建/待建 {todo}，已建 {ok}，"
              f"PDF 缺失 {missing}，PDF 损坏 {unreadable}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    ocr_extra = load_ocr_extra()
    built = skipped = failed = 0
    for i, doc_id in enumerate(scope_docs, 1):
        prev = docs_map.get(doc_id)
        if prev and prev.get("build_status") == "ok" and prev.get("sections") \
                and not args.rebuild:
            skipped += 1
            continue
        t0 = time.time()
        rec = build_one(doc_id, ocr_extra)
        docs_map[doc_id] = rec
        if rec.get("build_status") == "ok":
            built += 1
        else:
            failed += 1
        save_all(docs_map, scope_docs)
        print(f"[{i}/{len(scope_docs)}] {doc_id}: {rec.get('build_status')} "
              f"source={rec.get('toc_source')} sections={len(rec.get('sections') or [])} "
              f"({time.time() - t0:.1f}s) {rec.get('error', '')[:60]}")

    if not scope_docs:
        save_all(docs_map, scope_docs)
    print(f"\n本次新建 {built}，跳过已建 {skipped}，失败 {failed}")
    print(f"目录: {TOC_JSON}\n预览: {PREVIEW_MD}\n状态: {STATUS_JSON}")


if __name__ == "__main__":
    main()
