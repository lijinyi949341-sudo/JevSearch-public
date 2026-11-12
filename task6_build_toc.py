# -*- coding: utf-8 -*-
"""
文档结构解析（第一步）：从真实 PDF 构建结构化文档目录（TOC）。

构建优先级（与导师要求一致）：
  1. PyMuPDF 书签（PDF 自带目录）
  2. 正文中的“印刷目录页”（如 Annual Report / Newsletter 的 Contents 列表），
     并把印刷页码映射为 PDF 实际页码（1-based）
  3. 从正文标题的字号/编号特征识别一级、二级标题
  4. 以上都失败时，生成“页面主题索引”，显式标记为 pseudo-TOC（不伪装成原始目录）

说明：
  - 只读取 PDF 与可选的 OCR 补充结果，不读取任何问题、答案或 evidence_pages。
  - 所有页码均为 PDF 实际页码，从 1 开始。
  - 输出 outputs/task6/toc.json 与 outputs/task6/toc_preview.md。
  - 不修改其他模块的结果文件。

用法：
  py -3.12 task6_build_toc.py
"""
import argparse
import json
import os
import re
import time

import pymupdf

PDF_DIR = os.path.join(os.environ.get("TEMP", "."), "mmlongbench_pdfs")
OUT_DIR = os.path.join("outputs", "task6")
OCR_EXTRA = os.path.join(OUT_DIR, "ocr_extra.json")

# 第一轮最小实验选用的 4 份文档（均为本机真实存在、且与问答数据 doc_id 一致的 PDF）
DOCS = [
    "STEPBACK.pdf",
    "2023.findings-emnlp.248.pdf",
    "fdac8d1e9ef56519371df7e6532df27d.pdf",
    "0b85477387a9d0cc33fca0f4becaa0e5.pdf",
]

PURE_NUM = re.compile(r"^\d{1,3}$")
MARKER = re.compile(r"^(\d{1,2}|[A-Z])$")
DASHES = str.maketrans({"‐": "-", "‑": "-", "–": "-", "—": "-", "‒": "-"})


# ---------------------------------------------------------------- 基础工具
def norm(s):
    s = str(s).translate(DASHES).lower()
    s = s.replace("‟", "'").replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def tokens(s):
    return set(re.findall(r"[a-z0-9]+", norm(s)))


def page_lines(page):
    """返回 [(字号, y, 文本)]，按阅读顺序。"""
    out = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            txt = "".join(sp["text"] for sp in line["spans"]).strip()
            if not txt:
                continue
            size = max(sp["size"] for sp in line["spans"])
            out.append((round(size, 1), round(line["bbox"][1], 1), txt))
    return out


def body_size(doc):
    """按字符数加权的文档正文字号（众数）。"""
    import collections
    cnt = collections.Counter()
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for sp in line["spans"]:
                    cnt[round(sp["size"], 1)] += len(sp["text"])
    return cnt.most_common(1)[0][0] if cnt else 10.0


def detect_printed_offset(doc):
    """
    用每页页眉/页脚的独立数字推断“印刷页码 - PDF 页码”的偏移。
    例如年报 p3 顶部印着 5，则偏移 = 5-3 = 2。
    """
    import collections
    votes = collections.Counter()
    for i, page in enumerate(doc):
        if i == 0:              # 第 1 页通常是封面/目录，不参与投票
            continue
        h = page.rect.height
        for size, y, txt in page_lines(page):
            if PURE_NUM.match(txt) and (y < 0.12 * h or y > 0.90 * h):
                num = int(txt)
                if 1 <= num <= 300:
                    votes[num - (i + 1)] += 1
                break           # 每页只取第一个页眉/页脚数字
    if not votes:
        return 0, 0
    offset, n = votes.most_common(1)[0]
    return offset, n


def build_ranges(entries, num_pages):
    """
    为目录项补全 end_page。
    规则：本级或更高层级的下一项起始页 - 1；没有下一项则到文档末页。
    同一页开始的兄弟小节按“至少占 1 页”处理，避免出现空范围。
    """
    for i, e in enumerate(entries):
        end = num_pages
        for j in range(i + 1, len(entries)):
            if entries[j]["level"] <= e["level"]:
                end = entries[j]["start_page"] - 1
                break
        e["end_page"] = max(e["start_page"], end)


def verify_sections(doc, sections):
    """用起始页正文检查标题是否真的出现在该页（避免虚构标题）。"""
    for s in sections:
        p = doc[s["start_page"] - 1]
        pt = tokens(p.get_text())
        tt = tokens(s["title"])
        tt = {t for t in tt if t not in {"the", "of", "and", "a", "an", "to", "in", "for", "on"}}
        s["verified_start_page"] = bool(tt) and len(tt & pt) / len(tt) >= 0.5


# ---------------------------------------------------------------- 1. 书签
def from_bookmarks(doc, num_pages):
    toc = doc.get_toc(simple=True)
    if len(toc) < 3:
        return None
    junk = sum(1 for lvl, title, pg in toc if re.fullmatch(r"dianummer\s*\d*", norm(title)))
    if junk / len(toc) > 0.5:      # 这类书签只是页码占位符，不是真目录
        return None
    entries = []
    for lvl, title, pg in toc:
        title = re.sub(r"\s+", " ", title).strip()
        if not title or not (1 <= pg <= num_pages):
            continue
        entries.append({"title": title, "level": max(1, int(lvl)),
                        "start_page": int(pg)})
    if len(entries) < 3:
        return None
    build_ranges(entries, num_pages)
    return {
        "toc_source": "pdf_bookmark",
        "is_original_toc": True,
        "notes": "PDF 自带书签目录（PyMuPDF get_toc），页码为 PDF 实际页码。",
        "confidence": 0.95,
        "entries": entries,
    }


# ---------------------------------------------------------------- 2. 印刷目录页
CONTENTS_MARK = re.compile(r"(contents|in this issue)", re.I)


def parse_contents_page(doc):
    """从正文前两页的 Contents / In this issue 列表中解析 (标题, 印刷页码)。"""
    for pi in range(min(2, doc.page_count)):
        lines = [t for _, _, t in page_lines(doc[pi])]
        texts = "\n".join(lines)
        if not CONTENTS_MARK.search(texts):
            continue
        entries, prev = [], None
        started = False
        for ln in lines:
            t = ln.strip()
            if not t:
                continue
            if not started:
                if CONTENTS_MARK.search(t):
                    started = True
                continue
            if PURE_NUM.match(t):
                pg = int(t)
                if prev and 1 <= pg <= 300:
                    entries.append((prev, pg))
                prev = None
            elif re.search(r"[A-Za-z]{2}", t) and len(t) <= 90 \
                    and not t.lower().startswith(("volume", "issue")):
                # 形如 “Mission Profile  4” 的同一行写法
                m = re.match(r"^(.+?)\s{1,}(\d{1,3})$", t)
                if m:
                    entries.append((m.group(1).strip(), int(m.group(2))))
                    prev = None
                else:
                    prev = t
        if len(entries) >= 5:
            return entries, pi + 1
    return None, None


def from_contents_page(doc, num_pages):
    entries_raw, cpage = parse_contents_page(doc)
    if not entries_raw:
        return None
    offset, votes = detect_printed_offset(doc)
    if votes < 3:
        offset = 0
    mapped, skipped = [], 0
    for title, printed in entries_raw:
        start = printed - offset
        if 1 <= start <= num_pages:
            mapped.append({"title": title, "level": 1, "start_page": start})
        else:
            skipped += 1
    if len(mapped) < 3:
        return None
    # 同一标题重复时保留第一处
    seen, uniq = set(), []
    for e in mapped:
        key = norm(e["title"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    build_ranges(uniq, num_pages)
    return {
        "toc_source": "text_contents_page",
        "is_original_toc": True,
        "notes": (f"第 {cpage} 页存在印刷目录列表；印刷页码与 PDF 页码偏移 = "
                  f"{offset:+d}（页眉页码投票 {votes} 次），越界/未收录章节 {skipped} 项已排除。"),
        "confidence": 0.90 if votes >= 3 else 0.70,
        "entries": uniq,
        "printed_offset": offset,
    }


# ---------------------------------------------------------------- 3. 正文标题
def detect_headings(doc, num_pages):
    body = body_size(doc)
    per_page = []
    for i, page in enumerate(doc):
        cand = []
        for size, y, txt in page_lines(page):
            if not (body + 0.4 <= size <= 24 and len(txt) <= 100
                    and re.search(r"[A-Za-z0-9]", txt)):
                continue
            if "@" in txt or "university" in txt.lower():
                continue
            cand.append((size, y, txt))
        # 同一 y 坐标的片段（如 “2” 与 “Related Work” 被拆成两行）先合并
        merged, k = [], 0
        while k < len(cand):
            size, y, txt = cand[k]
            title, last_y = txt, y
            k += 1
            while k < len(cand) and abs(cand[k][0] - size) <= 0.1 \
                    and abs(cand[k][1] - last_y) <= 2.5:
                title += " " + cand[k][2]
                last_y = cand[k][1]
                k += 1
            merged.append({"title": re.sub(r"\s+", " ", title).strip(),
                           "level": 1, "start_page": i + 1, "font_size": size,
                           "_y": last_y})
        # 多行标题续行合并（同字号、行距接近）；编号/字母开头的标题不再向后吞并
        compact = []
        for e in merged:
            prev = compact[-1] if compact else None
            if prev and prev["start_page"] == e["start_page"] \
                    and abs(prev["font_size"] - e["font_size"]) <= 0.1 \
                    and e["_y"] - prev["_y"] <= e["font_size"] * 1.8 \
                    and len(prev["title"]) < 80 and not MARKER.match(prev["title"]) \
                    and not MARKER.match(e["title"]):
                prev["title"] += " " + e["title"]
                prev["_y"] = e["_y"]
            else:
                compact.append(e)
        # 首页（标题页）只保留 Abstract 或 “编号 + 标题”形式的真实章节标题
        final = []
        for e in compact:
            t = e["title"]
            if e["start_page"] == 1 and not (re.match(r"^\d{1,2}\s+\S", t)
                                            or t.lower() in {"abstract", "introduction"}):
                continue
            e.pop("_y", None)
            if len(t) >= 3:
                final.append(e)
        per_page.extend(final)

    # 去重（同一标题只保留第一次出现，通常是真实章节起点）
    seen, uniq = set(), []
    for e in per_page:
        key = norm(e["title"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    if len(uniq) < 4:
        return None
    build_ranges(uniq, num_pages)
    return {
        "toc_source": "text_heading",
        "is_original_toc": False,
        "notes": ("PDF 无可读书签与印刷目录；由正文字号/编号推导出的章节标题，"
                  "标题均为页面真实文字，非人工编造。"),
        "confidence": 0.75,
        "entries": uniq,
    }


# ---------------------------------------------------------------- 4. 页面主题索引
def from_pseudo_index(doc, num_pages):
    entries = []
    for i, page in enumerate(doc):
        line = None
        for size, y, txt in page_lines(page):
            if len(txt) >= 4 and re.search(r"[A-Za-z]", txt) and len(txt) <= 100:
                line = txt
                break
        title = re.sub(r"\s+", " ", line or f"Page {i + 1}").strip()
        entries.append({"title": title, "level": 1, "start_page": i + 1})
    build_ranges(entries, num_pages)
    return {
        "toc_source": "pseudo_toc",
        "is_original_toc": False,
        "notes": ("页面无正式目录且无法可靠识别标题，仅按“每页首个可读行”建立页面主题索引；"
                  "这是 pseudo-TOC，不代表原文目录。"),
        "confidence": 0.35,
        "entries": entries,
    }


# ---------------------------------------------------------------- 主流程
def load_ocr_extra():
    if not os.path.exists(OCR_EXTRA):
        return {}
    with open(OCR_EXTRA, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for item in data.get("docs", []):
        out[item["doc_id"]] = {p["page"]: p for p in item.get("pages", [])}
    return out


def build_doc(doc_id, ocr_extra):
    path = os.path.join(PDF_DIR, doc_id)
    doc = pymupdf.open(path)
    num_pages = doc.page_count
    result = from_bookmarks(doc, num_pages) \
        or from_contents_page(doc, num_pages) \
        or detect_headings(doc, num_pages) \
        or from_pseudo_index(doc, num_pages)
    entries = result.pop("entries")
    sections = []
    for idx, e in enumerate(entries, 1):
        sections.append({
            "section_id": f"S{idx:02d}",
            "title": e["title"],
            "level": e["level"],
            "start_page": e["start_page"],
            "end_page": e["end_page"],
            "source": result["toc_source"],
            "confidence": result["confidence"],
        })
    verify_sections(doc, sections)

    ocr_pages = sorted(ocr_extra.get(doc_id, {}).keys())
    record = {
        "doc_id": doc_id,
        "pdf_path": path,
        "num_pages": num_pages,
        "toc_source": result["toc_source"],
        "is_original_toc": result["is_original_toc"],
        "confidence": result["confidence"],
        "notes": result["notes"],
        "printed_offset": result.get("printed_offset"),
        "ocr_used_pages": ocr_pages,
        "sections": sections,
    }
    doc.close()
    return record


def write_preview(docs):
    lines = [
        "# 文档目录（TOC）预览",
        "",
        "> 页码均为 PDF 实际页码（从 1 开始）。"
        "`pdf_bookmark` / `text_contents_page` 属于原文档自带结构；"
        "`text_heading` 为正文标题推导；`pseudo_toc` 是页面主题索引，不代表原始目录。",
        "",
        "| 文档 | 页数 | 目录来源 | 原始目录 | 章节数 |",
        "|---|---|---|---|---|",
    ]
    for d in docs:
        lines.append(f"| {d['doc_id']} | {d['num_pages']} | {d['toc_source']} | "
                     f"{'是' if d['is_original_toc'] else '否'} | {len(d['sections'])} |")
    for d in docs:
        lines += ["", f"## {d['doc_id']}", "",
                  f"- 页数：{d['num_pages']}　　目录来源：`{d['toc_source']}`　　"
                  f"置信度：{d['confidence']}",
                  f"- 说明：{d['notes']}",
                  ""]
        if d["ocr_used_pages"]:
            lines.append(f"- 补充 OCR 页：{d['ocr_used_pages']}")
            lines.append("")
        lines += ["| ID | 标题 | 层级 | 页码 | 起始页标题核验 |",
                  "|---|---|---|---|---|"]
        for s in d["sections"]:
            mark = "通过" if s["verified_start_page"] else "未匹配（标题可能分栏/缩写）"
            lines.append(f"| {s['section_id']} | {s['title']} | {s['level']} | "
                         f"{s['start_page']}-{s['end_page']} | {mark} |")
    with open(os.path.join(OUT_DIR, "toc_preview.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default=",".join(DOCS))
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    ocr_extra = load_ocr_extra()
    docs = []
    for doc_id in [x.strip() for x in args.docs.split(",") if x.strip()]:
        if not os.path.exists(os.path.join(PDF_DIR, doc_id)):
            print(f"[跳过] PDF 不存在: {doc_id}")
            continue
        t0 = time.time()
        rec = build_doc(doc_id, ocr_extra)
        docs.append(rec)
        print(f"[OK] {doc_id}: pages={rec['num_pages']} source={rec['toc_source']} "
              f"sections={len(rec['sections'])} verify="
              f"{sum(1 for s in rec['sections'] if s['verified_start_page'])}"
              f" ocr_pages={rec['ocr_used_pages']} ({time.time() - t0:.1f}s)")
    with open(os.path.join(OUT_DIR, "toc.json"), "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "page_definition": "PDF 实际页码，1-based",
                   "docs": docs}, f, ensure_ascii=False, indent=2)
    write_preview(docs)
    print(f"\n已写入 {os.path.join(OUT_DIR, 'toc.json')}")
    print(f"已写入 {os.path.join(OUT_DIR, 'toc_preview.md')}")


if __name__ == "__main__":
    main()
