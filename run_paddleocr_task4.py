# -*- coding: utf-8 -*-
"""
OCR：PaddleOCR 3.x 最小可运行实验（真实推理）。

- 输入：outputs/ocr/images/ 下已渲染的 PDF 页面 PNG（第 1/3/8/15 页）。
- 使用 PaddleOCR 3.7.0 预训练模型做本地文字检测 + 识别（不联网调用大模型）。
- 输出：
    outputs/ocr/ocr_raw_results.json   原始识别结果（文字/置信度/坐标框）
    outputs/ocr/vis/*.png              带检测框的可视化图
    outputs/ocr/ocr_vs_pymupdf.json    OCR 与 PyMuPDF 文本层提取的对比
- 不修改任何原有实验文件。
"""
import argparse
import json
import os
import re
import time

# PaddleX 默认缓存目录是 ~/.paddlex，本机该路径不可写；改到可写位置再导入 paddleocr
os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    os.path.join(os.path.expanduser("~"), ".cache", "paddlex"),
)

import pymupdf  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

PDF_PATH = os.path.join("pdfs", "379f44022bb27aa53efd5d322c7b57bf.pdf")
DOC_ID = os.path.basename(PDF_PATH)
IMG_DIR = os.path.join("outputs", "ocr", "images")
VIS_DIR = os.path.join("outputs", "ocr", "vis")
RAW_JSON = os.path.join("outputs", "ocr", "ocr_raw_results.json")
CMP_JSON = os.path.join("outputs", "ocr", "ocr_vs_pymupdf.json")
DEFAULT_PAGES = [1, 3, 8, 15]


def image_path(page_no):
    stem = DOC_ID.replace(".pdf", "")
    return os.path.join(IMG_DIR, f"{stem}_p{page_no}.png")


def tokenize(s):
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))


def extract_items(res):
    """兼容 PaddleOCR 3.x 的 OCRResult 结构，返回 texts/scores/polys。"""
    d = None
    try:
        d = dict(res)
    except Exception:
        d = getattr(res, "res", None)
        if d is None:
            try:
                d = res.to_dict()
            except Exception:
                d = {}
    texts = list(d.get("rec_texts") or [])
    scores = list(d.get("rec_scores") or [])
    polys = d.get("rec_polys")
    if polys is None:
        polys = d.get("dt_polys")
    polys = [] if polys is None else [
        (p.tolist() if hasattr(p, "tolist") else p) for p in polys
    ]
    return texts, scores, polys


def draw_boxes(png_path, items, out_path):
    img = Image.open(png_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, it in enumerate(items):
        box = it["box"]
        if not box:
            continue
        pts = [(float(x), float(y)) for x, y in box]
        pts.append(pts[0])
        draw.line(pts, fill=(0, 200, 0), width=3)
        draw.text((pts[0][0] + 2, max(0, pts[0][1] - 12)),
                  str(i), fill=(255, 0, 0))
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", default="all",
                        help="all 或逗号分隔页码，如 1 或 1,3")
    args = parser.parse_args()
    pages = DEFAULT_PAGES if args.pages.lower() == "all" else \
        [int(x) for x in args.pages.split(",") if x.strip()]

    os.makedirs(VIS_DIR, exist_ok=True)

    print("初始化 PaddleOCR（首次运行会自动下载预训练模型）...")
    t_init = time.time()
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        # PaddlePaddle 3.3.1 的 oneDNN 后端与 PP-OCRv6 模型不兼容，关闭 MKLDNN 走普通 CPU 推理
        enable_mkldnn=False,
    )
    print(f"模型初始化完成，耗时 {time.time() - t_init:.1f}s")

    doc = pymupdf.open(PDF_PATH)
    raw_pages, cmp_pages = [], []

    for pno in pages:
        ip = image_path(pno)
        if not os.path.exists(ip):
            print(f"[跳过] 图片不存在: {ip}")
            continue
        t0 = time.time()
        result = ocr.predict(ip)
        elapsed = round(time.time() - t0, 2)

        items = []
        for res in result:
            texts, scores, polys = extract_items(res)
            for txt, sc, poly in zip(texts, scores, polys):
                box = poly if poly else None
                items.append({
                    "text": txt,
                    "score": round(float(sc), 4) if sc is not None else None,
                    "box": box,
                })

        ocr_text = "\n".join(it["text"] for it in items)
        avg_score = round(sum(it["score"] for it in items if it["score"]) /
                          max(1, sum(1 for it in items if it["score"])), 4)

        vis_path = os.path.join(VIS_DIR, os.path.basename(ip).replace(".png", "_vis.png"))
        draw_boxes(ip, items, vis_path)

        pm_text = doc[pno - 1].get_text("text")
        pm_tokens = tokenize(pm_text)
        ocr_tokens = tokenize(ocr_text)
        ocr_only = sorted(ocr_tokens - pm_tokens)
        pm_only = sorted(pm_tokens - ocr_tokens)

        raw_pages.append({
            "page": pno,
            "image": ip,
            "n_boxes": len(items),
            "avg_score": avg_score,
            "elapsed_sec": elapsed,
            "ocr_text": ocr_text,
            "items": items,
        })
        cmp_pages.append({
            "page": pno,
            "ocr_char_count": len(ocr_text.replace("\n", "")),
            "pymupdf_char_count": len(pm_text.replace("\n", "")),
            "ocr_token_count": len(ocr_tokens),
            "pymupdf_token_count": len(pm_tokens),
            "token_overlap": len(ocr_tokens & pm_tokens),
            "ocr_only_token_count": len(ocr_only),
            "pymupdf_only_token_count": len(pm_only),
            "ocr_only_sample": ocr_only[:40],
            "pymupdf_only_sample": pm_only[:40],
        })
        print(f"page {pno}: boxes={len(items)} avg_score={avg_score} "
              f"ocr_chars={len(ocr_text.replace(chr(10), ''))} "
              f"pm_chars={len(pm_text.replace(chr(10), ''))} elapsed={elapsed}s")

    doc.close()

    raw = {
        "doc_id": DOC_ID,
        "pdf": PDF_PATH,
        "paddleocr_version": __import__("paddleocr").__version__,
        "paddlepaddle_version": __import__("paddle").__version__,
        "engine": "PaddleOCR 3.x pretrained (local inference, lang=en)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pages": raw_pages,
    }
    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    with open(CMP_JSON, "w", encoding="utf-8") as f:
        json.dump({"doc_id": DOC_ID, "pages": cmp_pages}, f, ensure_ascii=False, indent=2)

    print(f"\n原始结果: {RAW_JSON}")
    print(f"对比结果: {CMP_JSON}")
    print(f"可视化图: {VIS_DIR}")


if __name__ == "__main__":
    main()
