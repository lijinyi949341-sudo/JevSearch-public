# -*- coding: utf-8 -*-
"""
OCR 补充步骤：对“没有可提取文字”的少量页面补充 PaddleOCR。

为什么要做：早期 OCR 验证只覆盖了 379f...pdf 的第 1/3/8/15 页，并不是所有 PDF 都已 OCR。
目录构建优先使用 PDF 自带文字层；当某页文字层为空时（例如整页地图/图片），
用 PaddleOCR 识别该页，确认页面上是否存在可用作章节标题的文字。

复用同一 OCR 环境与参数（.venv_paddleocr + PaddleOCR 3.7.0，lang=en，关闭 MKLDNN）。
输出：outputs/task6/ocr_extra.json（供 task6_build_toc.py 读取）

用法（必须用 OCR 虚拟环境运行）：
  .venv_paddleocr\\Scripts\\python.exe task6_ocr_missing_pages.py
"""
import json
import os
import time

os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME",
    os.path.join(os.path.expanduser("~"), ".cache", "paddlex"),
)

import pymupdf  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402

PDF_DIR = os.path.join(os.environ.get("TEMP", "."), "mmlongbench_pdfs")
OUT_DIR = os.path.join("outputs", "task6")
IMG_DIR = os.path.join(OUT_DIR, "ocr_images")
OUT_JSON = os.path.join(OUT_DIR, "ocr_extra.json")

# 只补充“文字层为空、且属于第一轮实验文档”的页面
TARGETS = [
    ("fdac8d1e9ef56519371df7e6532df27d.pdf", [17]),
]


def extract_items(res):
    d = dict(res)
    texts = list(d.get("rec_texts") or [])
    scores = list(d.get("rec_scores") or [])
    return texts, scores


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    print("初始化 PaddleOCR ...")
    t0 = time.time()
    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    print(f"初始化完成 {time.time() - t0:.1f}s")

    docs = []
    for doc_id, pages in TARGETS:
        path = os.path.join(PDF_DIR, doc_id)
        doc = pymupdf.open(path)
        out_pages = []
        for pno in pages:
            img_path = os.path.join(IMG_DIR, f"{doc_id.replace('.pdf', '')}_p{pno}.png")
            pix = doc[pno - 1].get_pixmap(dpi=200)
            pix.save(img_path)
            t1 = time.time()
            result = ocr.predict(img_path)
            elapsed = round(time.time() - t1, 2)
            texts, scores = [], []
            for res in result:
                t, s = extract_items(res)
                texts.extend(t)
                scores.extend(s)
            avg = round(sum(scores) / max(1, len(scores)), 4)
            out_pages.append({
                "page": pno,
                "image": img_path,
                "n_boxes": len(texts),
                "avg_score": avg,
                "elapsed_sec": elapsed,
                "ocr_text": "\n".join(texts),
            })
            print(f"{doc_id} p{pno}: boxes={len(texts)} avg={avg} {elapsed}s")
        doc.close()
        docs.append({"doc_id": doc_id, "pages": out_pages})

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "engine": "PaddleOCR 3.x pretrained (lang=en, CPU)",
                   "docs": docs}, f, ensure_ascii=False, indent=2)
    print("已写入", OUT_JSON)


if __name__ == "__main__":
    main()
