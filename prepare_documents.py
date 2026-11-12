import json
import os
import urllib.request

import fitz  # PyMuPDF

# Hugging Face 数据集中 documents/ 目录下存放了 135 个真实 PDF
BASE_URL = "https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/resolve/main/documents/"

# 原始 PDF 属于中间产物，放到系统临时目录，不进项目目录
PDF_DIR = os.path.join(os.environ.get("TEMP", "."), "mmlongbench_pdfs")
OUT_FILE = "document_texts.json"  # 抽取后的正文，供 RAG 脚本消费


def main():
    # 1. 读取采样文件，得到所有用到的 doc_id（按出现顺序去重）
    doc_ids = []
    seen = set()
    with open("sampled_500_for_baseline.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            did = data.get("doc_id", "")
            if did and did not in seen:
                seen.add(did)
                doc_ids.append(did)

    print(f"需要处理 {len(doc_ids)} 个唯一文档...")
    os.makedirs(PDF_DIR, exist_ok=True)

    cache = {}
    failed = []
    for i, did in enumerate(doc_ids, 1):
        path = os.path.join(PDF_DIR, did)
        if not os.path.exists(path):
            try:
                urllib.request.urlretrieve(BASE_URL + did, path)
            except Exception as e:
                print(f"[下载失败] {did}: {e}")
                failed.append(did)
                continue

        try:
            doc = fitz.open(path)
            pages = [doc[p].get_text() for p in range(doc.page_count)]
            full_text = "\n".join(pages)
            cache[did] = {"full_text": full_text, "num_pages": doc.page_count}
            doc.close()
        except Exception as e:
            print(f"[解析失败] {did}: {e}")
            failed.append(did)
            continue

        if i % 20 == 0:
            print(f"已完成 {i}/{len(doc_ids)} ...")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)

    total_chars = sum(len(v["full_text"]) for v in cache.values())
    print(f"完成：成功 {len(cache)}/{len(doc_ids)}，失败 {len(failed)}")
    print(f"失败列表: {failed}")
    print(f"正文总字符数: {total_chars}")
    print(f"原文已缓存到 {OUT_FILE}")


if __name__ == "__main__":
    main()