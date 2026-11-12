# -*- coding: utf-8 -*-
"""
JevSearch 500 题正式实验的共用工具。

用途：
  - 统一题目编号（qid）与采样文件行号的一一映射；
  - 统一 PDF 查找、页码解析、结果文件读写；
  - 所有新脚本都复用这里的规则，保证不同方法用完全相同的一批题。

约定：
  - qid = Q0001 ... Q0500，由 sampled_500_for_baseline.jsonl 的物理行号决定（1-based）。
  - 禁止重新排序或重新抽样；qid 一旦确定不再变化。
  - 页码均为 PDF 实际页码（1-based）。
  - 本模块只读采样文件，不写、不覆盖任何旧结果。
"""
import ast
import json
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "sampled_500_for_baseline.jsonl")

# 原始 PDF 缓存：优先系统临时目录，其次项目内 pdfs/
PDF_DIR = os.path.join(os.environ.get("TEMP", "."), "mmlongbench_pdfs")
PROJECT_PDF_DIR = os.path.join(ROOT, "pdfs")

# 正式实验统一输出根目录
OUT_ROOT = os.path.join(ROOT, "outputs", "final500")

MODEL_DS = "deepseek-v4.1-flash-1017"
MODEL_QWEN = "qwen3-vl-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

TASK_NAME = "mmlongbench_doc_500"


def qid_of(sample_index_1based):
    return f"Q{sample_index_1based:04d}"


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def load_samples(path=SAMPLES):
    """按物理行序读取 500 题；sample_index 为 1-based。"""
    if not os.path.exists(path):
        raise SystemExit(
            "未找到 sampled_500_for_baseline.jsonl（500 题清单与 gold 标注）。\n"
            "该文件不随仓库发布，请先按 README 第 9 节获取 MMLongBench-Doc，"
            "再执行 python prepare_data.py 重新生成。"
        )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # 原采样文件没有这两个字段，按物理行号生成；
            # 若传入的是保留 qid/sample_index 的子集文件，则沿用原映射，避免 qid 错位。
            d.setdefault("qid", qid_of(i))
            d.setdefault("sample_index", i)
            rows.append(d)
    return rows


def parse_pages(ep_str):
    """'[4, 5]' -> [4, 5]；'[]' -> []。解析失败返回 []。"""
    try:
        v = ast.literal_eval(ep_str) if isinstance(ep_str, str) else ep_str
        if not isinstance(v, (list, tuple)):
            return []
        out = []
        for x in v:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                continue
        return out
    except Exception:
        return []


def parse_sources(es_str):
    try:
        v = ast.literal_eval(es_str) if isinstance(es_str, str) else es_str
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else []
    except Exception:
        return []


def resolve_pdf(doc_id):
    """返回 (pdf_path, location)；找不到返回 (None, None)。"""
    for tag, d in (("temp", PDF_DIR), ("project", PROJECT_PDF_DIR)):
        p = os.path.join(d, doc_id)
        if os.path.exists(p):
            return p, tag
    return None, None


def pdf_page_count(pdf_path):
    """打开 PDF 返回页数；失败或 0 页（下载截断）返回 None。"""
    try:
        import pymupdf
        doc = pymupdf.open(pdf_path)
        n = doc.page_count
        doc.close()
        return n if n > 0 else None
    except Exception:
        return None


def write_json_atomic(path, obj):
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path, record):
    """逐条落盘并 flush；断点续跑只追加，不覆盖已成功记录。"""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def last_by_qid(records):
    """同一 qid 可能有多次失败尝试；以最后一条为准。"""
    out = {}
    for r in records:
        if r.get("qid"):
            out[r["qid"]] = r
    return out


def attempts_by_qid(records):
    out = {}
    for r in records:
        q = r.get("qid")
        if q:
            out[q] = out.get(q, 0) + 1
    return out


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def doc_ids_in_order():
    seen, ids = set(), []
    for r in load_samples():
        d = r.get("doc_id")
        if d and d not in seen:
            seen.add(d)
            ids.append(d)
    return ids
