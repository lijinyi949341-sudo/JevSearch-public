# -*- coding: utf-8 -*-
"""
DeepSeek V4.1 Flash 500 题版多模态问答（断点续跑）。

与 run_advanced_rag_ds.py 的关系：
  - 模型、专家提示词、不可答规则、PDF 页面渲染参数（300DPI / 最长边 2000 /
    max_tokens 1024 / enable_thinking=False）、证据页选择方式完全一致；
  - 本文件只扩展到 500 题，并增加断点续跑、逐条落盘、失败状态记录；
  - 原 10 题结果（advanced_rag_results_ds_smoke10.jsonl）保持不动。

关键行为：
  * PDF 缺失 / 无法解析 / 证据页全部越界 → 不调用模型，记为失败状态，
    绝不把空输入当成“成功的多模态回答”。
  * 只有 API 成功且回答非空才 ds_success=True。
  * 结果文件 append-only；同一 qid 的成功记录永不重跑、永不覆盖。
  * 失败记录可重跑，默认每个 qid 最多尝试 --max-attempts 次（默认 2）。

标注：本实验使用数据集提供的 gold evidence_pages 定位关键页，
属于“已知证据页的问答”，不是完整检索系统的端到端表现。

用法：
  # 只检查 PDF/页码，不调用 API
  python run_advanced_rag_ds_500.py --dry-run --limit 50
  # 先试 50 题
  python run_advanced_rag_ds_500.py --limit 50
  # 续跑剩余全部 500 题（已成功的自动跳过）
  python run_advanced_rag_ds_500.py --limit 0
"""
import argparse
import base64
import json
import os
import sys
import time

import pymupdf
from openai import OpenAI

import final500_common as C

MODEL_NAME = C.MODEL_DS
BASE_URL = C.BASE_URL
OUT_PATH = os.path.join(C.OUT_ROOT, "qa_multimodal", "deepseek_v41", "ds_results.jsonl")
SUMMARY_PATH = os.path.join(C.OUT_ROOT, "qa_multimodal", "deepseek_v41", "run_summary.json")

MAX_PAGES = 10
RENDER_DPI = 300
MAX_RENDER_DIM = 2000


def get_expert_prompt(doc_type):
    doc_type_lower = doc_type.lower()
    if "financial" in doc_type_lower or "report" in doc_type_lower:
        return "你是一位精通财务与数据分析的专家。请极其仔细地核对数字、表格和百分比，给出精确、简洁的答案。"
    elif "academic" in doc_type_lower or "research" in doc_type_lower:
        return "你是一位严谨的学术研究员。请专注于提取论文中的核心方法、实验结论和定义，避免过度推测。"
    elif "tutorial" in doc_type_lower or "guidebook" in doc_type_lower:
        return "你是一位资深的技术文档专家。请精准定位步骤、操作指南和属性说明，直接给出正确结论。"
    else:
        return "你是一个精准的科研问答助手，请结合文档图片直接回答问题，尽可能简短准确。"


NOT_ANSWERABLE_RULE = (
    "【极度重要·不可答规则】只有当你能在提供的文档图片中找到确凿证据时，才给出具体答案。"
    "如果图片中找不到足够信息（信息缺失、页码对不上、图中不含相关内容），"
    "你的回答必须以短语 'Not answerable' 开头，表示此题不可答。"
    "严禁在这种情况下编造数字或结论、严禁臆测、不要使用外部常识。"
)


def effective_dpi(page, target_dpi=RENDER_DPI, max_dim=MAX_RENDER_DIM):
    longest = max(page.rect.width, page.rect.height)
    if longest * target_dpi / 72.0 <= max_dim:
        return target_dpi
    return int(max_dim * 72.0 / longest)


def prepare_input(sample, do_render=True):
    """检查 PDF 与页码，返回渲染好的图片及输入状态；不发起 API 调用。

    do_render=False 时只校验 PDF/页码，不实际渲染图片（供 dry-run 快速统计）。
    """
    doc_id = sample["doc_id"]
    pdf_path, loc = C.resolve_pdf(doc_id)
    requested = C.parse_pages(sample.get("evidence_pages", "[]"))
    input_mode = "gold_evidence_pages"
    if not requested:
        # 与 Qwen 版保持一致：证据页为空的题回退到第 1 页
        requested = [1]
        input_mode = "unanswerable_fallback_page1"
    requested = requested[:MAX_PAGES]

    if pdf_path is None:
        return {"status": "pdf_missing", "error": "PDF 不在临时目录或项目 pdfs/",
                "requested_pages": requested, "rendered_pages": [], "missing_pages": requested,
                "images": [], "num_pages": None, "pdf_location": None, "input_mode": input_mode}

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        return {"status": "pdf_unreadable", "error": f"{type(e).__name__}: {e}",
                "requested_pages": requested, "rendered_pages": [], "missing_pages": requested,
                "images": [], "num_pages": None, "pdf_location": loc, "input_mode": input_mode}

    num_pages = doc.page_count
    if num_pages <= 0:
        doc.close()
        return {"status": "pdf_unreadable", "error": "PDF 页数为 0（通常是下载截断）",
                "requested_pages": requested, "rendered_pages": [], "missing_pages": requested,
                "images": [], "num_pages": 0, "pdf_location": loc, "input_mode": input_mode}
    images, rendered, missing = [], [], []
    render_error = ""
    for pnum in requested:
        idx = pnum - 1
        if idx < 0 or idx >= num_pages:
            missing.append(pnum)
            continue
        if not do_render:
            rendered.append(pnum)
            continue
        try:
            page = doc[idx]
            pix = page.get_pixmap(dpi=effective_dpi(page))
            images.append("data:image/png;base64," +
                          base64.b64encode(pix.tobytes("png")).decode("ascii"))
            rendered.append(pnum)
        except Exception as e:
            render_error = f"{type(e).__name__}: {e}"
            missing.append(pnum)
    doc.close()

    if not rendered:
        return {"status": "no_images", "error": render_error or "所有请求页都无法渲染",
                "requested_pages": requested, "rendered_pages": rendered,
                "missing_pages": missing, "images": [], "num_pages": num_pages,
                "pdf_location": loc, "input_mode": input_mode}

    status = "ready" if not missing else "ready_partial_evidence"
    return {"status": status, "error": render_error, "requested_pages": requested,
            "rendered_pages": rendered, "missing_pages": missing, "images": images,
            "num_pages": num_pages, "pdf_location": loc, "input_mode": input_mode}


def call_model(client, sample, prepared):
    expert = get_expert_prompt(sample.get("doc_type", "General")) + "\n\n" + NOT_ANSWERABLE_RULE
    content = [{"type": "image_url", "image_url": {"url": img}}
               for img in prepared["images"]]
    content.append({"type": "text", "text": sample["question"]})
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": expert},
                      {"role": "user", "content": content}],
            max_tokens=1024,
            extra_body={"enable_thinking": False},
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        u = None
        if usage is not None:
            u = {"prompt_tokens": usage.prompt_tokens,
                 "completion_tokens": usage.completion_tokens,
                 "total_tokens": usage.total_tokens}
        return {
            "ds_answer": text,
            "ds_success": bool(text.strip()),
            "status": "ok" if text.strip() else "empty_response",
            "error": "" if text.strip() else "EMPTY_RESPONSE: 模型返回空回答",
            "model_returned": getattr(resp, "model", None),
            "usage": u,
        }
    except Exception as e:
        return {"ds_answer": "", "ds_success": False, "status": "api_error",
                "error": f"{type(e).__name__}: {str(e)[:500]}",
                "model_returned": None, "usage": None}
    finally:
        pass


def make_record(sample, prepared, result, attempt, latency):
    return {
        "qid": sample["qid"],
        "sample_index": sample["sample_index"],
        "task": "qa_multimodal_gold_pages",
        "model_requested": MODEL_NAME,
        "doc_id": sample["doc_id"],
        "doc_type": sample.get("doc_type", ""),
        "question": sample.get("question", ""),
        "input_mode": prepared["input_mode"],
        "requested_pages": prepared["requested_pages"],
        "rendered_pages": prepared["rendered_pages"],
        "missing_pages": prepared["missing_pages"],
        "image_count": len(prepared["images"]),
        "pdf_location": prepared.get("pdf_location"),
        "pdf_num_pages": prepared.get("num_pages"),
        "status": result["status"],
        "ds_success": result["ds_success"],
        "ds_answer": result["ds_answer"],
        "ds_error": result["error"],
        "model_returned": result["model_returned"],
        "usage": result["usage"],
        "latency_sec": round(latency, 3),
        "attempt": attempt,
        "run_at": C.now_str(),
        # 仅离线评测使用，绝不发送给模型
        "offline_only": {
            "answer": sample.get("answer", ""),
            "evidence_pages": sample.get("evidence_pages", "[]"),
            "evidence_sources": sample.get("evidence_sources", "[]"),
            "answer_format": sample.get("answer_format", ""),
        },
    }


def summarize(path):
    rows = C.load_jsonl(path)
    last = C.last_by_qid(rows)
    n = len(last)
    ok = sum(1 for r in last.values() if r.get("status") == "ok")
    partial = sum(1 for r in last.values()
                  if r.get("status") == "ok" and r.get("missing_pages"))
    status = {}
    for r in last.values():
        status[r.get("status")] = status.get(r.get("status"), 0) + 1
    tokens = sum((r.get("usage") or {}).get("total_tokens", 0) for r in last.values())
    lat = sum(float(r.get("latency_sec") or 0) for r in last.values())
    return {
        "records_total": len(rows),
        "unique_qids": n,
        "success": ok,
        "partial_evidence_success": partial,
        "status_counts": status,
        "total_tokens": tokens,
        "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in last.values()),
        "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in last.values()),
        "latency_total_sec": round(lat, 2),
        "latency_avg_sec": round(lat / ok, 3) if ok else 0,
        "generated_at": C.now_str(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=C.SAMPLES)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--summary-out", default=SUMMARY_PATH,
                    help="汇总 JSON 输出路径（联调时可指向独立测试目录）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题；0=全部 500")
    ap.add_argument("--start", type=int, default=1, help="起始 qid 序号（1-based，含）")
    ap.add_argument("--end", type=int, default=0, help="结束 qid 序号（1-based，含）；0=到末尾")
    ap.add_argument("--dry-run", action="store_true", help="只检查 PDF/页码，不调用 API、不写结果")
    ap.add_argument("--render-check-limit", type=int, default=0,
                    help="dry-run 时额外真实渲染前 N 题以验证图片可生成")
    ap.add_argument("--retry-failed", action="store_true",
                    help="忽略 --max-attempts，强制重试未成功的题")
    ap.add_argument("--max-attempts", type=int, default=2,
                    help="同一 qid 最多尝试次数（默认 2，防止反复失败持续花钱）")
    args = ap.parse_args()

    samples = C.load_samples(args.input)
    if args.limit:
        samples = samples[:args.limit]
    samples = [s for s in samples
               if s["sample_index"] >= args.start
               and (args.end == 0 or s["sample_index"] <= args.end)]

    existing = C.load_jsonl(args.out)
    last = C.last_by_qid(existing)
    attempts = C.attempts_by_qid(existing)

    if args.dry_run:
        plan = {"ready": 0, "ready_partial_evidence": 0, "pdf_missing": 0,
                "pdf_unreadable": 0, "no_images": 0}
        rendered_check = 0
        for i, s in enumerate(samples, 1):
            do_render = bool(args.render_check_limit) and i <= args.render_check_limit
            p = prepare_input(s, do_render=do_render)
            plan[p["status"]] = plan.get(p["status"], 0) + 1
            if do_render and p["status"] in ("ready", "ready_partial_evidence"):
                rendered_check += 1
            if p["status"] not in ("ready", "ready_partial_evidence"):
                print(f"[不可运行] {s['qid']} {s['doc_id']} status={p['status']} "
                      f"err={p['error'][:80]} requested={p['requested_pages']} "
                      f"num_pages={p['num_pages']}")
        print("\n=== DRY-RUN（未调用 API，未写结果） ===")
        print(f"题目范围: {samples[0]['qid']}..{samples[-1]['qid']}  共 {len(samples)} 题")
        print(f"可运行输入: {plan.get('ready', 0)}，部分页可用: "
              f"{plan.get('ready_partial_evidence', 0)}")
        print(f"PDF 缺失: {plan.get('pdf_missing', 0)}，PDF 损坏: "
              f"{plan.get('pdf_unreadable', 0)}，无可用图片: {plan.get('no_images', 0)}")
        print(f"已测试真实渲染: {rendered_check} 题")
        if last:
            print(f"已有结果文件: {len(existing)} 条记录 / {len(last)} 个 qid")
        return

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[错误] 未检测到 DASHSCOPE_API_KEY，未发出任何请求。")
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120.0, max_retries=1)

    C.ensure_dir(os.path.dirname(args.out))
    if args.summary_out:
        C.ensure_dir(os.path.dirname(args.summary_out))
    n_called = n_ok = n_fail = n_skip = 0
    for s in samples:
        prev = last.get(s["qid"])
        if prev and prev.get("ds_success"):
            n_skip += 1
            continue
        if prev and not args.retry_failed and attempts.get(s["qid"], 0) >= args.max_attempts:
            n_skip += 1
            print(f"[跳过] {s['qid']} 已有 {attempts[s['qid']]} 次失败，"
                  f"达到 --max-attempts={args.max_attempts}")
            continue

        prepared = prepare_input(s)
        attempt = attempts.get(s["qid"], 0) + 1
        t0 = time.time()
        if prepared["status"] not in ("ready", "ready_partial_evidence"):
            result = {"status": prepared["status"], "ds_success": False, "ds_answer": "",
                      "error": prepared["error"], "model_returned": None, "usage": None}
            latency = time.time() - t0
            n_fail += 1
            print(f"[未调用] {s['qid']} {s['doc_id']} status={result['status']} "
                  f"err={result['error'][:80]}")
        else:
            result = call_model(client, s, prepared)
            latency = time.time() - t0
            n_called += 1
            if result["ds_success"]:
                n_ok += 1
            else:
                n_fail += 1

        rec = make_record(s, prepared, result, attempt, latency)
        C.append_jsonl(args.out, rec)
        last[s["qid"]] = rec
        attempts[s["qid"]] = attempt

        shown = (result["ds_answer"] or "").replace("\n", " ")[:50]
        print(f"[{s['qid']}] status={result['status']} imgs={len(prepared['images'])} "
              f"latency={latency:.2f}s answer={shown!r} err={result['error'][:70]}")

    summary = summarize(args.out)
    summary["run_config"] = {
        "model": MODEL_NAME,
        "max_pages": MAX_PAGES, "render_dpi": RENDER_DPI,
        "max_render_dim": MAX_RENDER_DIM,
        "input_policy": "使用 gold evidence_pages；空证据页回退第 1 页（与 Qwen 版一致）；"
                        "不是完整检索系统的端到端结果。",
        "qid_scheme": "sample_index 1-based -> Q0001..Q0500",
    }
    summary["run_config"]["input_file"] = args.input
    summary["run_config"]["output_file"] = args.out
    C.write_json_atomic(args.summary_out, summary)

    print("\n" + "=" * 60)
    print(f"本次调用 {n_called} 题：成功 {n_ok}，失败 {n_fail}，跳过 {n_skip}")
    print(f"累计成功 {summary['success']} / {summary['unique_qids']} 个 qid")
    print(f"累计 token {summary['total_tokens']} | 平均耗时 {summary['latency_avg_sec']}s")
    print(f"结果: {args.out}\n汇总: {args.summary_out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
