# -*- coding: utf-8 -*-
"""
目录路由（500 题版，第二步）：只用 Question + 目录预测证据章节，并计算检索指标。

与 task6_route_and_eval.py 的关系：
  - 复用同一套提示词、JSON 解析、章节校验、指标与标题匹配 baseline；
  - 原 9 题实验的 outputs/task6/ 保持不变，本脚本写入 outputs/final500/run500/toc/。

严格约束：
  - 发送给模型的只有：问题文本 + 目录（section_id / 标题 / 层级 / 页码范围）；
  - 绝不发送 gold evidence_pages、标准答案、evidence_sources 或正文片段；
  - 不可答题（evidence_pages 为空）单独统计，不记为检索失败；
  - 证据页越界 / PDF 与题目不对应的题只保存状态，不进入 Page Recall 分母；
  - 失败（API 报错、非法 JSON、无有效章节）在正式指标中按“未命中”计，而不是静默剔除。

输出：
  outputs/final500/run500/toc/routes.jsonl
  outputs/final500/run500/toc/metrics.json

用法：
  python task6_route_and_eval_500.py --dry-run --limit 50   # 不调用 API，只统计 + 标题 baseline
  python task6_route_and_eval_500.py --limit 50             # 先试 50 题
  python task6_route_and_eval_500.py --limit 0              # 全部 500 题（可断点续跑）
  python task6_route_and_eval_500.py --reuse                # 不调用 API，重算指标
"""
import argparse
import collections
import json
import os
import sys
import time

import final500_common as C
import task6_route_and_eval as base6

TOC_JSON = os.path.join(C.OUT_ROOT, "toc", "toc.json")
ROUTE_DIR = os.path.join(C.OUT_ROOT, "run500", "toc")
ROUTES = os.path.join(ROUTE_DIR, "routes.jsonl")
METRICS = os.path.join(ROUTE_DIR, "metrics.json")
BASELINE_PREVIEW = os.path.join(C.OUT_ROOT, "toc", "routes", "deepseek_v41", "baseline_preview.json")

FINAL_STATUS = {"ok", "invalid_json", "no_valid_section"}
ZERO = {"section_hit": 0, "page_recall": 0, "full_evidence_recall": 0,
        "covered_pages": 0, "candidate_pages": 0, "candidate_ratio": 0}


def load_tocs():
    doc = C.read_json(TOC_JSON, default={"docs": []}) or {"docs": []}
    return {d["doc_id"]: d for d in doc.get("docs", [])
            if d.get("build_status") == "ok" and d.get("sections")}


def classify(sample, tocs):
    """返回 (status, eligible, reason)。

    不可回答的判定：evidence_pages 为空，或标准答案本身是 Not answerable。
    个别样本有空证据页但存在真实答案（数据标注不一致），单独标记为 missing_gold。
    """
    gold = C.parse_pages(sample.get("evidence_pages", "[]"))
    is_not_answerable = str(sample.get("answer", "")) == "Not answerable"
    doc = tocs.get(sample["doc_id"])
    if is_not_answerable:
        return "unanswerable_skipped", False, "答案为 Not answerable（不计检索失败）"
    if not gold:
        return "missing_gold_pages_skipped", False, "无 gold 证据页，无法计算 Page Recall"
    if doc is None:
        return "no_toc_skipped", False, "PDF 缺失或目录构建失败"
    n = doc.get("num_pages")
    if n and any(p < 1 or p > n for p in gold):
        return "invalid_gold_pages_skipped", False, "gold 证据页超出 PDF 实际页数"
    return "pending", True, ""


def record_base(sample, doc, gold, status, eligible, reason, record_gold=False):
    rec = {
        "qid": sample["qid"],
        "sample_index": sample["sample_index"],
        "task": "toc_routing_question_only",
        "model_requested": C.MODEL_DS,
        "doc_id": sample["doc_id"],
        "doc_type": sample.get("doc_type", ""),
        "question": sample["question"],
        "status": status,
        "success": False,
        "eval_eligible": eligible,
        "eval_reason": reason,
        "toc_source": (doc or {}).get("toc_source"),
        "pdf_num_pages": (doc or {}).get("num_pages"),
        "prompt_system": "",
        "prompt_user": "",
        "raw_response": "",
        "sections": [],
        "invalid_section_ids": [],
        "usage": None,
        "latency_sec": 0,
        "attempt": 1,
        "run_at": C.now_str(),
    }
    if record_gold:
        rec["offline_only"] = {
            "gold_pages": gold,
            "answer": sample.get("answer", ""),
            "evidence_sources": sample.get("evidence_sources", ""),
        }
    return rec


def title_baselines(question, doc):
    secs = doc["sections"]
    b1 = base6.baseline_rank(question, secs)[:1]
    b3 = base6.baseline_rank(question, secs)[:3]
    return ([{"section_id": s["section_id"], "title": s["title"],
              "start_page": s["start_page"], "end_page": s["end_page"]} for s in b1],
            [{"section_id": s["section_id"], "title": s["title"], "start_page": s["start_page"],
              "end_page": s["end_page"]} for s in b3])


def call_api(client, sample, doc, attempt, record_gold=False):
    gold = C.parse_pages(sample.get("evidence_pages", "[]"))
    rec = record_base(sample, doc, gold, "pending", True, "", record_gold)
    system, user = base6.build_prompt(sample["question"], doc)
    rec["prompt_system"] = system
    rec["prompt_user"] = user
    rec["attempt"] = attempt
    sections_by_id = {s["section_id"]: s for s in doc["sections"]}
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=C.MODEL_DS,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0,
            max_tokens=400,
            extra_body={"enable_thinking": False},
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        rec.update({
            "success": True,
            "model_returned": getattr(resp, "model", None),
            "raw_response": text,
            "usage": {"prompt_tokens": usage.prompt_tokens,
                      "completion_tokens": usage.completion_tokens,
                      "total_tokens": usage.total_tokens} if usage else None,
            "error": "",
        })
    except Exception as e:
        text = ""
        rec.update({"success": False, "model_returned": None, "raw_response": "",
                    "usage": None, "error": f"{type(e).__name__}: {str(e)[:400]}"})
    rec["latency_sec"] = round(time.time() - t0, 3)

    obj = base6.parse_model_json(text) if rec["success"] else None
    valid, invalid = base6.validate_sections(obj, sections_by_id) if obj else ([], [])
    if not rec["success"]:
        rec["status"] = "api_error"
    elif obj is None:
        rec["status"] = "invalid_json"
    elif not valid:
        rec["status"] = "no_valid_section"
    else:
        rec["status"] = "ok"
    rec["sections"] = valid
    rec["invalid_section_ids"] = invalid
    rec["baseline_top1"], rec["baseline_top3"] = title_baselines(sample["question"], doc)
    return rec


def evaluate(records, tocs, expected_eligible, smap=None, record_gold=False):
    """smap：qid -> 数据集样本。公开副本已移除 offline_only，gold 页从样本文件读取。"""
    last = C.last_by_qid(records)

    def status_of(r):
        return r.get("status")

    # 只保留“可评测”的题：有有效 gold 页 + 目录可用
    eligible = [r for r in last.values()
                if r.get("eval_eligible") and r.get("doc_id") in tocs]

    per_q = []
    for r in sorted(eligible, key=lambda x: x["sample_index"]):
        doc = tocs[r["doc_id"]]
        n = doc["num_pages"]
        gold = (r.get("offline_only") or {}).get("gold_pages")
        if gold is None and smap is not None and r["qid"] in smap:
            gold = C.parse_pages(smap[r["qid"]].get("evidence_pages", "[]"))
        if gold is None:
            raise SystemExit(
                "[错误] %s 缺少 gold 证据页，且未提供本地评测集，无法计算路由指标。\n"
                "请按 README 第 9 节获取 sampled_500_for_baseline.jsonl 后重试。" % r["qid"]
            )
        ok = r.get("status") == "ok"
        llm1 = base6.metrics(r.get("sections", [])[:1], gold, n) if ok else dict(ZERO)
        llm3 = base6.metrics(r.get("sections", [])[:3], gold, n) if ok else dict(ZERO)

        base_all = r.get("baseline_top3") or []
        b1 = base6.metrics(base_all[:1], gold, n) if base_all else dict(ZERO)
        b3 = base6.metrics(base_all[:3], gold, n) if base_all else dict(ZERO)

        # 同页面预算对照：按标题得分排序，逐章节加入直到候选页数达到 LLM top3
        budget = llm3["candidate_pages"]
        all_ranked = [{"section_id": s["section_id"], "title": s["title"],
                       "start_page": s["start_page"], "end_page": s["end_page"]}
                      for s in base6.baseline_rank_all(r["question"], doc["sections"])]
        chosen, acc = [], set()
        for s in all_ranked:
            if len(acc) >= budget:
                break
            chosen.append(s)
            acc |= set(range(s["start_page"], s["end_page"] + 1))
        bm = base6.metrics(chosen, gold, n) if chosen else dict(ZERO)

        per_q.append({
            "qid": r["qid"], "doc_id": r["doc_id"], "question": r["question"],
            "status": r["status"], "success": bool(r.get("success")),
            "gold_pages": gold, "num_doc_pages": n,
            "llm_sections": r.get("sections", []),
            "llm_invalid_section_ids": r.get("invalid_section_ids", []),
            "llm_top1": llm1, "llm_top3": llm3,
            "baseline_top1": b1, "baseline_top3": b3,
            "baseline_top3_sections": base_all[:3],
            "baseline_budget_matched": {**bm, "budget_target": budget,
                                        "used_sections": len(chosen)},
            "latency_sec": r.get("latency_sec", 0), "usage": r.get("usage"),
            "multi_page_evidence": len(gold) > 1,
            "failure_as_miss": not ok,
        })

    ok_rows = [p for p in per_q if p["status"] == "ok"]

    def avg(rows, key, outer):
        vals = [x[outer][key] for x in rows]
        return round(sum(vals) / len(vals), 4) if vals else 0

    def tot(rows, key, outer):
        return sum(x[outer][key] for x in rows)

    def block(rows):
        return {
            "n": len(rows),
            "section_hit@1": tot(rows, "section_hit", "llm_top1"),
            "section_hit@3": tot(rows, "section_hit", "llm_top3"),
            "page_recall@1_macro": avg(rows, "page_recall", "llm_top1"),
            "page_recall@3_macro": avg(rows, "page_recall", "llm_top3"),
            "page_recall@1_micro": round(
                sum(x["llm_top1"]["covered_pages"] for x in rows) /
                sum(len(x["gold_pages"]) for x in rows), 4)
            if sum(len(x["gold_pages"]) for x in rows) else 0,
            "page_recall@3_micro": round(
                sum(x["llm_top3"]["covered_pages"] for x in rows) /
                sum(len(x["gold_pages"]) for x in rows), 4)
            if sum(len(x["gold_pages"]) for x in rows) else 0,
            "full_evidence_recall@1": tot(rows, "full_evidence_recall", "llm_top1"),
            "full_evidence_recall@3": tot(rows, "full_evidence_recall", "llm_top3"),
            "avg_candidate_pages@1": avg(rows, "candidate_pages", "llm_top1"),
            "avg_candidate_pages@3": avg(rows, "candidate_pages", "llm_top3"),
        }

    all_status = collections.Counter(r.get("status") for r in last.values())
    summary = {
        "expected_eligible": expected_eligible,
        "records_present": len(last),
        "eligible_in_records": len(per_q),
        "evaluated_ok": len(ok_rows),
        "status_counts_records": dict(all_status),
        "unanswerable_excluded_from_recall": all_status.get("unanswerable_skipped", 0),
        "missing_gold_pages_excluded": all_status.get("missing_gold_pages_skipped", 0),
        "invalid_gold_excluded_from_recall": all_status.get("invalid_gold_pages_skipped", 0),
        "no_toc_excluded": all_status.get("no_toc_skipped", 0),
        "api_error_counted_as_miss": sum(1 for p in per_q if p["status"] == "api_error"),
        "invalid_json_counted_as_miss": sum(1 for p in per_q if p["status"] == "invalid_json"),
        "no_valid_section_counted_as_miss": sum(
            1 for p in per_q if p["status"] == "no_valid_section"),
        "invalid_section_id_total": sum(len(p["llm_invalid_section_ids"]) for p in per_q),
        "token_usage_total": sum((p.get("usage") or {}).get("total_tokens", 0) for p in per_q),
        "latency_total_sec": round(sum(p["latency_sec"] for p in per_q), 2),
        "latency_avg_sec": round(sum(p["latency_sec"] for p in per_q) / len(per_q), 3)
        if per_q else 0,
        # 正式口径：可评测题全算，失败按 0 计
        "llm_all_eligible": block(per_q),
        # 参考口径：只统计成功返回有效章节的题（会偏乐观，不作为正式结论）
        "llm_ok_only_reference": block(ok_rows),
        "baseline_title_match_all_eligible": {
            "n": len(per_q),
            "section_hit@3": sum(1 for p in per_q
                                 if p["baseline_top3_sections"] and
                                 any(s["start_page"] <= gp <= s["end_page"]
                                     for s in p["baseline_top3_sections"]
                                     for gp in p["gold_pages"])) if per_q else 0,
            "page_recall@3_macro": avg(per_q, "page_recall", "baseline_top3"),
            "avg_candidate_pages@3": avg(per_q, "candidate_pages", "baseline_top3"),
        },
        "baseline_budget_matched_all_eligible": {
            "n": len(per_q),
            "section_hit": sum(1 for p in per_q
                               if p["baseline_budget_matched"]["section_hit"]),
            "page_recall_macro": avg(per_q, "page_recall", "baseline_budget_matched"),
            "full_evidence_recall": sum(
                1 for p in per_q if p["baseline_budget_matched"]["full_evidence_recall"]),
            "avg_candidate_pages": avg(per_q, "candidate_pages", "baseline_budget_matched"),
        },
        "note": ("Page Recall 只在“不可答=否 + 目录可用 + gold 页码在 PDF 范围内”的题上计算；"
                 "API 报错/非法 JSON/无有效章节在该口径下记 0，不静默剔除。"),
    }
    if not record_gold:
        for p in per_q:
            p.pop("gold_pages", None)
    return summary, per_q


def write_baseline_preview(tocs):
    """不调用 API，计算所有可评测题的标题匹配 baseline。"""
    rows = []
    for s in C.load_samples():
        st, elig, reason = classify(s, tocs)
        if not elig:
            continue
        doc = tocs[s["doc_id"]]
        gold = C.parse_pages(s.get("evidence_pages", "[]"))
        n = doc["num_pages"]
        b1, b3 = title_baselines(s["question"], doc)
        rows.append({
            "qid": s["qid"], "doc_id": s["doc_id"],
            "baseline_top1": base6.metrics(b1, gold, n),
            "baseline_top3": base6.metrics(b3, gold, n),
            "candidate_pages_top3": base6.metrics(b3, gold, n)["candidate_pages"],
        })
    def avg(key, outer):
        vals = [r[outer][key] for r in rows]
        return round(sum(vals) / len(vals), 4) if vals else 0
    out = {
        "generated_at": C.now_str(),
        "n_eligible": len(rows),
        "definition": "标题词重合度排序（无 LLM），只作为公平对照的 baseline。",
        "baseline_title_match": {
            "section_hit@1": sum(r["baseline_top1"]["section_hit"] for r in rows),
            "section_hit@3": sum(r["baseline_top3"]["section_hit"] for r in rows),
            "page_recall@1_macro": avg("page_recall", "baseline_top1"),
            "page_recall@3_macro": avg("page_recall", "baseline_top3"),
            "full_evidence_recall@1": sum(
                r["baseline_top1"]["full_evidence_recall"] for r in rows),
            "full_evidence_recall@3": sum(
                r["baseline_top3"]["full_evidence_recall"] for r in rows),
            "avg_candidate_pages@3": avg("candidate_pages", "baseline_top3"),
        },
        "per_question": rows,
    }
    C.write_json_atomic(BASELINE_PREVIEW, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题；0=全部 500")
    ap.add_argument("--qids", default="", help="只跑指定 qid（逗号分隔）")
    ap.add_argument("--out", default=ROUTES)
    ap.add_argument("--eval-out", default=METRICS)
    ap.add_argument("--reuse", action="store_true", help="不调用 API，复用已有预测重算指标")
    ap.add_argument("--dry-run", action="store_true",
                    help="不调用 API：只统计可评测题并计算标题 baseline")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--record-gold", action="store_true",
                    help="内部使用：把 gold 写入预测/指标；公开仓库默认关闭")
    args = ap.parse_args()

    try:
        samples = C.load_samples()
    except SystemExit as e:
        if args.reuse:
            records = C.load_jsonl(args.out)
            last = C.last_by_qid(records)
            st = collections.Counter(r.get("status") for r in last.values())
            print("[复核] 未找到本地评测集，仅复核不依赖 gold 的执行统计。")
            print(f"[复核] 记录={len(records)} 唯一qid={len(last)} "
                  f"token总={sum((r.get('usage') or {}).get('total_tokens', 0) for r in last.values())} "
                  f"平均延迟="
                  f"{round(sum(r.get('latency_sec') or 0 for r in last.values()) / len(last), 3) if last else 0}s")
            print(f"[复核] 状态分布={dict(sorted(st.items()))}")
            print("[跳过] 缺少 gold 证据页，无法计算 Hit@K / Page Recall / FullEvidence。")
            return
        raise SystemExit(str(e))
    full_samples = samples
    # 先按 qid 过滤，再按 --limit 截断：避免 --limit 50 --qids <靠后的题> 时把目标题截掉。
    if args.qids:
        wanted = {x.strip().upper() for x in args.qids.split(",") if x.strip()}
        samples = [s for s in samples if s["qid"].upper() in wanted]
    if args.limit:
        samples = samples[:args.limit]

    tocs = load_tocs()

    if args.dry_run:
        st = collections.Counter()
        for s in samples:
            status, elig, _ = classify(s, tocs)
            st[status] += 1
        preview = write_baseline_preview(tocs)
        print("=== ROUTING DRY-RUN（未调用 API） ===")
        print(f"题目 {len(samples)}，可评测 {st.get('pending', 0)}；"
              f"不可答 {st.get('unanswerable_skipped', 0)}；"
              f"无目录 {st.get('no_toc_skipped', 0)}；"
              f"gold 越界 {st.get('invalid_gold_pages_skipped', 0)}")
        print(f"标题 baseline（全部可评测题 n={preview['n_eligible']}）："
              f"Hit@1={preview['baseline_title_match']['section_hit@1']}, "
              f"Hit@3={preview['baseline_title_match']['section_hit@3']}, "
              f"PageRecall@3={preview['baseline_title_match']['page_recall@3_macro']}")
        print(f"baseline 详情: {BASELINE_PREVIEW}")
        return

    if args.reuse:
        records = C.load_jsonl(args.out)
        print(f"[复用] {len(records)} 条预测，未调用 API")
    else:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            print("[错误] 未检测到 DASHSCOPE_API_KEY，未发出任何请求。")
            sys.exit(1)
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=C.BASE_URL, timeout=120.0, max_retries=1)
        records = C.load_jsonl(args.out)
        last = C.last_by_qid(records)
        attempts = C.attempts_by_qid(records)
        n_called = 0
        for s in samples:
            status, elig, reason = classify(s, tocs)
            prev = last.get(s["qid"])
            if prev:
                if prev.get("status") in FINAL_STATUS:
                    continue
                if not args.retry_failed and attempts.get(s["qid"], 0) >= args.max_attempts:
                    print(f"[跳过] {s['qid']} 已有 {attempts[s['qid']]} 次失败")
                    continue
            if not elig:
                if prev is None:
                    rec = record_base(s, tocs.get(s["doc_id"]),
                                      C.parse_pages(s.get("evidence_pages", "[]")),
                                      status, False, reason, args.record_gold)
                    C.append_jsonl(args.out, rec)
                    records.append(rec)
                    last[s["qid"]] = rec
                continue
            rec = call_api(client, s, tocs[s["doc_id"]],
                           attempts.get(s["qid"], 0) + 1, args.record_gold)
            C.append_jsonl(args.out, rec)
            records.append(rec)
            last[s["qid"]] = rec
            attempts[s["qid"]] = attempts.get(s["qid"], 0) + 1
            n_called += 1
            print(f"[{s['qid']}] status={rec['status']} "
                  f"sections={len(rec['sections'])} latency={rec['latency_sec']}s "
                  f"err={rec['error'][:60]}")
        print(f"本次实际调用 {n_called} 题")

    expected_eligible = sum(1 for s in full_samples if classify(s, tocs)[1])
    summary, per_q = evaluate(records, tocs, expected_eligible,
                              {s["qid"]: s for s in full_samples}, args.record_gold)
    eval_doc = {
        "generated_at": C.now_str(),
        "model_requested": C.MODEL_DS,
        "input_policy": "模型仅收到 Question + 目录（section_id/标题/层级/页码范围）；"
                        "未发送答案、gold 页、evidence_sources 或正文。",
        "page_definition": "PDF 实际页码，1-based",
        "gold_included": bool(args.record_gold),
        "summary": summary,
        "per_question": per_q,
    }
    C.write_json_atomic(args.eval_out, eval_doc)
    print("\n" + "=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n预测: {args.out}\n指标: {args.eval_out}")


if __name__ == "__main__":
    main()
