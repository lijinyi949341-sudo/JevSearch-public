# -*- coding: utf-8 -*-
"""
目录路由（第二步）：让 DeepSeek 只根据「Question + 结构化目录」预测证据所在章节，
并用 gold evidence_pages 计算检索指标；同时做“目录标题词匹配”的无 LLM baseline。

严格约束：
  - 发给模型的输入只有：问题文本 + 目录（章节 ID、标题、层级、页码范围）。
    不发送答案、gold evidence_pages、evidence_sources、Jev 预测或任何正文片段。
  - 模型返回的 section_id 必须存在于真实目录，否则记为“无效章节”并丢弃。
  - API Key 只从环境变量 DASHSCOPE_API_KEY 读取，不打印、不落盘。
  - 所有页码为 PDF 实际页码（1-based）。

用法：
  # 先跑 1 题确认流程
  py -3.12 task6_route_and_eval.py --limit 1 --out outputs/task6/predictions_smoke1.jsonl \
      --eval-out outputs/task6/evaluation_smoke1.json
  # 确认后跑全部 9 题
  py -3.12 task6_route_and_eval.py --limit 0
"""
import argparse
import json
import os
import re
import sys
import time

from openai import OpenAI

MODEL_NAME = "deepseek-v4.1-flash-1017"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
INPUT = "sampled_500_for_baseline.jsonl"
TOC_JSON = os.path.join("outputs", "task6", "toc.json")
OUT_DIR = os.path.join("outputs", "task6")

# 第一轮最小实验的 9 道真实问题（来自 MMLongBench-Doc，均有非空 gold evidence_pages）
PICKS = [
    ("STEPBACK.pdf", "How much lower is the accuracy of GPT-4 on SituatedQA"),
    ("STEPBACK.pdf", "What score did PaLM-2L + Step-Back achieve in MMLU Physics?"),
    ("2023.findings-emnlp.248.pdf", "For the dataset that uses Prover9 as the symbolic solver"),
    ("2023.findings-emnlp.248.pdf", "How many prompt examples are given in the Appendix?"),
    ("fdac8d1e9ef56519371df7e6532df27d.pdf", "What does the map in the report shows?"),
    ("fdac8d1e9ef56519371df7e6532df27d.pdf", "How many organisations does this report introduce in detail"),
    ("fdac8d1e9ef56519371df7e6532df27d.pdf", "What year is the report for?"),
    ("0b85477387a9d0cc33fca0f4becaa0e5.pdf", "Who is the commanding officer in the first figure on the second page?"),
    ("0b85477387a9d0cc33fca0f4becaa0e5.pdf", "With whom did the NAMRU-3 team visit Monrovia, Liberia"),
]

STOPWORDS = set("""what which who whom whose when where why how is are was were be been
the a an of in on at to for from by with and or but this that these those it its
does do did has have had can could should would will may might must about into over
under between among during after before than then there their they you your we our
following given based according document page pages number many much more most least
""".split())


def load_questions():
    if not os.path.exists(INPUT):
        raise SystemExit(
            "[错误] 未找到本地评测集 %s，无法定位 9 道题及其 gold 证据页。\n"
            "该文件不随仓库发布；请按 README 第 9 节获取 MMLongBench-Doc 数据，"
            "并运行 prepare_data.py。" % INPUT
        )
    rows = [json.loads(l) for l in open(INPUT, encoding="utf-8") if l.strip()]
    picked = []
    for i, (doc_id, prefix) in enumerate(PICKS, 1):
        hit = None
        for r in rows:
            if r["doc_id"] == doc_id and r["question"].strip().startswith(prefix):
                hit = r
                break
        if hit is None:
            raise SystemExit(f"[错误] 在数据集中找不到题目: {doc_id} | {prefix}")
        try:
            gold = [int(x) for x in eval(hit["evidence_pages"])]
        except Exception:
            gold = []
        picked.append({
            "qid": f"Q{i:02d}",
            "doc_id": doc_id,
            "question": hit["question"],
            "gold_pages": gold,
            "evidence_sources": hit.get("evidence_sources", ""),
            "answer": hit.get("answer", ""),
            "answer_format": hit.get("answer_format", ""),
        })
    return picked, rows


def build_prompt(question, toc_doc):
    lines = [f"{s['section_id']} | 层级{s['level']} | 第{s['start_page']}-{s['end_page']}页 | {s['title']}"
             for s in toc_doc["sections"]]
    user = (
        "你是文档检索助手。下面给出一个问题，以及该文档的章节目录（页码为 PDF 实际页码）。\n"
        "请判断答案最可能出现在哪 1-3 个章节中。只能从目录中选择，不得新增章节或改动页码。\n"
        "优先给出最可能的一个章节；如有多页证据或不确定，再补充后续章节。\n\n"
        f"【问题】\n{question}\n\n"
        f"【文档目录】（共 {len(toc_doc['sections'])} 个章节，文档共 {toc_doc['num_pages']} 页）\n"
        + "\n".join(lines) +
        "\n\n【输出格式】只输出一个 JSON 对象，不要解释，不要输出其他文字：\n"
        '{"sections":[{"section_id":"<ID>","rank":1},{"section_id":"<ID>","rank":2}]}\n'
        "其中 <ID> 必须替换成上面目录中真实存在的 section_id（形如 S03），rank 从 1 开始，最多 3 个。"
    )
    system = ("你是一个严谨的文档章节定位器。你只能依据提供的目录做判断，"
              "不得使用外部知识，不得编造目录中不存在的章节。")
    return system, user


def parse_model_json(text):
    """从模型输出中提取 JSON（容忍 ```json 包裹）。"""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def validate_sections(obj, sections_by_id):
    raw_ids = []
    if isinstance(obj, dict) and isinstance(obj.get("sections"), list):
        for item in obj["sections"]:
            if isinstance(item, dict) and item.get("section_id"):
                raw_ids.append(str(item["section_id"]).strip())
    elif isinstance(obj, dict) and obj.get("section_id"):
        raw_ids = [str(obj["section_id"]).strip()]

    valid, invalid = [], []
    for sid in raw_ids:
        if sid in sections_by_id and sid not in [v["section_id"] for v in valid]:
            s = sections_by_id[sid]
            valid.append({"section_id": sid, "title": s["title"], "level": s["level"],
                          "start_page": s["start_page"], "end_page": s["end_page"]})
        elif sid not in sections_by_id and sid not in invalid:
            invalid.append(sid)
        if len(valid) == 3:
            break
    return valid, invalid


def pages_of(sections):
    ps = set()
    for s in sections:
        ps.update(range(s["start_page"], s["end_page"] + 1))
    return ps


def metrics(sections, gold, num_pages):
    pred = pages_of(sections)
    covered = len(pred & set(gold))
    recall = covered / len(gold) if gold else 0.0
    hit = any(s["start_page"] <= p <= s["end_page"] for s in sections for p in gold)
    return {
        "section_hit": int(bool(hit)),
        "page_recall": round(recall, 4),
        "full_evidence_recall": int(bool(gold) and covered == len(gold)),
        "covered_pages": covered,
        "candidate_pages": len(pred),
        "candidate_ratio": round(len(pred) / num_pages, 4),
    }


def title_score(question, title):
    q = {t for t in re.findall(r"[a-z0-9]+", question.lower()) if t not in STOPWORDS}
    s = {t for t in re.findall(r"[a-z0-9]+", title.lower()) if t not in STOPWORDS}
    return len(q & s)


def baseline_rank(question, sections):
    """按标题词重合度排序，仅返回有词重合的章节（top-k 用这一列）。"""
    scored = [(title_score(question, s["title"]), i, s) for i, s in enumerate(sections)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for sc, _, s in scored if sc > 0]


def baseline_rank_all(question, sections):
    """全章节排序（含 0 分），用于“同等页面预算”对照。"""
    scored = [(title_score(question, s["title"]), i, s) for i, s in enumerate(sections)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [s for _, _, s in scored]


def run(args, questions, tocs):
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("[错误] 未检测到环境变量 DASHSCOPE_API_KEY。请先在当前终端设置后重试。")
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120.0, max_retries=1)

    os.makedirs(OUT_DIR, exist_ok=True)
    records = []
    with open(args.out, "w", encoding="utf-8") as f_out:
        for q in questions:
            doc = tocs[q["doc_id"]]
            sections_by_id = {s["section_id"]: s for s in doc["sections"]}
            system, user = build_prompt(q["question"], doc)

            rec = {**q, "model_requested": MODEL_NAME, "toc_source": doc["toc_source"],
                   "prompt_system": system, "prompt_user": user}
            if not getattr(args, "record_gold", False):
                rec.pop("gold_pages", None)
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
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
                rec.update({"success": False, "model_returned": None,
                            "raw_response": "", "usage": None,
                            "error": f"{type(e).__name__}: {str(e)[:400]}"})
            rec["latency_sec"] = round(time.time() - t0, 3)

            obj = parse_model_json(text) if rec["success"] else None
            valid, invalid = validate_sections(obj, sections_by_id) if obj else ([], [])
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
            rec["baseline_sections"] = [
                {"section_id": s["section_id"], "title": s["title"],
                 "start_page": s["start_page"], "end_page": s["end_page"]}
                for s in baseline_rank(q["question"], doc["sections"])[:3]
            ]
            records.append(rec)
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            top = ", ".join(f"{s['section_id']}({s['title'][:36]})" for s in valid) or "-"
            print(f"[{q['qid']}] {q['doc_id']} success={rec['success']} status={rec['status']} "
                  f"pred={top} latency={rec['latency_sec']}s err={rec['error'][:80]}")
    return records


def evaluate(records, tocs, gold_map=None, record_gold=False):
    per_q = []
    for r in records:
        doc = tocs[r["doc_id"]]
        n = doc["num_pages"]
        gold = r.get("gold_pages")
        if gold is None and gold_map is not None:
            gold = gold_map.get(r["qid"])
        if gold is None:
            raise SystemExit(
                "[错误] %s 缺少 gold_pages，且未提供本地评测集，无法计算路由指标。"
                "请恢复 sampled_500_for_baseline.jsonl 或使用 --reuse 前先补齐 gold。" % r["qid"]
            )
        llm1 = metrics(r["sections"][:1], gold, n) if r["sections"] else \
            {"section_hit": 0, "page_recall": 0, "full_evidence_recall": 0,
             "covered_pages": 0, "candidate_pages": 0, "candidate_ratio": 0}
        llm3 = metrics(r["sections"][:3], gold, n) if r["sections"] else llm1

        base_all = r["baseline_sections"]
        b1 = metrics(base_all[:1], gold, n) if base_all else \
            {"section_hit": 0, "page_recall": 0, "full_evidence_recall": 0,
             "covered_pages": 0, "candidate_pages": 0, "candidate_ratio": 0}
        b3 = metrics(base_all[:3], gold, n) if base_all else b1

        # 预算对齐 baseline：按标题得分（含 0 分章节）依次加入，直到候选页数达到 LLM top3
        budget = llm3["candidate_pages"]
        all_ranked = [{"section_id": s["section_id"], "title": s["title"],
                       "start_page": s["start_page"], "end_page": s["end_page"]}
                      for s in baseline_rank_all(r["question"], doc["sections"])]
        chosen, acc = [], set()
        for s in all_ranked:
            if len(acc) >= budget:
                break
            chosen.append(s)
            acc |= set(range(s["start_page"], s["end_page"] + 1))
        bm = metrics(chosen, gold, n) if chosen else b3

        item = {
            "qid": r["qid"], "doc_id": r["doc_id"], "question": r["question"],
            "num_doc_pages": n,
            "status": r["status"], "success": r["success"],
            "llm_sections": r["sections"], "llm_invalid_section_ids": r["invalid_section_ids"],
            "llm_top1": llm1, "llm_top3": llm3,
            "baseline_sections": base_all, "baseline_top1": b1, "baseline_top3": b3,
            "baseline_budget_matched": {**bm, "budget_target": budget,
                                        "used_sections": len(chosen),
                                        "sections": chosen},
            "latency_sec": r["latency_sec"], "usage": r["usage"],
            "multi_page_evidence": len(gold) > 1,
            "note": "",
        }
        if record_gold:
            item["gold_pages"] = gold
        per_q.append(item)

    ok = [p for p in per_q if p["status"] == "ok"]
    def avg(rows, key, outer):
        vals = [x[outer][key] for x in rows]
        return round(sum(vals) / len(vals), 4) if vals else 0
    def tot(rows, key, outer):
        return sum(x[outer][key] for x in rows)

    n = len(per_q)
    summary = {
        "n_questions": n,
        "api_success": sum(1 for x in per_q if x["success"]),
        "ok_with_valid_section": len(ok),
        "api_error": sum(1 for x in per_q if x["status"] == "api_error"),
        "invalid_json": sum(1 for x in per_q if x["status"] == "invalid_json"),
        "no_valid_section": sum(1 for x in per_q if x["status"] == "no_valid_section"),
        "invalid_section_id_total": sum(len(x["llm_invalid_section_ids"]) for x in per_q),
        "token_usage_total": sum((x["usage"] or {}).get("total_tokens", 0) for x in per_q),
        "token_usage_prompt": sum((x["usage"] or {}).get("prompt_tokens", 0) for x in per_q),
        "token_usage_completion": sum((x["usage"] or {}).get("completion_tokens", 0) for x in per_q),
        "latency_total_sec": round(sum(x["latency_sec"] for x in per_q), 2),
        "latency_avg_sec": round(sum(x["latency_sec"] for x in per_q) / n, 3) if n else 0,
        "llm": {
            "section_hit@1": tot(ok, "section_hit", "llm_top1"),
            "section_hit@3": tot(ok, "section_hit", "llm_top3"),
            "page_recall@1": avg(ok, "page_recall", "llm_top1"),
            "page_recall@3": avg(ok, "page_recall", "llm_top3"),
            "full_evidence_recall@1": tot(ok, "full_evidence_recall", "llm_top1"),
            "full_evidence_recall@3": tot(ok, "full_evidence_recall", "llm_top3"),
            "avg_candidate_pages@1": avg(ok, "candidate_pages", "llm_top1"),
            "avg_candidate_pages@3": avg(ok, "candidate_pages", "llm_top3"),
            "avg_candidate_ratio@1": avg(ok, "candidate_ratio", "llm_top1"),
            "avg_candidate_ratio@3": avg(ok, "candidate_ratio", "llm_top3"),
        },
        "baseline_title_match": {
            "section_hit@1": tot(ok, "section_hit", "baseline_top1"),
            "section_hit@3": tot(ok, "section_hit", "baseline_top3"),
            "page_recall@1": avg(ok, "page_recall", "baseline_top1"),
            "page_recall@3": avg(ok, "page_recall", "baseline_top3"),
            "full_evidence_recall@1": tot(ok, "full_evidence_recall", "baseline_top1"),
            "full_evidence_recall@3": tot(ok, "full_evidence_recall", "baseline_top3"),
            "avg_candidate_pages@1": avg(ok, "candidate_pages", "baseline_top1"),
            "avg_candidate_pages@3": avg(ok, "candidate_pages", "baseline_top3"),
            "avg_candidate_ratio@1": avg(ok, "candidate_ratio", "baseline_top1"),
            "avg_candidate_ratio@3": avg(ok, "candidate_ratio", "baseline_top3"),
        },
        "baseline_budget_matched": {
            "section_hit": tot(ok, "section_hit", "baseline_budget_matched"),
            "page_recall": avg(ok, "page_recall", "baseline_budget_matched"),
            "full_evidence_recall": tot(ok, "full_evidence_recall", "baseline_budget_matched"),
            "avg_candidate_pages": avg(ok, "candidate_pages", "baseline_budget_matched"),
            "avg_sections_used": avg(ok, "used_sections", "baseline_budget_matched"),
        },
    }
    return summary, per_q


def inventory(rows, tocs, record_gold=False):
    """记录这 4 份文档里被排除/未使用的题目，避免把不可答题当作普通题计算。"""
    out = []
    for r in rows:
        if r["doc_id"] not in tocs:
            continue
        try:
            gold = [int(x) for x in eval(r["evidence_pages"])]
        except Exception:
            gold = []
        used = any(r["question"].strip().startswith(p) and r["doc_id"] == d for d, p in PICKS)
        n = tocs[r["doc_id"]]["num_pages"]
        if used:
            continue
        if not gold:
            reason = "unanswerable / 数据集 evidence_pages 为空，不纳入检索指标"
        elif any(p > n for p in gold):
            reason = "evidence_pages 超出 PDF 实际页数（doc_id 与 PDF 不对应），不纳入"
        else:
            reason = "可回答问题，但未进入9 题小样本实验"
        item = {"doc_id": r["doc_id"], "question": r["question"],
                "exclude_reason": reason}
        if record_gold:
            item["evidence_pages"] = r["evidence_pages"]
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0=全部 9 题")
    parser.add_argument("--out", default=os.path.join(OUT_DIR, "predictions.jsonl"))
    parser.add_argument("--eval-out", default=os.path.join(OUT_DIR, "evaluation.json"))
    parser.add_argument("--reuse", action="store_true",
                        help="不调用 API，直接复用 --out 中已保存的预测重新计算指标")
    parser.add_argument("--record-gold", action="store_true",
                        help="内部使用：把 gold_pages 写入评价文件；公开仓库默认关闭")
    args = parser.parse_args()

    if args.reuse and not os.path.exists(INPUT):
        records = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip()]
        if not records:
            raise SystemExit("[错误] %s 为空，无法复核执行情况。" % args.out)
        qids = [r.get("qid") for r in records]
        print("[复核] 未找到本地评测集 %s，仅复核不依赖 gold 的执行统计。" % INPUT)
        print("[复核] 记录=%d 唯一qid=%d 成功=%d 失败=%d token总=%d 平均延迟=%.3fs"
              % (len(records), len(set(qids)), sum(1 for r in records if r.get("success")),
                 sum(1 for r in records if not r.get("success")),
                 sum((r.get("usage") or {}).get("total_tokens", 0) for r in records),
                 sum(r.get("latency_sec") or 0 for r in records) / len(records)))
        print("[跳过] 缺少 gold 证据页，无法计算 Hit@K / Page Recall / FullEvidence。")
        return

    questions, all_rows = load_questions()
    with open(TOC_JSON, encoding="utf-8") as f:
        tocs = {d["doc_id"]: d for d in json.load(f)["docs"]}
    missing = [q["doc_id"] for q in questions if q["doc_id"] not in tocs]
    if missing:
        raise SystemExit(f"[错误] 缺少目录: {missing}")

    used = questions[:args.limit] if args.limit else questions
    if args.reuse:
        records = [json.loads(l) for l in open(args.out, encoding="utf-8") if l.strip()]
        print(f"[复用] 读取已保存预测 {len(records)} 条，不调用 API")
    else:
        print(f"开始调用 DeepSeek：model={MODEL_NAME}，本次 {len(used)} 题（共 {len(questions)} 题）")
        records = run(args, used, tocs)

    gold_map = {q["qid"]: q["gold_pages"] for q in questions}
    for r in records:
        if r.get("gold_pages") is None:
            if r["qid"] not in gold_map:
                raise SystemExit(f"[错误] 预测中的 qid {r['qid']} 不在本地评测集，无法对齐 gold。")
            r["gold_pages"] = gold_map[r["qid"]]

    summary, per_q = evaluate(records, tocs, gold_map, args.record_gold)
    eval_doc = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_requested": MODEL_NAME,
        "model_returned_values": sorted({r.get("model_returned") for r in records if r.get("model_returned")}),
        "input_policy": "模型仅收到 Question + 目录（章节 ID/标题/层级/页码），未发送答案、gold 页、来源或正文。",
        "page_definition": "PDF 实际页码，1-based",
        "gold_included": bool(args.record_gold),
        "summary": summary,
        "per_question": per_q,
        "excluded_and_unused_same_docs": inventory(all_rows, tocs, args.record_gold),
    }
    with open(args.eval_out, "w", encoding="utf-8") as f:
        json.dump(eval_doc, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n预测: {args.out}\n评价: {args.eval_out}")


if __name__ == "__main__":
    main()
