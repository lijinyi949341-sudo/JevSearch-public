# -*- coding: utf-8 -*-
"""
500 题统一结果汇总与比较（不调用任何模型 API）。

读取：
  - sampled_500_for_baseline.jsonl（qid 映射，只读）
  - baseline_results.jsonl / advanced_rag_results.jsonl（第一轮 Qwen 结果，只读）
  - outputs/final500/qa_multimodal/deepseek_v41/ds_results.jsonl（DeepSeek 新结果）
  - outputs/final500/toc/routes/deepseek_v41/metrics.json（目录检索指标）
  - outputs/final500/jev/agreement_*.json（Jev 人工标注一致性）

输出：
  outputs/final500/manifest.json   统一结果清单与比较口径
  outputs/final500/compare/summary.json
  outputs/final500/compare/summary.md

比较原则：
  1. 所有方法都按 qid 对齐到同一批 500 题；
  2. 主口径把 DeepSeek 的失败/缺失样本记为“错”，避免只挑成功样本抬高准确率；
  3. 同时给出“双方都成功”的参考口径，便于定位模型能力差异；
  4. 不可答题、gold 页码越界题单独统计；
  5. 未有人工确认标注时，不输出 Jev 准确率结论。

用法：
  python compare_500.py            # 允许部分结果，缺失项标记 pending
"""
import argparse
import ast
import collections
import json
import os
import re

import final500_common as C

OUT_DIR = os.path.join(C.OUT_ROOT, "compare")
SUMMARY_JSON = os.path.join(OUT_DIR, "summary.json")
SUMMARY_MD = os.path.join(OUT_DIR, "summary.md")
MANIFEST = os.path.join(C.OUT_ROOT, "manifest.json")

DS_RESULTS = os.path.join(C.OUT_ROOT, "run500", "qa", "ds_results.jsonl")
TOC_METRICS = os.path.join(C.OUT_ROOT, "run500", "toc", "metrics.json")
TOC_BASELINE = os.path.join(C.OUT_ROOT, "toc", "routes", "deepseek_v41", "baseline_preview.json")
JEV_DIR = os.path.join(C.OUT_ROOT, "run500", "jev")

# 500 题正式实验的 6 个 Jev 片段来源（文件名与 qid 一一对应，来源之间无重复题）
JEV_FILES = {
    "title": "jev_predictions_title_500.jsonl",
    "route": "jev_predictions_route_500.jsonl",
    "ocr": "jev_predictions_ocr_500.jsonl",
    "recover": "jev_predictions_recover_500.jsonl",
    "pilot1": "jev_predictions_ocr_pilot_500.jsonl",
    "pilot2": "jev_predictions_ocr_pilot2_500.jsonl",
}


# ---------- 与第一轮完全一致的判分口径 ----------
def loose_norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def gold_items(gold):
    s = str(gold).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return [str(x) for x in v]
        except Exception:
            pass
    return [s]


def match_strict(gold, pred):
    g = str(gold).strip().lower()
    p = str(pred).strip().lower()
    return bool(g) and g in p


def match_lenient(gold, pred):
    p = loose_norm(pred)
    return all(loose_norm(x) in p for x in gold_items(gold) if loose_norm(x))


def acc(rows):
    strict = sum(1 for g, p in rows if match_strict(g, p))
    lenient = sum(1 for g, p in rows if match_lenient(g, p))
    return {"n": len(rows), "strict": strict,
            "strict_rate": round(strict / len(rows), 4) if rows else None,
            "lenient": lenient,
            "lenient_rate": round(lenient / len(rows), 4) if rows else None}


def load_aligned(path, key, samples):
    """旧结果按行对齐；同时校验 doc_id/question，避免错位。"""
    if not os.path.exists(path):
        return None, 0
    rows = C.load_jsonl(path)
    mismatch = 0
    out = {}
    for s, r in zip(samples, rows):
        if s["doc_id"] != r.get("doc_id") or s["question"].strip() != str(r.get("question", "")).strip():
            mismatch += 1
        out[s["qid"]] = str(r.get(key, "") or "")
    return out, mismatch


def count_failures(records, qids):
    last = C.last_by_qid(records)
    missing = [q for q in qids if q not in last]
    failed = [q for q in qids if q in last and not last[q].get("ds_success")]
    return missing, failed, last


def jev_summary():
    out = {}
    for mode, fn in JEV_FILES.items():
        pred = os.path.join(JEV_DIR, fn)
        rows = C.load_jsonl(pred)
        last = {}
        for r in rows:
            last[r["snippet_id"]] = r
        ok = [r for r in last.values() if r.get("success")]
        out[mode] = {
            "predictions_total": len(rows),
            "unique_snippets": len(last),
            "success": len(ok),
            "input_tokens": sum((r.get("input_tokens") or 0) for r in ok),
            "output_tokens": sum((r.get("output_tokens") or 0) for r in ok),
            "latency_total_sec": round(sum((r.get("latency_sec") or 0) for r in ok), 2),
            "latency_avg_sec": round(sum((r.get("latency_sec") or 0) for r in ok) / len(ok), 3)
            if ok else 0,
            "next_action_counts": dict(collections.Counter(r.get("next_action") for r in ok)),
            "agreement": {},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true", default=True)
    args = ap.parse_args()

    samples = C.load_samples()
    qids = [s["qid"] for s in samples]
    not_answerable = {s["qid"] for s in samples
                      if str(s.get("answer", "")) == "Not answerable"}
    no_gold = {s["qid"] for s in samples
               if str(s.get("answer", "")) != "Not answerable"
               and not C.parse_pages(s.get("evidence_pages", "[]"))}
    answerable = set(qids) - not_answerable
    unanswerable = not_answerable

    qwen, qwen_mismatch = load_aligned(
        os.path.join(C.ROOT, "advanced_rag_results.jsonl"), "advanced_answer", samples)
    text_base, base_mismatch = load_aligned(
        os.path.join(C.ROOT, "baseline_results.jsonl"), "deepseek_answer", samples)

    ds_rows = C.load_jsonl(DS_RESULTS)
    missing, failed, ds_last = count_failures(ds_rows, qids)
    ds_ok = {q for q in qids if q in ds_last and ds_last[q].get("ds_success")}

    gold = {s["qid"]: s.get("answer", "") for s in samples}
    ds_ans = {q: (ds_last[q].get("ds_answer", "") if q in ds_last else "") for q in qids}

    def pair(qid_set, use_ds_failure_as_wrong=True):
        rows_q, rows_ds = [], []
        for q in qids:
            if q not in qid_set:
                continue
            if q not in qwen or not qwen[q].strip():
                continue
            rows_q.append((gold[q], qwen[q]))
            if q in ds_ok:
                rows_ds.append((gold[q], ds_ans[q]))
            elif use_ds_failure_as_wrong:
                rows_ds.append((gold[q], ""))
            else:
                continue
        return acc(rows_q), acc(rows_ds)

    subsets = {
        "all_500": set(qids),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "answerable_without_gold_pages": no_gold,
    }
    comparison = {}
    for name, sset in subsets.items():
        q_acc, ds_acc = pair(sset, use_ds_failure_as_wrong=True)
        q_acc_ok, ds_acc_ok = pair(sset, use_ds_failure_as_wrong=False)
        if not ds_ok:
            # 还没跑 DeepSeek 时不要输出“0%”，明确标记 pending
            ds_acc = acc([])
            ds_acc_ok = acc([])
        comparison[name] = {
            "qwen_advanced": q_acc,
            "deepseek_failure_as_wrong": ds_acc,
            "deepseek_ok_only_reference": ds_acc_ok,
            "deepseek_pending": not ds_ok,
        }

    text_acc = None
    if text_base:
        text_rows = [(gold[q], text_base[q]) for q in qids if q in text_base]
        text_acc = acc(text_rows)

    ds_usage = collections.defaultdict(int)
    ds_lat = 0.0
    ds_status = collections.Counter()
    ds_partial = 0
    for r in ds_last.values():
        ds_status[r.get("status")] += 1
        u = r.get("usage") or {}
        ds_usage["prompt"] += u.get("prompt_tokens", 0)
        ds_usage["completion"] += u.get("completion_tokens", 0)
        ds_usage["total"] += u.get("total_tokens", 0)
        ds_lat += float(r.get("latency_sec") or 0)
        if r.get("status") == "ok" and r.get("missing_pages"):
            ds_partial += 1

    toc_metrics = C.read_json(TOC_METRICS)
    toc_baseline = C.read_json(TOC_BASELINE)
    jev = jev_summary()

    artifacts = [
        ("500 题采样（只读）", C.SAMPLES, len(samples)),
        ("第一轮纯文本 baseline（只读）", os.path.join(C.ROOT, "baseline_results.jsonl"),
         len(text_base) if text_base else 0),
        ("第一轮 Qwen3-VL-Plus（只读）", os.path.join(C.ROOT, "advanced_rag_results.jsonl"),
         len(qwen) if qwen else 0),
        ("DeepSeek 500 题问答", DS_RESULTS, len(ds_last)),
        ("500 题目录", os.path.join(C.OUT_ROOT, "toc", "toc.json"), None),
        ("目录路由预测", os.path.join(C.OUT_ROOT, "run500", "toc", "routes.jsonl"), None),
        ("目录检索指标", TOC_METRICS, None),
        ("Jev 判断汇总", JEV_DIR, None),
    ] + [("Jev 预测({})".format(mode), os.path.join(JEV_DIR, fn), None)
         for mode, fn in JEV_FILES.items()]
    manifest = {
        "generated_at": C.now_str(),
        "dataset": "MMLongBench-Doc 500 题（同一采样文件）",
        "qid_scheme": "Q0001..Q0500，按采样文件物理行号 1-based",
        "n_questions": len(samples),
        "n_unique_docs": len(set(s["doc_id"] for s in samples)),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_answerable_but_no_gold_pages": len(no_gold),
        "comparison_policy": {
            "same_qids": True,
            "primary_accuracy": "DeepSeek 失败/缺失样本记为错（denominator=Qwen 有回答的同一 qid 集）",
            "reference_accuracy": "双方都成功的 qid 子集",
            "retrieval_denominator": "不可答题与 gold 页码越界题不计入 Page Recall",
            "qa_note": "多模态问答使用 gold evidence_pages，属于已知证据页问答，不是端到端检索系统的结果",
            "jev_policy": "只有 review_status=confirmed 且字段完整的人工标注才可计算准确率",
        },
        "models": {
            "qwen3-vl-plus": {"role": "第一轮多模态问答/基线（已完成，不重复付费）"},
            "deepseek-v4.1-flash-1017": {
                "role": "500 题多模态问答 + 目录路由 + Jev 调用",
                "base_url": C.BASE_URL,
            },
            "jev-latest": {"role": "上下文必要性判断", "api": "https://api.typesafe.ai"},
        },
        "artifacts": [{"purpose": p, "path": os.path.relpath(path, C.ROOT),
                       "exists": os.path.exists(path), "records": n}
                      for p, path, n in artifacts],
    }
    C.write_json_atomic(MANIFEST, manifest)

    summary = {
        "generated_at": C.now_str(),
        "partial_result": bool(missing or failed),
        "coverage": {
            "total_qids": len(qids),
            "ds_records_present": len(ds_last),
            "ds_success": len(ds_ok),
            "ds_missing": len(missing),
            "ds_failed": len(failed),
            "qwen_rows": len(qwen) if qwen else 0,
            "text_baseline_rows": len(text_base) if text_base else 0,
            "alignment_mismatch_qwen": qwen_mismatch,
            "alignment_mismatch_text": base_mismatch,
        },
        "accuracy": {
            "text_only_baseline_all_500": text_acc,
            "pairwise": comparison,
        },
        "deepseek_stats": {
            "status_counts": dict(ds_status),
            "success_partial_evidence_pages": ds_partial,
            "tokens": dict(ds_usage),
            "latency_total_sec": round(ds_lat, 2),
            "latency_avg_sec": round(ds_lat / len(ds_ok), 3) if ds_ok else 0,
        },
        "retrieval": {
            "metrics_present": bool(toc_metrics),
            "summary": (toc_metrics or {}).get("summary"),
            "title_baseline_preview": (toc_baseline or {}).get("baseline_title_match"),
        },
        "jev": jev,
        "billing_note": ("脚本只记录 token/耗时；人民币成本需按 DashScope 与 TypeSafe 的"
                         "官方单价乘以实际计费 token/请求数，本文不臆造单价。"),
    }
    C.write_json_atomic(SUMMARY_JSON, summary)

    # ---------- Markdown ----------
    lines = [
        "# 500 题统一结果汇总", "",
        f"生成时间：{summary['generated_at']}", "",
        "> 多模态问答使用数据集 gold evidence_pages，属于“已知证据页问答”，"
        "不是完整检索系统的端到端表现。", "",
        "## 1. 样本与覆盖", "",
        f"- 题目：{len(qids)}；唯一 PDF：{manifest['n_unique_docs']}；"
        f"可回答：{len(answerable)}；不可回答：{len(unanswerable)}；"
        f"可回答但无 gold 页：{len(no_gold)}",
        f"- Qwen 旧结果：{len(qwen) if qwen else 0} 条（对齐错位 {qwen_mismatch}）",
        f"- DeepSeek 新结果：成功 {len(ds_ok)} / {len(qids)}，"
        f"缺失 {len(missing)}，失败 {len(failed)}",
        "",
        "## 2. 问答准确率（严格子串 / 宽松匹配）", "",
        "| 子集 | 方法 | n | strict | lenient |",
        "|---|---|---|---|---|",
    ]
    if text_acc:
        lines.append(f"| 全部 500 | 纯文本 baseline | {text_acc['n']} | "
                     f"{text_acc['strict']} ({text_acc['strict_rate']*100:.2f}%) | "
                     f"{text_acc['lenient']} ({text_acc['lenient_rate']*100:.2f}%) |")
    for name, blk in comparison.items():
        for label, key in (("Qwen3-VL-Plus", "qwen_advanced"),
                           ("DeepSeek(失败记错)", "deepseek_failure_as_wrong"),
                           ("DeepSeek(仅成功)", "deepseek_ok_only_reference")):
            a = blk[key]
            if a['n']:
                lines.append(f"| {name} | {label} | {a['n']} | "
                             f"{a['strict']} ({a['strict_rate']*100:.2f}%) | "
                             f"{a['lenient']} ({a['lenient_rate']*100:.2f}%) |")
            else:
                lines.append(f"| {name} | {label} | 0 | pending | pending |")
    lines += ["", "主口径为 “DeepSeek(失败记错)”；“仅成功”只用于参考，不能单独作为结论。", ""]

    lines += ["## 3. 目录检索（Question + TOC）", ""]
    if toc_metrics:
        s = toc_metrics["summary"]
        lines += [
            f"- 可评测题：{s['eligible_in_records']} / 预期 {s['expected_eligible']}；"
            f"成功返回：{s['evaluated_ok']}",
            f"- LLM Hit@1：{s['llm_all_eligible']['section_hit@1']}，"
            f"Hit@3：{s['llm_all_eligible']['section_hit@3']}",
            f"- LLM Page Recall@3（macro）：{s['llm_all_eligible']['page_recall@3_macro']}，"
            f"完整召回：{s['llm_all_eligible']['full_evidence_recall@3']}",
            f"- 同预算标题 baseline：Hit {s['baseline_budget_matched_all_eligible']['section_hit']}，"
            f"Page Recall {s['baseline_budget_matched_all_eligible']['page_recall_macro']}",
            f"- 不可答题（不计检索失败）：{s['unanswerable_excluded_from_recall']}；"
            f"gold 越界（不计分母）：{s['invalid_gold_excluded_from_recall']}",
        ]
    else:
        lines.append("- 尚未运行目录路由，指标 pending。")

    lines += ["", "## 4. Jev 上下文判断", ""]
    lines += ["| 片段来源 | 预测数 | 成功 | 平均耗时 | 关键动作分布 | 人工确认 | 可宣称准确率 |",
              "|---|---|---|---|---|---|---|"]
    for mode, js in jev.items():
        ag = js.get("agreement") or {}
        lines.append(f"| {mode} | {js['unique_snippets']} | {js['success']} | "
                     f"{js['latency_avg_sec']}s | {js['next_action_counts']} | "
                     f"{ag.get('confirmed_total', 0)} | "
                     f"{'是' if ag.get('claimable') else '否（样本不足/未确认）'} |")
    lines += ["", "未达到确认门槛前，不计算也不宣称 Jev 准确率。", ""]

    lines += ["## 5. 资源与成本", "",
              f"- DeepSeek token：{dict(ds_usage)}；平均耗时 {summary['deepseek_stats']['latency_avg_sec']}s",
              f"- {summary['billing_note']}", ""]
    C.ensure_dir(OUT_DIR)
    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n清单: {MANIFEST}\n汇总: {SUMMARY_JSON}\n报告: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
