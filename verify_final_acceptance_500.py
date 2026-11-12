# -*- coding: utf-8 -*-
"""JevSearch 500 题离线验收脚本（零付费、不修改任何原始预测）。

用法：
    py -3.12 verify_final_acceptance_500.py

功能：
  1. 从全部 jev_predictions_*_500.jsonl 独立复算唯一 qid、成功数、失败数、重复调用；
  2. 校验每条记录是否具备真实 API 证据（model / run_at / tokens / state_sent）；
  3. 将 state_sent 中的证据文本回溯到已保存的片段文件（含历史版本）；
  4. 重算 DeepSeek QA、目录路由、Jev 覆盖率与分布指标；
  5. 校验原始结果文件哈希未被覆盖。

输出：outputs/final500/audit/ACCEPTANCE_CHECKS_500.json
"""
import ast
import collections
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import final500_common as C
import compare_500 as cmp500
import task6_route_and_eval_500 as t6500

JD = os.path.join(C.OUT_ROOT, "run500", "jev")
AUD = os.path.join(C.OUT_ROOT, "audit")


def load_dataset():
    """加载本地评测集；缺失时返回空集合，由调用方决定只做不依赖 gold 的检查。"""
    try:
        rows = C.load_samples()
    except SystemExit as e:
        print("[GOLD] 未找到本地评测集，无法复算 QA / 路由指标。")
        print("[GOLD] " + str(e).splitlines()[0])
        return [], {}, {}
    return rows, {s["qid"]: s for s in rows}, \
        {s["qid"]: str(s.get("answer", "")) for s in rows}


def jev_check(smap, ans):
    dataset_available = bool(smap)
    files = sorted(f for f in os.listdir(JD)
                   if f.startswith("jev_predictions_") and f.endswith(".jsonl"))
    ok, failed, per_file = [], [], {}
    for f in files:
        rows = C.load_jsonl(os.path.join(JD, f))
        good = [r for r in rows if r.get("success")]
        per_file[f] = {"records": len(rows), "success": len(good)}
        ok += good
        failed += [r for r in rows if not r.get("success")]

    # 片段回溯：允许历史版本
    snips = {}
    for f in os.listdir(JD):
        if f.startswith("snippets_") and f.endswith(".jsonl"):
            for r in C.load_jsonl(os.path.join(JD, f)):
                snips.setdefault(r["snippet_id"], []).append(r)

    untraceable, wrong_page = [], []
    for r in ok:
        variants = snips.get(r["snippet_id"], [])
        if not any(s.get("evidence_snippet") and s["evidence_snippet"] in r["state_sent"]
                   for s in variants):
            untraceable.append(r["snippet_id"])
        if not any(s.get("page") == r.get("page") for s in variants):
            wrong_page.append(r["snippet_id"])

    calls = collections.Counter(r["qid"] for r in ok)
    integrity = {
        "missing_model": [r["qid"] for r in ok if not r.get("model")],
        "missing_tokens": [r["qid"] for r in ok
                           if r.get("input_tokens") is None or r.get("output_tokens") is None],
        "missing_run_at": [r["qid"] for r in ok if not r.get("run_at")],
        "empty_state": [r["qid"] for r in ok if not r.get("state_sent")],
        "invalid_action": [r["qid"] for r in ok
                           if r.get("next_action") not in ("answer", "expand", "search")],
    }
    gold_in_evidence, short = 0, 0
    for r in ok:
        g = ans.get(r["qid"], "")
        ev = r["state_sent"].split("Evidence snippet:", 1)[-1]
        if g and g != "Not answerable" and g in ev:
            gold_in_evidence += 1
            short += 1 if len(g) <= 2 else 0

    unans = [r for r in ok if ans.get(r["qid"]) == "Not answerable"]
    ansable = [r for r in ok if ans.get(r["qid"]) != "Not answerable"]
    return {
        "dataset_available": dataset_available,
        "per_file": per_file,
        "success_records": len(ok),
        "unique_qid": len(calls),
        "duplicate_qid": {q: n for q, n in calls.items() if n > 1},
        "failed_records": len(failed),
        "missing_qid": [q for q in smap if q not in calls] if dataset_available else None,
        "integrity": {k: len(v) for k, v in integrity.items()},
        "untraceable_snippet": untraceable,
        "page_mismatch_snippet": wrong_page,
        "selection_uses_gold": sum(1 for r in ok if r.get("selection_uses_gold") is True),
        "gold_answer_in_evidence": gold_in_evidence if dataset_available else None,
        "gold_answer_in_evidence_len_le_2": short if dataset_available else None,
        "input_tokens": sum(r.get("input_tokens") or 0 for r in ok),
        "output_tokens": sum(r.get("output_tokens") or 0 for r in ok),
        "models": dict(collections.Counter(r.get("model") for r in ok)),
        "action_all": dict(collections.Counter(r.get("next_action") for r in ok)),
        "relevant_true": sum(1 for r in ok if r.get("relevant") is True),
        "self_contained_true": sum(1 for r in ok if r.get("self_contained") is True),
        "sufficient_true": sum(1 for r in ok if r.get("sufficient") is True),
        "unanswerable": {
            "n": len(unans),
            "action": dict(collections.Counter(r.get("next_action") for r in unans)),
            "sufficient_true_qids": [r["qid"] for r in unans if r.get("sufficient") is True],
        } if dataset_available else None,
        "answerable": {
            "n": len(ansable),
            "action": dict(collections.Counter(r.get("next_action") for r in ansable)),
            "sufficient_true": sum(1 for r in ansable if r.get("sufficient") is True),
        } if dataset_available else None,
    }


def qa_check(smap, ans):
    rows = C.load_jsonl(os.path.join(C.OUT_ROOT, "run500", "qa", "ds_results.jsonl"))
    last = C.last_by_qid(rows)
    failed = [q for q in smap
              if q not in last or not last[q].get("ds_success")]
    unans = [q for q in smap if ans[q] == "Not answerable"]
    ansable = [q for q in smap if ans[q] != "Not answerable"]

    def pred(q):
        r = last.get(q)
        return str(r.get("ds_answer", "") or "") if r and r.get("ds_success") else ""

    def block(qids, skip_failed=False):
        pair = [(ans[q], pred(q)) for q in qids if not (skip_failed and q in failed)]
        return cmp500.acc(pair)

    tok = {"prompt": 0, "completion": 0}
    for r in last.values():
        u = r.get("usage")
        if not u:
            continue
        u = ast.literal_eval(u) if isinstance(u, str) else u
        tok["prompt"] += u.get("prompt_tokens", 0)
        tok["completion"] += u.get("completion_tokens", 0)
    return {
        "records": len(rows), "unique_qid": len(last),
        "success": len(last) - len(failed), "failed_qids": failed,
        "status_counts": dict(collections.Counter(r.get("status") for r in last.values())),
        "all_500_failure_as_wrong": block(list(smap)),
        "all_500_success_only": block(list(smap), True),
        "answerable_success_only": block(ansable, True),
        "unanswerable_refusal_correctness": block(unans),
        "tokens": tok,
        "note": "输入使用 gold evidence_pages（oracle），空证据页回退第 1 页；gold answer 不发送。",
    }


def route_check(samples):
    tocs = t6500.load_tocs()
    routes = C.load_jsonl(os.path.join(C.OUT_ROOT, "run500", "toc", "routes.jsonl"))
    fill = C.load_jsonl(os.path.join(C.OUT_ROOT, "run500", "toc", "routes_fill_126.jsonl"))
    expected = sum(1 for s in samples if t6500.classify(s, tocs)[1])
    smap = {s["qid"]: s for s in samples}
    summary, _ = t6500.evaluate(routes, tocs, expected, smap)
    fm = C.read_json(os.path.join(AUD, "route_fill_metrics.json"), default={}) or {}
    corr = fm.get("corrected_gold_eval", [])
    base = [r for r in routes if r.get("status") == "ok"]
    hit1 = hit3 = full3 = 0
    r1 = r3 = 0.0
    for r in base:
        off = r.get("offline_only") or {}
        gold = C.parse_pages(off.get("gold_pages", "[]")) if off else \
            C.parse_pages(smap.get(r["qid"], {}).get("evidence_pages", "[]"))
        secs = r.get("sections") or []
        p1 = [p for s in secs[:1] for p in range(s["start_page"], s["end_page"] + 1)]
        p3 = [p for s in secs[:3] for p in range(s["start_page"], s["end_page"] + 1)]
        if not gold:
            continue
        a, b = len(set(p1) & set(gold)) / len(gold), len(set(p3) & set(gold)) / len(gold)
        r1 += a; r3 += b
        hit1 += a > 0; hit3 += b > 0
        full3 += set(gold) <= set(p3)
    return {
        "exec_records": len(routes),
        "exec_ok": len(base),
        "fill_records": len(fill),
        "fill_ok": sum(1 for r in fill if r.get("success") and not r.get("abstained")),
        "fill_abstain": sum(1 for r in fill if r.get("abstained")),
        "executed_unique_qid": len({r["qid"] for r in routes} | {r["qid"] for r in fill}),
        "eval_summary": summary,
        "recomputed_base_374": {"n": len(base), "hit@1": hit1, "hit@3": hit3,
                                "pr@1": round(r1 / len(base), 4),
                                "pr@3": round(r3 / len(base), 4),
                                "full@3": full3},
        "recomputed_376_non_abstain": {
            "n": len(base) + sum(1 for c in corr if not c["abstained"]),
            "hit@3": hit3 + sum(1 for c in corr if c["hit@3"]),
            "pr@3": round((r3 + sum(c["page_recall@3"] for c in corr if not c["abstained"]))
                          / (len(base) + sum(1 for c in corr if not c["abstained"])), 4),
        },
        "recomputed_379_strict": {
            "n": len(base) + len(corr),
            "hit@3": hit3 + sum(1 for c in corr if c["hit@3"]),
            "pr@3": round((r3 + sum(c["page_recall@3"] for c in corr))
                          / (len(base) + len(corr)), 4),
        },
    }


def hash_check():
    """优先比对公开副本基线；缺失时回退到实验前的原始哈希。"""
    pub = C.read_json(os.path.join(AUD, "public_files_sha256.json"), default={}) or {}
    pre = C.read_json(os.path.join(AUD, "pre_run_check.json"), default={}) or {}
    baseline = pub.get("files") or (pre.get("E_hashes_before") or {})
    out = {}
    for rel, h in baseline.items():
        p = os.path.join(ROOT, rel)
        now = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else "MISSING"
        out[rel] = "UNCHANGED" if now == h else "CHANGED"
    return out


def shared_check():
    qa_last = C.last_by_qid(C.load_jsonl(os.path.join(C.OUT_ROOT, "run500", "qa", "ds_results.jsonl")))
    qa_ok = {q for q, r in qa_last.items() if r.get("ds_success")}

    jd = os.path.join(C.OUT_ROOT, "run500", "jev")
    jev_ok = set()
    for f in os.listdir(jd):
        if f.startswith("jev_predictions_") and f.endswith(".jsonl"):
            for r in C.load_jsonl(os.path.join(jd, f)):
                if r.get("success"):
                    jev_ok.add(r["qid"])

    routes = C.load_jsonl(os.path.join(C.OUT_ROOT, "run500", "toc", "routes.jsonl"))
    route_ok = {r["qid"] for r in routes if r.get("status") == "ok"}
    fm = C.read_json(os.path.join(AUD, "route_fill_metrics.json"), default={}) or {}
    route379 = route_ok | {c["qid"] for c in fm.get("corrected_gold_eval", [])}

    return {
        "jev_and_qa": len(jev_ok & qa_ok),
        "jev_and_route_eval": len(jev_ok & route379),
        "all_three": len(jev_ok & qa_ok & route379),
        "note": "route_eval 采用 379 严格口径（原生 374 + 5 道页码修正题）。",
    }


def main():
    samples, smap, ans = load_dataset()
    dataset_available = bool(smap)
    result = {
        "generated_at": C.now_str(),
        "dataset_available": dataset_available,
        "samples": len(samples),
        "jev": jev_check(smap, ans),
        "original_files_hash": hash_check(),
    }
    if dataset_available:
        result["qa"] = qa_check(smap, ans)
        result["route"] = route_check(samples)
        result["shared_evaluable_qids"] = shared_check()

    j = result["jev"]
    print("[JEV] success=%d unique=%d dup=%d fail=%d untraceable=%d gold=%d"
          % (j["success_records"], j["unique_qid"], len(j["duplicate_qid"]),
             j["failed_records"], len(j["untraceable_snippet"]),
             j["selection_uses_gold"]))
    changed = [k for k, v in result["original_files_hash"].items() if v != "UNCHANGED"]
    hash_line = "[HASH] changed=%s" % (changed or "none")

    if not dataset_available:
        print("[QA ] [SKIP] 缺少本地评测集，无法复算准确率。")
        print("[RTE] [SKIP] 缺少本地评测集，无法复算 Hit@3 / Page Recall。")
        print(hash_line)
        print("[OUT] 未写入 ACCEPTANCE_CHECKS_500.json（缺少 gold，指标不完整）。")
        return

    print("[QA ] success=%d/%d strict=%.4f lenient=%.4f"
          % (result["qa"]["success"], result["qa"]["unique_qid"],
             result["qa"]["all_500_success_only"]["strict_rate"],
             result["qa"]["all_500_success_only"]["lenient_rate"]))
    print("[RTE] exec=%d fill_ok=%d abstain=%d eval376_hit@3=%d pr@3=%.4f"
          % (result["route"]["executed_unique_qid"], result["route"]["fill_ok"],
             result["route"]["fill_abstain"],
             result["route"]["recomputed_376_non_abstain"]["hit@3"],
             result["route"]["recomputed_376_non_abstain"]["pr@3"]))
    print(hash_line)
    if j["missing_qid"]:
        print("[JEV] missing_qid=%s" % j["missing_qid"])
    C.write_json_atomic(os.path.join(AUD, "ACCEPTANCE_CHECKS_500.json"), result)
    print("[OUT] 已写入 outputs/final500/audit/ACCEPTANCE_CHECKS_500.json")


if __name__ == "__main__":
    main()
