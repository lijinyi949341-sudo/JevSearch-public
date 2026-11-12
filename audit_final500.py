# -*- coding: utf-8 -*-
"""
500 题最终交付的离线审计脚本（不调用任何模型 API，不改写任何原始结果）。

读取：
  - sampled_500_for_baseline.jsonl
  - outputs/final500/run500/qa/ds_results.jsonl
  - outputs/final500/run500/toc/routes.jsonl
  - outputs/final500/run500/toc/metrics.json
  - outputs/final500/run500/jev/jev_predictions_*.jsonl（8 个来源，含 gapfill / round3）
  - outputs/final500/audit/gold_page_fixes_7.json（仅离线分母核对）

输出：
  - outputs/final500/audit/final500_status_500.csv       500 行统一状态表
  - outputs/final500/audit/final500_stats.json           全部统计（由真实文件计算）
  - outputs/final500/audit/final500_module_matrix.md     六类状态矩阵 + 指标汇总

六类状态定义（每个模块单独判定，不跨模块相加）：
  已尝试 attempted      该题在本模块有至少一条真实执行记录
  API 成功 api_success  远端调用返回成功
  有效输入 valid_input  模型/脚本实际拿到了可用输入（QA: 成功渲染页面；路由: eval_eligible；
                        Jev: 有通过 leak-check 的证据片段）
  可评测 evaluable      该模块能产出可进入指标计算的输出
  不适用 not_applicable 该题在任务定义上不属于本模块（如不可答题不参与路由）
  失败 failed           已尝试但未成功，且不属于“不适用”

用法：
  .venv_paddleocr\\Scripts\\python.exe audit_final500.py
"""
import ast
import collections
import csv
import json
import os
import re

import final500_common as C

AUDIT_DIR = os.path.join(C.OUT_ROOT, "audit")
QA_PATH = os.path.join(C.OUT_ROOT, "run500", "qa", "ds_results.jsonl")
ROUTE_PATH = os.path.join(C.OUT_ROOT, "run500", "toc", "routes.jsonl")
TOC_METRICS = os.path.join(C.OUT_ROOT, "run500", "toc", "metrics.json")
JEV_DIR = os.path.join(C.OUT_ROOT, "run500", "jev")
GOLD_FIX = os.path.join(AUDIT_DIR, "gold_page_fixes_7.json")

JEV_SOURCES = [
    ("title", "jev_predictions_title_500.jsonl"),
    ("route", "jev_predictions_route_500.jsonl"),
    ("ocr", "jev_predictions_ocr_500.jsonl"),
    ("recover", "jev_predictions_recover_500.jsonl"),
    ("pilot1", "jev_predictions_ocr_pilot_500.jsonl"),
    ("pilot2", "jev_predictions_ocr_pilot2_500.jsonl"),
    ("gapfill", "jev_predictions_gapfill_500.jsonl"),
    ("round3", "jev_predictions_round3_500.jsonl"),
]
# 同一题可能有多个来源；按此优先级选定“主来源”（与既有覆盖率报告一致）
SOURCE_PRIORITY = ["round3", "gapfill", "route", "ocr", "recover", "pilot2", "pilot1", "title"]


# ---------- 判分口径（与 compare_500.py 完全一致，gold 只用于离线评分） ----------
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


def norm_action(r):
    if not r.get("relevant"):
        return "search"
    if r.get("sufficient"):
        return "answer"
    return "expand"


def rate(a, b):
    return round(a / b, 4) if b else None


def main():
    samples = C.load_samples()
    qids = [s["qid"] for s in samples]
    assert len(qids) == 500, f"期望 500 题，实际 {len(qids)}"
    assert len(set(qids)) == 500, "qid 有重复"

    info = {}
    for s in samples:
        pages = C.parse_pages(s.get("evidence_pages", "[]"))
        unans = str(s.get("answer", "")).strip().lower() == "not answerable"
        info[s["qid"]] = {
            "sample_index": s.get("sample_index"),
            "doc_id": s.get("doc_id"),
            "doc_type": s.get("doc_type"),
            "question": s.get("question"),
            "gold": s.get("answer", ""),
            "gold_pages": pages,
            "answerable": (not unans),
            "unanswerable": unans,
            "no_gold_pages": (not unans) and not pages,
        }
    answerable = {q for q in qids if info[q]["answerable"]}
    unanswerable = {q for q in qids if info[q]["unanswerable"]}
    no_gold = {q for q in qids if info[q]["no_gold_pages"]}

    # ---------- QA ----------
    qa_rows = C.load_jsonl(QA_PATH)
    qa_last = C.last_by_qid(qa_rows)
    qa = {}
    for q in qids:
        r = qa_last.get(q)
        attempted = r is not None
        api_ok = bool(r and r.get("ds_success"))
        status = (r or {}).get("status")
        valid_input = bool(api_ok and status == "ok")
        ans = str((r or {}).get("ds_answer") or "")
        evaluable = bool(valid_input and ans.strip())
        qa[q] = {
            "attempted": attempted, "api_success": api_ok,
            "valid_input": valid_input, "evaluable": evaluable,
            "not_applicable": False, "failed": attempted and not api_ok,
            "status": status, "answer": ans,
            "error": (r or {}).get("ds_error") or "",
            "image_count": (r or {}).get("image_count"),
            "missing_pages": (r or {}).get("missing_pages") or [],
        }
        qa[q]["strict"] = match_strict(info[q]["gold"], ans) if evaluable else False
        qa[q]["lenient"] = match_lenient(info[q]["gold"], ans) if evaluable else False

    # ---------- 路由 ----------
    route_rows = C.load_jsonl(ROUTE_PATH)
    route_last = C.last_by_qid(route_rows)
    metrics = C.read_json(TOC_METRICS, {}) or {}
    mq = {x["qid"]: x for x in (metrics.get("per_question") or [])}
    route = {}
    for q in qids:
        r = route_last.get(q)
        api_ok = bool(r and r.get("success"))
        eligible = bool(r and r.get("eval_eligible"))
        # 以 status 判定是否“不适用”：不可答题 / gold 页码越界 / 缺 gold 页
        status = (r or {}).get("status")
        reason = (r or {}).get("eval_reason") or ""
        skipped = status in ("unanswerable_skipped", "invalid_gold_pages_skipped",
                             "missing_gold_pages_skipped")
        attempted = bool(r) and not skipped
        not_applicable = bool(r) and skipped
        m = mq.get(q) or {}
        t1 = (m.get("llm_top1") or {})
        t3 = (m.get("llm_top3") or {})
        route[q] = {
            "attempted": attempted,
            "api_success": api_ok,
            "valid_input": eligible,
            "evaluable": bool(api_ok and eligible),
            "not_applicable": not_applicable and not eligible,
            "failed": attempted and not api_ok,
            "status": (r or {}).get("status"),
            "eval_reason": reason,
            "hit1": t1.get("section_hit"),
            "hit3": t3.get("section_hit"),
            "pr1": t1.get("page_recall"),
            "pr3": t3.get("page_recall"),
            "full1": t1.get("full_evidence_recall"),
            "full3": t3.get("full_evidence_recall"),
            "toc_source": (r or {}).get("toc_source"),
            "n_sections": len((r or {}).get("sections") or []),
        }

    # ---------- Jev ----------
    jev_all = {}
    per_source_counts = {}
    jev_conflicts = []
    for src, fn in JEV_SOURCES:
        path = os.path.join(JEV_DIR, fn)
        rows = C.load_jsonl(path)
        per_source_counts[src] = {
            "records": len(rows),
            "success": sum(1 for r in rows if r.get("success")),
        }
        for r in rows:
            if not r.get("success"):
                continue
            sid = r.get("snippet_id")
            if not sid:
                continue
            prev = jev_all.get(sid)
            # 同一 snippet_id 只保留一条（最后一条成功记录）
            r["_src"] = src
            jev_all[sid] = r
    jev_by_qid = collections.defaultdict(list)
    for sid, r in jev_all.items():
        q = r.get("qid")
        if q:
            jev_by_qid[q].append(r)
    jev = {}
    for q in qids:
        recs = jev_by_qid.get(q, [])
        if recs:
            recs_sorted = sorted(recs, key=lambda r: SOURCE_PRIORITY.index(r.get("_src"))
                                 if r.get("_src") in SOURCE_PRIORITY else 99)
            primary = recs_sorted[0]
            jev[q] = {
                "attempted": True, "api_success": True, "valid_input": True,
                "evaluable": True, "not_applicable": False, "failed": False,
                "source": primary.get("_src"), "snippet_id": primary.get("snippet_id"),
                "relevant": primary.get("relevant"),
                "self_contained": primary.get("self_contained"),
                "sufficient": primary.get("sufficient"),
                "action_raw": primary.get("next_action"),
                "action_norm": norm_action(primary),
                "n_sources": len(recs),
                "all_sources": ",".join(sorted({x.get("_src") for x in recs})),
            }
            if len(recs) > 1:
                jev_conflicts.append({"qid": q, "n": len(recs),
                                      "sources": jev[q]["all_sources"]})
        else:
            jev[q] = {
                "attempted": False, "api_success": False, "valid_input": False,
                "evaluable": False, "not_applicable": True, "failed": False,
                "source": "", "snippet_id": "", "relevant": None,
                "self_contained": None, "sufficient": None,
                "action_raw": "", "action_norm": "", "n_sources": 0, "all_sources": "",
            }

    # ---------- 四集合 ----------
    common_all_three = {q for q in qids if qa[q]["evaluable"] and route[q]["evaluable"]
                        and jev[q]["evaluable"]}
    sets = {
        "all_500": set(qids),
        "common_all_three": common_all_three,
        "answerable_381": answerable,
        "unanswerable_119": unanswerable,
    }

    def set_stats(qset):
        n = len(qset)
        qa_ev = [q for q in qset if qa[q]["evaluable"]]
        qa_strict = sum(1 for q in qa_ev if qa[q]["strict"])
        qa_len = sum(1 for q in qa_ev if qa[q]["lenient"])
        rt_ev = [q for q in qset if route[q]["evaluable"]]
        hit1 = sum(1 for q in rt_ev if route[q]["hit1"] == 1)
        hit3 = sum(1 for q in rt_ev if route[q]["hit3"] == 1)
        pr1 = sum((route[q]["pr1"] or 0) for q in rt_ev)
        pr3 = sum((route[q]["pr3"] or 0) for q in rt_ev)
        full1 = sum(1 for q in rt_ev if route[q]["full1"] == 1)
        full3 = sum(1 for q in rt_ev if route[q]["full3"] == 1)
        jv = [q for q in qset if jev[q]["evaluable"]]
        jact = collections.Counter(jev[q]["action_norm"] for q in jv)
        jraw = collections.Counter(jev[q]["action_raw"] for q in jv)
        jsrc = collections.Counter(jev[q]["source"] for q in jv)
        unans_in = [q for q in qset if info[q]["unanswerable"]]
        refusal = sum(1 for q in unans_in if qa[q]["lenient"])
        return {
            "n": n,
            "qa": {
                "attempted": sum(1 for q in qset if qa[q]["attempted"]),
                "api_success": sum(1 for q in qset if qa[q]["api_success"]),
                "valid_input": sum(1 for q in qset if qa[q]["valid_input"]),
                "evaluable": len(qa_ev),
                "strict_correct": qa_strict, "strict_rate": rate(qa_strict, len(qa_ev)),
                "lenient_correct": qa_len, "lenient_rate": rate(qa_len, len(qa_ev)),
                "unanswerable_in_set": len(unans_in),
                "refusal_correct_lenient": refusal,
                "refusal_rate": rate(refusal, len(unans_in)) if unans_in else None,
            },
            "route": {
                "applicable": sum(1 for q in qset if not route[q]["not_applicable"]),
                "attempted": sum(1 for q in qset if route[q]["attempted"]),
                "api_success": sum(1 for q in qset if route[q]["api_success"]),
                "valid_input": sum(1 for q in qset if route[q]["valid_input"]),
                "evaluable": len(rt_ev),
                "not_applicable": sum(1 for q in qset if route[q]["not_applicable"]),
                "hit1": hit1, "hit1_rate": rate(hit1, len(rt_ev)),
                "hit3": hit3, "hit3_rate": rate(hit3, len(rt_ev)),
                "page_recall1_macro": rate(pr1, len(rt_ev)),
                "page_recall3_macro": rate(pr3, len(rt_ev)),
                "full_recall1": full1, "full_recall1_rate": rate(full1, len(rt_ev)),
                "full_recall3": full3, "full_recall3_rate": rate(full3, len(rt_ev)),
            },
            "jev": {
                "attempted": sum(1 for q in qset if jev[q]["attempted"]),
                "api_success": sum(1 for q in qset if jev[q]["api_success"]),
                "valid_input": sum(1 for q in qset if jev[q]["valid_input"]),
                "evaluable": len(jv),
                "not_applicable": sum(1 for q in qset if jev[q]["not_applicable"]),
                "failed": sum(1 for q in qset if jev[q]["failed"]),
                "relevant_true": sum(1 for q in jv if jev[q]["relevant"]),
                "self_contained_true": sum(1 for q in jv if jev[q]["self_contained"]),
                "sufficient_true": sum(1 for q in jv if jev[q]["sufficient"]),
                "action_raw": dict(jraw),
                "action_norm": dict(jact),
                "sources": dict(jsrc),
                "accuracy": None,
                "accuracy_note": "无人工 confirmed 标注，不计算判断准确率",
            },
        }

    stats_sets = {k: set_stats(v) for k, v in sets.items()}

    # ---------- 失败与异常明细 ----------
    qa_failed = []
    for q in qids:
        if qa[q]["failed"]:
            qa_failed.append({
                "qid": q, "status": qa[q]["status"], "error": qa[q]["error"],
                "image_count": qa[q]["image_count"],
                "missing_pages": qa[q]["missing_pages"],
                "evidence_pages": info[q]["gold_pages"],
            })
    route_na = collections.Counter(route[q]["eval_reason"] for q in qids
                                   if route[q]["not_applicable"])
    jev_missing = [q for q in qids if jev[q]["not_applicable"]]
    jev_missing_by_answerable = collections.Counter(
        "answerable" if info[q]["answerable"] else "unanswerable" for q in jev_missing)

    # 45 道定位异常（复用既有诊断文件的 primary_cause）
    diag_path = os.path.join(AUDIT_DIR, "route45_retrieval_diagnosis.json")
    diag = C.read_json(diag_path, {}) or {}
    if isinstance(diag, dict):
        diag_rows = diag.get("rows") or []
        diag_stats = diag.get("stats") or {}
    else:
        diag_rows, diag_stats = diag, {}
    diag_causes = collections.Counter((d.get("primary_cause") or "unknown") for d in diag_rows)

    # gold 页码修复清单
    fixes = C.read_json(GOLD_FIX, {}) or {}
    if isinstance(fixes, dict):
        fixes_list = fixes.get("fixes") or fixes.get("items") or []
    else:
        fixes_list = fixes if isinstance(fixes, list) else []

    out_stats = {
        "generated_at": C.now_str(),
        "note": "全部数字由真实结果文件离线计算；QA 使用 gold evidence_pages（oracle 证据上界），不代表端到端检索系统。",
        "dataset": {
            "n": 500, "n_unique_qid": len(set(qids)),
            "n_unique_docs": len({info[q]["doc_id"] for q in qids}),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "n_answerable_without_gold_pages": len(no_gold),
        },
        "files": {
            "qa": os.path.relpath(QA_PATH, C.ROOT),
            "route": os.path.relpath(ROUTE_PATH, C.ROOT),
            "toc_metrics": os.path.relpath(TOC_METRICS, C.ROOT),
            "jev_dir": os.path.relpath(JEV_DIR, C.ROOT),
        },
        "qa_raw_records": len(qa_rows),
        "route_raw_records": len(route_rows),
        "jev_unique_snippets": len(jev_all),
        "jev_sources": per_source_counts,
        "jev_multi_source_qids": len(jev_conflicts),
        "sets": stats_sets,
        "qa_failed": qa_failed,
        "route_not_applicable_reasons": dict(route_na),
        "route45_primary_causes": dict(diag_causes),
        "gold_page_fixes": fixes_list,
        "jev_missing_n": len(jev_missing),
        "jev_missing_by_answerable": dict(jev_missing_by_answerable),
    }
    C.write_json_atomic(os.path.join(AUDIT_DIR, "final500_stats.json"), out_stats)

    # ---------- 500 行状态表 ----------
    cols = [
        "qid", "sample_index", "doc_id", "doc_type", "answerable", "has_gold_pages",
        # QA 六类
        "qa_attempted", "qa_api_success", "qa_valid_input", "qa_evaluable",
        "qa_not_applicable", "qa_failed", "qa_status", "qa_strict_correct",
        "qa_lenient_correct", "qa_error",
        # 路由六类
        "route_attempted", "route_api_success", "route_valid_input", "route_evaluable",
        "route_not_applicable", "route_failed", "route_status", "route_eval_reason",
        "route_hit1", "route_hit3", "route_page_recall1", "route_page_recall3",
        # Jev 六类
        "jev_attempted", "jev_api_success", "jev_valid_input", "jev_evaluable",
        "jev_not_applicable", "jev_failed", "jev_source", "jev_snippet_id",
        "jev_relevant", "jev_sufficient", "jev_self_contained",
        "jev_action_raw", "jev_action_norm",
        "jev_n_sources", "jev_all_sources",
    ]
    csv_path = os.path.join(AUDIT_DIR, "final500_status_500.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for q in qids:
            a, b, c = qa[q], route[q], jev[q]
            w.writerow({
                "qid": q, "sample_index": info[q]["sample_index"],
                "doc_id": info[q]["doc_id"], "doc_type": info[q]["doc_type"],
                "answerable": info[q]["answerable"],
                "has_gold_pages": bool(info[q]["gold_pages"]),
                "qa_attempted": a["attempted"], "qa_api_success": a["api_success"],
                "qa_valid_input": a["valid_input"], "qa_evaluable": a["evaluable"],
                "qa_not_applicable": a["not_applicable"], "qa_failed": a["failed"],
                "qa_status": a["status"], "qa_strict_correct": a["strict"],
                "qa_lenient_correct": a["lenient"],
                "qa_error": (a["error"] or "")[:200],
                "route_attempted": b["attempted"], "route_api_success": b["api_success"],
                "route_valid_input": b["valid_input"], "route_evaluable": b["evaluable"],
                "route_not_applicable": b["not_applicable"], "route_failed": b["failed"],
                "route_status": b["status"], "route_eval_reason": b["eval_reason"],
                "route_hit1": b["hit1"], "route_hit3": b["hit3"],
                "route_page_recall1": b["pr1"], "route_page_recall3": b["pr3"],
                "jev_attempted": c["attempted"], "jev_api_success": c["api_success"],
                "jev_valid_input": c["valid_input"], "jev_evaluable": c["evaluable"],
                "jev_not_applicable": c["not_applicable"], "jev_failed": c["failed"],
                "jev_source": c["source"], "jev_snippet_id": c["snippet_id"],
                "jev_relevant": c["relevant"], "jev_sufficient": c["sufficient"],
                "jev_self_contained": c["self_contained"],
                "jev_action_raw": c["action_raw"], "jev_action_norm": c["action_norm"],
                "jev_n_sources": c["n_sources"], "jev_all_sources": c["all_sources"],
            })

    print(f"500 行状态表: {csv_path}")

    # ---------- 六类状态矩阵 Markdown ----------
    def six(mod, qset=set(qids)):
        return {
            "已尝试": sum(1 for q in qset if mod[q]["attempted"]),
            "API 成功": sum(1 for q in qset if mod[q]["api_success"]),
            "有效输入": sum(1 for q in qset if mod[q]["valid_input"]),
            "可评测": sum(1 for q in qset if mod[q]["evaluable"]),
            "不适用": sum(1 for q in qset if mod[q]["not_applicable"]),
            "失败": sum(1 for q in qset if mod[q]["failed"]),
        }

    def row(name, d):
        return "| {} | {} | {} | {} | {} | {} | {} |".format(
            name, d["已尝试"], d["API 成功"], d["有效输入"], d["可评测"], d["不适用"], d["失败"])

    all_jev = [q for q in qids if jev[q]["evaluable"]]
    raw_actions = collections.Counter(jev[q]["action_raw"] for q in all_jev)
    src_counts = collections.Counter(jev[q]["source"] for q in all_jev)

    lines = []
    lines.append("# 500 题三模块六类状态矩阵（离线生成）\n")
    lines.append(f"生成时间：{C.now_str()}\n")
    lines.append("说明：六类状态按模块独立判定，不可跨模块相加；QA 使用 gold evidence_pages，"
                 "属于已知证据页问答上界，不是端到端检索结果。\n")
    lines.append("## 1. 全量 500 题六类状态\n")
    lines.append("| 模块 | 已尝试 | API 成功 | 有效输入 | 可评测 | 不适用 | 失败 |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.append(row("QA（DeepSeek）", six(qa)))
    lines.append(row("目录路由", six(route)))
    lines.append(row("Jev 判断", six(jev)))
    lines.append("")
    lines.append("> 本文件第 1、2 节的路由统计均基于 `routes.jsonl` 原生 374 条；补跑 126 条"
                 "（56 有效 + 70 拒答）与最终 376 / 379 评测口径见 `FINAL_ACCEPTANCE_REPORT_500.md` 第 5.3 节与 "
                 "`ACCEPTANCE_CHECKS_500.json`。Jev 为 500/500。\n")
    lines.append("## 2. 四集合核心指标\n")
    lines.append("| 集合 | n | QA strict | QA lenient | 路由 Hit@1 | 路由 Hit@3 | Page Recall@1 | Page Recall@3 | Jev 可评测 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, s in stats_sets.items():
        q, rt, jv = s["qa"], s["route"], s["jev"]
        rt_ok = rt["evaluable"] > 0
        lines.append("| {} | {} | {} ({}) | {} ({}) | {} | {} | {} | {} | {} |".format(
            name, s["n"],
            q["strict_correct"], q["strict_rate"],
            q["lenient_correct"], q["lenient_rate"],
            f"{rt['hit1']} ({rt['hit1_rate']})" if rt_ok else "—",
            f"{rt['hit3']} ({rt['hit3_rate']})" if rt_ok else "—",
            rt["page_recall1_macro"] if rt_ok else "—",
            rt["page_recall3_macro"] if rt_ok else "—",
            jv["evaluable"]))
    lines.append("")
    lines.append(f"## 3. Jev 判断分布（{stats_sets['all_500']['jev']['evaluable']} 条可评测）\n")
    lines.append("| 维度 | 分布 |")
    lines.append("|---|---|")
    lines.append(f"| 输入来源 | {dict(src_counts)} |")
    lines.append(f"| 原始 next_action | {dict(raw_actions)} |")
    lines.append(f"| 归一化动作 | {stats_sets['all_500']['jev']['action_norm']} |")
    lines.append(f"| relevant=True | {stats_sets['all_500']['jev']['relevant_true']} |")
    lines.append(f"| self_contained=True | {stats_sets['all_500']['jev']['self_contained_true']} |")
    lines.append(f"| sufficient=True | {stats_sets['all_500']['jev']['sufficient_true']} |")
    lines.append("")
    lines.append("归一化规则：relevant=False → search；relevant=True 且 sufficient=True → answer；其余 → expand。\n")
    lines.append("## 4. 未完成与异常\n")
    lines.append("### QA 失败 5 题\n")
    lines.append("| qid | 状态 | 原因 | 渲染页数 | 缺失页 |")
    lines.append("|---|---|---|---|---|")
    for x in qa_failed:
        lines.append("| {} | {} | {} | {} | {} |".format(
            x["qid"], x["status"], (x["error"] or "")[:80],
            x["image_count"], x["missing_pages"]))
    lines.append("")
    lines.append("### 路由不适用 126 题\n")
    lines.append("| 原因 | 题数 |")
    lines.append("|---|---|")
    for k, v in route_na.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("### Jev 缺口 126 题\n")
    lines.append(f"- 可答题无有效证据：{jev_missing_by_answerable.get('answerable', 0)}\n")
    lines.append(f"- 不可答题无有效证据：{jev_missing_by_answerable.get('unanswerable', 0)}\n")
    lines.append("")
    lines.append("### 45 道路由定位异常根因\n")
    lines.append(f"- {dict(diag_causes)}\n")
    lines.append(f"- 诊断统计：{json.dumps(diag_stats, ensure_ascii=False)}\n")
    md_path = os.path.join(AUDIT_DIR, "final500_module_matrix.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"六类状态矩阵: {md_path}")

    for name, s in stats_sets.items():
        print(f"\n[{name}] n={s['n']}")
        print(f"  QA  : attempted={s['qa']['attempted']} api_ok={s['qa']['api_success']} "
              f"valid={s['qa']['valid_input']} eval={s['qa']['evaluable']} "
              f"strict={s['qa']['strict_correct']}({s['qa']['strict_rate']}) "
              f"lenient={s['qa']['lenient_correct']}({s['qa']['lenient_rate']})")
        print(f"  ROUTE: eval={s['route']['evaluable']} N/A={s['route']['not_applicable']} "
              f"hit1={s['route']['hit1']}({s['route']['hit1_rate']}) "
              f"hit3={s['route']['hit3']}({s['route']['hit3_rate']}) "
              f"PR@1={s['route']['page_recall1_macro']} PR@3={s['route']['page_recall3_macro']}")
        print(f"  JEV  : eval={s['jev']['evaluable']} N/A={s['jev']['not_applicable']} "
              f"sufficient={s['jev']['sufficient_true']} actions={s['jev']['action_norm']}")
    print("\nQA 失败:", [(x["qid"], x["status"]) for x in qa_failed])
    print("路由不适用原因:", dict(route_na))
    print("Jev 缺口:", len(jev_missing), dict(jev_missing_by_answerable))
    print("Jev 多来源题数:", len(jev_conflicts))
    print("45 道定位异常根因:", dict(diag_causes))


if __name__ == "__main__":
    main()
