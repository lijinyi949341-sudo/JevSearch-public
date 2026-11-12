# -*- coding: utf-8 -*-
"""
JevSearch 500 题实验总控脚本。

目标：
  - 用固定 500 题清单（Q0001..Q0500，按 sampled_500_for_baseline.jsonl 行号）
    依次运行 DeepSeek 问答、目录路由、Jev 判断；
  - 自动分批、断点续跑、成功后去重、失败重试、每批落盘、自动汇总与 Markdown 报告；
  - 不修改任何模块的原始预测，只在独立目录 outputs/final500/run500/ 下追加。

设计约定：
  * 三个模块共用同一 500 题清单，qid 不重新抽样、不重新排序；
  * 问答：gold evidence_pages；空证据页回退第 1 页；
  * 路由：不可答题不调用 API，记 unanswerable_skipped（N/A，不是失败）；
  * Jev：500 题全部参与（含 119 道不可答），片段为 title 选页，单独统计误判；
  * 已有 stage50 第一批结果会“复制”进 run500 结果文件，避免重复付费；
    原文件保持不变。

用法：
  # 1) 无付费预检：构建清单/片段，统计剩余题数、已有成功、跳过规则与路径
  py -3.12 run_experiment_500.py --preflight --build

  # 2) 无付费自检：模拟断点恢复、去重与批次状态
  py -3.12 run_experiment_500.py --selftest

  # 3) 正式运行剩余题（需要 DASHSCOPE_API_KEY 与 TYPESAFE_API_KEY）
  py -3.12 run_experiment_500.py --run

  # 只跑前 N 批（调试）
  py -3.12 run_experiment_500.py --run --max-batches 2
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

import final500_common as C

ROOT = C.ROOT
RUN_DIR = os.path.join(C.OUT_ROOT, "run500")
QA_INPUT = os.path.join(RUN_DIR, "qa_input_500.jsonl")
QA_OUT = os.path.join(RUN_DIR, "qa", "ds_results.jsonl")
QA_SUMMARY = os.path.join(RUN_DIR, "qa", "run_summary.json")
ROUTE_OUT = os.path.join(RUN_DIR, "toc", "routes.jsonl")
ROUTE_METRICS = os.path.join(RUN_DIR, "toc", "metrics.json")
JEV_SNIPPETS = os.path.join(RUN_DIR, "jev", "snippets_title_500.jsonl")
JEV_OUT = os.path.join(RUN_DIR, "jev", "jev_predictions_title_500.jsonl")
MANIFEST = os.path.join(RUN_DIR, "manifest.json")
QIDS_TXT = os.path.join(RUN_DIR, "qids_500.txt")
STATE = os.path.join(RUN_DIR, "state.json")
REPORT_DIR = os.path.join(RUN_DIR, "reports")
BATCH_DIR = os.path.join(RUN_DIR, "batches")

STAGE50 = os.path.join(C.OUT_ROOT, "stage50")
SEED_FILES = {
    "qa": os.path.join(STAGE50, "qa", "ds_results.jsonl"),
    "route": os.path.join(STAGE50, "toc", "routes.jsonl"),
    "jev": os.path.join(STAGE50, "jev", "jev_predictions_title_50.jsonl"),
}

ROUTE_DONE = {"ok", "invalid_json", "no_valid_section",
              "unanswerable_skipped", "missing_gold_pages_skipped",
              "no_toc_skipped", "invalid_gold_pages_skipped"}

# 实测单价（与第一批报告一致）
DS_IN_BUSY, DS_OUT_BUSY = 2.0, 8.0          # 元 / 百万 token
JEV_IN_USD, USD_CNY = 0.042, 7.1            # 美元 / 百万 token


# ---------------------------------------------------------------- 通用
def load_compare_module():
    path = os.path.join(ROOT, "compare_500.py")
    spec = importlib.util.spec_from_file_location("compare500_orch", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_task6():
    import task6_route_and_eval_500 as T6
    return T6


def load_task5():
    import task5_jev_500 as T5
    return T5


def samples_and_qids():
    samples = C.load_samples()
    return samples, [s["qid"] for s in samples]


def chunked(qids, size):
    return [qids[i:i + size] for i in range(0, len(qids), size)]


def last_of(path, key="qid"):
    rows = C.load_jsonl(path)
    out = {}
    for r in rows:
        k = r.get(key)
        if k:
            out[k] = r
    return out


def write_text(path, text):
    C.ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def usage_tokens(rec):
    u = rec.get("usage") or {}
    return int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def run_cmd(args):
    print("\n[执行] " + " ".join(args))
    t0 = time.time()
    proc = subprocess.run([sys.executable] + args, cwd=ROOT)
    print(f"[退出码] {proc.returncode}  用时 {time.time() - t0:.1f}s")
    return proc.returncode


# ---------------------------------------------------------------- 产物构建
def build_manifest_and_input():
    samples, qids = samples_and_qids()
    C.ensure_dir(RUN_DIR)
    write_text(QIDS_TXT, "\n".join(qids) + "\n")
    qa_tmp = QA_INPUT + ".tmp"
    with open(qa_tmp, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    os.replace(qa_tmp, QA_INPUT)
    C.write_json_atomic(MANIFEST, {
        "generated_at": C.now_str(),
        "source": C.SAMPLES,
        "qid_scheme": "Q0001..Q0500，按采样文件物理行号 1-based",
        "total": len(qids),
        "qids": qids,
        "unanswerable": sum(1 for s in samples
                            if str(s.get("answer", "")).strip() == "Not answerable"),
        "modules": {
            "qa": "gold evidence_pages；空证据页回退第 1 页",
            "route": "Question + TOC；不可答题记 N/A，不调用",
            "jev": "Question + title 片段；500 题全部参与，含不可答题",
        },
    })
    return samples, qids


def build_jev_snippets():
    """构建 500 题 title 片段；不可答题同样选页并抽取（不提前过滤）。"""
    T5 = load_task5()
    T6 = load_task6()
    samples, _ = samples_and_qids()
    tocs = T5.load_tocs()
    ocr = T5.load_ocr()
    tmp = JEV_SNIPPETS + ".tmp"
    C.ensure_dir(os.path.dirname(JEV_SNIPPETS))
    counts = {"ok": 0, "not_ok": 0}
    with open(tmp, "w", encoding="utf-8") as f:
        for s in samples:
            gold = C.parse_pages(s.get("evidence_pages", "[]"))
            doc = tocs.get(s["doc_id"])
            reason, status, page, section_id = "", "ok", None, None
            if doc is None:
                status, reason = "no_toc", "目录不可用"
            else:
                ranked = T5.base6.baseline_rank(s["question"], doc["sections"])
                if not ranked:
                    status, reason = "no_title_match", "目录标题与问题无词重合"
                else:
                    page = ranked[0]["start_page"]
                    section_id = ranked[0]["section_id"]
            rec = {
                "snippet_id": f"{s['qid']}__title",
                "qid": s["qid"], "sample_index": s["sample_index"],
                "doc_id": s["doc_id"], "doc_type": s.get("doc_type", ""),
                "question": s["question"], "mode": "title",
                "page": page, "section_id": section_id,
                "selection_uses_gold": False,
                "selection_rule": "目录标题词匹配 top-1 章节首页（真实检索，无 LLM，不使用 gold）；"
                                  "不可答题目同样参与，用于统计是否误判 sufficient=True",
                "status": status, "reason": reason,
                "evidence_snippet": "", "char_len": 0,
                "extraction_method": "", "has_page_images": False,
                "truncated": False,
                "evidence_type_primary": T5.primary_type(
                    C.parse_sources(s.get("evidence_sources", "[]"))),
                "cross_page": len(gold) > 1,
                "created_at": C.now_str(),
                "offline_only": {
                    "answer": s.get("answer", ""),
                    "gold_answer": s.get("answer", ""),
                    "evidence_pages": s.get("evidence_pages", "[]"),
                    "evidence_sources": s.get("evidence_sources", ""),
                },
            }
            if status == "ok":
                text, method, has_images, truncated = T5.extract_page_text(
                    s["doc_id"], page, ocr)
                if text:
                    rec.update({"evidence_snippet": text, "char_len": len(text),
                                "extraction_method": method,
                                "has_page_images": has_images,
                                "truncated": truncated})
                else:
                    rec.update({"status": "no_snippet", "reason": method,
                                "has_page_images": has_images})
            counts["ok" if rec["status"] == "ok" else "not_ok"] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, JEV_SNIPPETS)
    print(f"[片段] 500 题：可用 {counts['ok']}，不可用 {counts['not_ok']} -> {JEV_SNIPPETS}")
    return counts


def seed_from_stage50():
    """把第一批真实结果复制进 run500 结果文件；只追加缺失 qid，原文件不改。"""
    copies = {"qa": (SEED_FILES["qa"], QA_OUT, "qid"),
              "route": (SEED_FILES["route"], ROUTE_OUT, "qid"),
              "jev": (SEED_FILES["jev"], JEV_OUT, "snippet_id")}
    total = 0
    for name, (src, dst, key) in copies.items():
        if not os.path.exists(src):
            continue
        existing = last_of(dst, key)
        before = len(C.load_jsonl(dst))
        n = 0
        for r in C.load_jsonl(src):
            k = r.get(key)
            if k and k not in existing:
                C.append_jsonl(dst, r)
                existing[k] = r
                n += 1
        total += n
        print(f"[seed] {name}: 复制 {n} 条，当前文件 {before + n} 条"
              f"（{os.path.basename(src)} -> {os.path.basename(dst)}）")
    if total:
        print(f"[seed] 合计复制 {total} 条，已成功记录不会被再次调用")
    return total


# ---------------------------------------------------------------- 状态计算
def qa_last():
    return last_of(QA_OUT, "qid")


def route_last():
    return last_of(ROUTE_OUT, "qid")


def jev_last():
    return last_of(JEV_OUT, "snippet_id")


def snippet_ok_map():
    return {r["qid"]: r for r in C.load_jsonl(JEV_SNIPPETS)
            if r.get("status") == "ok"}


def remaining(qids, samples_by_qid, tocs=None):
    """返回各模块尚未完成的题号。路由按可评测性区分 N/A。"""
    qmap = qa_last()
    rmap = route_last()
    jmap = jev_last()
    sok = snippet_ok_map() if os.path.exists(JEV_SNIPPETS) else {}

    qa_missing = [q for q in qids
                  if not (qmap.get(q) or {}).get("ds_success")]

    route_missing, route_na = [], []
    if tocs is not None:
        T6 = load_task6()
        for q in qids:
            s = samples_by_qid[q]
            status, elig, _ = T6.classify(s, tocs)
            prev = rmap.get(q)
            if not elig:
                route_na.append(q)
            elif prev is None or prev.get("status") not in ROUTE_DONE:
                route_missing.append(q)
    else:
        route_missing = []

    jev_missing = []
    jev_input_missing = []
    for q in qids:
        if q not in sok:
            jev_input_missing.append(q)
        elif not (jmap.get(q + "__title") or {}).get("success"):
            jev_missing.append(q)

    return {
        "qa": qa_missing,
        "route": route_missing,
        "route_na": route_na,
        "jev": jev_missing,
        "jev_input_missing": jev_input_missing,
    }


def batch_status(qids, samples_by_qid, tocs):
    rem = remaining(qids, samples_by_qid, tocs)
    flags = {
        "qa": bool(rem["qa"]),
        "route": bool(rem["route"]),
        # Jev 片段不可用属于检索输入缺失（N/A），不阻塞批次完成，但单独计入覆盖率
        "jev": bool(rem["jev"]),
    }
    return {
        "qids": qids,
        "remaining": rem,
        "needs_run": [k for k, v in flags.items() if v],
        "complete": not any(flags.values()),
    }


# ---------------------------------------------------------------- 预检
def preflight(build=False):
    samples, qids = build_manifest_and_input()
    samples_by_qid = {s["qid"]: s for s in samples}
    T6 = load_task6()
    tocs = T6.load_tocs()

    if build or not os.path.exists(JEV_SNIPPETS):
        build_jev_snippets()
    seed_from_stage50()

    rem = remaining(qids, samples_by_qid, tocs)
    qa_map, route_map, jev_map = qa_last(), route_last(), jev_last()
    sok = snippet_ok_map()

    qa_success = sum(1 for q in qids if (qa_map.get(q) or {}).get("ds_success"))
    route_ok = sum(1 for q in qids
                   if (route_map.get(q) or {}).get("status") in ROUTE_DONE
                   and (route_map.get(q) or {}).get("status") != "unanswerable_skipped")
    route_na = sum(1 for q in qids
                   if (route_map.get(q) or {}).get("status") == "unanswerable_skipped")
    jev_success = sum(1 for q in qids
                      if (jev_map.get(q + "__title") or {}).get("success"))

    # qid 完整性核对
    def unexplained(path, key, expected):
        rows = C.load_jsonl(path)
        got = {r.get(key) for r in rows if r.get(key)}
        return sorted(got - set(expected)), sorted(set(expected) - got)

    qa_extra, qa_missing_rows = unexplained(QA_OUT, "qid", qids)
    route_extra, route_missing_rows = unexplained(ROUTE_OUT, "qid", qids)
    jev_extra, jev_missing_rows = unexplained(JEV_OUT, "snippet_id",
                                             [q + "__title" for q in qids])

    batches = chunked(qids, 10)
    bstat = [batch_status(b, samples_by_qid, tocs) for b in batches]

    n_calls = len(rem["qa"]) + len(rem["route"]) + len(rem["jev"])
    est = estimate_cost()

    print("\n=== PRE-FLIGHT（未调用任何付费 API） ===")
    print(f"固定清单: {len(qids)} 题  -> {QIDS_TXT}")
    print(f"QA 输入: {len(C.load_jsonl(QA_INPUT))} 条 -> {QA_INPUT}")
    print(f"Jev 片段: 可用 {len(sok)}，不可用 {len(C.load_jsonl(JEV_SNIPPETS)) - len(sok)} -> {JEV_SNIPPETS}")
    print(f"分批: {len(batches)} 批 × 最多 10 题")
    print(f"问答: 已成功 {qa_success}/{len(qids)}，剩余 {len(rem['qa'])}")
    print(f"路由: 已完成 {route_ok}，N/A {route_na}，剩余 {len(rem['route'])}")
    print(f"Jev: 已成功 {jev_success}/{len(sok)} 可运行片段，剩余 {len(rem['jev'])}，"
          f"输入不可用 {len(rem['jev_input_missing'])}")
    print(f"预计新增请求: {n_calls}（问答 {len(rem['qa'])} + 路由 {len(rem['route'])} + Jev {len(rem['jev'])}）")
    print(f"预计新增费用: 忙时 {est['busy']:.4f} 元，闲时 {est['idle']:.4f} 元")
    print(f"qid 核对: 问答 extra={len(qa_extra)} missing={len(qa_missing_rows)}"
          f"（missing 示例 {qa_missing_rows[:5]}）")
    print(f"          路由 extra={len(route_extra)} missing={len(route_missing_rows)}"
          f"（missing 示例 {route_missing_rows[:5]}）")
    print(f"          Jev  extra={len(jev_extra)} missing={len(jev_missing_rows)}"
          f"（missing 示例 {jev_missing_rows[:5]}）")
    print(f"完成批次: {sum(1 for b in bstat if b['complete'])}/{len(batches)}")
    print(f"结果路径: {RUN_DIR}")
    print("跳过规则: 问答 ds_success=True；路由 status∈完成集合（含 N/A）；"
          "Jev success=True；均已成功的题目不再发请求。")

    pre = {
        "generated_at": C.now_str(),
        "total": len(qids),
        "qa_input": len(C.load_jsonl(QA_INPUT)),
        "jev_snippets_ok": len(sok),
        "jev_snippets_not_ok": len(C.load_jsonl(JEV_SNIPPETS)) - len(sok),
        "existing": {"qa_success": qa_success, "route_done": route_ok,
                     "route_na": route_na, "jev_success": jev_success},
        "remaining": {k: len(v) if isinstance(v, list) else v
                      for k, v in rem.items()},
        "estimated_requests": n_calls,
        "estimated_cost_cny": est,
        "qid_alignment": {"qa_extra": qa_extra, "qa_missing": qa_missing_rows,
                          "route_extra": route_extra, "route_missing": route_missing_rows,
                          "jev_extra": jev_extra, "jev_missing": jev_missing_rows},
        "batches_complete": sum(1 for b in bstat if b["complete"]),
        "paths": {"qa_input": QA_INPUT, "qa_out": QA_OUT, "route_out": ROUTE_OUT,
                  "route_metrics": ROUTE_METRICS, "jev_snippets": JEV_SNIPPETS,
                  "jev_out": JEV_OUT, "manifest": MANIFEST, "state": STATE},
    }
    C.write_json_atomic(os.path.join(RUN_DIR, "preflight.json"), pre)
    return pre


def sum_tokens(path, module):
    last = last_of(path, "snippet_id" if module == "jev" else "qid")
    pin = pout = 0
    for r in last.values():
        if module == "jev":
            pin += int(r.get("input_tokens") or 0)
            pout += int(r.get("output_tokens") or 0)
        else:
            a, b = usage_tokens(r)
            pin += a
            pout += b
    return pin, pout


def estimate_cost():
    """按已跑题目的平均 token 估算剩余费用；无样本时用第一批实测均值。"""
    qa_p, qa_c = sum_tokens(QA_OUT, "qa")
    n_qa = sum(1 for r in qa_last().values() if r.get("ds_success"))
    qa_p = qa_p / n_qa if n_qa else 1310.4
    qa_c = qa_c / n_qa if n_qa else 51.0
    rt_p, rt_c = sum_tokens(ROUTE_OUT, "route")
    n_rt = sum(1 for r in route_last().values() if r.get("status") == "ok")
    rt_p = rt_p / n_rt if n_rt else 1958.0
    rt_c = rt_c / n_rt if n_rt else 24.5
    jv_p, _ = sum_tokens(JEV_OUT, "jev")
    n_jv = sum(1 for r in jev_last().values() if r.get("success"))
    jv_p = jv_p / n_jv if n_jv else 1349.5

    samples, qids = samples_and_qids()
    samples_by_qid = {s["qid"]: s for s in samples}
    tocs = load_task6().load_tocs()
    rem = remaining(qids, samples_by_qid, tocs)
    n_q, n_r, n_j = len(rem["qa"]), len(rem["route"]), len(rem["jev"])
    busy = (n_q * (qa_p * DS_IN_BUSY + qa_c * DS_OUT_BUSY) / 1e6
            + n_r * (rt_p * DS_IN_BUSY + rt_c * DS_OUT_BUSY) / 1e6
            + n_j * jv_p * JEV_IN_USD * USD_CNY / 1e6)
    return {"busy": round(busy, 4), "idle": round(busy / 2, 4),
            "requests": {"qa": n_q, "route": n_r, "jev": n_j},
            "avg_tokens": {"qa": [round(qa_p, 1), round(qa_c, 1)],
                           "route": [round(rt_p, 1), round(rt_c, 1)],
                           "jev_input": round(jv_p, 1)}}


# ---------------------------------------------------------------- 正式运行
def run(max_batches=0, batch_size=10, max_rounds=2):
    for key in ("DASHSCOPE_API_KEY", "TYPESAFE_API_KEY"):
        if not os.environ.get(key):
            print(f"[停止] 未检测到 {key}，未发出任何请求。")
            return 2

    preflight(build=False)
    samples, qids = samples_and_qids()
    samples_by_qid = {s["qid"]: s for s in samples}
    tocs = load_task6().load_tocs()
    batches = chunked(qids, batch_size)
    if max_batches:
        batches = batches[:max_batches]

    state = {"started_at": C.now_str(), "batch_size": batch_size,
             "total_batches": len(chunked(qids, batch_size)), "batches": {}}

    for rnd in range(1, max_rounds + 1):
        print(f"\n########## 第 {rnd}/{max_rounds} 轮 ##########")
        progressed = False
        for bi, bq in enumerate(batches, 1):
            st = batch_status(bq, samples_by_qid, tocs)
            if st["complete"]:
                write_batch_report(bi, bq, samples_by_qid, tocs, state)
                continue
            print(f"\n=== 批 {bi}/{len(batches)}：{bq[0]}..{bq[-1]}，"
                  f"待跑 {st['needs_run']} ===")
            idx = [samples_by_qid[q]["sample_index"] for q in bq]

            if st["remaining"]["qa"]:
                run_cmd(["run_advanced_rag_ds_500.py",
                         "--input", QA_INPUT, "--out", QA_OUT,
                         "--summary-out", QA_SUMMARY,
                         "--start", str(min(idx)), "--end", str(max(idx)),
                         "--limit", "0"])
                progressed = True

            if st["remaining"]["route"]:
                run_cmd(["task6_route_and_eval_500.py",
                         "--qids", ",".join(bq),
                         "--out", ROUTE_OUT, "--eval-out", ROUTE_METRICS])
                progressed = True

            if st["remaining"]["jev"]:
                run_cmd(["task5_jev_500.py", "--run", "--mode", "title",
                         "--limit", "0", "--qids", ",".join(bq),
                         "--snippets", JEV_SNIPPETS, "--out", JEV_OUT])
                progressed = True

            state["batches"][str(bi)] = {
                "qids": bq,
                "updated_at": C.now_str(),
                "round": rnd,
                "status": batch_status(bq, samples_by_qid, tocs),
            }
            C.write_json_atomic(STATE, state)
            write_batch_report(bi, bq, samples_by_qid, tocs, state)

        if not progressed:
            break

    # 全量指标：路由用已有预测重算，不调用 API
    run_cmd(["task6_route_and_eval_500.py", "--reuse",
             "--qids", ",".join(qids),
             "--out", ROUTE_OUT, "--eval-out", ROUTE_METRICS])

    write_full_report(qids, samples_by_qid, tocs, state)
    print("\n全部批次结束。全量报告: " + os.path.join(REPORT_DIR, "FULL_REPORT.md"))
    return 0


# ---------------------------------------------------------------- 统计与报告
def module_stats(qids, samples_by_qid):
    CMP = load_compare_module()
    qa_map, route_map, jev_map = qa_last(), route_last(), jev_last()
    sok = snippet_ok_map()

    qa_recs = [qa_map[q] for q in qids if (qa_map.get(q) or {}).get("ds_success")]
    qa_rows = [(str(samples_by_qid[q].get("answer", "")), r.get("ds_answer", ""))
               for q in qids if (qa_map.get(q) or {}).get("ds_success")
               for r in [qa_map[q]]]
    ans_rows = [(str(samples_by_qid[q].get("answer", "")), qa_map[q].get("ds_answer", ""))
                for q in qids if q in qa_map and qa_map[q].get("ds_success")
                and str(samples_by_qid[q].get("answer", "")).strip() != "Not answerable"]
    un_rows = [(str(samples_by_qid[q].get("answer", "")), qa_map[q].get("ds_answer", ""))
               for q in qids if q in qa_map and qa_map[q].get("ds_success")
               and str(samples_by_qid[q].get("answer", "")).strip() == "Not answerable"]
    qa_acc_all = CMP.acc(qa_rows) if qa_rows else {"n": 0}
    qa_acc_ans = CMP.acc(ans_rows) if ans_rows else {"n": 0}
    qa_reject = sum(1 for g, p in un_rows
                    if "not answerable" in str(p).strip().lower())

    qa_tokens = [usage_tokens(r) for r in qa_recs]
    qa_lat = sum(float(r.get("latency_sec") or 0) for r in qa_recs)

    route_recs = [route_map[q] for q in qids if q in route_map]
    route_ok = [r for r in route_recs if r.get("status") == "ok"]
    route_na = [r for r in route_recs if r.get("status") == "unanswerable_skipped"]
    rt_tokens = [usage_tokens(r) for r in route_ok]
    rt_lat = sum(float(r.get("latency_sec") or 0) for r in route_ok)

    jev_recs = [jev_map[q + "__title"] for q in qids
                if (jev_map.get(q + "__title") or {}).get("success")]
    jv_in = sum(int(r.get("input_tokens") or 0) for r in jev_recs)
    jv_out = sum(int(r.get("output_tokens") or 0) for r in jev_recs)
    jv_lat = sum(float(r.get("latency_sec") or 0) for r in jev_recs)
    raw = {}
    norm = {}
    for r in jev_recs:
        a = r.get("next_action")
        raw[a] = raw.get(a, 0) + 1
        n = ("search" if not r.get("relevant")
             else ("answer" if r.get("sufficient") else "expand"))
        norm[n] = norm.get(n, 0) + 1
    un_jev = [r for r in jev_recs
              if str(samples_by_qid.get(r.get("qid"), {}).get("answer", "")).strip()
              == "Not answerable"]
    un_false_suff = sum(1 for r in un_jev if r.get("sufficient"))

    pin = sum(a for a, _ in qa_tokens) + sum(a for a, _ in rt_tokens) + jv_in
    pout = sum(b for _, b in qa_tokens) + sum(b for _, b in rt_tokens) + jv_out
    busy = (sum(a for a, _ in qa_tokens) * DS_IN_BUSY
            + sum(b for _, b in qa_tokens) * DS_OUT_BUSY
            + sum(a for a, _ in rt_tokens) * DS_IN_BUSY
            + sum(b for _, b in rt_tokens) * DS_OUT_BUSY) / 1e6 \
        + jv_in * JEV_IN_USD * USD_CNY / 1e6

    metrics = C.read_json(ROUTE_METRICS, default={}) or {}
    rsum = (metrics.get("summary") or {}) if metrics else {}
    llm = rsum.get("llm_all_eligible") or {}

    return {
        "qa": {"success": len(qa_recs), "planned": len(qids),
               "accuracy_all": qa_acc_all, "accuracy_answerable": qa_acc_ans,
               "unanswerable_n": len(un_rows), "reject_correct": qa_reject,
               "input_tokens": sum(a for a, _ in qa_tokens),
               "output_tokens": sum(b for _, b in qa_tokens),
               "latency_sec": round(qa_lat, 2)},
        "route": {"ok": len(route_ok), "na": len(route_na),
                  "planned_answerable": rsum.get("expected_eligible"),
                  "hit@1": llm.get("section_hit@1"), "hit@3": llm.get("section_hit@3"),
                  "page_recall@1": llm.get("page_recall@1_macro"),
                  "page_recall@3": llm.get("page_recall@3_macro"),
                  "input_tokens": sum(a for a, _ in rt_tokens),
                  "output_tokens": sum(b for _, b in rt_tokens),
                  "latency_sec": round(rt_lat, 2)},
        "jev": {"success": len(jev_recs), "snippets_ok": len(sok),
                "input_missing": len(qids) - len(sok),
                "raw_next_action": raw, "normalized_action": norm,
                "relevant_true": sum(1 for r in jev_recs if r.get("relevant")),
                "self_contained_true": sum(1 for r in jev_recs if r.get("self_contained")),
                "sufficient_true": sum(1 for r in jev_recs if r.get("sufficient")),
                "unanswerable_n": len(un_jev),
                "unanswerable_false_sufficient": un_false_suff,
                "accuracy": "不可计算（无人工 confirmed 标注）",
                "input_tokens": jv_in, "output_tokens": jv_out,
                "latency_sec": round(jv_lat, 2)},
        "tokens_total": [pin, pout],
        "cost_cny_busy": round(busy, 4),
        "cost_cny_idle": round(busy / 2, 4),
    }


def write_batch_report(bi, bq, samples_by_qid, tocs, state):
    st = batch_status(bq, samples_by_qid, tocs)
    stats = module_stats(bq, samples_by_qid)
    lines = [
        f"# 第 {bi} 批实验报告（{bq[0]}..{bq[-1]}）",
        "",
        f"生成时间：{C.now_str()}",
        "",
        "| 模块 | 计划 | 成功/完成 | N/A | 本次剩余 |",
        "|---|---|---|---|---|",
        f"| 问答 | {len(bq)} | {stats['qa']['success']} | 0 | {len(st['remaining']['qa'])} |",
        f"| 路由 | {len(bq)} | {stats['route']['ok']} | {stats['route']['na']} | "
        f"{len(st['remaining']['route'])} |",
        f"| Jev | {len(bq)} | {stats['jev']['success']} | {stats['jev']['input_missing']}（无可用片段） | "
        f"{len(st['remaining']['jev'])} |",
        "",
        f"批次完成：{'是' if st['complete'] else '否'}；"
        f"Token {stats['tokens_total'][0]} 入 / {stats['tokens_total'][1]} 出；"
        f"忙时费用 {stats['cost_cny_busy']} 元。",
        "",
        "原始预测文件：",
        f"- {QA_OUT}",
        f"- {ROUTE_OUT}",
        f"- {JEV_OUT}",
    ]
    write_text(os.path.join(REPORT_DIR, f"BATCH_{bi:02d}.md"), "\n".join(lines) + "\n")
    C.write_json_atomic(os.path.join(BATCH_DIR, f"batch_{bi:02d}.json"),
                        {"batch": bi, "qids": bq, "status": st, "stats": stats})


def write_full_report(qids, samples_by_qid, tocs, state):
    stats = module_stats(qids, samples_by_qid)
    rem = remaining(qids, samples_by_qid, tocs)
    qa, rt, jv = stats["qa"], stats["route"], stats["jev"]
    coverage = {
        "qa": f"{qa['success']}/{len(qids)}",
        "route": f"{rt['ok']}+{rt['na']}N/A / {len(qids)}",
        "jev": f"{jv['success']}/{jv['snippets_ok']}可运行片段",
    }
    lines = [
        "# JevSearch 500 题实验全量报告",
        "",
        f"生成时间：{C.now_str()}",
        "",
        "## 一、运行成功率 / 覆盖率 / 准确率（分开统计）",
        "",
        "| 模块 | 运行成功 | 覆盖率 | 准确率/命中率 |",
        "|---|---|---|---|",
        f"| 问答 | {qa['success']} | {coverage['qa']} | 严格 "
        f"{qa['accuracy_all'].get('strict')}/{qa['accuracy_all'].get('n')}；"
        f"可回答 {qa['accuracy_answerable'].get('strict')}/"
        f"{qa['accuracy_answerable'].get('n')} |",
        f"| 路由 | {rt['ok']} | {coverage['route']} | Hit@1 {rt['hit@1']}，"
        f"Hit@3 {rt['hit@3']}，PR@1 {rt['page_recall@1']}，PR@3 {rt['page_recall@3']} |",
        f"| Jev | {jv['success']} | {coverage['jev']} | {jv['accuracy']} |",
        "",
        f"问答不可答：{qa['reject_correct']}/{qa['unanswerable_n']} 正确拒答。",
        f"Jev 不可答：{jv['unanswerable_n']} 条中 sufficient=True "
        f"{jv['unanswerable_false_sufficient']} 条（误判率）。",
        "",
        "## 二、动作分布（Jev）",
        "",
        f"- 原始 next_action：{jv['raw_next_action']}",
        f"- 归一化动作（离线派生，未覆盖原始）：{jv['normalized_action']}",
        "",
        "## 三、Token、耗时与费用",
        "",
        f"- 问答：{qa['input_tokens']} 入 / {qa['output_tokens']} 出，"
        f"{qa['latency_sec']}s",
        f"- 路由：{rt['input_tokens']} 入 / {rt['output_tokens']} 出，"
        f"{rt['latency_sec']}s",
        f"- Jev：{jv['input_tokens']} 入 / {jv['output_tokens']} 出，"
        f"{jv['latency_sec']}s",
        f"- 合计忙时 **{stats['cost_cny_busy']} 元**，闲时约 {stats['cost_cny_idle']} 元",
        "",
        "## 四、剩余与异常",
        "",
        f"- 问答剩余 {len(rem['qa'])}；路由剩余 {len(rem['route'])}；"
        f"Jev 剩余 {len(rem['jev'])}；Jev 无可用片段 {len(rem['jev_input_missing'])}。",
        "- 不可答题规则：问答需正确拒答；路由记 N/A 不调用；Jev 正常评估并统计误判。",
        "- Jev 准确率未计算：人工标注未完成（review_status 未确认）。",
        "",
        "## 五、复现",
        "",
        "```powershell",
        "py -3.12 run_experiment_500.py --run",
        "```",
        "",
        "所有模块结果均为 append-only；已成功记录不会重复调用。",
    ]
    write_text(os.path.join(REPORT_DIR, "FULL_REPORT.md"), "\n".join(lines) + "\n")
    C.write_json_atomic(os.path.join(RUN_DIR, "final_summary.json"), stats)


# ---------------------------------------------------------------- 无付费自检
def selftest():
    """用临时目录模拟 10 题、8 成功 + 2 待补，验证去重/断点/批次状态。"""
    tmp = os.path.join(RUN_DIR, "_selftest")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    C.ensure_dir(tmp)
    samples, qids = samples_and_qids()
    # 优先用第一批固定 10 题（含 2 道不可答），否则退化为前 10 题
    plan = C.read_json(os.path.join(STAGE50, "plan.json"), default=None)
    batch = (plan["batches"][0]["qids"] if plan and plan.get("batches")
             else qids[:10])
    samples_by_qid = {s["qid"]: s for s in samples}
    T6 = load_task6()
    tocs = T6.load_tocs()

    # 伪造记录（只写临时目录，不碰真实结果）
    qa_tmp = os.path.join(tmp, "qa.jsonl")
    route_tmp = os.path.join(tmp, "route.jsonl")
    jev_tmp = os.path.join(tmp, "jev.jsonl")
    for i, q in enumerate(batch):
        s = samples_by_qid[q]
        C.append_jsonl(qa_tmp, {"qid": q, "ds_success": True, "usage": {
            "prompt_tokens": 1000, "completion_tokens": 50}})
        status = "unanswerable_skipped" if str(s.get("answer", "")).strip() == "Not answerable" else "ok"
        C.append_jsonl(route_tmp, {"qid": q, "status": status})
        if str(s.get("answer", "")).strip() != "Not answerable":
            C.append_jsonl(jev_tmp, {"snippet_id": q + "__title", "qid": q,
                                     "success": True, "relevant": True,
                                     "sufficient": False, "next_action": "search",
                                     "input_tokens": 1300, "output_tokens": 90})

    def rem(qids_in, qamap, rmap, jmap, sok):
        return {
            "qa": [q for q in qids_in if not (qamap.get(q) or {}).get("ds_success")],
            "route": [q for q in qids_in if rmap.get(q, {}).get("status") not in ROUTE_DONE],
            "jev": [q for q in qids_in if q + "__title" in sok
                    and not (jmap.get(q + "__title") or {}).get("success")],
        }

    qamap = last_of(qa_tmp)
    rmap = last_of(route_tmp)
    jmap = last_of(jev_tmp, "snippet_id")
    sok = {q + "__title": 1 for q in batch}  # 模拟 10 条可用片段
    r1 = rem(batch, qamap, rmap, jmap, sok)
    checks = {
        "qa_complete": len(r1["qa"]) == 0,
        "route_complete_with_na": len(r1["route"]) == 0,
        "jev_dedupe_skips_8": len(r1["jev"]) == 2,
        "jev_pending_are_unanswerable": sorted(r1["jev"]) == sorted(
            [q for q in batch
             if str(samples_by_qid[q].get("answer", "")).strip() == "Not answerable"]),
    }
    # 模拟补跑 2 条后断点恢复
    for q in r1["jev"]:
        C.append_jsonl(jev_tmp, {"snippet_id": q + "__title", "qid": q,
                                 "success": True, "relevant": True,
                                 "sufficient": False, "next_action": "search",
                                 "input_tokens": 1300, "output_tokens": 90})
    r2 = rem(batch, last_of(qa_tmp), last_of(route_tmp),
             last_of(jev_tmp, "snippet_id"), sok)
    checks["resume_after_two_calls"] = all(len(v) == 0 for v in r2.values())

    out = {"generated_at": C.now_str(), "batch_qids": batch,
           "first_remaining": {k: len(v) for k, v in r1.items()},
           "after_resume_remaining": {k: len(v) for k, v in r2.items()},
           "checks": checks, "all_pass": all(checks.values()),
           "note": "自检只使用临时文件，未调用任何 API，未修改真实结果"}
    C.write_json_atomic(os.path.join(tmp, "selftest.json"), out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["all_pass"] else 1


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true",
                   help="不调用 API：构建清单/片段并统计剩余、跳过、路径")
    g.add_argument("--selftest", action="store_true",
                   help="不调用 API：验证去重、断点恢复与批次状态")
    g.add_argument("--run", action="store_true", help="正式运行（需要两个 API Key）")
    ap.add_argument("--build", action="store_true", help="preflight 时重建 500 题 Jev 片段")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=0, help="只跑前 N 批；0=全部")
    ap.add_argument("--max-rounds", type=int, default=2, help="失败重试轮数")
    args = ap.parse_args()

    C.ensure_dir(RUN_DIR)
    if args.preflight:
        preflight(build=args.build)
        return 0
    if args.selftest:
        return selftest()
    return run(args.max_batches, args.batch_size, args.max_rounds)


if __name__ == "__main__":
    sys.exit(main())
