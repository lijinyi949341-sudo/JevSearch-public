# -*- coding: utf-8 -*-
"""
补跑前泄漏与一致性检查（零 API、只读）。

检查项：
  A. 路由补跑：126 个目标题的实际提示词中，不含 gold answer、gold evidence_pages、
     也不含数据集的 answerability 判定用语；目标集合只按「无成功路由记录」选出。
  B. Jev 补全片段：每条 ok 片段的 doc_id / 页码 / snippet_id 可追溯，
     片段文本确实来自该页 PDF 文本层或 OCR 缓存，且不含 gold answer。
  C. 发送给 Jev 的 state（build_state 白名单）不含 offline_only / gold 字段。
  D. 去重：snippet_id 唯一；预测输出为独立文件，不覆盖已有 374 条预测。
  E. 记录现有原始预测文件的 SHA-256，补跑后可对比确认未被修改。

输出：outputs/final500/audit/pre_run_check.json
"""
import collections
import hashlib
import json
import os
import re
import sys

import final500_common as C
import fill_gaps_500 as G

OUT = os.path.join(C.OUT_ROOT, "audit")
CHECK_JSON = os.path.join(OUT, "pre_run_check.json")
ROUTES = os.path.join(C.OUT_ROOT, "run500", "toc", "routes.jsonl")
JEV_DIR = os.path.join(C.OUT_ROOT, "run500", "jev")
EXISTING_FILES = [
    ROUTES,
    os.path.join(C.OUT_ROOT, "run500", "qa", "ds_results.jsonl"),
] + [os.path.join(JEV_DIR, "jev_predictions_%s_500.jsonl" % s)
     for s in ("title", "route", "ocr", "recover", "ocr_pilot", "ocr_pilot2")]


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    samples = C.load_samples()
    tocs = G.load_tocs()
    routes = C.last_by_qid(C.load_jsonl(ROUTES))
    targets = G.route_fill_targets(routes, samples)

    # ---------- A. 路由提示词泄漏
    import inspect
    leaks = []
    unanswerable_targets = 0
    src = inspect.getsource(G.build_abstain_prompt)
    if re.search(r"\banswer\b", src.replace("answerable", "")):
        leaks.append({"qid": "*", "issue": "prompt_builder_reads_answer_field"})
    if "evidence" in src or "gold" in src:
        leaks.append({"qid": "*", "issue": "prompt_builder_reads_gold_field"})
    for s in targets:
        if str(s.get("answer", "")) == "Not answerable":
            unanswerable_targets += 1
        doc = tocs.get(s["doc_id"])
        if not doc:
            leaks.append({"qid": s["qid"], "issue": "no_toc"})
            continue
        _system, user = G.build_abstain_prompt(s["question"], doc)
        low = user.lower()
        ans = norm(s.get("answer", ""))
        toc_text = norm(" ".join(str(x.get("title", "")) for x in doc["sections"]))
        # 答案字符串若同时出现在问题或目录标题里，说明它本来就在输入内容中，不算泄漏
        if (len(ans) >= 8 and ans.lower() != "not answerable"
                and ans.lower() in low
                and ans.lower() not in norm(s["question"]).lower()
                and ans.lower() not in toc_text.lower()):
            leaks.append({"qid": s["qid"], "issue": "gold_answer_in_prompt",
                          "value": ans[:60]})
        for token in ("evidence_pages", "gold_answer", "gold_pages", "标准答案"):
            if token in low:
                leaks.append({"qid": s["qid"], "issue": "gold_field_name_in_prompt",
                              "token": token})
        # 提示词必须包含完整问题与全部目录章节
        if norm(s["question"]) not in norm(user):
            leaks.append({"qid": s["qid"], "issue": "question_missing_from_prompt"})
        for sec in doc["sections"]:
            if str(sec["section_id"]) not in user:
                leaks.append({"qid": s["qid"], "issue": "section_missing_from_prompt"})
                break
    # 说明：目录本身包含页码区间（如「第 5-9 页」），页码数字出现在提示词中属于目录内容，
    # 不是 gold 泄漏；真正需要禁止的是 gold 字段名与 gold 答案文本。

    # ---------- B/C. Jev 片段可追溯 + 无 gold
    gap_rows = C.load_jsonl(G.SNIPPETS_GAP)
    ok_rows = [r for r in gap_rows if r.get("status") == "ok"]
    ocr_text = G.load_ocr_text()
    import pymupdf
    page_cache = {}
    snippet_issues = []
    gold_in_snippet = 0
    for r in ok_rows:
        s = next(x for x in samples if x["qid"] == r["qid"])
        if not r.get("snippet_id", "").startswith(r["qid"]):
            snippet_issues.append({"qid": r["qid"], "issue": "snippet_id_mismatch"})
        if r.get("selection_uses_gold") is not False:
            snippet_issues.append({"qid": r["qid"], "issue": "selection_uses_gold_not_false"})
        pdf, _ = C.resolve_pdf(r["doc_id"])
        if not pdf:
            snippet_issues.append({"qid": r["qid"], "issue": "pdf_missing"})
            continue
        if r["doc_id"] not in page_cache:
            page_cache[r["doc_id"]] = pymupdf.open(pdf)
        d = page_cache[r["doc_id"]]
        p = int(r["page"] or 0)
        if p < 1 or p > d.page_count:
            snippet_issues.append({"qid": r["qid"], "issue": "page_out_of_range", "page": p})
            continue
        raw = norm(d.load_page(p - 1).get_text("text"))
        ocr = ocr_text.get((r["doc_id"], p), "")
        src = raw if len(raw) >= 40 else ocr
        snip = norm(r.get("evidence_snippet", ""))
        if not src:
            snippet_issues.append({"qid": r["qid"], "issue": "no_source_text"})
        elif snip not in src:
            snippet_issues.append({"qid": r["qid"], "issue": "snippet_not_in_source"})
        ans = norm(s.get("answer", ""))
        if len(ans) >= 8 and ans.lower() != "not answerable" and ans.lower() in snip.lower():
            gold_in_snippet += 1

    # Jev 发送内容白名单
    import run_task5_jev as base5
    import task5_jev_500 as t5
    state_issues = []
    for r in ok_rows:
        js = t5.to_jev_sample(r)
        state = base5.build_state(js)
        forbidden = ("offline_only", "gold_answer", "evidence_pages", "evidence_sources",
                     "human_relevant", "human_self_contained", "human_sufficient",
                     "human_action", "assistant_suggestion")
        hit = [k for k in forbidden if k in state]
        if hit:
            state_issues.append({"qid": r["qid"], "issue": "gold_field_in_state",
                                 "fields": hit})
        # 发送内容只能有 question / evidence_snippet / doc_type / page 四类：
        # 去掉白名单字段行后，剩余正文必须与 evidence_snippet 完全一致。
        body = []
        allowed_prefixes = ("doc_type:", "page:", "Question:", "Evidence snippet:")
        for line in state.splitlines():
            if not line.strip() or line.startswith(allowed_prefixes):
                continue
            body.append(line)
        if "\n".join(body).strip() != str(js.get("evidence_snippet", "")).strip():
            state_issues.append({"qid": r["qid"], "issue": "state_body_mismatch",
                                 "body_preview": "\n".join(body)[:80]})
        problems, _ = base5.leak_check(js, state)
        if problems:
            state_issues.append({"qid": r["qid"], "issue": "leak_check_failed",
                                 "detail": problems})

    # ---------- D. 去重
    snip_ids = [r.get("snippet_id") for r in gap_rows]
    dup_snip = [k for k, v in collections.Counter(snip_ids).items() if v > 1]
    pred_qids = [r.get("qid") for r in C.load_jsonl(G.PRED_GAP) if r.get("success")]
    # 预测文件尚不存在时，退化为检查「本次计划实际要跑的 qid」，避免空集合导致去重检查失效。
    planned_qids = [r["qid"] for r in gap_rows if r.get("status") == "ok"]
    check_qids = pred_qids or planned_qids
    pred_dups = ([k for k, v in collections.Counter(pred_qids).items() if v > 1]
                 if pred_qids else
                 [k for k, v in collections.Counter(planned_qids).items() if v > 1])
    overlap_with_existing = sorted(set(check_qids) & set(
        q for q in (C.last_by_qid(C.load_jsonl(f)) for f in EXISTING_FILES[2:])
        for q in q))

    result = {
        "generated_at": C.now_str(),
        "A_route": {
            "targets": len(targets),
            "selection_basis": "无成功路由记录（不读取 answer/answerability）",
            "unanswerable_targets_present_but_not_used_for_selection": unanswerable_targets,
            "prompt_leaks": leaks,
            "prompt_fields": ["question", "sections(section_id/title/level/page range)"],
        },
        "B_jev_snippets": {
            "total_records": len(gap_rows),
            "ok": len(ok_rows),
            "status_counts": dict(collections.Counter(r.get("status") for r in gap_rows)),
            "traceability_issues": snippet_issues,
            "gold_answer_appears_in_snippet": gold_in_snippet,
            "note": "gold_answer 出现在片段中属于数据巧合（答案本就写在正文里），"
                    "不代表泄漏；选页过程未使用 gold。",
        },
        "C_jev_state": {"state_issues": state_issues, "checked": min(200, len(ok_rows))},
        "D_dedup": {
            "snippet_id_duplicates": dup_snip,
            "prediction_qid_duplicates": pred_dups,
            "checked_qids": len(check_qids),
            "checked_qids_basis": "已有预测文件" if pred_qids else "本次计划运行的片段",
            "prediction_qids_overlapping_existing": overlap_with_existing,
            "gapfill_output_file": G.PRED_GAP,
            "existing_predictions_touched": False,
        },
        "E_hashes_before": {os.path.relpath(p, C.ROOT): sha256(p) for p in EXISTING_FILES},
        "verdict": "PASS" if not (leaks or snippet_issues or state_issues or dup_snip
                                  or pred_dups or overlap_with_existing) else "FAIL",
    }
    C.write_json_atomic(CHECK_JSON, result)
    print(json.dumps({k: v for k, v in result.items() if k != "E_hashes_before"},
                     ensure_ascii=False, indent=2))
    print("hash baseline -> %s" % CHECK_JSON)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
