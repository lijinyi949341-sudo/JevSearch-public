# -*- coding: utf-8 -*-
"""
Jev 证据判断（500 题版）：构造 Question + Evidence snippet，并调用 Jev 判断上下文必要性。

与第一轮的关系：
  - 原来的 12 条样本、结果与人工标注（task5_samples.jsonl / task5_annotations.jsonl /
    outputs/task5/jev_predictions.jsonl）完全保留，本脚本只写 outputs/final500/jev/。
  - 复用 run_task5_jev.py 完全相同的 Jev 问题定义、输入拼接与泄漏自检。

三种证据片段来源（严格区分）：
  gold       —— 数据集 gold evidence_pages 的第一页（已知证据页，oracle 上界，
                选择过程使用 gold，但发送给 Jev 的只有 question + snippet）；
  title      —— 目录标题词匹配的 top-1 章节首页（真实检索结果，不使用 gold）；
  llm        —— DeepSeek 目录路由 top-1 预测章节的首页（真实检索结果，不使用 gold）。

安全约束：
  - 发送给 Jev 的内容只有 question、evidence snippet 和 doc_type/page 元数据；
  - 绝不发送 gold_answer、evidence_pages、evidence_sources、人工标签；
  - 片段选择规则与是否使用 gold 明确记录在 selection_uses_gold。

人工标注：
  - 先冻结 jev500_v1 标注标准与分层抽样方案；
  - 未经人工确认（review_status=confirmed）的样本，绝不计算或宣称 Jev 准确率。

用法：
  python task5_jev_500.py --build-snippets --mode gold
  python task5_jev_500.py --build-snippets --mode title
  python task5_jev_500.py --dry-run --mode gold --limit 5
  python task5_jev_500.py --run --mode gold --limit 50
  python task5_jev_500.py --annotation-sample 60 --mode gold
  python task5_jev_500.py --agreement --mode gold
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time

import pymupdf

import final500_common as C
import run_task5_jev as base5
import task6_route_and_eval as base6

JEV_DIR = os.path.join(C.OUT_ROOT, "jev")
TOC_JSON = os.path.join(C.OUT_ROOT, "toc", "toc.json")
ROUTES = os.path.join(C.OUT_ROOT, "toc", "routes", "deepseek_v41", "routes.jsonl")
PROTOCOL_JSON = os.path.join(JEV_DIR, "annotation_protocol.json")

MODES = ("gold", "title", "llm", "route", "ocr", "recover", "ocr_pilot")
MODE_DESC = {
    "gold": "gold evidence_pages 第一页（oracle 上界，选择使用 gold，输入不发送 gold 字段）",
    "title": "目录标题词匹配 top-1 章节首页（真实检索，无 LLM，不使用 gold）",
    "llm": "DeepSeek 目录路由 top-1 预测章节首页（真实检索，不使用 gold）",
    "route": "目录路由 top-1~3 预测章节的真实 PDF 文本片段（不使用 gold，见 task5_jev_route_500.py）",
    "ocr": "路由预测页经本地 PaddleOCR 识别后的文本片段（不使用 gold，见 task5_jev_ocr_500.py）",
    "recover": "路由邻域/题干页码的真实 PDF 文本层片段（不使用 gold，见 task5_jev_recover_500.py）",
    "ocr_pilot": "40 页 OCR 试点中语义复核通过的片段（不使用 gold，见 task5_jev_ocr_pilot.py）",
}
MAX_CHARS = 900
MIN_CONFIRMED_FOR_CLAIM = 30
PROMPT_VERSION = "jev500_v1"

TYPE_PRIORITY = ["Chart", "Table", "Figure", "Generalized-text (Layout)", "Pure-text (Plain-text)"]


# ---------------------------------------------------------------- 通用
def snippets_path(mode):
    return os.path.join(JEV_DIR, f"snippets_{mode}.jsonl")


def predictions_path(mode):
    return os.path.join(JEV_DIR, f"jev_predictions_{mode}.jsonl")


def annotation_sample_path(mode):
    return os.path.join(JEV_DIR, f"annotation_sample_{mode}.jsonl")


def annotations_path(mode):
    return os.path.join(JEV_DIR, f"annotations_{mode}.jsonl")


def load_ocr():
    path = os.path.join(C.ROOT, "outputs", "task6", "ocr_extra.json")
    out = {}
    if os.path.exists(path):
        try:
            data = json.load(open(path, encoding="utf-8"))
            for item in data.get("docs", []):
                out[item["doc_id"]] = {p["page"]: p for p in item.get("pages", [])}
        except Exception as e:
            print(f"[警告] OCR 读取失败: {e}")
    return out


def primary_type(sources):
    for t in TYPE_PRIORITY:
        if t in sources:
            return t
    return "Unknown"


def norm_text(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def extract_page_text(doc_id, page_1based, ocr):
    """确定性规则：优先文本层；不足 40 字符时用已有 OCR；否则标记 empty。

    返回 (snippet, method, has_images, truncated)；snippet 为 None 表示无法提取。
    """
    pdf, _ = C.resolve_pdf(doc_id)
    if pdf is None:
        return None, "pdf_missing", False, False
    try:
        doc = pymupdf.open(pdf)
        if page_1based < 1 or page_1based > doc.page_count:
            doc.close()
            return None, "page_out_of_range", False, False
        p = doc.load_page(page_1based - 1)
        raw = norm_text(p.get_text("text"))
        has_images = bool(p.get_images())
        doc.close()
    except Exception as e:
        return None, f"pdf_unreadable:{type(e).__name__}", False, False
    if len(raw) >= 40:
        return raw[:MAX_CHARS], "text_layer", has_images, len(raw) > MAX_CHARS
    item = ocr.get(doc_id, {}).get(page_1based)
    if item and norm_text(item.get("ocr_text", "")):
        ocr_raw = norm_text(item["ocr_text"])
        return ocr_raw[:MAX_CHARS], "ocr", has_images, len(ocr_raw) > MAX_CHARS
    return None, "empty_text_page", has_images, False


def load_tocs():
    d = C.read_json(TOC_JSON, default={"docs": []}) or {"docs": []}
    return {x["doc_id"]: x for x in d.get("docs", [])
            if x.get("build_status") == "ok" and x.get("sections")}


def load_routes():
    return {r["qid"]: r for r in C.load_jsonl(ROUTES)}


def select_page(sample, mode, tocs, routes):
    """返回 (page, section_id, status, reason)。"""
    gold = C.parse_pages(sample.get("evidence_pages", "[]"))
    doc = tocs.get(sample["doc_id"])
    if str(sample.get("answer", "")) == "Not answerable":
        return None, None, "unanswerable", "答案为 Not answerable，不纳入 Jev 测试"
    if mode == "gold":
        valid = [p for p in gold if doc and p >= 1 and p <= doc["num_pages"]]
        if not valid:
            return None, None, "missing_gold_page", "无有效 gold 证据页"
        return valid[0], None, "ok", ""
    if doc is None:
        return None, None, "no_toc", "目录不可用"
    if mode == "title":
        ranked = base6.baseline_rank(sample["question"], doc["sections"])
        if not ranked:
            return None, None, "no_title_match", "目录标题与问题无词重合"
        s = ranked[0]
        return s["start_page"], s["section_id"], "ok", ""
    # llm
    r = routes.get(sample["qid"])
    if not r:
        return None, None, "no_route", "尚未运行目录路由"
    if r.get("status") != "ok" or not r.get("sections"):
        return None, None, "route_not_ok", f"路由状态={r.get('status')}"
    s = r["sections"][0]
    return s["start_page"], s["section_id"], "ok", ""


# ---------------------------------------------------------------- 片段构建
def build_snippets(mode, limit=0):
    samples = C.load_samples()
    if limit:
        samples = samples[:limit]
    tocs = load_tocs()
    routes = load_routes()
    ocr = load_ocr()
    out_path = snippets_path(mode)
    # 重建：先写临时文件，完成后原子替换；这是可重复派生产物，不是旧实验结果
    tmp = out_path + ".tmp"
    C.ensure_dir(JEV_DIR)
    counts = {"ok": 0, "skipped": 0}
    with open(tmp, "w", encoding="utf-8") as f:
        for s in samples:
            page, section_id, status, reason = select_page(s, mode, tocs, routes)
            rec = {
                "snippet_id": f"{s['qid']}__{mode}",
                "qid": s["qid"], "sample_index": s["sample_index"],
                "doc_id": s["doc_id"], "doc_type": s.get("doc_type", ""),
                "question": s["question"], "mode": mode,
                "page": page, "section_id": section_id,
                "selection_uses_gold": mode == "gold",
                "selection_rule": MODE_DESC[mode],
                "status": status, "reason": reason,
                "evidence_snippet": "", "char_len": 0,
                "extraction_method": "", "has_page_images": False,
                "evidence_type_primary": primary_type(
                    C.parse_sources(s.get("evidence_sources", "[]")),
                ),
                "cross_page": len(C.parse_pages(s.get("evidence_pages", "[]"))) > 1,
                "created_at": C.now_str(),
                "offline_only": {
                    "answer": s.get("answer", ""),
                    "gold_answer": s.get("answer", ""),
                    "evidence_pages": s.get("evidence_pages", "[]"),
                    "evidence_sources": s.get("evidence_sources", ""),
                },
            }
            if status == "ok":
                text, method, has_images, truncated = extract_page_text(
                    s["doc_id"], page, ocr)
                if text:
                    rec.update({"evidence_snippet": text, "char_len": len(text),
                                "extraction_method": method, "has_page_images": has_images,
                                "truncated": truncated})
                    rec["status"] = "ok"
                else:
                    rec.update({"status": "no_snippet", "reason": method,
                                "has_page_images": has_images})
            counts["ok" if rec["status"] == "ok" else "skipped"] += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    print(f"[片段] mode={mode} 共 {len(samples)} 条：可用 {counts['ok']}，跳过 {counts['skipped']}")
    print(f"输出: {out_path}")
    return out_path


# ---------------------------------------------------------------- Jev 运行
def to_jev_sample(rec):
    return {
        "sample_id": rec["snippet_id"],
        "doc_id": rec["doc_id"],
        "doc_type": rec.get("doc_type"),
        "page": rec.get("page"),
        "question": rec["question"],
        "evidence_snippet": rec["evidence_snippet"],
        "offline_only": rec.get("offline_only", {}),
    }


def run_jev(mode, limit=0, retry_failed=False, max_attempts=2,
            snippets_file=None, out_file=None, qids=None, exclude_qids=None):
    path = snippets_file or snippets_path(mode)
    snippets = [r for r in C.load_jsonl(path) if r.get("status") == "ok"]
    if exclude_qids:
        snippets = [r for r in snippets if r["qid"] not in exclude_qids]
    if qids:
        wanted = {x.strip().upper() for x in qids.split(",") if x.strip()}
        snippets = [r for r in snippets if r["qid"].upper() in wanted]
    if limit:
        snippets = snippets[:limit]
    if len(snippets) > 500:
        print(f"[错误] 单次最多 500 条 Jev 预测，当前 {len(snippets)} 条。")
        sys.exit(1)
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("[错误] 未检测到 TYPESAFE_API_KEY，未发出任何请求。")
        sys.exit(1)
    from typesafe_sdk import TypeSafeClient
    client = TypeSafeClient(api_key=api_key, base_url=base5.BASE_URL, model=base5.MODEL_NAME)
    questions = base5.build_questions()

    out = out_file or predictions_path(mode)
    C.ensure_dir(os.path.dirname(out) or ".")
    existing = C.load_jsonl(out)
    last = {r["snippet_id"]: r for r in existing}
    attempts = {}
    for r in existing:
        attempts[r["snippet_id"]] = attempts.get(r["snippet_id"], 0) + 1

    n_ok = n_fail = n_skip = 0
    for rec in snippets:
        sid = rec["snippet_id"]
        prev = last.get(sid)
        if prev and prev.get("success"):
            n_skip += 1
            continue
        if prev and not retry_failed and attempts.get(sid, 0) >= max_attempts:
            n_skip += 1
            continue

        js = to_jev_sample(rec)
        state = base5.build_state(js)
        problems, gold_in_snippet = base5.leak_check(js, state)
        entry = {
            "snippet_id": sid, "qid": rec["qid"], "mode": mode,
            "doc_id": rec["doc_id"], "page": rec["page"],
            "section_id": rec.get("section_id"),
            "selection_uses_gold": rec.get("selection_uses_gold"),
            "prompt_version": PROMPT_VERSION,
            "requested_model": base5.MODEL_NAME,
            "state_sent": state,
            "leak_check_passed": not problems,
            "leak_check_problems": problems,
            "gold_answer_visible_in_snippet": bool(gold_in_snippet),
            "model": None,
            "relevant_raw": None, "relevant": None,
            "self_contained_raw": None, "self_contained": None,
            "sufficient_raw": None, "sufficient": None,
            "next_action": None, "next_action_probabilities": None,
            "next_action_confidence": None,
            "action_normalized": None,
            "input_tokens": None, "output_tokens": None,
            "latency_sec": None, "success": False, "error": "",
            "attempt": attempts.get(sid, 0) + 1,
            "run_at": C.now_str(),
        }
        t0 = time.time()
        if problems:
            entry["error"] = "LEAK_CHECK_FAILED: " + "; ".join(problems)
        else:
            try:
                resp = client.system_one(state=state, questions=questions,
                                         model=base5.MODEL_NAME)
                ans = resp.answers
                entry["model"] = getattr(resp, "model", None)
                for key in ("relevant", "self_contained", "sufficient"):
                    a = ans.get(key)
                    if a is not None:
                        entry[f"{key}_raw"] = float(a.noul)
                        entry[key] = base5.as_decision(entry[f"{key}_raw"])
                na = ans.get("next_action")
                if na is not None:
                    entry["next_action"] = na.choice
                    entry["next_action_probabilities"] = dict(na.probabilities)
                    entry["next_action_confidence"] = float(na.confidence)
                # 归一化动作另存一列，不覆盖 next_action 原始返回
                if entry["relevant"] is False:
                    entry["action_normalized"] = "search"
                elif entry["sufficient"] is True:
                    entry["action_normalized"] = "answer"
                elif entry["relevant"] is True:
                    entry["action_normalized"] = "expand"
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    entry["input_tokens"] = getattr(usage, "input_tokens", None)
                    entry["output_tokens"] = getattr(usage, "output_tokens", None)
                entry["success"] = True
                n_ok += 1
            except Exception as e:
                msg = str(e).replace(api_key, "***")
                entry["error"] = f"{type(e).__name__}: {msg[:400]}"
                n_fail += 1
        entry["latency_sec"] = round(time.time() - t0, 3)
        C.append_jsonl(out, entry)
        last[sid] = entry
        attempts[sid] = entry["attempt"]
        print(f"[{sid}] success={entry['success']} action={entry['next_action']} "
              f"rel={entry['relevant']} self={entry['self_contained']} "
              f"suff={entry['sufficient']} {entry['latency_sec']}s "
              f"err={entry['error'][:60]}")

    print(f"\n本次成功 {n_ok}，失败 {n_fail}，跳过 {n_skip}；预测文件: {out}")


# ---------------------------------------------------------------- 标注冻结与抽样
def protocol_text():
    return "\n".join([
        "Jev 500 人工标注标准 v1（冻结）",
        "维度定义：relevant=片段主题与问题是否相关；self_contained=片段脱离上下文能否被正确理解；"
        "sufficient=片段是否已包含足以回答问题的信息；next_action=answer/expand/search。",
        "判定顺序：先判 relevant；不相关时 sufficient 必为 false，action=search；"
        "相关但依赖指代/图注/表头时 self_contained=false，action=expand；"
        "相关且信息完备时 sufficient=true，action=answer。",
        "证据边界：只依据 snippet 本身，不允许使用外部知识；图表页只依据片段中可见文字与图注。",
        "不可答样本不纳入准确率分母；空片段、PDF 缺失样本不进入抽样。",
        "抽样：按 evidence type × 跨页与否分层，固定随机种子，按比例分配，至少覆盖 Chart/Table/Figure/文本。",
        "准确率门槛：只有 review_status=confirmed 且四个字段均非空的人工标注才能计算；"
        f"确认样本少于 {MIN_CONFIRMED_FOR_CLAIM} 条时不得宣称准确率。",
        "独立性：标注人不得预先查看 Jev 预测结果；预测与人工标签分开存放。",
    ])


def annotation_sample(n, mode):
    snippets = [r for r in C.load_jsonl(snippets_path(mode)) if r.get("status") == "ok"]
    if not snippets:
        print("[错误] 没有可用片段，请先 --build-snippets。")
        sys.exit(1)
    n = min(n, len(snippets), 500)
    strata = {}
    for r in snippets:
        key = f"{r['evidence_type_primary']}|cross={int(bool(r['cross_page']))}"
        strata.setdefault(key, []).append(r)

    # 固定种子 + 哈希排序，保证可复现
    rng = random.Random(20261108)
    for key in strata:
        strata[key].sort(key=lambda x: hashlib.sha256(
            (x["snippet_id"] + "|jev500_v1").encode()).hexdigest())

    alloc = {}
    total = len(snippets)
    frac = []
    for key, rows in strata.items():
        exact = n * len(rows) / total
        alloc[key] = min(len(rows), int(exact))
        frac.append((exact - int(exact), key))
    # 先保证每个层至少 1 条（若 n 足够）
    if n >= len(strata):
        for key in strata:
            if alloc[key] == 0:
                alloc[key] = 1
    while sum(alloc.values()) < n:
        frac.sort(reverse=True)
        progressed = False
        for _, key in frac:
            if alloc[key] < len(strata[key]) and sum(alloc.values()) < n:
                alloc[key] += 1
                progressed = True
        if not progressed:
            break
    while sum(alloc.values()) > n:
        for key in sorted(alloc, key=lambda k: -alloc[k]):
            if alloc[key] > 1:
                alloc[key] -= 1
                if sum(alloc.values()) == n:
                    break

    rules = protocol_text()
    digest = hashlib.sha256(rules.encode("utf-8")).hexdigest()
    if not os.path.exists(PROTOCOL_JSON):
        C.write_json_atomic(PROTOCOL_JSON, {
            "version": PROMPT_VERSION,
            "frozen_at": C.now_str(),
            "mode": mode,
            "rules": rules,
            "rule_digest": digest,
            "dimensions": ["relevant", "self_contained", "sufficient", "next_action"],
            "decision_threshold": base5.DECISION_THRESHOLD,
            "stratification": "evidence_type_primary × cross_page",
            "sampling_seed": 20261108,
            "min_confirmed_for_claim": MIN_CONFIRMED_FOR_CLAIM,
            "accuracy_policy": "仅 confirmed 且字段完整的人工标注可计算准确率",
        })

    selected = []
    for key, rows in strata.items():
        selected.extend(rows[:alloc[key]])
    selected.sort(key=lambda x: x["snippet_id"])
    ann_sample = annotation_sample_path(mode)
    tmp = ann_sample + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in selected:
            js = to_jev_sample(r)
            state = base5.build_state(js)
            problems, gold_visible = base5.leak_check(js, state)
            f.write(json.dumps({
                "snippet_id": r["snippet_id"], "qid": r["qid"],
                "doc_id": r["doc_id"], "page": r["page"], "mode": mode,
                "stratum": f"{r['evidence_type_primary']}|cross={int(bool(r['cross_page']))}",
                "evidence_type_primary": r["evidence_type_primary"],
                "cross_page": r["cross_page"],
                "question": r["question"],
                "evidence_snippet": r["evidence_snippet"],
                "selection_uses_gold": r["selection_uses_gold"],
                "state_to_jev": state,
                "leak_check_passed": not problems,
                "annotation_version": PROMPT_VERSION,
                "rule_digest": digest,
                "label_status": "pending",
                "review_status": "",
                "annotator": "",
                "human_relevant": None,
                "human_self_contained": None,
                "human_sufficient": None,
                "human_action": None,
                "missing_context_type": None,
                "annotation_reason": "",
                "offline_only": r.get("offline_only", {}),
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, ann_sample)
    print(f"[抽样] mode={mode} 共 {len(snippets)} 条可用片段，抽取 {len(selected)} 条")
    for k in sorted(alloc):
        print(f"  {k}: {alloc[k]}")
    print(f"标注样本: {ann_sample}\n冻结标准: {PROTOCOL_JSON}")


def cohen_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(1 for x, y in pairs if x == y) / n
    pa = sum(1 for x, _ in pairs if x) / n
    pb = sum(1 for _, y in pairs if y) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 4) if pe != 1 else None


def agreement(mode):
    preds = {r["snippet_id"]: r for r in C.load_jsonl(predictions_path(mode))
             if r.get("success")}
    anns = C.load_jsonl(annotations_path(mode))
    confirmed = [a for a in anns
                 if a.get("review_status") == "confirmed"
                 and a.get("human_relevant") is not None
                 and a.get("human_self_contained") is not None
                 and a.get("human_sufficient") is not None
                 and a.get("human_action") in ("answer", "expand", "search")]
    sample = C.load_jsonl(annotation_sample_path(mode))
    used = []
    for a in confirmed:
        p = preds.get(a["snippet_id"])
        if not p:
            continue
        used.append((a, p))

    def acc(key_h, key_p):
        vals = [(a[key_h], p[key_p]) for a, p in used if p.get(key_p) is not None]
        return (round(sum(1 for x, y in vals if x == y) / len(vals), 4) if vals else None,
                len(vals))

    rel, n_rel = acc("human_relevant", "relevant")
    sc, n_sc = acc("human_self_contained", "self_contained")
    su, n_su = acc("human_sufficient", "sufficient")
    act, n_act = acc("human_action", "next_action")
    kappa = {
        "relevant": cohen_kappa([a["human_relevant"] for a, _ in used],
                                [p.get("relevant") for _, p in used]),
        "self_contained": cohen_kappa([a["human_self_contained"] for a, _ in used],
                                      [p.get("self_contained") for _, p in used]),
        "sufficient": cohen_kappa([a["human_sufficient"] for a, _ in used],
                                  [p.get("sufficient") for _, p in used]),
    }
    claimable = (len(confirmed) >= MIN_CONFIRMED_FOR_CLAIM
                 and sample and len(confirmed) >= 0.5 * len(sample))
    out = {
        "generated_at": C.now_str(),
        "mode": mode,
        "annotation_version": PROMPT_VERSION,
        "annotation_sample_size": len(sample),
        "annotations_total": len(anns),
        "confirmed_total": len(confirmed),
        "confirmed_with_predictions": len(used),
        "claimable": bool(claimable),
        "claim_policy": (f"仅 confirmed 且字段完整；确认样本 < {MIN_CONFIRMED_FOR_CLAIM} "
                         "或不足抽样的一半时，不得宣称 Jev 准确率。"),
        "metrics_on_confirmed_only": {
            "relevant_accuracy": rel, "relevant_n": n_rel,
            "self_contained_accuracy": sc, "self_contained_n": n_sc,
            "sufficient_accuracy": su, "sufficient_n": n_su,
            "next_action_accuracy": act, "next_action_n": n_act,
        },
        "cohen_kappa": kappa,
        "warning": "" if claimable else "未达到可宣称准确率的条件；以上数字仅供参考，不能作为结论。",
    }
    C.write_json_atomic(os.path.join(JEV_DIR, f"agreement_{mode}.json"), out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- dry-run
def dry_run(mode, limit=5, snippets_file=None, qids=None):
    rows = [r for r in C.load_jsonl(snippets_file or snippets_path(mode))
            if r.get("status") == "ok"]
    if qids:
        wanted = {x.strip().upper() for x in qids.split(",") if x.strip()}
        rows = [r for r in rows if r["qid"].upper() in wanted]
    snippets = rows[:limit] if limit else rows
    if not snippets:
        print("[错误] 没有可用片段，请先 --build-snippets。")
        sys.exit(1)
    print(f"【DRY-RUN】mode={mode}，将检查 {len(snippets)} 条，不调用 API。")
    all_ok = True
    for r in snippets:
        js = to_jev_sample(r)
        state = base5.build_state(js)
        problems, gold_visible = base5.leak_check(js, state)
        print("=" * 72)
        print(f"{r['snippet_id']}  page={r['page']}  method={r['extraction_method']}  "
              f"uses_gold={r['selection_uses_gold']}")
        print(f"--- 发送给 Jev 的 state ---\n{state}\n--- end ---")
        if problems:
            all_ok = False
            print("[泄漏检查] 失败: " + "; ".join(problems))
        else:
            print("[泄漏检查] 通过：无 gold_answer/evidence_pages/human_* 字段。")
        if gold_visible:
            print("[说明] 答案文本是证据的一部分，天然出现在片段中；发送的仍只有 question+片段。")
    print("=" * 72)
    print("输入自检" + ("全部通过" if all_ok else "存在问题"))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build-snippets", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--annotation-sample", type=int, metavar="N")
    g.add_argument("--agreement", action="store_true")
    ap.add_argument("--mode", choices=MODES, default="gold")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qids", default="", help="只跑指定 qid（逗号分隔，联调时可精确选样）")
    ap.add_argument("--snippets", default=None,
                    help="自定义片段 JSONL（默认 outputs/final500/jev/snippets_<mode>.jsonl）")
    ap.add_argument("--out", default=None,
                    help="自定义预测输出 JSONL（默认 outputs/final500/jev/jev_predictions_<mode>.jsonl）")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=2)
    args = ap.parse_args()

    C.ensure_dir(JEV_DIR)
    if args.build_snippets:
        build_snippets(args.mode, args.limit)
    elif args.run:
        run_jev(args.mode, args.limit, args.retry_failed, args.max_attempts,
                args.snippets, args.out, args.qids)
    elif args.dry_run:
        dry_run(args.mode, args.limit, args.snippets, args.qids)
    elif args.annotation_sample:
        annotation_sample(args.annotation_sample, args.mode)
    elif args.agreement:
        agreement(args.mode)


if __name__ == "__main__":
    main()
