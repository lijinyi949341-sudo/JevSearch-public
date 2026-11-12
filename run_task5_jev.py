# -*- coding: utf-8 -*-
"""
Jev：用 TypeSafe 官方 Jev API 判断「当前证据片段是否缺少必要上下文」。

三个互不相同的判断维度：
  relevant        —— 片段与问题是否「相关」（只看主题对应，不看信息是否够用）
  self_contained  —— 片段脱离相邻上下文后能否「被正确理解」（看指代、图注、表头等）
  sufficient      —— 片段是否「包含足以回答问题的信息」（看信息完备度）

一个动作决策：
  next_action     —— answer / expand / search

安全约束：
  * 发送给 Jev 的内容只由 question、evidence_snippet 以及 md_type/page 两个最小元数据拼成；
  * 绝不发送 offline_only（gold_answer / evidence_pages / evidence_sources）、
    人工标签（human_*）或辅助候选标签（assistant_*）；
  * 失败如实记录，不重试、不伪造结果；
  * 密钥只从环境变量 TYPESAFE_API_KEY 读取，不打印、不落盘。

用法（在已配置 TYPESAFE_API_KEY 的终端）：
    py -3.12 run_task5_jev.py --dry-run --limit 5      # 只打印将要发送的内容，不调用 API
    py -3.12 run_task5_jev.py --limit 5                # 第一阶段：5 条真实样本
    py -3.12 run_task5_jev.py --ids S01,S02,S03,S04,S05
"""
import argparse
import json
import os
import sys
import time

from typesafe_sdk import Choice, Noul, TypeSafeClient

BASE_URL = "https://api.typesafe.ai"
MODEL_NAME = "jev-latest"          # SDK 默认模型；实际版本以 resp.model 为准
PROMPT_VERSION = "task5_v1"        # prompt 版本号，便于后续对照与复现

SAMPLES_FILE = "task5_samples.jsonl"
OUT_DIR = os.path.join("outputs", "task5")
PRED_FILE = os.path.join(OUT_DIR, "jev_predictions.jsonl")

# 允许进入模型输入的最小元数据（不含任何答案/标签）
ALLOWED_METADATA = ("doc_type", "page")

# 判断阈值：Noul 输出 0~1 分数，>=0.5 记为 true。
# 该阈值是我们自己的决策规则，原始分数会一并保存，便于后续调整。
DECISION_THRESHOLD = 0.5


# ---------------------------------------------------------------- 输入构造
def build_state(sample):
    """只由 question + evidence_snippet + 最小元数据拼成模型输入。"""
    lines = []
    for key in ALLOWED_METADATA:
        value = sample.get(key)
        if value is not None:
            lines.append(f"{key}: {value}")
    if lines:
        lines.append("")
    lines.append(f"Question: {sample['question']}")
    lines.append("")
    lines.append("Evidence snippet:")
    lines.append(sample["evidence_snippet"])
    return "\n".join(lines)


def leak_check(sample, state):
    """检查模型输入是否混入了禁止字段；返回 (问题列表, gold 是否自然出现在片段中)。"""
    problems = []
    for token in ("offline_only", "gold_answer", "evidence_pages", "evidence_sources",
                  "human_relevant", "human_self_contained", "human_sufficient",
                  "human_action", "missing_context_type", "assistant_suggestion"):
        if token in state:
            problems.append(f"输入中出现禁止字段名: {token}")

    gold = str(sample.get("offline_only", {}).get("gold_answer", "") or "")
    gold_in_snippet = bool(gold) and gold in sample.get("evidence_snippet", "")
    return problems, gold_in_snippet


# ---------------------------------------------------------------- 判断问题
def build_questions():
    """三个 Noul + 一个 Choice，刻意让三个维度互不重叠。"""
    relevant = Noul(
        instructions=(
            "判断【证据片段是否与问题相关】。"
            "相关性只考察主题是否对应：片段讨论的对象/主题是否是问题所问的对象/主题。"
            "不要考虑片段信息是否够用，也不要考虑片段是否依赖上下文才能读懂。"
        ),
        criteria={
            "true": "片段讨论的主题与问题所问主题一致，且片段在该主题下提供了相关信息。",
            "false": "片段讨论的主题与问题所问主题不一致，或完全没有涉及问题所问的对象。",
        },
    )

    self_contained = Noul(
        instructions=(
            "判断【证据片段脱离相邻上下文后是否仍能被正确理解】。"
            "只考察片段本身的可读性与指代是否消解，例如："
            "是否以 These/This/It/Such 等指代词开头却不知道指什么；"
            "是否只是图表标题、表头、图注或脚注碎片；"
            "是否依赖前文定义或前一句才能成立。"
            "不要考虑片段是否包含回答问题的关键数据，也不要判断它与问题是否相关。"
        ),
        criteria={
            "true": "片段自身语义完整，读者无需查看相邻文字/图表即可正确理解它说了什么。",
            "false": "片段缺少必要的指代对象、图表本体、表头或邻近文字，单独看不出它指的是什么。",
        },
    )

    sufficient = Noul(
        instructions=(
            "判断【仅凭当前证据片段，是否已经包含足以准确回答问题的信息】。"
            "只考察信息完备度：回答所需的数值、结论、定义或条目是否直接出现在片段里。"
            "不要考虑片段与问题是否相关，也不要考虑片段是否依赖上下文才能读懂。"
        ),
        criteria={
            "true": "片段中直接出现了回答问题所需的关键信息，可以据此给出具体答案。",
            "false": "片段没有直接给出回答问题所需的关键信息（例如只提到图表却未给出其中数值）。",
        },
    )

    next_action = Choice(
        instructions=(
            "综合判断证据与问题的关系，选择最合适的下一步动作。"
            "answer=证据充分可直接作答；"
            "expand=证据相关但需补充同页/邻近的图表、表格、图注或上下文；"
            "search=当前证据无关或需要到其他位置寻找证据。"
        ),
        criteria={
            "answer": "当前片段已经包含回答问题所需的信息，可以直接回答。",
            "expand": "当前片段与问题相关，但缺少同页或邻近的图表、表格、图注、指代对象或必要说明，应补充局部上下文。",
            "search": "当前片段与问题无关，或者所需信息不在当前位置，应到文档其他页面检索。",
        },
    )

    return {
        "relevant": relevant,
        "self_contained": self_contained,
        "sufficient": sufficient,
        "next_action": next_action,
    }


# ---------------------------------------------------------------- 结果读写
def load_samples(path=SAMPLES_FILE):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_predictions(path=PRED_FILE):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_predictions(rows, path=PRED_FILE):
    """按 sample_id 覆盖写回（同一条样本重跑时替换旧记录，不产生重复）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    order, merged = [], {}
    for row in rows:
        sid = row["sample_id"]
        if sid not in merged:
            order.append(sid)
        merged[sid] = row
    with open(path, "w", encoding="utf-8") as f:
        for sid in order:
            f.write(json.dumps(merged[sid], ensure_ascii=False) + "\n")


def upsert(rows, entry):
    for i, row in enumerate(rows):
        if row["sample_id"] == entry["sample_id"]:
            rows[i] = entry
            return rows
    rows.append(entry)
    return rows


def as_decision(score):
    if score is None:
        return None
    return score >= DECISION_THRESHOLD


# ---------------------------------------------------------------- 主流程
def main():
    parser = argparse.ArgumentParser(description="Jev：Jev 上下文必要性判断")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条样本")
    parser.add_argument("--ids", default=None, help="指定样本编号，逗号分隔，如 S01,S02")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印将要发送给 Jev 的内容，不调用 API、不写结果")
    args = parser.parse_args()

    samples = load_samples()
    if args.ids:
        wanted = [x.strip().upper() for x in args.ids.split(",") if x.strip()]
        samples = [s for s in samples if s["sample_id"].upper() in wanted]
    if args.limit is not None:
        samples = samples[: args.limit]

    if not samples:
        print("[错误] 没有匹配到任何样本。")
        sys.exit(1)

    # ---------- dry-run：完整展示输入，并做泄漏自检 ----------
    if args.dry_run:
        print(f"【DRY-RUN】将处理 {len(samples)} 条样本，不调用 API。\n")
        all_ok = True
        for s in samples:
            state = build_state(s)
            problems, gold_in_snippet = leak_check(s, state)
            print("=" * 72)
            print(f"sample_id = {s['sample_id']}  doc_id = {s['doc_id']}  1-based page = {s['page']}")
            print(f"--- 实际发送给 Jev 的 state ---\n{state}\n--- state 结束 ---")
            if problems:
                all_ok = False
                print("[泄漏检查] 失败：" + "；".join(problems))
            else:
                print("[泄漏检查] 通过：未出现 offline_only / gold_answer / evidence_pages / human_* / assistant_* 字段。")
            if gold_in_snippet:
                print("[提示] 答案文本本身是证据的一部分（片段天然包含答案），这不属于标签泄漏；"
                      "发送的仍然只有 question 与 evidence_snippet。")
        print("=" * 72)
        print(f"总计 {len(samples)} 条，输入自检{'全部通过' if all_ok else '存在问题，请先修正'}。")
        return

    # ---------- 真实调用 ----------
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("[错误] 当前进程读不到环境变量 TYPESAFE_API_KEY，未发出任何请求。")
        print("请在你已配置该变量的 PowerShell 终端里运行，例如：")
        print("    py -3.12 run_task5_jev.py --limit 5")
        print("本脚本不会生成假密钥，也不会有任何模拟返回。")
        sys.exit(1)

    client = TypeSafeClient(api_key=api_key, base_url=BASE_URL, model=MODEL_NAME)
    questions = build_questions()
    rows = load_predictions()

    ok_count = 0
    for s in samples:
        sid = s["sample_id"]
        state = build_state(s)
        entry = {
            "sample_id": sid,
            "doc_id": s["doc_id"],
            "page": s["page"],
            "prompt_version": PROMPT_VERSION,
            "requested_model": MODEL_NAME,
            "state_sent": state,          # 便于审计：只含 question + evidence_snippet + 元数据
            "model": None,
            "relevant_raw": None,
            "relevant": None,
            "self_contained_raw": None,
            "self_contained": None,
            "sufficient_raw": None,
            "sufficient": None,
            "next_action": None,
            "next_action_probabilities": None,
            "next_action_confidence": None,
            "input_tokens": None,
            "output_tokens": None,
            "latency_sec": None,
            "success": False,
            "error": "",
            "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        t0 = time.time()
        try:
            resp = client.system_one(state=state, questions=questions, model=MODEL_NAME)
            answers = resp.answers
            ra = answers.get("relevant")
            ca = answers.get("self_contained")
            sa = answers.get("sufficient")
            na = answers.get("next_action")

            entry["model"] = getattr(resp, "model", None)

            if ra is not None:
                entry["relevant_raw"] = float(ra.noul)
                entry["relevant"] = as_decision(entry["relevant_raw"])
            if ca is not None:
                entry["self_contained_raw"] = float(ca.noul)
                entry["self_contained"] = as_decision(entry["self_contained_raw"])
            if sa is not None:
                entry["sufficient_raw"] = float(sa.noul)
                entry["sufficient"] = as_decision(entry["sufficient_raw"])
            if na is not None:
                entry["next_action"] = na.choice
                entry["next_action_probabilities"] = dict(na.probabilities)
                entry["next_action_confidence"] = float(na.confidence)

            usage = getattr(resp, "usage", None)
            if usage is not None:
                entry["input_tokens"] = getattr(usage, "input_tokens", None)
                entry["output_tokens"] = getattr(usage, "output_tokens", None)

            entry["success"] = True
            ok_count += 1
        except Exception as e:  # 失败如实记录：不重试、不伪造
            msg = str(e).replace(api_key, "***")  # 防止密钥出现在错误信息里
            entry["error"] = f"{type(e).__name__}: {msg[:400]}"

        entry["latency_sec"] = round(time.time() - t0, 3)
        rows = upsert(rows, entry)
        save_predictions(rows)

        if entry["success"]:
            print(f"[{sid}] ok model={entry['model']} "
                  f"rel={entry['relevant_raw']}→{entry['relevant']} "
                  f"self={entry['self_contained_raw']}→{entry['self_contained']} "
                  f"suff={entry['sufficient_raw']}→{entry['sufficient']} "
                  f"action={entry['next_action']}({entry['next_action_confidence']}) "
                  f"tokens={entry['input_tokens']}/{entry['output_tokens']} "
                  f"{entry['latency_sec']}s")
        else:
            print(f"[{sid}] FAILED {entry['latency_sec']}s error={entry['error']}")

    print(f"\n成功 {ok_count}/{len(samples)} 条，真实预测已写入 {PRED_FILE}")


if __name__ == "__main__":
    main()
