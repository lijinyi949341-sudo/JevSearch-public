# -*- coding: utf-8 -*-
"""
TypeSafe 官方 Jev API 真实连通性测试（接口连通性测试）。

- 只读取环境变量 TYPESAFE_API_KEY；不打印、不写入任何文件、不硬编码。
- 使用官方 typesafe-sdk 0.7.5 的 TypeSafeClient.system_one()。
- 三个判断：
    relevant    -> Noul  (0~1 分数)
    sufficient  -> Noul  (0~1 分数)
    next_action -> Choice(answer / expand / search)
- 结果为真实 API 返回；失败时如实记录错误，绝不生成模拟结果。

用法（请在你已配置密钥的 PowerShell 中运行）：
    py -3.12 test_jev.py --sample A
    py -3.12 test_jev.py --sample B
    py -3.12 test_jev.py --sample C
    py -3.12 test_jev.py --sample all
"""
import argparse
import json
import os
import sys
import time

from typesafe_sdk import Choice, Noul, TypeSafeClient  # noqa: F401

BASE_URL = "https://api.typesafe.ai"
MODEL_NAME = "jev-latest"

OUT_DIR = os.path.join("outputs", "jev")
SAMPLES_FILE = os.path.join(OUT_DIR, "jev_test_samples.json")
RESULTS_FILE = os.path.join(OUT_DIR, "jev_live_results.json")


def build_questions():
    """按任务要求构造三个 Jev 判断（不使用固定规则替代模型决策）。"""
    relevant = Noul(
        instructions="判断当前证据片段是否与用户的问题相关。",
        criteria={
            "true": "证据片段讨论的核心主题与问题一致，可用于回答该问题。",
            "false": "证据片段与问题主题无关，无法用于回答该问题。",
        },
    )
    sufficient = Noul(
        instructions=(
            "判断仅凭当前证据片段，是否已经包含足以准确回答问题的信息。"
            "注意：相关不代表充分；若片段只提到图表/表格却没有给出其中的关键数据，应判为不充分。"
        ),
        criteria={
            "true": "证据片段直接包含回答所需的关键信息（例如具体数值、结论、定义）。",
            "false": "证据片段只是提及相关对象或图表，但缺少回答所需的具体数据或说明。",
        },
    )
    next_action = Choice(
        instructions="根据证据与问题的关系，选择最合适的下一步动作。",
        criteria={
            "answer": "当前证据已经足够，可以直接进入回答阶段。",
            "expand": "当前片段相关，但缺少相邻上下文、图表、表格或必要说明，应补充局部信息。",
            "search": "当前证据不相关，或者还需要寻找其他页面中的信息。",
        },
    )
    return {"relevant": relevant, "sufficient": sufficient, "next_action": next_action}


def load_samples():
    with open(SAMPLES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def upsert(results, entry):
    for i, r in enumerate(results):
        if r.get("id") == entry.get("id"):
            results[i] = entry
            return results
    results.append(entry)
    return results


def save_results(results):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TypeSafe Jev 真实 API 测试")
    parser.add_argument("--sample", default="A",
                        help="A / B / C / all，或逗号分隔（如 A,B）")
    args = parser.parse_args()

    # 1) 密钥只从环境变量读取；缺失则明确提示，绝不生成假密钥
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        print("[错误] 当前进程读不到环境变量 TYPESAFE_API_KEY。")
        print("请在你已配置该变量的那个 PowerShell 终端里运行：")
        print("    py -3.12 test_jev.py --sample A")
        print("本脚本不会生成假密钥，也不会有任何模拟返回。")
        sys.exit(1)

    samples = load_samples()
    by_id = {s["id"]: s for s in samples}
    chosen = list(by_id.keys()) if args.sample.lower() == "all" \
        else [x.strip().upper() for x in args.sample.split(",") if x.strip()]

    client = TypeSafeClient(api_key=api_key, base_url=BASE_URL, model=MODEL_NAME)

    try:
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
    except Exception:
        results = []

    for sid in chosen:
        sample = by_id.get(sid)
        if not sample:
            print(f"[跳过] 未知样本编号: {sid}")
            continue

        state = f"Question: {sample['question']}\n\nEvidence:\n{sample['evidence']}"
        entry = {
            "id": sid,
            "question": sample["question"],
            "evidence": sample["evidence"],
            "human_expected": sample.get("human_expected"),
            "requested_model": MODEL_NAME,
            "model": None,
            "relevant": None,
            "sufficient": None,
            "next_action": None,
            "next_action_probabilities": None,
            "confidence": None,
            "input_tokens": None,
            "output_tokens": None,
            "latency_sec": None,
            "success": False,
            "error": "",
        }

        t0 = time.time()
        try:
            resp = client.system_one(
                state=state,
                questions=build_questions(),
                model=MODEL_NAME,
            )
            answers = resp.answers
            ra = answers.get("relevant")
            sa = answers.get("sufficient")
            na = answers.get("next_action")

            entry["model"] = getattr(resp, "model", None)
            entry["relevant"] = None if ra is None else float(ra.noul)
            entry["sufficient"] = None if sa is None else float(sa.noul)
            if na is not None:
                entry["next_action"] = na.choice
                entry["next_action_probabilities"] = dict(na.probabilities)
                entry["confidence"] = float(na.confidence)
            usage = getattr(resp, "usage", None)
            if usage is not None:
                entry["input_tokens"] = getattr(usage, "input_tokens", None)
                entry["output_tokens"] = getattr(usage, "output_tokens", None)
            entry["success"] = True
        except Exception as e:  # 失败如实记录，不重试、不伪造
            msg = str(e)
            if api_key:
                msg = msg.replace(api_key, "***")  # 防止密钥意外出现在错误信息里
            entry["error"] = f"{type(e).__name__}: {msg[:400]}"

        entry["latency_sec"] = round(time.time() - t0, 3)
        results = upsert(results, entry)
        save_results(results)

        if entry["success"]:
            print(f"[{sid}] success=True model={entry['model']} "
                  f"relevant={entry['relevant']} sufficient={entry['sufficient']} "
                  f"next_action={entry['next_action']} conf={entry['confidence']} "
                  f"tokens={entry['input_tokens']}/{entry['output_tokens']} "
                  f"latency={entry['latency_sec']}s")
            print(f"     预期={entry['human_expected']} 概率={entry['next_action_probabilities']}")
        else:
            print(f"[{sid}] success=False latency={entry['latency_sec']}s error={entry['error']}")

    print(f"\n结果已写入: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
