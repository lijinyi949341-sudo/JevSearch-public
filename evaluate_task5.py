#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jev 判断评价程序。

设计原则：
1. 只把 task5_annotations.jsonl 中的 human_* 字段当作人工真值；
2. suggestion_unverified 是 AI 候选，绝不参与任何指标计算；
3. 只要还有样本 review_status != confirmed 或 human_* 为 null，就拒绝计算正式准确率。
"""
import argparse
import json
import os
from collections import Counter

HUMAN_FIELDS = ["human_relevant", "human_self_contained", "human_sufficient", "human_action"]
LABELS = ["answer", "expand", "search"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def integrity_report(samples, annos, preds):
    s_ids = [r["sample_id"] for r in samples]
    a_ids = [r["sample_id"] for r in annos]
    p_ids = [r["sample_id"] for r in preds]
    dup = [k for k, v in Counter(p_ids).items() if v > 1]
    missing_pred = sorted(set(s_ids) - set(p_ids))
    extra_pred = sorted(set(p_ids) - set(s_ids))
    missing_anno = sorted(set(s_ids) - set(a_ids))
    failed = sorted(r["sample_id"] for r in preds if not r.get("success"))

    print("=== 完整性检查 ===")
    print(f"样本 {len(samples)} 条 / 标注 {len(annos)} 条 / 预测 {len(preds)} 条")
    print(f"重复预测: {dup or '无'}")
    print(f"缺失预测: {missing_pred or '无'}")
    print(f"多余预测: {extra_pred or '无'}")
    print(f"缺失标注: {missing_anno or '无'}")
    print(f"调用失败: {failed or '无'}")
    ok = not (dup or missing_pred or extra_pred or missing_anno or failed)
    print(f"结论: {'通过' if ok else '不通过'}")
    return ok


def human_complete(anno):
    if anno.get("review_status") != "confirmed":
        return False
    return all(anno.get(f) is not None for f in HUMAN_FIELDS)


def main():
    ap = argparse.ArgumentParser(description="Jev 判断评价（需人工标签全部确认）")
    ap.add_argument("--samples", default="task5_samples.jsonl")
    ap.add_argument("--annotations", default="task5_annotations.jsonl")
    ap.add_argument("--predictions", default=os.path.join("outputs", "task5", "jev_predictions.jsonl"))
    ap.add_argument("--out", default=os.path.join("outputs", "task5", "evaluation.json"))
    args = ap.parse_args()

    samples = load_jsonl(args.samples)
    annos = load_jsonl(args.annotations)
    preds = load_jsonl(args.predictions)

    if not integrity_report(samples, annos, preds):
        print("\n存在完整性问题，终止。")
        return 1

    confirmed = [a for a in annos if human_complete(a)]
    excluded = [a for a in annos if not human_complete(a)]
    print(f"\n=== 人工标签状态 ===")
    print(f"有效（已确认）: {len(confirmed)}/{len(annos)}")
    print(f"排除: {len(excluded)} 条 -> {', '.join(a['sample_id'] for a in excluded) or '无'}")
    print(f"排除原因分布: {dict(Counter(a.get('review_status') for a in excluded))}")
    if not confirmed:
        print("\n没有已确认样本，无法计算指标。候选标签不能当作 Ground Truth。")
        print("请先填写 task5_annotations.jsonl 的 human_* 并把 review_status 改为 confirmed。")
        return 2

    by_id = {a["sample_id"]: a for a in confirmed}
    pmap = {p["sample_id"]: p for p in preds}
    preds_eval = [pmap[sid] for sid in by_id]

    cm = {h: {p: 0 for p in LABELS} for h in LABELS}
    errors = []
    for sid, anno in by_id.items():
        pred = pmap[sid]
        h, p = anno["human_action"], pred["next_action"]
        cm[h][p] += 1
        row = {
            "sample_id": sid,
            "human_action": h,
            "pred_action": p,
            "human": {f.replace("human_", ""): anno.get(f) for f in HUMAN_FIELDS},
            "pred": {
                "relevant": pred["relevant"],
                "self_contained": pred["self_contained"],
                "sufficient": pred["sufficient"],
                "probabilities": pred.get("next_action_probabilities"),
            },
        }
        if h != p or any(row["human"][k] != pred[k] for k in ("relevant", "self_contained", "sufficient")):
            errors.append(row)

    n = len(by_id)
    n_total = len(annos)
    acc = sum(cm[c][c] for c in LABELS) / n

    per_class = {}
    for c in LABELS:
        tp = cm[c][c]
        pred_n = sum(cm[h][c] for h in LABELS)
        human_n = sum(cm[c][p] for p in LABELS)
        precision = tp / pred_n if pred_n else None
        recall = tp / human_n if human_n else None
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1,
                        "support_human": human_n, "predicted_n": pred_n}

    expand_support = sum(1 for a in by_id.values() if a["human_action"] == "expand")
    answer_support = sum(1 for a in by_id.values() if a["human_action"] == "answer")
    missed = [sid for sid, a in by_id.items() if a["human_action"] == "expand" and pmap[sid]["next_action"] == "answer"]
    over = [sid for sid, a in by_id.items() if a["human_action"] == "answer" and pmap[sid]["next_action"] == "expand"]
    missed_search = [sid for sid, a in by_id.items() if a["human_action"] == "search" and pmap[sid]["next_action"] == "answer"]

    total_in = sum(p.get("input_tokens", 0) for p in preds_eval)
    total_out = sum(p.get("output_tokens", 0) for p in preds_eval)
    result = {
        "meta": {
            "n_samples_total": n_total,
            "n_evaluated": n,
            "n_excluded": n_total - n,
            "excluded_samples": [
                {"sample_id": a["sample_id"], "review_status": a.get("review_status")}
                for a in excluded
            ],
            "models": sorted({p.get("model") for p in preds_eval}),
            "prompt_versions": sorted({p.get("prompt_version") for p in preds_eval}),
            "note": "指标仅使用已确认样本的 human_* 人工标签；未确认/uncertain 样本被排除；suggestion_unverified 未参与计算。",
        },
        "action_accuracy": acc,
        "confusion_matrix": cm,
        "per_class": per_class,
        "expand": per_class["expand"],
        "under_expansion": {
            "definition": "human=expand 且 pred=answer 的比例（需要扩展却被判可直接回答）",
            "count": len(missed), "support": expand_support,
            "rate": len(missed) / expand_support if expand_support else None,
            "samples": missed,
        },
        "over_expansion": {
            "definition": "human=answer 且 pred=expand 的比例（已足够回答却被判需扩展）",
            "count": len(over), "support": answer_support,
            "rate": len(over) / answer_support if answer_support else None,
            "samples": over,
        },
        "missed_search": {
            "definition": "human=search 且 pred=answer 的比例",
            "count": len(missed_search),
            "support": sum(1 for a in by_id.values() if a["human_action"] == "search"),
            "samples": missed_search,
        },
        "errors": errors,
        "calls": {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "avg_latency_sec": sum(p.get("latency_sec", 0) for p in preds_eval) / len(preds_eval),
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== 正式评价 ===")
    print(f"有效样本: {n}/{n_total}（排除 {n_total - n} 条未确认/uncertain）")
    print(f"动作三分类准确率: {acc:.2%} ({sum(cm[c][c] for c in LABELS)}/{n})")
    print("混淆矩阵（行=人工，列=预测）:")
    print("          " + " ".join(f"{c:>8s}" for c in LABELS))
    for h in LABELS:
        print(f"{h:>8s}  " + " ".join(f"{cm[h][p]:>8d}" for p in LABELS))
    print(f"expand Precision={per_class['expand']['precision']} Recall={per_class['expand']['recall']}")
    print(f"漏扩展率={result['under_expansion']['rate']} 过度扩展率={result['over_expansion']['rate']}")
    print(f"结果写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
