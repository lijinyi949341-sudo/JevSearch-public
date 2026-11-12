import ast
import json
import os
import re

baseline_file = "baseline_results.jsonl"
advanced_file = "advanced_rag_results.jsonl"
# 本地评测集，不随仓库发布；可用 JEV_GOLD_FILE 指向其他位置。
gold_file = os.environ.get("JEV_GOLD_FILE", "sampled_500_for_baseline.jsonl")


def loose_norm(s):
    """宽松归一化：仅保留字母数字，忽略大小写、标点、空格、括号、连字符、单位等。"""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def gold_items(gold):
    """gold 可能是 Python list 字面量字符串（如 "['a', 'b']"），拆成元素逐一匹配。"""
    s = str(gold).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            items = ast.literal_eval(s)
            if isinstance(items, (list, tuple)):
                return [str(x) for x in items]
        except Exception:
            pass
    return [s]


def match_strict(gold, pred):
    """旧口径：gold 精确子串匹配（与上一版 compare_results.py 完全一致）。"""
    g = str(gold).strip().lower()
    p = str(pred).strip().lower()
    return bool(g) and g in p


def match_lenient(gold, pred):
    """新口径：宽松匹配——List 逐元素 + 忽略大小写/标点/空格/括号/单位。"""
    p = loose_norm(pred)
    for item in gold_items(gold):
        g = loose_norm(item)
        if not g:
            continue
        if g not in p:
            return False
    return True


def load_gold():
    """从本地未提交的评测集读取 gold；缺失时明确失败，避免误报 0%/100%。"""
    if not os.path.exists(gold_file):
        raise SystemExit(
            "[错误] 未找到本地评测集 %s，无法计算准确率。\n"
            "该文件不随仓库发布；请按 README 第 9 节获取 MMLongBench-Doc 数据，"
            "并运行 prepare_data.py 生成同一 500 题样本。" % gold_file
        )
    rows = []
    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit("[错误] 本地评测集 %s 为空，无法计算准确率。" % gold_file)
    return rows


def load_rows(path, answer_key, gold_rows):
    """按行对齐预测与本地 gold；doc_id/question 不一致时直接报错，不静默计算。"""
    preds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    if len(preds) != len(gold_rows):
        raise SystemExit(
            "[错误] %s 有 %d 行，本地评测集有 %d 行，无法对齐。"
            % (path, len(preds), len(gold_rows))
        )
    rows = []
    for i, (d, g) in enumerate(zip(preds, gold_rows), 1):
        if d.get("doc_id") != g.get("doc_id") or \
                str(d.get("question", "")).strip() != str(g.get("question", "")).strip():
            raise SystemExit(
                "[错误] %s 第 %d 行与本地评测集不一致（doc_id/question 不匹配），"
                "请确认预测文件与同一 500 题样本对应。" % (path, i)
            )
        rows.append((g.get("answer", ""), d.get(answer_key, "")))
    return rows


def evaluate(rows):
    strict = sum(1 for g, p in rows if match_strict(g, p))
    lenient = sum(1 for g, p in rows if match_lenient(g, p))
    return strict, lenient, len(rows)


def mcnemar_lenient(rows_a, rows_b):
    """配对 McNemar 检验（基于宽松匹配口径的 discordant 对 b/c）。"""
    b = c = 0
    for (ga, pa), (gb, pb) in zip(rows_a, rows_b):
        a_ok = match_lenient(ga, pa)
        b_ok = match_lenient(gb, pb)
        if a_ok and not b_ok:
            b += 1  # baseline 对、advanced 错
        elif not a_ok and b_ok:
            c += 1  # baseline 错、advanced 对
    if b + c == 0:
        return None
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = chi2_dist.sf(chi2, df=1)
    except Exception:
        p = None
    return chi2, p, b, c


gold_rows = load_gold()
base_rows = load_rows(baseline_file, "deepseek_answer", gold_rows)
adv_rows = load_rows(advanced_file, "advanced_answer", gold_rows)

base_strict, base_lenient, total = evaluate(base_rows)
adv_strict, adv_lenient, _ = evaluate(adv_rows)

print("================ 实验效果对比报告（宽松匹配口径） ================")
print(f"总测试样本数: {total}")
print("判分口径: 宽松匹配 = 忽略大小写/标点/空格/括号/单位 + List 逐元素匹配")
print("-" * 60)
print(f"{'指标':<26}{'Baseline':>14}{'多模态 VLM':>14}")
print(f"{'严格子串匹配(strict)':<24}{base_strict/total*100:>13.2f}%{adv_strict/total*100:>13.2f}%")
print(f"{'宽松匹配(lenient)':<24}{base_lenient/total*100:>13.2f}%{adv_lenient/total*100:>13.2f}%")
print("-" * 60)
print(f"严格口径下: Baseline {base_strict}/{total} vs VLM {adv_strict}/{total}")
print(f"宽松口径下: Baseline {base_lenient}/{total} vs VLM {adv_lenient}/{total}")
print("-" * 60)
diff_strict = adv_strict - base_strict
diff_lenient = adv_lenient - base_lenient
print(f"宽松口径提升幅度: {'+' if diff_lenient >= 0 else ''}{diff_lenient/total*100:.2f}%  "
      f"({base_lenient} -> {adv_lenient})")
print(f"（对比：严格口径提升幅度 {'+' if diff_strict >= 0 else ''}{diff_strict/total*100:.2f}%）")

res = mcnemar_lenient(base_rows, adv_rows)
if res:
    chi2, p, b, c = res
    print("-" * 60)
    print(f"McNemar 显著性检验(宽松口径): 进步 {c} 例 / 退步 {b} 例, chi2={chi2:.2f}", end="")
    if p is not None:
        print(f", p={p:.4f}" + (" (显著)" if p < 0.05 else " (不显著)"))
    else:
        print(" (scipy 未安装，未计算 p 值)")

# 输出对比结论
print("=" * 60)
print("结论: 宽松匹配口径对 List/单位/标点/大小写差异更宽容，可更公平地反映模型能力。")
print("      严格子串匹配会把上述格式差异误判为错误。")
print("=" * 60)