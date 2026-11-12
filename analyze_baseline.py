import json

# 读取我们刚刚生成的成绩单
results_file = "baseline_results.jsonl"

correct_count = 0
total_count = 0
error_cases = []

print("正在批改考卷并生成错题本...")

with open(results_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        total_count += 1
        
        question = data.get("question", "")
        gold_answer = str(data.get("answer", "")).strip()
        model_answer = str(data.get("deepseek_answer", "")).strip()
        doc_type = data.get("doc_type", "Unknown")
        
        # 简单的包含匹配评估（看看标准答案是否包含在模型回答中，或者完全相等）
        # 你也可以根据实际情况调整判分逻辑
        if gold_answer.lower() in model_answer.lower():
            correct_count += 1
        else:
            # 把错题记录下来
            error_cases.append({
                "doc_type": doc_type,
                "question": question,
                "gold_answer": gold_answer,
                "model_answer": model_answer
            })

# 计算准确率
accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

print(f"\n================ 考试成绩统计 ================")
print(f"总题数: {total_count}")
print(f"答对题数: {correct_count}")
print(f"Baseline 准确率: {accuracy:.2f}%")
print(f"错题总数: {len(error_cases)}")
print(f"==============================================\n")

# 把前 5 个错题打印出来，方便我们人工分析原因
print("【典型错题抽样分析（前 5 个）：】\n")
for i, case in enumerate(error_cases[:5], 1):
    print(f"--- 错题 {i} [文档类型: {case['doc_type']}] ---")
    print(f"题目: {case['question']}")
    print(f"标准答案: {case['gold_answer']}")
    print(f"模型回答: {case['model_answer']}")
    print("-" * 50)

# 顺便把所有错题单独保存到一个文件里，方便以后写论文用
with open("error_analysis_report.jsonl", "w", encoding="utf-8") as f_err:
    for case in error_cases:
        f_err.write(json.dumps(case, ensure_ascii=False) + "\n")

print("\n完整的错题本已保存到 error_analysis_report.jsonl 文件中！")