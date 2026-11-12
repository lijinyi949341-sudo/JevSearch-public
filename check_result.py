import json

# 打开我们的进阶实验结果文件
with open('advanced_rag_results.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 3:  # 只看前 3 道题
            break
        data = json.loads(line)
        print(f"题目 {i+1}: {data.get('question')}")
        print(f"标准答案: {data.get('answer')}")
        print(f"模型回答: {data.get('advanced_answer')}")
        print("-" * 50)