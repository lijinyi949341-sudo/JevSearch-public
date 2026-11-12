import json
import os
import sys

from openai import OpenAI

# 密钥只从环境变量读取，不写入仓库
YOUR_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not YOUR_API_KEY:
    sys.exit("[错误] 未检测到环境变量 DASHSCOPE_API_KEY。")
MODEL_NAME = "deepseek-v4.1-flash-1017"

client = OpenAI(
    api_key=YOUR_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0,
    max_retries=1
)

# 2. 设定考卷文件和要生成的成绩单文件
input_file = "sampled_500_for_baseline.jsonl"
output_file = "baseline_results.jsonl"

print(f"开始让 DeepSeek 狂刷 500 道题，请耐心等待...")

# 3. 开启循环：一边读题，一边把答案写进新文件
with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    count = 1
    for line in f_in:
        data = json.loads(line)
        question_text = data["question"]
        
        # 呼叫大模型做题
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个精准的科研问答助手，请直接回答问题，尽可能简短。"},
                    {"role": "user", "content": question_text}
                ],
                max_tokens=1024,
                extra_body={"enable_thinking": False}
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            answer = f"API请求报错: {str(e)}"
            
        # 将大模型的答案塞进原来的数据字典里
        data["deepseek_answer"] = answer
        
        # 把带有答案的完整数据保存到新文件
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"已完成第 {count} 题...")
        count += 1

print(f"\n大功告成！所有答案已存入左侧的 {output_file} 文件中！")