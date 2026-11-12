from datasets import load_dataset

# 这里的名字要和你在 Hugging Face 网页上看到的名字完全一致
# 比如 "VLDB/MMLongBench-Doc" (具体看网页标题)
print("正在从 Hugging Face 下载数据，请耐心等待...")
dataset = load_dataset("yubo2333/MMLongBench-Doc", split="train") 
# 注意：通常评测用 test 集或者 validation(val) 集

print(f"下载成功！这个数据集一共有 {len(dataset)} 条数据。")

# 设置 seed=42 是科研里的不成文规定，保证每次“随机”抽出来的 500 条都是同一批
sampled_dataset = dataset.shuffle(seed=42).select(range(500))

print("已成功随机抽取 500 条数据！")

# 打印第一条数据看看长什么样，观察它的“键(keys)”
print("第一条数据的格式如下：")
print(sampled_dataset[0].keys())

import json

# 将抽取的数据保存到本地
output_file = "sampled_500_for_baseline.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for item in sampled_dataset:
        # 将字典转换成 JSON 字符串写入文件
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"大功告成！数据已保存到 {output_file}，快去看看吧！")