# -*- coding: utf-8 -*-
"""
DeepSeek V4.1 Flash 版多模态问答脚本。

说明：
- 本文件是 run_advanced_rag.py 的副本，不改动原文件。
- 仅把模型从 qwen3-vl-plus 换成 deepseek-v4.1-flash-1017；
  题目、PDF 页面渲染、图片输入格式、专家提示词、evidence_pages 选择方式全部保持不变。
- API Key 只从环境变量 DASHSCOPE_API_KEY 读取，不硬编码、不打印真实密钥。
- 结果写入独立的 JSONL（默认 advanced_rag_results_ds.jsonl），不覆盖任何原始结果文件。

每条结果新增字段：
  ds_answer         模型回答文本（失败时为空字符串）
  ds_success        调用是否成功（API 成功且回答非空才为 True）
  ds_response_model 服务端实际返回的模型名
  ds_usage          token 用量（prompt/completion/total）
  ds_latency_sec    本次调用耗时（秒）
  ds_error          失败原因（API 报错或空回答），成功时为空字符串
  ds_image_count    本次实际送入的页面图片数量
  ds_pages_rendered 本次实际渲染的页码（1-indexed）

用法示例：
  python run_advanced_rag_ds.py --limit 1  --out advanced_rag_results_ds_smoke1.jsonl
  python run_advanced_rag_ds.py --limit 10 --out advanced_rag_results_ds_smoke10.jsonl
  python run_advanced_rag_ds.py --limit 0  --out advanced_rag_results_ds.jsonl   # 0 = 全部（暂不运行）
"""
import argparse
import ast
import base64
import json
import os
import sys
import time

import fitz  # PyMuPDF
from openai import OpenAI

# 1. 基础配置
MODEL_NAME = "deepseek-v4.1-flash-1017"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 密钥只从环境变量读取；缺失时明确提示，绝不生成/硬编码假密钥
api_key = os.environ.get("DASHSCOPE_API_KEY")
if not api_key:
    print("[错误] 未检测到环境变量 DASHSCOPE_API_KEY。")
    print("请先在系统环境变量中配置新的 DashScope 密钥（并撤销可能已泄露的旧密钥），")
    print("然后关闭并重新打开终端，再运行本脚本。")
    sys.exit(1)

client = OpenAI(
    api_key=api_key,
    base_url=BASE_URL,
    timeout=120.0,
    max_retries=1,
)

# prepare_documents.py 下载好的 PDF 缓存目录（系统临时目录，不进项目）
PDF_DIR = os.path.join(os.environ.get("TEMP", "."), "mmlongbench_pdfs")
MAX_PAGES = 10        # 单题最多渲染关键页数
RENDER_DPI = 300      # 目标渲染 DPI
MAX_RENDER_DIM = 2000 # 单页最长边像素上限


# 2. 多专家路由器（与原始脚本完全一致）
def get_expert_prompt(doc_type):
    doc_type_lower = doc_type.lower()
    if "financial" in doc_type_lower or "report" in doc_type_lower:
        return "你是一位精通财务与数据分析的专家。请极其仔细地核对数字、表格和百分比，给出精确、简洁的答案。"
    elif "academic" in doc_type_lower or "research" in doc_type_lower:
        return "你是一位严谨的学术研究员。请专注于提取论文中的核心方法、实验结论和定义，避免过度推测。"
    elif "tutorial" in doc_type_lower or "guidebook" in doc_type_lower:
        return "你是一位资深的技术文档专家。请精准定位步骤、操作指南和属性说明，直接给出正确结论。"
    else:
        return "你是一个精准的科研问答助手，请结合文档图片直接回答问题，尽可能简短准确。"


# 2.5 不可答题强约束（与原始脚本完全一致）
NOT_ANSWERABLE_RULE = (
    "【极度重要·不可答规则】只有当你能在提供的文档图片中找到确凿证据时，才给出具体答案。"
    "如果图片中找不到足够信息（信息缺失、页码对不上、图中不含相关内容），"
    "你的回答必须以短语 'Not answerable' 开头，表示此题不可答。"
    "严禁在这种情况下编造数字或结论、严禁臆测、不要使用外部常识。"
)


# 3. 解析 evidence_pages 字符串（与原始脚本完全一致）
def parse_evidence_pages(ep_str):
    try:
        lst = ast.literal_eval(ep_str)
        return [int(x) for x in lst]
    except Exception:
        return []


# 4. 页面渲染（与原始脚本完全一致）
def effective_dpi(page, target_dpi=RENDER_DPI, max_dim=MAX_RENDER_DIM):
    longest = max(page.rect.width, page.rect.height)
    if longest * target_dpi / 72.0 <= max_dim:
        return target_dpi
    return int(max_dim * 72.0 / longest)


def render_pages_to_base64(doc_id, page_numbers, pdf_dir=PDF_DIR, dpi=RENDER_DPI):
    pdf_path = os.path.join(pdf_dir, doc_id)
    images = []
    try:
        doc = fitz.open(pdf_path)
        for pnum in page_numbers:
            idx = pnum - 1  # evidence_pages 是 1-indexed
            if idx < 0 or idx >= doc.page_count:
                continue
            page = doc[idx]
            d = effective_dpi(page, dpi)
            pix = page.get_pixmap(dpi=d)
            b64 = base64.b64encode(pix.tobytes("png")).decode("ascii")
            images.append("data:image/png;base64," + b64)
        doc.close()
    except Exception as e:
        print(f"[渲染失败] {doc_id}: {e}")
    return images


def main():
    parser = argparse.ArgumentParser(description="DeepSeek V4.1 Flash 多模态问答")
    parser.add_argument("--limit", type=int, default=10,
                        help="只评测前 N 道题；0 表示全部（默认 10，防止误跑 500 题）")
    parser.add_argument("--input", default="sampled_500_for_baseline.jsonl")
    parser.add_argument("--out", default="advanced_rag_results_ds.jsonl")
    args = parser.parse_args()

    print(f"开始运行【DeepSeek V4.1 Flash 多模态版】 model={MODEL_NAME} limit={args.limit}")

    n_done = n_success = n_error = n_empty = 0
    sum_tokens = 0
    sum_latency = 0.0

    with open(args.input, "r", encoding="utf-8") as f_in, \
            open(args.out, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            if args.limit and idx >= args.limit:
                break
            data = json.loads(line)
            question = data.get("question", "")
            doc_type = data.get("doc_type", "General")
            doc_id = data.get("doc_id", "")

            page_numbers = parse_evidence_pages(data.get("evidence_pages", "[]"))
            if not page_numbers:
                page_numbers = [1]
            page_numbers = page_numbers[:MAX_PAGES]

            images = render_pages_to_base64(doc_id, page_numbers)
            if not images:
                print(f"[警告] 第 {idx + 1} 题未能渲染任何页面图片（PDF 缺失或页码越界）: {doc_id}")

            expert_system_prompt = get_expert_prompt(doc_type) + "\n\n" + NOT_ANSWERABLE_RULE

            content = [{"type": "image_url", "image_url": {"url": img}} for img in images]
            content.append({"type": "text", "text": question})

            # 先置为失败态，确保报错不会被当作正常回答
            ds_success = False
            ds_answer = ""
            ds_error = ""
            ds_resp_model = None
            ds_usage = None

            t0 = time.time()
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": expert_system_prompt},
                        {"role": "user", "content": content},
                    ],
                    max_tokens=1024,
                    extra_body={"enable_thinking": False},
                )
                ds_resp_model = getattr(response, "model", None)
                ds_answer = response.choices[0].message.content or ""
                if ds_answer.strip():
                    ds_success = True
                else:
                    ds_error = "EMPTY_RESPONSE: 模型返回空回答"
                u = getattr(response, "usage", None)
                if u is not None:
                    ds_usage = {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                    }
                    sum_tokens += u.total_tokens
            except Exception as e:
                ds_error = f"{type(e).__name__}: {str(e)[:500]}"
                ds_answer = ""

            latency = round(time.time() - t0, 3)
            sum_latency += latency

            data["ds_answer"] = ds_answer
            data["ds_success"] = ds_success
            data["ds_response_model"] = ds_resp_model
            data["ds_usage"] = ds_usage
            data["ds_latency_sec"] = latency
            data["ds_error"] = ds_error
            data["ds_image_count"] = len(images)
            data["ds_pages_rendered"] = page_numbers
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
            f_out.flush()

            n_done += 1
            if ds_success:
                n_success += 1
            else:
                n_error += 1
                if ds_error.startswith("EMPTY_RESPONSE"):
                    n_empty += 1

            shown = ds_answer.replace("\n", " ")[:50]
            print(f"[{idx + 1}] success={ds_success} imgs={len(images)} "
                  f"model={ds_resp_model} latency={latency}s "
                  f"answer={shown!r} err={ds_error[:90]}")

    print("\n" + "=" * 60)
    print(f"完成 {n_done} 题：成功 {n_success}，失败 {n_error}（其中空回答 {n_empty}）")
    print(f"总 token 用量: {sum_tokens} | 平均耗时: "
          f"{(sum_latency / n_done) if n_done else 0:.2f}s")
    print(f"结果已写入: {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
