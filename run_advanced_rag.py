import ast
import base64
import json
import os

import fitz  # PyMuPDF
from openai import OpenAI

# 1. 基础配置：密钥只从环境变量读取，不写入仓库
YOUR_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
if not YOUR_API_KEY:
    raise SystemExit("[错误] 未检测到环境变量 DASHSCOPE_API_KEY。")
# DashScope 多模态理解模型（qwen-vl-max 已下线，用 qwen3-vl-plus 替代）
MODEL_NAME = "qwen3-vl-plus"

client = OpenAI(
    api_key=YOUR_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=120.0,
    max_retries=1
)

input_file = "sampled_500_for_baseline.jsonl"
output_file = "advanced_rag_results.jsonl"

# prepare_documents.py 下载好的 PDF 缓存目录（系统临时目录，不进项目）
PDF_DIR = os.path.join(os.environ.get("TEMP", "."), "mmlongbench_pdfs")
MAX_PAGES = 10   # 单个问题最多渲染的关键页面数，防止图片过大/过多
RENDER_DPI = 300      # 目标渲染 DPI（提升图表细节清晰度）
MAX_RENDER_DIM = 2000 # 单页图片最长边像素上限，防止体积过大触发 API 限制


# 2. 定义【多专家路由器】
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


# 2.5 不可答题强约束（防止模型对无法从图中作答的题盲目幻觉）
NOT_ANSWERABLE_RULE = (
    "【极度重要·不可答规则】只有当你能在提供的文档图片中找到确凿证据时，才给出具体答案。"
    "如果图片中找不到足够信息（信息缺失、页码对不上、图中不含相关内容），"
    "你的回答必须以短语 'Not answerable' 开头，表示此题不可答。"
    "严禁在这种情况下编造数字或结论、严禁臆测、不要使用外部常识。"
)


# 3. 解析 evidence_pages 字符串（形如 "[81]" / "[4, 5]" / "[]"）为 1-indexed 页码列表
def parse_evidence_pages(ep_str):
    try:
        lst = ast.literal_eval(ep_str)
        return [int(x) for x in lst]
    except Exception:
        return []


# 4. 辅助函数：用 PyMuPDF 把 PDF 的指定页渲染成 Base64 图片（data URI）
def effective_dpi(page, target_dpi=RENDER_DPI, max_dim=MAX_RENDER_DIM):
    """按页面尺寸计算实际渲染 DPI：小页面用满目标 DPI，大页面沿最长边封顶 max_dim 像素。"""
    longest = max(page.rect.width, page.rect.height)  # 单位 pt（1 inch = 72 pt）
    if longest * target_dpi / 72.0 <= max_dim:
        return target_dpi
    return int(max_dim * 72.0 / longest)


def render_pages_to_base64(doc_id, page_numbers, pdf_dir=PDF_DIR, dpi=RENDER_DPI):
    pdf_path = os.path.join(pdf_dir, doc_id)
    images = []
    try:
        doc = fitz.open(pdf_path)
        for pnum in page_numbers:
            idx = pnum - 1  # evidence_pages 是 1-indexed，PyMuPDF 是 0-indexed
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


print("开始运行【多模态 VLM 版】关键页面图片 + 专家路由评测...")

# 5. 主循环：逐条渲染关键页面图片并送入 VLM
with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for idx, line in enumerate(f_in):
        data = json.loads(line)
        question = data.get("question", "")
        doc_type = data.get("doc_type", "General")
        doc_id = data.get("doc_id", "")

        # 依据数据集 evidence_pages 定位关键页面；为空则回退到第 1 页
        page_numbers = parse_evidence_pages(data.get("evidence_pages", "[]"))
        if not page_numbers:
            page_numbers = [1]
        page_numbers = page_numbers[:MAX_PAGES]

        images = render_pages_to_base64(doc_id, page_numbers)

        expert_system_prompt = get_expert_prompt(doc_type) + "\n\n" + NOT_ANSWERABLE_RULE

        # OpenAI 兼容多模态格式：image_url 图片 + 文本问题
        content = [{"type": "image_url", "image_url": {"url": img}} for img in images]
        content.append({"type": "text", "text": question})

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": expert_system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=1024
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            answer = f"API请求报错: {str(e)}"

        data["advanced_answer"] = answer
        f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

        if (idx + 1) % 50 == 0:
            print(f"已完成评测第 {idx + 1} 题...")

print(f"\n大功告成！带有真实页面图片的多模态结果已存入 {output_file}！")