#!/usr/bin/env python3
# framework/batch_classify_bugs.py
# 
# 缺陷库构建核心模块：使用 SiliconFlow Batch API 进行低成本、大规模缺陷分类。
# 架构特性：映射器模式解耦元数据、自动分块(5000/file)、支持断点续传监控。

import os
import json
import time
import uuid
import requests
import re
from openai import OpenAI

# ==========================================
# 1. 核心配置与全局变量
# ==========================================
API_KEY = os.getenv("SILICONCLOUD_API_KEY")
if not API_KEY:
    raise ValueError("[Error] 请设置环境变量 'SILICONCLOUD_API_KEY'")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.siliconflow.cn/v1",
)

# 使用官方 Batch API 明确支持的模型列表中的模型
MODEL_NAME = "deepseek-ai/DeepSeek-V3" 
MAX_LINES_PER_BATCH = 4900 # 留出安全余量（官方限制 5000）

# 路径配置
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bug_classification'))
INPUT_FILE = os.path.join(BASE_DIR, 'parsed_data_tmp.jsonl')
OUTPUT_FILE = os.path.join(BASE_DIR, 'classified_data_batch.jsonl')

# 状态管理文件（用于断点续传）
STATE_FILE = os.path.join(BASE_DIR, 'batch_state.json')
MAPPING_FILE = os.path.join(BASE_DIR, 'batch_mapping.json')

# ODC 与 CWE 分类标准
ODC_LABELS = [
    "Assignment", "Checking", "Algorithm", "Interface", 
    "Timing/Serialization", "Build/Package/Merge", "Documentation", 
    "Function/LogicFlow", "Other"
]

CWE_LABELS = [
    "CWE-20: Improper Input Validation", "CWE-16: Configuration", "CWE-119: Memory/Buffer Boundary",
    "CWE-284: Access Control", "CWE-362: Race Condition", "CWE-400: Resource Exhaustion / Memory Leak",
    "CWE-573: Logic Mismatch", "CWE-628: API Misuse", "CWE-664: Resource Lifecycle",
    "CWE-665: Improper Initialization", "CWE-682: Calculation Error", "CWE-704: Type/Cast Error",
    "CWE-707: Data Encoding", "CWE-754: Missing Check", "CWE-1116: Wrong Comments",
    "CWE-1357: Dependency Issue", "Other"
]

# 在提示词中更加强硬地要求仅输出 JSON
SYSTEM_PROMPT = f"""
You are an expert software engineer and researcher specializing in bug triaging.
Your task is to classify software bug reports into TWO distinct, independent taxonomies:
1. ODC (Orthogonal Defect Classification): Based on the nature of the *code fix*.
2. CWE (Common Weakness Enumeration): Based on the *root cause weakness*.

[Taxonomy 1: ODC Categories]
{json.dumps(ODC_LABELS)}

[Taxonomy 2: CWE Categories]
{json.dumps(CWE_LABELS)}

Instructions:
1. Select EXACTLY ONE category from Taxonomy 1 (ODC) and Taxonomy 2 (CWE).
2. You MUST output your response strictly as a JSON object with two keys: "odc" and "cwe". 
3. Output ONLY the raw JSON object. Do NOT wrap it in markdown code blocks (e.g., no ```json). Do not add any conversational text.
"""

# ==========================================
# 2. 准备阶段：数据封装与上传
# ==========================================
def prepare_and_submit_batches():
    print(f"--- 阶段 1: 解析输入并生成 Batch 格式数据 ---")
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

    original_data_map = {}
    batch_requests = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            try:
                data = json.loads(line)
                bug_text = data.get("llm_input_text")
                if not bug_text: continue

                # 创建唯一的 custom_id
                custom_id = f"req-{uuid.uuid4().hex[:8]}-{idx}"
                
                # 剥离并保存原始元数据
                original_data_map[custom_id] = {
                    "project_id": data.get("project_id"),
                    "bug_id": data.get("bug_id"),
                    "source_type": data.get("source_type"),
                    "llm_input_text": bug_text
                }

                # 按照 SiliconFlow Batch API 格式组装
                req_obj = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL_NAME,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": bug_text}
                        ],
                        "temperature": 0
                        # 【完全移除 JSON Mode】因当前模型批处理不支持，我们在后续解析时使用正则弥补
                    }
                }
                batch_requests.append(req_obj)
            except json.JSONDecodeError:
                print(f"[Warning] 跳过无效 JSON 行: {idx}")

    # 保存映射字典到本地
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(original_data_map, f, ensure_ascii=False)
    
    print(f"共解析 {len(batch_requests)} 条有效缺陷记录。正在分块并提交...")

    # 分块提交
    batch_ids = []
    chunks = [batch_requests[i:i + MAX_LINES_PER_BATCH] for i in range(0, len(batch_requests), MAX_LINES_PER_BATCH)]
    
    for i, chunk in enumerate(chunks):
        temp_file = os.path.join(BASE_DIR, f'temp_batch_input_{i}.jsonl')
        with open(temp_file, 'w', encoding='utf-8') as f:
            for req in chunk:
                f.write(json.dumps(req, ensure_ascii=False) + '\n')
        
        print(f"  -> 正在上传分块 {i+1}/{len(chunks)}...")
        with open(temp_file, "rb") as file_data:
            batch_input_file = client.files.create(file=file_data, purpose="batch")
            
            # 兼容 SiliconFlow 特殊的响应结构提取真实 file_id
            file_id = None
            if hasattr(batch_input_file, 'model_dump'):
                dump = batch_input_file.model_dump()
                if 'data' in dump and isinstance(dump['data'], dict) and 'id' in dump['data']:
                    file_id = dump['data']['id']
                elif 'id' in dump and dump['id']:
                    file_id = dump['id']
            
            if not file_id and getattr(batch_input_file, 'id', None):
                file_id = batch_input_file.id
                
            if not file_id:
                raise ValueError(f"[Error] 无法从响应中解析 file_id。原始响应: {batch_input_file}")

            print(f"  -> 文件上传成功，获取到 File ID: {file_id}")
            time.sleep(2) 
        
        print(f"  -> 正在创建批处理任务 {i+1}...")
        batch_job = client.batches.create(
            input_file_id=file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Bug Classification Part {i+1}"},
            extra_body={"replace": {"model": MODEL_NAME}} 
        )
        batch_ids.append(batch_job.id)
        
        # 清理临时文件
        os.remove(temp_file)

    # 保存任务状态
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump({"batch_ids": batch_ids}, f)
    
    print(f"成功提交 {len(batch_ids)} 个 Batch 任务。")
    return batch_ids

# ==========================================
# 3. 监控阶段：轮询任务状态
# ==========================================
def monitor_batches(batch_ids):
    print(f"\n--- 阶段 2: 监控 Batch 任务状态 ---")
    print("你可以随时按 Ctrl+C 中止此脚本。下次运行时，它将自动从此处恢复。")
    
    completed_batches = []
    
    while len(completed_batches) < len(batch_ids):
        all_done = True
        for b_id in batch_ids:
            if b_id in completed_batches: continue
            
            try:
                batch_status = client.batches.retrieve(b_id)
                status = batch_status.status
                
                req_counts = getattr(batch_status, 'request_counts', None)
                if req_counts:
                    prog_str = f"{getattr(req_counts, 'completed', 0)}/{getattr(req_counts, 'total', 0)}"
                else:
                    prog_str = "排队/准备中"
                
                print(f"任务 [{b_id}]: {status} (进度: {prog_str})")
                
                if status in ["completed", "failed", "expired", "cancelled"]:
                    completed_batches.append(batch_status)
                else:
                    all_done = False
            except Exception as e:
                print(f"[Error] 获取状态失败 {b_id}: {e}")
                all_done = False
                
        if not all_done:
            time.sleep(30) # 每30秒轮询一次

    return completed_batches

# ==========================================
# 4. 汇总阶段：解析结果并映射还原
# ==========================================
def process_results(completed_batches):
    print(f"\n--- 阶段 3: 下载并还原结果 ---")
    
    # 读取原始映射字典
    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        original_data_map = json.load(f)

    processed_count = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for batch in completed_batches:
            if batch.status != "completed":
                print(f"[Warning] 任务 {batch.id} 未成功完成，状态为 {batch.status}。")
                
            error_file_id = getattr(batch, 'error_file_id', None)
            if error_file_id:
                print(f"\n  -> [排障分析] 发现 API 返回了错误文件！正在下载具体错误原因 (File ID/URL: {error_file_id})...")
                try:
                    if error_file_id.startswith("http://") or error_file_id.startswith("https://"):
                        err_content = requests.get(error_file_id).text
                    else:
                        err_content = client.files.content(error_file_id).text
                        
                    print(f"  --- 错误日志预览 (前 3 条) ---")
                    lines = [line for line in err_content.strip().split('\n') if line]
                    for err_line in lines[:3]:
                        print(f"     {err_line}")
                    if len(lines) > 3:
                        print(f"     ... (共拦截了 {len(lines)} 条失败记录)")
                    print(f"  ---------------------------\n")
                except Exception as e:
                    print(f"  -> [Error] 无法下载错误日志: {e}")

            output_file_id = getattr(batch, 'output_file_id', None)
            if not output_file_id:
                print(f"  -> [Info] 任务 {batch.id} 没有生成有效的成功输出文件。")
                continue
            
            print(f"  -> 正在下载结果文件: {output_file_id}")
            try:
                if output_file_id.startswith("http://") or output_file_id.startswith("https://"):
                    result_content = requests.get(output_file_id).text
                else:
                    result_content = client.files.content(output_file_id).text
            except Exception as e:
                print(f"  -> [Error] 无法下载结果文件: {e}")
                continue
            
            for line in result_content.strip().split('\n'):
                if not line: continue
                try:
                    res_data = json.loads(line)
                    custom_id = res_data.get("custom_id")
                    
                    # 【核心抗脆弱设计】容错提取模型输出，使用正则对抗不稳定的格式
                    try:
                        raw_llm_output = res_data["response"]["body"]["choices"][0]["message"]["content"].strip()
                        
                        # 正则提取 JSON 对象：找到第一个 '{' 到最后一个 '}' 之间的内容
                        json_match = re.search(r'\{.*\}', raw_llm_output, re.DOTALL)
                        if json_match:
                            raw_llm_output = json_match.group(0)
                            
                        parsed_json = json.loads(raw_llm_output)
                        odc_final = parsed_json.get("odc", "Other") if parsed_json.get("odc") in ODC_LABELS else "Other"
                        
                        raw_cwe = parsed_json.get("cwe", "Other")
                        cwe_final = next((valid for valid in CWE_LABELS if raw_cwe.split(':')[0] in valid), "Other")
                    except (KeyError, json.JSONDecodeError, TypeError) as e:
                        # 解析失败退化为 Other
                        odc_final, cwe_final = "Other", "Other"

                    # 还原为原格式
                    original_record = original_data_map.get(custom_id)
                    if original_record:
                        final_record = {
                            "project_id": original_record["project_id"],
                            "bug_id": original_record["bug_id"],
                            "source_type": original_record["source_type"],
                            "label": [odc_final, cwe_final],
                            "llm_input_text": original_record["llm_input_text"]
                        }
                        out_f.write(json.dumps(final_record, ensure_ascii=False) + '\n')
                        processed_count += 1

                except json.JSONDecodeError:
                    print(f"[Warning] 批量结果行读取失败。")

    print(f"\n✅ 处理完成！共生成 {processed_count} 条分类数据。")
    print(f"文件已保存至: {OUTPUT_FILE}")
    
    # 善后清理
    if os.path.exists(STATE_FILE): os.remove(STATE_FILE)
    if os.path.exists(MAPPING_FILE): os.remove(MAPPING_FILE)

def main():
    # 断点续传逻辑
    if os.path.exists(STATE_FILE):
        print("检测到未完成的 Batch 任务记录，跳过上传阶段，尝试恢复监控...")
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
            batch_ids = state.get("batch_ids", [])
    else:
        batch_ids = prepare_and_submit_batches()
        
    if batch_ids:
        completed_batches = monitor_batches(batch_ids)
        process_results(completed_batches)

if __name__ == "__main__":
    main()