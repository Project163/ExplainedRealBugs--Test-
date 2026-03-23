import os
import json
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. Configuration & Taxonomies
# ==========================================

# API Key - 保持您的原始配置
api_key = os.getenv("SILICONCLOUD_API_KEY")
if not api_key:
    raise ValueError("Please set the 'SILICONCLOUD_API_KEY' environment variable")

# Initialize OpenAI client with SiliconCloud base URL
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
)

ODC_LABELS = [
    "Assignment",          # 赋值、初始化、值错误
    "Checking",            # 验证、条件判断、缺失的校验
    "Algorithm",           # 算法逻辑、计算、效率
    "Interface",           # 接口交互、参数传递、协议格式
    "Timing/Serialization",# 并发时序、竞态条件、死锁
    "Build/Package/Merge", # 构建、配置、依赖
    "Documentation",       # 文档、注释错误
    "Function/LogicFlow",  # 宏观功能、复杂业务逻辑流
    "Other"
]

CWE_LABELS = [
    "CWE-20: Improper Input Validation",
    "CWE-16: Configuration",
    "CWE-119: Memory/Buffer Boundary",
    "CWE-284: Access Control",
    "CWE-362: Race Condition",
    "CWE-400: Resource Exhaustion / Memory Leak",
    "CWE-573: Logic Mismatch",
    "CWE-628: API Misuse",
    "CWE-664: Resource Lifecycle",
    "CWE-665: Improper Initialization",
    "CWE-682: Calculation Error",
    "CWE-704: Type/Cast Error",
    "CWE-707: Data Encoding",
    "CWE-754: Missing Check",
    "CWE-1116: Wrong Comments",
    "CWE-1357: Dependency Issue",
    "Other"
]

# Input and output files
INPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'bug_classification', 'semantic_patches_ready.jsonl'))
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'bug_classification', 'classified_data_llm_cot.jsonl'))

# Concurrency settings
MAX_WORKERS = 10  
REQUEST_DELAY = 0.1

# ==========================================
# 2. Advanced System Prompt (引入 CoT 机制)
# ==========================================
# 核心优化：要求模型先输出 root_cause_analysis，再输出 odc 和 cwe，强制模型自我推理
SYSTEM_PROMPT = f"""
You are an expert software engineer and researcher specializing in bug triaging.
Your task is to classify software bug reports into TWO distinct, independent taxonomies:
1. ODC (Orthogonal Defect Classification): Based on the nature of the *code fix*.
2. CWE (Common Weakness Enumeration): Based on the *root cause weakness*.

[Taxonomy 1: ODC Categories]
{json.dumps(ODC_LABELS, indent=2)}

[Taxonomy 2: CWE Categories]
{json.dumps(CWE_LABELS, indent=2)}

Instructions:
1. Carefully analyze the bug report's Title, Symptom, Context, and Patch (if available).
2. You MUST perform a step-by-step reasoning (Chain of Thought) to identify the true root cause before assigning labels.
3. Select EXACTLY ONE category from Taxonomy 1 (ODC) and EXACTLY ONE from Taxonomy 2 (CWE).
4. Strictly return a valid JSON object with EXACTLY three keys in the following order:
   - "root_cause_analysis": A brief 1-2 sentence explanation of what went wrong and how it was/should be fixed.
   - "odc": The selected ODC category.
   - "cwe": The selected CWE category.

Example JSON output:
{{
  "root_cause_analysis": "The code fails to verify if the user input pointer is null before dereferencing it, which was fixed by adding a null check.",
  "odc": "Checking",
  "cwe": "CWE-754: Missing Check"
}}

WARNING: Do NOT use "Other" unless the text provides absolutely zero technical context. Explain missing context in `root_cause_analysis` if you must use "Other".
"""

# ==========================================
# 3. API Call & Parsing Function
# ==========================================
def get_bug_classification(bug_text):
    """
    Call API to classify using JSON mode with CoT validation.
    Returns: [labelODC, labelCWE, explanation]
    """
    try:
        completion = client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3.2",  
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": bug_text}
            ],
            temperature=0.1, # 设定较低的温度以保证分类和推理的稳定性
            response_format={"type": "json_object"}
        )
        
        raw_response = completion.choices[0].message.content.strip()
        
        # Parse the JSON response
        try:
            parsed_json = json.loads(raw_response)
            
            # 提取思维链分析过程
            explanation = parsed_json.get("root_cause_analysis", "No explanation generated.")
            
            # 提取并验证 ODC
            odc_result = parsed_json.get("odc", "Other")
            odc_final = odc_result if odc_result in ODC_LABELS else "Other"
            
            # 提取并验证 CWE (前缀模糊匹配防越界)
            cwe_result = parsed_json.get("cwe", "Other")
            cwe_final = "Other"
            for valid_cwe in CWE_LABELS:
                if cwe_result.split(':')[0] in valid_cwe:
                    cwe_final = valid_cwe
                    break

            return [odc_final, cwe_final, explanation]
            
        except json.JSONDecodeError:
            print(f"[Warning]: Failed to parse JSON from LLM: {raw_response}")
            return ["Other", "Other", "JSON parsing failed."]
            
    except Exception as e:
        print(f"[Error]: API call failed: {e}")
        return ["Error", "Error", f"API Error: {str(e)}"]

# ==========================================
# 4. Main Processing Logic
# ==========================================
def process_bug_file():
    print(f"Processing {INPUT_FILE}...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"[Error]: Input file {INPUT_FILE} not found. Please ensure pipeline steps have run successfully.")
        return
        
    print(f"Found a total of {len(lines)} bug reports.")

    tasks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                
                parsed_data = data.get("parsed_data", {})
                title = parsed_data.get("title", "No Title")
                description = parsed_data.get("description", "No Description")
                
                # 1. 提取并格式化 Comments (开发者讨论历史)
                comments_list = parsed_data.get("comments", [])
                comments_text = ""
                if comments_list:
                    formatted_comments = []
                    for c in comments_list:
                        author = c.get("author", "Unknown")
                        body = c.get("body", "").strip()
                        # 防止个别超长 comment 占用过多 token，可酌情做单条截断
                        if len(body) > 1000:
                            body = body[:1000] + "... [Truncated]"
                        formatted_comments.append(f"[{author}]: {body}")
                    comments_text = "\n".join(formatted_comments)
                else:
                    comments_text = "No comments available."
                
                # 2. 提取并格式化 Semantic Patches (代码补丁)
                patches = data.get("semantic_patches", [])
                patch_texts = []
                for p in patches:
                    file_name = p.get("file", "Unknown")
                    body = p.get("body", "")
                    patch_texts.append(f"--- File: {file_name} ---\n{body}")
                patch_str = "\n\n".join(patch_texts) if patch_texts else "No patches available."
                
                # 3. 最终组装高信息密度字符串
                bug_text = (
                    f"### Issue Title\n{title}\n\n"
                    f"### Issue Description\n{description}\n\n"
                    f"### Discussion Comments\n{comments_text}\n\n"
                    f"### Code Patches\n{patch_str}"
                )
                # ==========================================
                
                # 确保组装后的文本不是空的
                if bug_text and bug_text.strip():
                    future = executor.submit(get_bug_classification, bug_text)
                    tasks.append((data, future))
                else:
                    print(f"Line {i+1} generated empty text, skipping.")
                    
            except json.JSONDecodeError:
                print(f"Line {i+1} JSON decode error, skipping.")
            except Exception as e:
                print(f"Line {i+1} processing error: {e}")

        print("All tasks submitted, waiting for results and writing in order...")
        
        # 为了防丢失，确保目录存在
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            processed_count = 0
            
            for original_data, future in tasks:
                odc_label, cwe_label, explanation = future.result()

                new_data = {
                    "project_id": original_data.get("project_id"),
                    "bug_id": original_data.get("bug_id"),
                    "source_type": original_data.get("source_type"),
                    "label": [odc_label, cwe_label], 
                    "explanation": explanation,    
                    "parsed_data": original_data.get("parsed_data")
                }
                
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(tasks)} records...")
                
                time.sleep(REQUEST_DELAY)

    print(f"Processing completed!")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_bug_file()