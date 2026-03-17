import os
import json
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. Configuration & Taxonomies
# ==========================================

# API Key
api_key = os.getenv("SILICONCLOUD_API_KEY")
if not api_key:
    raise ValueError("Please set the 'SILICONCLOUD_API_KEY' environment variable")

# Initialize OpenAI client
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
INPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bug_classification', 'parsed_data_tmp.jsonl'))
OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bug_classification', 'classified_data_llm.jsonl'))

# Concurrency settings
MAX_WORKERS = 10  
REQUEST_DELAY = 0.1

# ==========================================
# 2. Advanced System Prompt
# ==========================================
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
1. Carefully analyze the bug report's Title, Symptom, and Context.
2. Select EXACTLY ONE category from Taxonomy 1 (ODC).
3. Select EXACTLY ONE category from Taxonomy 2 (CWE).
4. You MUST output your response strictly as a JSON object with two keys: "odc" and "cwe". Do NOT wrap the JSON in markdown code blocks.

Example output format:
{{
  "odc": "Checking",
  "cwe": "CWE-20: Improper Input Validation"
}}
"""

# ==========================================
# 3. API Call & Parsing Function
# ==========================================
def get_bug_classification(bug_text):
    """
    Call API to classify a single bug text using JSON mode.
    Returns a list: [labelODC, labelCWE]
    """
    try:
        completion = client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3.2",  
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": bug_text}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        raw_response = completion.choices[0].message.content.strip()
        
        # Parse the JSON response
        try:
            parsed_json = json.loads(raw_response)
            
            # If the model returns a category not in our predefined list, we will classify it as "Other"
            odc_result = parsed_json.get("odc", "Other")
            cwe_result = parsed_json.get("cwe", "Other")
            
            odc_final = odc_result if odc_result in ODC_LABELS else "Other"
            
            # For CWE, perform prefix fuzzy matching (to prevent the model from changing the description text after CWE)
            cwe_final = "Other"
            for valid_cwe in CWE_LABELS:
                if cwe_result.split(':')[0] in valid_cwe:
                    cwe_final = valid_cwe
                    break

            return [odc_final, cwe_final]
            
        except json.JSONDecodeError:
            print(f"[Warning]: Failed to parse JSON from LLM: {raw_response}")
            return ["Other", "Other"]
            
    except Exception as e:
        print(f"[Error]: API call failed: {e}")
        return ["Error", "Error"]

# ==========================================
# 4. Main Processing Logic
# ==========================================
def process_bug_file():
    print(f"Processing {INPUT_FILE}...")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"[Error]: Input file {INPUT_FILE} not found. Please ensure fast_bug_miner.py and clean_report.py have run successfully.")
        return
        
    print(f"Found a total of {len(lines)} bug reports.")

    tasks = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                bug_text = data.get("llm_input_text")
                
                if bug_text:
                    future = executor.submit(get_bug_classification, bug_text)
                    tasks.append((data, future))
                else:
                    print(f"Line {i+1} missing 'llm_input_text', skipping.")
            except json.JSONDecodeError:
                print(f"Line {i+1} JSON decode error, skipping.")

        print("All tasks submitted, waiting for results and writing in order...")
        
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            processed_count = 0
            
            for original_data, future in tasks:
                predicted_labels = future.result()

                new_data = {
                    "project_id": original_data.get("project_id"),
                    "bug_id": original_data.get("bug_id"),
                    "source_type": original_data.get("source_type"),
                    "label": predicted_labels,
                    "llm_input_text": original_data.get("llm_input_text")
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