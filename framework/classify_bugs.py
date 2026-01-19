import os
import json
import time
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# 1. Configuration
# API Key
api_key = os.getenv("SILICONCLOUD_API_KEY")
if not api_key:
    print("Warning: 'SILICONCLOUD_API_KEY' environment variable is not set.")
    # raise ValueError("Please set the 'SILICONCLOUD_API_KEY' environment variable")

# Initialize OpenAI client pointing to Qwen's compatible API
client = OpenAI(
    api_key=api_key if api_key else "dummy_key", # 防止初始化报错，实际调用会失败如果key无效
    base_url="https://api.siliconflow.cn/v1",
)

# Labels to classify
DEFAULT_LABELS = [
    "Function :: Access Control (CWE-284)",
    "Function :: Logic Mismatch (CWE-573)",

    "Algorithm :: Calculation Error (CWE-682)",
    "Algorithm :: Resource/Memory Leak (CWE-400)",

    "Checking :: Input Validation (CWE-20)",
    "Checking :: Boundary/Buffer (CWE-119)",
    "Checking :: Missing Check (CWE-754)",

    "Assignment :: Initialization (CWE-665)",
    "Assignment :: Type/Cast Error (CWE-704)",

    "Interface :: API Misuse (CWE-628)",
    "Interface :: Data Encoding (CWE-707)",

    "Timing :: Race Condition (CWE-362)",
    "Timing :: Resource Lifecycle (CWE-664)",

    "Build :: Configuration (CWE-16)",
    "Build :: Dependency (CWE-1357)",

    "Documentation :: Wrong Comments (CWE-1116)",

    "Other"
]

LABELS_STRING = "\n".join(DEFAULT_LABELS)

# Input and output files
# 使用相对路径，确保在不同目录下运行时的稳健性
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, '..', 'bug_classification', 'parsed_data_test_sample.jsonl')
OUTPUT_FILE = os.path.join(BASE_DIR, '..', 'bug_classification', 'classified_data_llm.jsonl')

# Concurrency settings
MAX_WORKERS = 10  # Adjust according to API limits
REQUEST_DELAY = 0.1

# 2. System Prompt
# 修改提示词，强制要求 JSON 输出包含置信度
SYSTEM_PROMPT = f"""
You are an expert software engineer specializing in bug triaging and classification.
Your task is to classify bug reports into one of the following categories based on their content.

Categories:
{LABELS_STRING}

Instructions:
1. Analyze the 'Title' and 'Description' of the bug report.
2. Determine the *root cause* or the *nature* of the defect.
3. Select EXACTLY ONE category from the list above.
4. **Confidence Calibration (CRITICAL)**:
   - **0.9 - 1.0 (Certain)**: The report explicitly states the root cause using standard terminology (e.g., "race condition", "buffer overflow") that perfectly matches the category.
   - **0.7 - 0.8 (Likely)**: You are reasonably sure based on symptoms, but the exact root cause is inferred. **Most clear bug reports should fall here.**
   - **0.4 - 0.6 (Uncertain)**: The description is vague, lacks logs, or the bug could plausibly fit into 2+ categories.
   - **< 0.4 (Guessing)**: The report is completely ambiguous (e.g., "it crashed").
   - **Avoid Artificial High Confidence**: Do not default to 0.95+ unless it is a textbook example. Be conservative.
5. **Output Format**: Output the result in STRICT JSON format using the keys "confidence" and "category". Do not add any markdown formatting (like ```json).

Examples:
Bug Report: "User can access admin panel without logging in."
Response: {{"confidence": 0.95, "category": "Function :: Access Control (CWE-284)"}}

Bug Report: "The app crashes with NullPointerException when username is empty."
Response: {{"confidence": 0.98, "category": "Checking :: Missing Check (CWE-754)"}}

Bug Report: "Sorting a large list causes the UI to freeze forever."
Response: {{"confidence": 0.75, "category": "Algorithm :: Resource/Memory Leak (CWE-400)"}}

Bug Report: "Application behaves weirdly after restart."
Response: {{"confidence": 0.4, "category": "Other"}}

Bug Report: "Typo in the help message."
Response: {{"confidence": 0.99, "category": "Documentation :: Wrong Comments (CWE-1116)"}}
"""

# 3. API Call Function
def get_bug_classification(bug_text):
    """
    Call API to classify a single bug text and get confidence.
    Returns: tuple (confidence, label)
    """
    try:
        completion = client.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3.2",  
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": bug_text}
            ],
            temperature=0, # Low temperature for consistent JSON output
            response_format={"type": "json_object"} # 尝试启用 JSON 模式（如果模型支持）
        )
        
        raw_content = completion.choices[0].message.content.strip()
        
        # Parse JSON output
        try:
            result = json.loads(raw_content)
            confidence = result.get("confidence", 0.0)
            label = result.get("category", "Other")
        except json.JSONDecodeError:
            # Fallback: Try regex if JSON parsing fails (e.g. if model wrapped in markdown code blocks)
            # Look for "confidence": <number> and "category": "<string>"
            conf_match = re.search(r'"confidence":\s*([0-9.]+)', raw_content)
            cat_match = re.search(r'"category":\s*"([^"]+)"', raw_content)
            
            confidence = float(conf_match.group(1)) if conf_match else 0.0
            label = cat_match.group(1) if cat_match else raw_content # Fallback to raw text if very broken

        # Validate if the returned label is in our list
        # Remove extra whitespace or quotes if regex failed slightly
        label = label.strip()
        
        final_label = "Other"
        if label in DEFAULT_LABELS:
            final_label = label
        else:
            # Fuzzy match attempt
            for d_label in DEFAULT_LABELS:
                if d_label in label:
                    final_label = d_label
                    break
        
        return confidence, final_label
            
    except Exception as e:
        print(f"[Error]: API call failed: {e}")
        return 0.0, "Error"


# 4. Main Processing Logic
def process_bug_file():
    print(f"Processing {INPUT_FILE}...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"[Error]: Input file {INPUT_FILE} not found.")
        return
        
    print(f"Found a total of {len(lines)} bug reports.")

    # List to store (original data, Future object) in order
    tasks = []

    # Use ThreadPoolExecutor to submit tasks
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 1. Submit all tasks
        for i, line in enumerate(lines):
            try:
                line = line.strip()
                if not line: continue
                
                data = json.loads(line)
                bug_text = data.get("llm_input_text")
                
                if bug_text:
                    future = executor.submit(get_bug_classification, bug_text)
                    # Store the original data and future together to maintain order
                    tasks.append((data, future))
                else:
                    print(f"Line {i+1} missing 'llm_input_text', skipping.")
            except json.JSONDecodeError:
                print(f"Line {i+1} JSON decode error, skipping.")

        print("All tasks submitted, waiting for results and writing in order...")
        
        # 2. Retrieve results in order and write
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            processed_count = 0
            
            for original_data, future in tasks:
                # Get result from future
                confidence, predicted_label = future.result()

                # Construct new data dict
                # Python 3.7+ dicts preserve insertion order. 
                # We insert 'confidence' before 'label'.
                new_data = {
                    "project_id": original_data.get("project_id"),
                    "bug_id": original_data.get("bug_id"),
                    "source_type": original_data.get("source_type"),
                    "confidence": confidence,  # <--- 置信度在前
                    "label": predicted_label,  # <--- 分类在后
                    "llm_input_text": original_data.get("llm_input_text")
                }
                
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(tasks)} records...")
                
                # Avoid rate limiting due to high concurrency
                time.sleep(REQUEST_DELAY)

    print(f"Processing completed!")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_bug_file()