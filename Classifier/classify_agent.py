#!/usr/bin/env python3
# framework/classify_bugs_agent.py

import os
import json
import asyncio
import argparse
from openai import AsyncOpenAI

# ==========================================
# 1. Configuration & Taxonomies
# ==========================================

api_key = os.getenv("SILICONCLOUD_API_KEY")
if not api_key:
    raise ValueError("Please set the 'SILICONCLOUD_API_KEY' environment variable")

client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
)

ODC_LABELS = [
    "Assignment", "Checking", "Algorithm", "Interface", 
    "Timing/Serialization", "Build/Package/Merge", 
    "Documentation", "Function/LogicFlow", "Other"
]

CWE_LABELS = [
    "CWE-20: Improper Input Validation", "CWE-16: Configuration",
    "CWE-119: Memory/Buffer Boundary", "CWE-284: Access Control",
    "CWE-362: Race Condition", "CWE-400: Resource Exhaustion / Memory Leak",
    "CWE-573: Logic Mismatch", "CWE-628: API Misuse",
    "CWE-664: Resource Lifecycle", "CWE-665: Improper Initialization",
    "CWE-682: Calculation Error", "CWE-704: Type/Cast Error",
    "CWE-707: Data Encoding", "CWE-754: Missing Check",
    "CWE-1116: Wrong Comments", "CWE-1357: Dependency Issue", "Other"
]

SCRIPT_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '.', 'bug_classification', 'parsed_data_2.jsonl'))
OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '.', 'bug_classification', 'classified_data_agent.jsonl'))
BUG_MINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'bug-mining'))

MAX_CONCURRENT_AGENTS = 15  
MAX_AGENT_TURNS = 4 

# ==========================================
# 2. Agent Tools & Prompt Factory
# ==========================================

def get_agent_tools(use_patch=False):
    """根据是否开启 patch 查看功能，动态生成工具列表"""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_full_description",
                "description": "获取缺陷报告的完整症状描述（Symptom）。如果初始截断的描述不足以诊断问题，请调用此工具。",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_discussion_comments",
                "description": "获取开发者在 Issue 中的讨论和评论。可以通过分页获取。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "offset": {"type": "integer", "description": "起始索引 (从 0 开始)"},
                        "limit": {"type": "integer", "description": "要获取的评论条数 (建议每次 2-3 条)"}
                    },
                    "required": ["offset", "limit"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "submit_classification",
                "description": "【终结动作】当你收集到足够的信息，确定了缺陷类别时，调用此工具提交最终分类并结束诊断。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "odc": {"type": "string", "enum": ODC_LABELS, "description": "代码修复类型 (ODC)"},
                        "cwe": {"type": "string", "enum": CWE_LABELS, "description": "根本软件弱点 (CWE)"}
                    },
                    "required": ["odc", "cwe"]
                }
            }
        }
    ]
    
    # 如果允许查看 Patch，追加提取 Patch 的工具
    if use_patch:
        tools.append({
            "type": "function",
            "function": {
                "name": "get_bug_patch",
                "description": "获取该缺陷对应的源代码修复补丁 (Diff Patch)。查看开发者实际修改了哪几行代码，以进行最准确的 ODC 和 CWE 归类。",
                "parameters": {"type": "object", "properties": {}, "required": []}
            }
        })
        
    return tools

def get_system_prompt(use_patch=False):
    """根据模式动态调整 System Prompt"""
    base_prompt = """你是一个“高级软件缺陷诊断智能体”。
你的任务是将软件缺陷分类到指定的 ODC 体系 (代码修改类型) 和 CWE 体系 (底层漏洞根因)。
【初始信息】：你目前只能看到标题和截断的症状描述。
【工具调用】：如果初始信息不足以做出高置信度的分类，你必须调用提供的工具收集线索。"""

    if use_patch:
        constraint = "\n【模式设定】：你处于**后验分类模式**。强烈建议你调用 `get_bug_patch` 工具查看实际的代码修复补丁，这将极大提升你判断 ODC 的准确性。"
    else:
        constraint = "\n【模式设定】：你处于**修复前诊断模式**，无法查看补丁。你需要通过症状和开发者的讨论预测可能的修复动作。"

    ending = "\n【终结约束】：当你确定分类后，必须调用 `submit_classification` 工具来提交 JSON 格式的最终结果并结束对话。绝不要在普通对话中输出分类。"
    
    return base_prompt + constraint + ending

# ==========================================
# 3. Async Agent Logic
# ==========================================

async def run_diagnostic_agent(record, semaphore, use_patch):
    project_id = record.get("project_id")
    bug_id = record.get("bug_id")
    parsed_data = record.get("parsed_data", {})
    
    title = parsed_data.get("title", "")
    full_desc = parsed_data.get("description", "")
    comments = parsed_data.get("comments", [])
    
    short_desc = full_desc[:300] + ("...\n[Description Truncated]" if len(full_desc) > 300 else "")
    
    system_prompt = get_system_prompt(use_patch)
    active_tools = get_agent_tools(use_patch)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Bug Title: {title}\nSymptom Preview: {short_desc}\n\n请开始你的诊断。"}
    ]

    result_odc = "Other"
    result_cwe = "Other"

    async with semaphore:
        for turn in range(MAX_AGENT_TURNS):
            try:
                response = await client.chat.completions.create(
                    model="Pro/deepseek-ai/DeepSeek-V3.2",  
                    messages=messages,
                    tools=active_tools,
                    tool_choice="auto",
                    temperature=0.1
                )
                
                message = response.choices[0].message
                
                if not message.tool_calls:
                    messages.append({"role": "assistant", "content": message.content or ""})
                    messages.append({"role": "user", "content": "请继续收集信息，或者使用 `submit_classification` 工具提交最终结果。"})
                    continue

                messages.append(message) 

                classification_submitted = False
                
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    
                    if func_name == "submit_classification":
                        result_odc = args.get("odc", "Other")
                        result_cwe = args.get("cwe", "Other")
                        classification_submitted = True
                        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": "Success"})
                        break
                        
                    elif func_name == "get_full_description":
                        messages.append({
                            "role": "tool", "tool_call_id": tool_call.id,
                            "content": full_desc if full_desc else "No further description available."
                        })
                        
                    elif func_name == "get_discussion_comments":
                        offset = args.get("offset", 0)
                        limit = args.get("limit", 2)
                        slice_comments = comments[offset : offset + limit]
                        if not slice_comments:
                            tool_response = "No more comments available."
                        else:
                            tool_response = "\n".join([f"[{c['author']}]: {c['body']}" for c in slice_comments])
                        
                        messages.append({
                            "role": "tool", "tool_call_id": tool_call.id, "content": tool_response
                        })
                        
                    elif func_name == "get_bug_patch" and use_patch:
                        # 定位并读取 Patch 文件
                        patch_file = os.path.join(BUG_MINING_DIR, project_id, 'patches', f"{bug_id}.src.patch")
                        if os.path.exists(patch_file):
                            with open(patch_file, 'r', encoding='utf-8') as pf:
                                patch_content = pf.read()
                                # 安全截断机制：Patch可能极大，限制前 4000 个字符
                                if len(patch_content) > 4000:
                                    patch_content = patch_content[:4000] + "\n...[Patch Snipped due to length limit]..."
                        else:
                            patch_content = f"Patch file not found for {project_id}/{bug_id}."
                            
                        messages.append({
                            "role": "tool", "tool_call_id": tool_call.id, "content": patch_content
                        })

                if classification_submitted:
                    break

            except Exception as e:
                print(f"[Error] Bug {bug_id} API 调用失败: {e}")
                break

    return {
        "project_id": project_id,
        "bug_id": bug_id,
        "source_type": record.get("source_type"),
        "label": [result_odc, result_cwe],
        "turns_used": turn + 1
    }

# ==========================================
# 4. Main Async Pipeline & CLI Entry
# ==========================================

async def process_bug_file(use_patch):
    print(f"--- Starting Agentic Bug Classification ---")
    print(f"Mode: {'POST-FIX (Patch Tool Enabled)' if use_patch else 'PRE-FIX (No Patch Tool)'}")
    print(f"Reading from: {INPUT_FILE}")
    
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"[Error]: Input file {INPUT_FILE} not found.")
        return
        
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    print(f"Loaded {len(records)} bug reports. Dispatching to Agents...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_AGENTS)
    tasks = [run_diagnostic_agent(record, semaphore, use_patch) for record in records]
    
    results = []
    for count, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        results.append(result)
        if count % 10 == 0:
            print(f"  -> Agents completed {count}/{len(records)} bugs...")

    print("All Agents finished. Writing results...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for res in results:
            outfile.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"Classification completed! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="运行基于 Agent 的缺陷分类器")
    parser.add_argument('-p', '--use-patch', action='store_true', 
                        help="开启后验模式，允许 Agent 读取和分析 patches 目录下的 .src.patch 文件")
    args = parser.parse_args()

    asyncio.run(process_bug_file(use_patch=args.use_patch))