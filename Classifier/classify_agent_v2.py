#!/usr/bin/env python3
"""
基于 Agent 的缺陷分类引擎 (Bug Classification Agent Engine)

功能：
- 作为一个可独立运行的脚本，或作为一个可被表层调用的核心类。
- 支持按需（Tool Calling）获取缺陷描述、讨论历史。
- 智能 Patch 分析：优先使用 RAG 流水线注入的 semantic_patches，支持回退到物理补丁文件。
- 要求 LLM 在提交分类结果时，输出自然语言解释 (Explanation)，以供未来的人工 Review。
"""

import os
import json
import asyncio
import argparse
from typing import Dict, Any, List
from openai import AsyncOpenAI

# ==========================================
# 1. Configuration & Taxonomies
# ==========================================
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
DEFAULT_INPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '.', 'bug_classification', 'semantic_patches_ready.jsonl'))
DEFAULT_OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '.', 'bug_classification', 'classified_data_agent_v2.jsonl'))
BUG_MINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'bug-mining'))


# ==========================================
# 2. 核心 Agent 类 (供表层控制脚本调用)
# ==========================================
class BugClassificationAgent:
    """
    缺陷分类智能体核心引擎
    设计为无状态服务类，方便被外部并发调用。
    """
    def __init__(self, api_key: str = None, max_turns: int = 4):
        self.api_key = api_key or os.getenv("SILICONCLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("启动失败: 未提供 API Key 且未设置 SILICONCLOUD_API_KEY")
            
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.siliconflow.cn/v1",
        )
        self.max_turns = max_turns

    def _get_agent_tools(self, use_patch: bool = False) -> List[Dict]:
        """动态生成工具列表，强制增加 explanation 字段"""
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
                            "cwe": {"type": "string", "enum": CWE_LABELS, "description": "根本软件弱点 (CWE)"},
                            "explanation": {
                                "type": "string", 
                                "description": "【极其重要】详细的自然语言推理过程（不少于50字）。严格按以下三步结构作答：\n1. 症状分析：缺陷的表面现象或报错是什么？\n2. 根因/修复分析：补丁实际修改了什么逻辑（如果没有补丁则推测代码是如何写错的）？\n3. 分类依据：为什么上述逻辑修改严格符合你选择的 ODC 和 CWE 类别？"
                            }
                        },
                        "required": ["odc", "cwe", "explanation"]
                    }
                }
            }
        ]
        
        if use_patch:
            tools.append({
                "type": "function",
                "function": {
                    "name": "get_bug_patch",
                    "description": "获取该缺陷对应的源代码修复补丁 (Diff Patch)。查看开发者实际修改了哪几行代码，以进行最准确的归类。",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            })
            
        return tools

    def _get_system_prompt(self, use_patch: bool = False) -> str:
        base_prompt = """你是一个“高级软件缺陷诊断智能体”。
你的任务是将软件缺陷分类到指定的 ODC 体系 (代码修改类型) 和 CWE 体系 (底层漏洞根因)。
【初始信息】：你目前只能看到标题和截断的症状描述。
【工具调用】：如果初始信息不足以做出高置信度的分类，你必须调用提供的工具收集线索。

【推理与思维链约束】(极其重要)：
你的分类结果将被用于学术研究，必须具备极高的严谨性。在调用最终的 `submit_classification` 工具时，你的 `explanation` 参数绝不能是简单的复述！
你必须在脑海中完成以下推演后，再将推演过程写入 `explanation`：
- Step 1: 提炼 Issue 中的崩溃、错误或异常表现。
- Step 2: 结合 Patch (若有) 或讨论，指出代码层面的缺陷（如：变量未初始化、边界判断遗漏、API调用错误）。
- Step 3: 将缺陷事实映射到 ODC 和 CWE 的具体定义上进行论证。"""

        if use_patch:
            constraint = "\n【模式设定】：你处于**后验分类模式**。强烈建议你调用 `get_bug_patch` 工具查看实际的代码修复上下文，这将极大提升你判断 ODC 的准确性。"
        else:
            constraint = "\n【模式设定】：你处于**修复前诊断模式**，无法查看补丁。你需要通过症状预测可能的修复动作。"

        ending = "\n【终结约束】：当你确定分类后，必须调用 `submit_classification` 工具提交 JSON 格式的最终结果，并务必提供详细的 `explanation`（推理过程），结束对话。"
        return base_prompt + constraint + ending

    async def classify(self, record: Dict[str, Any], use_patch: bool = False) -> Dict[str, Any]:
        """
        核心开放接口：供外部脚本并发调用。
        传入单条缺陷记录，返回包含分类结果和解释的字典。
        """
        project_id = record.get("project_id")
        bug_id = record.get("bug_id")
        parsed_data = record.get("parsed_data", {})
        
        title = parsed_data.get("title", "")
        full_desc = parsed_data.get("description", "")
        comments = parsed_data.get("comments", [])
        
        short_desc = full_desc[:300] + ("...\n[Description Truncated]" if len(full_desc) > 300 else "")
        
        system_prompt = self._get_system_prompt(use_patch)
        active_tools = self._get_agent_tools(use_patch)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Bug Title: {title}\nSymptom Preview: {short_desc}\n\n请开始你的诊断。"}
        ]

        result_odc = "Other"
        result_cwe = "Other"
        explanation = "未能成功生成解释。"
        turns_used = 0

        for turn in range(self.max_turns):
            turns_used = turn + 1
            try:
                response = await self.client.chat.completions.create(
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
                        explanation = args.get("explanation", "无详细解释。")
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
                            tool_response = "\n".join([f"[{c.get('author','Unknown')}]: {c.get('body','')}" for c in slice_comments])
                        messages.append({
                            "role": "tool", "tool_call_id": tool_call.id, "content": tool_response
                        })
                        
                    elif func_name == "get_bug_patch" and use_patch:
                        patch_response = ""
                        MAX_PATCH_CHARS = 5000  # 设定全局 Patch 上下文安全水位线
                        
                        # 优先：检查是否经过了 RAG 流水线的提纯 (semantic_patches)
                        if "semantic_patches" in record and isinstance(record["semantic_patches"], list):
                            sp_list = record["semantic_patches"]
                            blocks = []
                            current_length = 0
                            
                            for sp in sp_list:
                                block_text = f"--- File: {sp.get('file')} (Relevance: {sp.get('relevance_score')}) ---\n{sp.get('body')}"
                                
                                # 如果加上这个代码块会超限
                                if current_length + len(block_text) > MAX_PATCH_CHARS:
                                    remaining_space = MAX_PATCH_CHARS - current_length
                                    if remaining_space > 150:  # 如果还有一点空间，截断当前块保留一部分
                                        blocks.append(block_text[:remaining_space] + "\n...[单块代码过长，已截断]...")
                                    blocks.append("\n...[后续低相关性补丁块因长度限制已省略]...")
                                    break  # 达到水位线，停止添加后续代码块
                                
                                blocks.append(block_text)
                                current_length += len(block_text)
                                
                            patch_response = "\n\n".join(blocks)
                            
                        else:
                            # 降级：读取物理文件
                            patch_file = os.path.join(BUG_MINING_DIR, str(project_id), 'patches', f"{bug_id}.src.patch")
                            if os.path.exists(patch_file):
                                with open(patch_file, 'r', encoding='utf-8') as pf:
                                    raw_patch = pf.read()
                                    if len(raw_patch) > MAX_PATCH_CHARS:
                                        # 优雅截断：截取到限制长度，并回溯到最近的一个换行符，保证代码行完整
                                        truncated = raw_patch[:MAX_PATCH_CHARS]
                                        last_newline = truncated.rfind('\n')
                                        if last_newline != -1:
                                            truncated = truncated[:last_newline]
                                        patch_response = truncated + "\n\n...[原始 Patch 过长，已触发安全按行截断机制]..."
                                    else:
                                        patch_response = raw_patch
                            else:
                                patch_response = f"Patch file not found for {project_id}/{bug_id}."
                                
                        messages.append({
                            "role": "tool", "tool_call_id": tool_call.id, "content": patch_response
                        })

                if classification_submitted:
                    break

            except Exception as e:
                print(f"[Error] Project {project_id} Bug {bug_id} API 调用失败: {e}")
                explanation = f"Error during Agent execution: {str(e)}"
                break

        return {
            "project_id": project_id,
            "bug_id": bug_id,
            "source_type": record.get("source_type"),
            "label": [result_odc, result_cwe],
            "explanation": explanation,
            "turns_used": turns_used
        }


# ==========================================
# 3. Standalone CLI Entry (独立运行模式)
# ==========================================
async def process_bug_file_standalone(input_file: str, output_file: str, use_patch: bool, max_concurrent: int = 15):
    print(f"--- Starting Standalone Agentic Bug Classification ---")
    print(f"Mode: {'POST-FIX (Patch Tool Enabled)' if use_patch else 'PRE-FIX (No Patch Tool)'}")
    print(f"Reading from: {input_file}")
    
    records = []
    if os.path.exists(input_file):
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                if line.strip():
                    records.append(json.loads(line))
    else:
        print(f"[Error]: Input file {input_file} not found.")
        return

    print(f"Loaded {len(records)} bug reports. Dispatching to Agents...")

    agent_engine = BugClassificationAgent()
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_classify(record):
        async with semaphore:
            return await agent_engine.classify(record, use_patch=use_patch)

    tasks = [bounded_classify(record) for record in records]
    results = []
    
    for count, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        results.append(result)
        if count % 10 == 0:
            print(f"  -> Agents completed {count}/{len(records)} bugs...")

    print("All Agents finished. Writing results...")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for res in results:
            outfile.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"Classification completed! Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行基于 Agent 的缺陷分类器 (支持独立运行与 API 调用)")
    parser.add_argument('-i', '--input', default=DEFAULT_INPUT_FILE, help="输入数据文件路径")
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_FILE, help="输出数据文件路径")
    parser.add_argument('-p', '--use-patch', action='store_true', 
                        help="开启后验模式，允许 Agent 读取 RAG 注入的 Semantic Patches 或物理文件")
    parser.add_argument('-c', '--concurrency', type=int, default=15, help="最大并发数")
    args = parser.parse_args()

    asyncio.run(process_bug_file_standalone(
        input_file=args.input,
        output_file=args.output,
        use_patch=args.use_patch,
        max_concurrent=args.concurrency
    ))