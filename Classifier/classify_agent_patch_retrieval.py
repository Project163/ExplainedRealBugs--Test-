#!/usr/bin/env python3
# Classifier/classify_agent_patch_retrieval.py

import os
import re
import json
import asyncio
import argparse
from collections import Counter
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
OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, '.', 'bug_classification', 'classified_data_agent_patch_retrieval.jsonl'))
BUG_MINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'bug-mining'))

MAX_CONCURRENT_AGENTS = 15
MAX_AGENT_TURNS = 4

# Patch retrieval parameters
PATCH_MAX_TOTAL_CHARS = 5000
PATCH_MAX_HUNKS_DEFAULT = 4
PATCH_MAX_HUNKS_CAP = 8
PATCH_HEAD_FALLBACK_CHARS = 3000
COMMENT_QUERY_TAIL = 5

# ==========================================
# 2. Patch Retrieval (Chunk + Relevance)
# ==========================================

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def tokenize(text: str):
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def term_counter(text: str) -> Counter:
    return Counter(tokenize(text))


def overlap_score(query_terms: Counter, doc_terms: Counter) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    common = set(query_terms.keys()) & set(doc_terms.keys())
    if not common:
        return 0.0

    # 基础词频重叠分数
    tf_overlap = sum(min(query_terms[t], doc_terms[t]) for t in common)

    # 稀疏归一化，降低超长 hunk 对分数的天然优势
    q_norm = sum(query_terms.values()) ** 0.5
    d_norm = sum(doc_terms.values()) ** 0.5
    if q_norm == 0 or d_norm == 0:
        return 0.0

    return tf_overlap / (q_norm * d_norm)


def parse_patch_hunks(patch_text: str):
    """将 patch 解析成 hunk 列表。每个 hunk 保留文件名 + @@header + body。"""
    lines = patch_text.splitlines()
    hunks = []

    current_file = "unknown"
    current_hunk_header = None
    current_hunk_lines = []

    def flush_hunk():
        nonlocal current_hunk_header, current_hunk_lines
        if current_hunk_header is not None:
            body = "\n".join(current_hunk_lines)
            hunks.append({
                "file": current_file,
                "header": current_hunk_header,
                "body": body,
            })
        current_hunk_header = None
        current_hunk_lines = []

    for ln in lines:
        if ln.startswith("diff --git "):
            flush_hunk()
            parts = ln.split()
            if len(parts) >= 4:
                # diff --git a/xxx b/xxx
                current_file = parts[3].replace("b/", "", 1)
            continue

        if ln.startswith("+++ b/"):
            # 更精确地更新文件名
            current_file = ln[6:]
            continue

        if ln.startswith("@@"):
            flush_hunk()
            current_hunk_header = ln
            current_hunk_lines = []
            continue

        if current_hunk_header is not None:
            current_hunk_lines.append(ln)

    flush_hunk()
    return hunks


def build_patch_query(title: str, full_desc: str, comments: list, explicit_query: str = "") -> str:
    recent_comments = comments[-COMMENT_QUERY_TAIL:] if comments else []
    comment_text = "\n".join(c.get("body", "") for c in recent_comments)

    query_parts = [
        explicit_query or "",
        title or "",
        (full_desc or "")[:1200],
        comment_text[:1200],
    ]
    return "\n".join(p for p in query_parts if p).strip()


def rank_hunks(hunks, query_text: str):
    q_terms = term_counter(query_text)
    ranked = []

    for i, h in enumerate(hunks):
        h_text = f"{h['file']}\n{h['header']}\n{h['body']}"
        h_terms = term_counter(h_text)
        score = overlap_score(q_terms, h_terms)

        # 轻量结构先验：含修改行（+/-）越多，信息密度通常越高
        add_cnt = h["body"].count("\n+") + (1 if h["body"].startswith("+") else 0)
        del_cnt = h["body"].count("\n-") + (1 if h["body"].startswith("-") else 0)
        density_bonus = min(add_cnt + del_cnt, 30) * 0.002

        ranked.append((score + density_bonus, i, h))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked


def format_selected_hunks(ranked_hunks, max_hunks: int, max_total_chars: int):
    selected = []
    used_chars = 0

    for score, _, h in ranked_hunks:
        snippet = (
            f"--- FILE: {h['file']}\n"
            f"--- HUNK: {h['header']}\n"
            f"{h['body']}\n"
        )
        if not snippet.strip():
            continue

        # 预留分隔符空间
        projected = used_chars + len(snippet) + 40
        if selected and projected > max_total_chars:
            break

        if len(snippet) > max_total_chars:
            snippet = snippet[:max_total_chars] + "\n...[HUNK TRUNCATED]..."

        selected.append((score, snippet))
        used_chars += len(snippet) + 40

        if len(selected) >= max_hunks:
            break

    if not selected:
        return ""

    blocks = []
    for idx, (score, text) in enumerate(selected, 1):
        blocks.append(f"[PatchChunk #{idx} | relevance={score:.3f}]\n{text}")

    return "\n".join(blocks)


def retrieve_relevant_patch_excerpt(
    patch_file: str,
    title: str,
    full_desc: str,
    comments: list,
    explicit_query: str = "",
    max_hunks: int = PATCH_MAX_HUNKS_DEFAULT,
):
    if not os.path.exists(patch_file):
        return "", f"Patch file not found: {patch_file}"

    with open(patch_file, 'r', encoding='utf-8', errors='replace') as pf:
        patch_content = pf.read()

    if not patch_content.strip():
        return "", "Patch file is empty."

    max_hunks = max(1, min(int(max_hunks), PATCH_MAX_HUNKS_CAP))

    hunks = parse_patch_hunks(patch_content)
    query = build_patch_query(title, full_desc, comments, explicit_query)

    if hunks:
        ranked = rank_hunks(hunks, query)
        excerpt = format_selected_hunks(
            ranked_hunks=ranked,
            max_hunks=max_hunks,
            max_total_chars=PATCH_MAX_TOTAL_CHARS,
        )
        if excerpt:
            return excerpt, ""

    # 兜底：无法解析 hunk 时返回 patch 头部片段
    head = patch_content[:PATCH_HEAD_FALLBACK_CHARS]
    if len(patch_content) > PATCH_HEAD_FALLBACK_CHARS:
        head += "\n...[Patch Snipped due to fallback length limit]..."
    return head, ""

# ==========================================
# 3. Agent Tools & Prompt Factory
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
                "description": (
                    "按相关性返回该缺陷 patch 的关键 hunk，而不是固定返回 patch 开头。"
                    "可选传入 query（当前怀疑点/关键词）与 max_hunks。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "可选。想聚焦的关键词，例如 null pointer / race / bounds / parse 等"
                        },
                        "max_hunks": {
                            "type": "integer",
                            "description": "可选。最多返回多少个相关 hunk，建议 2-6",
                            "minimum": 1,
                            "maximum": PATCH_MAX_HUNKS_CAP,
                        }
                    },
                    "required": []
                }
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
        constraint = "\n【模式设定】：你处于**后验分类模式**。强烈建议你调用 `get_bug_patch` 工具查看相关 patch hunk，并可通过 query 指定怀疑点。"
    else:
        constraint = "\n【模式设定】：你处于**修复前诊断模式**，无法查看补丁。你需要通过症状和开发者的讨论预测可能的修复动作。"

    ending = "\n【终结约束】：当你确定分类后，必须调用 `submit_classification` 工具来提交 JSON 格式的最终结果并结束对话。绝不要在普通对话中输出分类。"

    return base_prompt + constraint + ending

# ==========================================
# 4. Async Agent Logic
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
                    args = json.loads(tool_call.function.arguments or "{}")

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
                        offset = max(0, int(args.get("offset", 0)))
                        limit = max(1, int(args.get("limit", 2)))
                        slice_comments = comments[offset: offset + limit]
                        if not slice_comments:
                            tool_response = "No more comments available."
                        else:
                            tool_response = "\n".join([f"[{c.get('author', 'unknown')}]: {c.get('body', '')}" for c in slice_comments])

                        messages.append({
                            "role": "tool", "tool_call_id": tool_call.id, "content": tool_response
                        })

                    elif func_name == "get_bug_patch" and use_patch:
                        patch_file = os.path.join(BUG_MINING_DIR, project_id, 'patches', f"{bug_id}.src.patch")
                        explicit_query = args.get("query", "") if isinstance(args, dict) else ""
                        max_hunks = args.get("max_hunks", PATCH_MAX_HUNKS_DEFAULT) if isinstance(args, dict) else PATCH_MAX_HUNKS_DEFAULT

                        patch_excerpt, err = retrieve_relevant_patch_excerpt(
                            patch_file=patch_file,
                            title=title,
                            full_desc=full_desc,
                            comments=comments,
                            explicit_query=explicit_query,
                            max_hunks=max_hunks,
                        )

                        if err:
                            tool_response = err
                        else:
                            tool_response = patch_excerpt

                        messages.append({
                            "role": "tool", "tool_call_id": tool_call.id, "content": tool_response
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
# 5. Main Async Pipeline & CLI Entry
# ==========================================


async def process_bug_file(use_patch):
    print(f"--- Starting Agentic Bug Classification (Patch Retrieval v2) ---")
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
    parser = argparse.ArgumentParser(description="运行基于 Agent 的缺陷分类器（Patch 检索增强版）")
    parser.add_argument('-p', '--use-patch', action='store_true',
                        help="开启后验模式，允许 Agent 读取和分析 patches 目录下的 .src.patch 文件")
    args = parser.parse_args()

    asyncio.run(process_bug_file(use_patch=args.use_patch))
