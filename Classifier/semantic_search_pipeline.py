#!/usr/bin/env python3
"""
缺陷库语义检索总控流水线 (Semantic Search Controller Pipeline)

功能：
- 作为表层调度脚本，组合 Retriever(召回) 和 Reranker(精排) 模块。
- 从 parsed_data.jsonl 读取缺陷元数据。
- 为每个缺陷构建自然语言 Query。
- 执行完整的 [召回 -> 重排 -> 截断] 管道。
- 将最终提纯的代码上下文写入输出文件，供下游分类 LLM 使用。
"""

import os
import json
import time
import argparse
from typing import Dict, Any

# 导入我们之前编写的底层执行引擎
from online_patch_retriever import OnlinePatchRetriever
from semantic_patch_reranker import OnlinePatchReranker

# -----------------------------
# Configuration & Defaults
# -----------------------------
SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_INPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, 'bug_classification', 'parsed_data_2.jsonl'))
DEFAULT_OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, 'bug_classification', 'semantic_patches_ready.jsonl'))

def build_query_text(parsed_data: Dict[str, Any]) -> str:
    """
    根据 Issue 元数据构建高质量的自然语言查询 (Query)
    """
    title = parsed_data.get("title", "")
    desc = parsed_data.get("description", "")[:1000] # 适度截断描述
    comments = parsed_data.get("comments", [])
    
    # 提取最近的几条核心讨论作为上下文补充
    recent_comments = "\n".join(c.get("body", "") for c in comments[-3:])[:800]
    
    return f"Title: {title}\nDescription: {desc}\nDiscussions: {recent_comments}".strip()

def main():
    parser = argparse.ArgumentParser(description="缺陷补丁语义检索与精排总控流水线")
    parser.add_argument('-i', '--input', default=DEFAULT_INPUT_FILE, help="输入文件 (parsed_data.jsonl)")
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_FILE, help="输出文件 (提纯后的上下文)")
    parser.add_argument('--top-k', type=int, default=20, help="召回阶段的粗筛数量 (Recall Top-K)")
    parser.add_argument('--top-n', type=int, default=4, help="精排阶段的最终保留数量 (Rerank Top-N)")
    parser.add_argument('--limit', type=int, default=None, help="仅处理前 N 条记录用于测试")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[Error] 输入文件不存在: {args.input}")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("==================================================")
    print(" 启动语义检索流水线 (Semantic Search Pipeline)")
    print("==================================================")

    # 1. 系统初始化：加载服务引擎
    print("[Pipeline] 正在初始化在线召回引擎 (Retriever)...")
    try:
        retriever = OnlinePatchRetriever()
    except Exception as e:
        print(f"[Fatal] 召回引擎初始化失败: {e}")
        return

    print("[Pipeline] 正在初始化在线精排引擎 (Reranker)...")
    try:
        reranker = OnlinePatchReranker()
    except Exception as e:
        print(f"[Fatal] 精排引擎初始化失败: {e}")
        return

    # 2. 读取输入数据
    records = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if args.limit:
        records = records[:args.limit]
        print(f"[Pipeline] 测试模式：仅处理前 {args.limit} 条数据。")

    print(f"\n[Pipeline] 开始处理，共 {len(records)} 条缺陷记录...")
    processed_count = 0
    start_time = time.time()

    # 3. 执行核心流水线
    with open(args.output, 'w', encoding='utf-8') as out_f:
        for idx, record in enumerate(records, 1):
            project_id = record.get("project_id")
            bug_id = record.get("bug_id")
            parsed_data = record.get("parsed_data", {})
            
            print(f"\n[{idx}/{len(records)}] 正在处理: {project_id} - Bug {bug_id}")
            
            # Step A: 构建查询
            query_text = build_query_text(parsed_data)
            if not query_text:
                print("  -> [Skip] 查询内容为空。")
                continue

            try:
                # Step B: 召回阶段 (Recall) -> Top-K 粗筛
                t1 = time.time()
                candidates = retriever.search(query_text, top_k=args.top_k)
                recall_time = time.time() - t1
                
                if not candidates:
                    print("  -> [Warning] 召回阶段未找到相关代码块。")
                    continue
                    
                print(f"  -> [Recall] 成功召回 {len(candidates)} 个补丁块，耗时 {recall_time:.3f}s")

                # Step C: 精排阶段 (Rerank) -> Top-N 提纯
                t2 = time.time()
                best_hunks = reranker.rerank(query_text, candidates, top_n=args.top_n)
                rerank_time = time.time() - t2
                
                print(f"  -> [Rerank] 成功精排 Top-{args.top_n} 核心代码，耗时 {rerank_time:.3f}s")
                
                # Step D: 组装最终的大模型上下文
                semantic_patches = []
                for rank, h in enumerate(best_hunks, 1):
                    semantic_patches.append({
                        "rank": rank,                                # 排名优先级
                        "file": h["file"],                           # 所属文件
                        "relevance_score": round(h["rerank_score"], 4), # 最终相关性得分
                        "body": h["body"]                            # 提纯后的代码片段
                    })
                
                # 更新 record 结构，注入结构化后的补丁列表
                record["semantic_patches"] = semantic_patches
                
                # 写入输出文件
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed_count += 1
                
            except Exception as e:
                print(f"  -> [Error] 处理失败: {e}")

    total_time = time.time() - start_time
    print("\n==================================================")
    print(f" 流水线执行完毕！")
    print(f" 成功处理: {processed_count}/{len(records)} 个缺陷")
    print(f" 总耗时: {total_time:.2f} 秒 (平均 {total_time/max(1, processed_count):.2f}s/缺陷)")
    print(f" 输出文件已保存至: {args.output}")
    print("==================================================")

if __name__ == "__main__":
    main()