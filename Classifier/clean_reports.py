#!/usr/bin/env python3
# framework/clean_report.py (Refactored for Agentic Workflow)

import os
import json
import argparse
import re

# ==========================================
# Configuration & Constants 
# ==========================================

# 社交噪音/无意义短语 (Stop Phrases)
LOW_VALUE_PHRASES = [
    "thanks", "thank you", "thx", "lgtm", "+1", "bump", 
    "great work", "awesome", "sent from my", "dupe", "duplicate"
]

# 高价值关键词 (确保包含诊断价值的短句不被误删)
HIGH_VALUE_KEYWORDS = [
    "fix", "patch", "bisect", "regression", "workaround", 
    "repro", "crash", "panic", "segfault", "assert", 
    "exception", "error", "fail", "root cause", "caused by"
]

class TextCleaner:
    @staticmethod
    def normalize_technical_data(text):
        """归一化技术数据，将高熵字符串替换为通用占位符，降低 Token 噪声"""
        if not text: return ""
        text = re.sub(r'\b0x[0-9a-fA-F]{4,}\b', '<PTR>', text)
        text = re.sub(r'\b[0-9a-fA-F]{16,}\b', '<HASH>', text)
        text = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>', text)
        return text

    @staticmethod
    def simplify_links(text):
        """简化 markdown 链接和图片"""
        if not text: return ""
        def img_repl(match):
            alt = match.group(1).strip()
            return f"[Image: {alt}]" if alt else "[Image]"
        text = re.sub(r'!\[(.*?)\]\(.*?\)', img_repl, text)

        def link_repl(match):
            anchor_text = match.group(1).strip()
            if anchor_text.startswith('http') and len(anchor_text) > 20:
                return "[Link]"
            if anchor_text.startswith('#') and len(anchor_text) < 10:
                return f"[Ref: {anchor_text}]"
            return f"[Link: {anchor_text}]" if anchor_text else "[Link]"
            
        text = re.sub(r'\[(.*?)\]\((.*?)\)', link_repl, text)
        text = re.sub(r'(?<![\[\(])https?://\S+', '[URL]', text)
        return text

    @staticmethod
    def remove_quotes(text):
        """移除引用文本以减少上下文冗余"""
        if not text: return ""
        lines = [line for line in text.split('\n') if not line.strip().startswith('>')]
        return '\n'.join(lines)

    @staticmethod
    def truncate_code_blocks(text, max_lines=30):
        """
        Agent 模式下放宽截断限制：将原先的 8 行放宽至 30 行，
        以保留关键的 Stack Trace 堆栈和错误日志供 Agent 分析。
        """
        if not text: return ""
        def replacement(match):
            content = match.group(1)
            lines = content.strip().split('\n')
            if len(lines) > max_lines:
                # 保留头 15 行，尾 10 行
                head = '\n'.join(lines[:15]) 
                tail = '\n'.join(lines[-10:])
                return f"```\n{head}\n... [Long Log/Code Snipped] ...\n{tail}\n```"
            return match.group(0)
        return re.sub(r'```(.*?)```', replacement, text, flags=re.DOTALL)

    @staticmethod
    def clean(text):
        """流水线式文本清洗"""
        if not text: return ""
        text = TextCleaner.truncate_code_blocks(text)
        text = TextCleaner.remove_quotes(text)
        text = TextCleaner.simplify_links(text)
        text = TextCleaner.normalize_technical_data(text)
        text = re.sub(r'<[^>]+>', ' ', text) # Remove HTML tags
        text = re.sub(r'\n{3,}', '\n\n', text) # Merge excessive newlines
        return text.strip()

def is_useful_comment(text):
    """启发式过滤器：判断清洗后的文本是否具有保留价值"""
    clean_t = text.lower().strip()
    if any(kw in clean_t for kw in HIGH_VALUE_KEYWORDS):
        return True
    if len(clean_t) < 60 and any(p in clean_t for p in LOW_VALUE_PHRASES):
        return False
    return len(clean_t) > 20

# ==========================================
# Main Processing Pipeline
# ==========================================

def main(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"[Error]: Input file not found: {input_file}")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_count = 0
    skipped_count = 0

    print(f"--- Starting Agentic Data Cleaning & Structuring ---")
    print(f"Reading from: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
                
            try:
                record = json.loads(line)
                raw_data = record.get("raw_data", {})
                
                # 1. 清洗 Title
                clean_title = TextCleaner.clean(raw_data.get("title", ""))
                
                # 2. 清洗 Description (保留完整内容，不做截断)
                clean_desc = TextCleaner.clean(raw_data.get("description", ""))
                
                # 3. 清洗并过滤 Comments，保持列表结构，不拼接
                clean_comments = []
                for comment_obj in raw_data.get("comments", []):
                    author = comment_obj.get("author", "Unknown")
                    raw_body = comment_obj.get("body", "")
                    
                    clean_body = TextCleaner.clean(raw_body)
                    # 保留有价值的评论，并保持结构化
                    if clean_body and is_useful_comment(clean_body):
                        clean_comments.append({
                            "author": author,
                            "body": clean_body
                        })
                
                # 4. 空数据过滤
                if not clean_title and not clean_desc and not clean_comments:
                    skipped_count += 1
                    continue

                # 5. 构建供 Agent 使用的结构化 payload
                # 注意：此处不再拼接 llm_input_text，而是封装在 parsed_data 字典中
                output_record = {
                    "project_id": record.get("project_id"),
                    "bug_id": record.get("bug_id"),
                    "source_type": record.get("source_type"),
                    "parsed_data": {
                        "title": clean_title,
                        "description": clean_desc,
                        "comments": clean_comments
                    }
                }
                
                f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"  -> Processed {processed_count} records...")

            except json.JSONDecodeError:
                print(f"[Warning]: Line {line_num} is not valid JSON, skipping.")
            except Exception as e:
                print(f"[Error]: Failed processing line {line_num}: {e}")

    print(f"\n=================================================")
    print(f"Agentic Data Structuring Complete.")
    print(f"Processed: {processed_count} reports.")
    print(f"Skipped (Empty after cleaning): {skipped_count}")
    print(f"Output saved to: {output_file}")
    print(f"=================================================")

if __name__ == "__main__":
    DEFAULT_INPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'bug_classification', 'extracted_data.jsonl'))
    DEFAULT_OUTPUT_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'bug_classification', 'parsed_data_1.jsonl'))

    parser = argparse.ArgumentParser(description="Step 2: Clean raw JSONL data and structure it for Agent APIs.")
    parser.add_argument('-i', '--input_file', default=DEFAULT_INPUT_FILE, help="Path to raw_extracted_data.jsonl")
    parser.add_argument('-o', '--output_file', default=DEFAULT_OUTPUT_FILE, help="Path to parsed_data.jsonl")
    args = parser.parse_args()

    main(args.input_file, args.output_file)