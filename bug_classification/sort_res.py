import json
import argparse
import sys

def get_sort_key(record):
    """
    定义排序键值：
    1. Label (主要排序依据)
    2. Project ID (当 Label 相同时，按项目分组)
    3. Bug ID (当项目也相同时，按 Bug ID 排序，优先尝试数字排序)
    """
    # 1. Label: 处理缺失的情况，排在最后或最前均可，这里空字符串通常排最前
    label = record.get('label', '')
    
    # 2. Project ID
    project = record.get('project_id', '')
    
    # 3. Bug ID: 尝试转换为整数，以便 "2" 排在 "10" 之前，而不是之后
    bug_id_raw = record.get('bug_id', '0')
    try:
        bug_id = int(bug_id_raw)
    except ValueError:
        bug_id = str(bug_id_raw)
        
    return (label, project, bug_id)

def sort_jsonl(input_file, output_file):
    print(f"Processing: {input_file} -> {output_file}")
    
    data = []
    try:
        # 读取数据
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError:
                    print(f"[Warning] Skipped invalid JSON on line {i}")
    except FileNotFoundError:
        print(f"[Error] File not found: {input_file}")
        sys.exit(1)

    print(f"Loaded {len(data)} records. Sorting...")

    # 执行排序
    data.sort(key=get_sort_key)

    # 写入结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Done. Sorted {len(data)} records.")
    except IOError as e:
        print(f"[Error] Failed to write output: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sort LLM classification results by label.")
    parser.add_argument('input_file', help="Path to input JSONL file")
    parser.add_argument('output_file', help="Path to output JSONL file")
    
    args = parser.parse_args()
    
    sort_jsonl(args.input_file, args.output_file)