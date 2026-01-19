import json
from collections import defaultdict

# 输入文件路径
input_files = [
    "confidence/classified_data_test_front30_tem1.jsonl",
    "confidence/classified_data_test_front30_tem2.jsonl",
    "confidence/classified_data_test_front30_tem3.jsonl",
]

# 输出文件路径
output_file = "confidence/label_differences.jsonl"

# 数据结构：{(project_id, bug_id): [label1, label2, label3]}
data = defaultdict(list)

# 读取所有输入文件
for file_path in input_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                key = (entry['project_id'], entry['bug_id'])
                data[key].append(entry)

# 筛选出label不同的条目
different_entries = []

for key, entries in data.items():
    # 获取所有的labels
    labels = [entry['label'] for entry in entries]
    
    # 检查labels是否都相同
    if len(set(labels)) > 1:
        # labels不同，记录这个条目
        combined_entry = {
            'project_id': entries[0]['project_id'],
            'bug_id': entries[0]['bug_id'],
            'source_type': entries[0]['source_type'],
            'labels': {
                f'label{i+1}': {
                    'label': entries[i]['label']
                }
                for i in range(len(entries))
            },
        }
        different_entries.append(combined_entry)

# 按project_id和bug_id排序
different_entries.sort(key=lambda x: (x['project_id'], int(x['bug_id'])))

# 输出到文件
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in different_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 打印统计信息
print(f"找到 {len(different_entries)} 个label不同的条目")
print(f"结果已输出到: {output_file}")
