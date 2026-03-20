#!/usr/bin/env python3
"""
离线补丁向量化与增量构建引擎 (Offline Patch Embedding Engine)

功能：
- 扫描 bug-mining 目录下的所有补丁文件。
- 通过 processed_state.json 实现增量过滤（跳过已处理的补丁）。
- 批量调用 SiliconCloud API 获取 Hunks 向量。
- 将向量追加至 FAISS 索引，将元数据追加至 JSONL。
- 定期保存 Checkpoint，保证中断后可无缝续跑。
"""

import os
import json
import time
import requests
import numpy as np
import faiss
from typing import List, Dict, Any

# -----------------------------
# Configuration
# -----------------------------
API_KEY = os.getenv("SILICONCLOUD_API_KEY")
EMBEDDING_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
VECTOR_DIMENSION = 1024  # BGE-large 维度
BATCH_SIZE = 50          # 每次发给 API 的 Hunk 数量
SAVE_INTERVAL = 10       # 每处理 10 个 Bug 保存一次 Checkpoint

SCRIPT_DIR = os.path.dirname(__file__)
BUG_MINING_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'bug-mining'))
DB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'vector_db'))

# 持久化文件路径
STATE_FILE = os.path.join(DB_DIR, 'processed_state.json')
FAISS_INDEX_FILE = os.path.join(DB_DIR, 'patch_hunks.index')
METADATA_FILE = os.path.join(DB_DIR, 'hunks_metadata.jsonl')

# -----------------------------
# 基础解析组件 (复用您的 extract_patch_preview.py 逻辑)
# -----------------------------
def parse_patch_hunks(patch_text: str) -> List[Dict[str, str]]:
    """提取 Patch 中的 Hunks (此处简化，同之前的逻辑)"""
    lines = patch_text.splitlines()
    hunks = []
    current_file = "unknown"
    current_hunk_header = None
    current_hunk_lines = []

    def flush():
        if current_hunk_header:
            hunks.append({"file": current_file, "header": current_hunk_header, "body": "\n".join(current_hunk_lines)})
    
    for ln in lines:
        if ln.startswith("diff --git "):
            flush()
            parts = ln.split()
            if len(parts) >= 4: current_file = parts[3].replace("b/", "", 1)
            current_hunk_header, current_hunk_lines = None, []
        elif ln.startswith("+++ b/"):
            current_file = ln[6:]
        elif ln.startswith("@@"):
            flush()
            current_hunk_header, current_hunk_lines = ln, []
        elif current_hunk_header is not None:
            current_hunk_lines.append(ln)
    flush()
    return hunks

# -----------------------------
# API 调用组件
# -----------------------------
def fetch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """批量获取向量并实施简单的重试机制"""
    if not texts: return []
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": EMBEDDING_MODEL, "input": texts}
    
    for attempt in range(3):
        try:
            response = requests.post(EMBEDDING_API_URL, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return [np.array(item["embedding"], dtype=np.float32) for item in data.get("data", [])]
        except Exception as e:
            print(f"  -> [Warning] API 请求失败 (Attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    
    print("  -> [Error] API 连续请求失败，放弃当前批次。")
    return []

# -----------------------------
# 离线增量引擎
# -----------------------------
class OfflineEmbeddingEngine:
    def __init__(self):
        os.makedirs(DB_DIR, exist_ok=True)
        self.state = self._load_state()
        self.index, self.current_id = self._load_or_create_faiss()
        self.meta_fp = open(METADATA_FILE, 'a', encoding='utf-8') # 追加模式

    def _load_state(self) -> set:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return set(json.load(f))
        return set()

    def _load_or_create_faiss(self) -> tuple:
        if os.path.exists(FAISS_INDEX_FILE):
            print(f"[*] 加载现有 FAISS 索引: {FAISS_INDEX_FILE}")
            index = faiss.read_index(FAISS_INDEX_FILE)
            current_id = index.ntotal # 以当前总向量数作为下一个起步 ID
            return index, current_id
        else:
            print("[*] 创建全新的 FAISS 索引")
            # IndexFlatIP 用于余弦相似度计算 (前提是向量已被 L2 归一化)
            base_index = faiss.IndexFlatIP(VECTOR_DIMENSION)
            # 使用 IndexIDMap 使我们能自定义分配 ID
            index = faiss.IndexIDMap(base_index)
            return index, 0

    def _save_checkpoints(self):
        faiss.write_index(self.index, FAISS_INDEX_FILE)
        with open(STATE_FILE, 'w') as f:
            json.dump(list(self.state), f)
        self.meta_fp.flush()
        print(f"  -> [Checkpoint] 状态已保存。当前向量总数: {self.current_id}")

    def run(self):
        if not API_KEY:
            raise ValueError("未设置 SILICONCLOUD_API_KEY")

        print("=== 开始离线补丁向量化 ===")
        processed_count_this_run = 0
        projects = sorted([d for d in os.listdir(BUG_MINING_DIR) if os.path.isdir(os.path.join(BUG_MINING_DIR, d))])
        
        for project_id in projects:
            patch_dir = os.path.join(BUG_MINING_DIR, project_id, 'patches')
            if not os.path.isdir(patch_dir): continue
            
            for patch_file in os.listdir(patch_dir):
                if not patch_file.endswith('.src.patch'): continue
                
                bug_id = patch_file.replace('.src.patch', '')
                unique_key = f"{project_id}_{bug_id}"
                
                # 【增量核心】如果已经在状态记录中，直接跳过
                if unique_key in self.state:
                    continue
                
                print(f"处理新增: {unique_key}")
                patch_path = os.path.join(patch_dir, patch_file)
                with open(patch_path, 'r', encoding='utf-8', errors='replace') as f:
                    hunks = parse_patch_hunks(f.read())
                
                if not hunks:
                    self.state.add(unique_key)
                    continue

                # 提取 Hunk 文本
                hunk_texts = [f"File: {h['file']}\n{h['body']}" for h in hunks]
                
                # 应对异常大的 Patch 进行分批请求
                hunk_embeddings = []
                for i in range(0, len(hunk_texts), BATCH_SIZE):
                    batch_texts = hunk_texts[i:i+BATCH_SIZE]
                    batch_embs = fetch_embeddings(batch_texts)
                    if batch_embs:
                        hunk_embeddings.extend(batch_embs)
                    time.sleep(0.1) # 速率限制
                
                if len(hunk_embeddings) != len(hunks):
                    print(f"  -> [Error] {unique_key} 向量化不完整，跳过以待下次重试。")
                    continue

                # 将向量转换为 Numpy 矩阵并进行 L2 归一化 (配合 FlatIP 计算余弦相似度)
                emb_matrix = np.vstack(hunk_embeddings)
                faiss.normalize_L2(emb_matrix)
                
                # 分配唯一的递增 ID
                ids = np.arange(self.current_id, self.current_id + len(hunks), dtype=np.int64)
                
                # 写入 FAISS 和元数据
                self.index.add_with_ids(emb_matrix, ids)
                for i, hunk in enumerate(hunks):
                    meta_record = {
                        "global_id": int(ids[i]),
                        "project_id": project_id,
                        "bug_id": bug_id,
                        "file": hunk['file'],
                        "header": hunk['header'],
                        "body": hunk['body']
                    }
                    self.meta_fp.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
                
                self.current_id += len(hunks)
                self.state.add(unique_key)
                processed_count_this_run += 1
                
                # 定期 Checkpoint
                if processed_count_this_run % SAVE_INTERVAL == 0:
                    self._save_checkpoints()

        # 运行结束做最终保存
        self._save_checkpoints()
        self.meta_fp.close()
        print("=== 离线向量化完成 ===")

if __name__ == "__main__":
    engine = OfflineEmbeddingEngine()
    engine.run()