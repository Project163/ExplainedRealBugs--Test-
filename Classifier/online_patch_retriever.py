#!/usr/bin/env python3
"""
在线补丁语义召回模块 (Online Patch Retriever)

功能：
- 供表层控制脚本 (Controller) 调用的执行引擎。
- 在内存中预加载 FAISS 索引和 JSONL 元数据字典。
- 接收自然语言 Query，调用 API 进行单条向量化。
- 在 FAISS 中执行极速余弦相似度检索。
- 将检索到的 Global ID 映射回真实的代码补丁片段并返回。
"""

import os
import json
import requests
import numpy as np
import faiss
from typing import List, Dict, Any

# -----------------------------
# Configuration
# -----------------------------
EMBEDDING_API_URL = "https://api.siliconflow.cn/v1/embeddings"
EMBEDDING_MODEL = "BAAI/bge-m3"

SCRIPT_DIR = os.path.dirname(__file__)
DEFAULT_DB_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'vector_db'))

class OnlinePatchRetriever:
    """
    在线补丁检索器
    设计模式：单例/常驻内存的服务对象，避免每次查询重复加载几百MB的索引。
    """
    def __init__(self, db_dir: str = DEFAULT_DB_DIR):
        self.api_key = os.getenv("SILICONCLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("启动失败: 未设置 SILICONCLOUD_API_KEY 环境变量")
            
        self.index_path = os.path.join(db_dir, 'patch_hunks.index')
        self.meta_path = os.path.join(db_dir, 'hunks_metadata.jsonl')
        
        if not os.path.exists(self.index_path) or not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"向量数据库文件不完整，请检查目录: {db_dir}")

        print("[Retriever] 正在将 FAISS 索引加载至内存...")
        self.index = faiss.read_index(self.index_path)
        
        print("[Retriever] 正在构建元数据内存哈希表...")
        self.metadata_store = self._load_metadata(self.meta_path)
        print(f"[Retriever] 初始化完成. 当前向量库容量: {self.index.ntotal} 个补丁块.")

    def _load_metadata(self, meta_path: str) -> Dict[int, Dict[str, Any]]:
        """
        将 JSONL 元数据加载为以 global_id 为键的内存字典。
        对于十万级数据，占用内存极小 (约百MB级)，检索时间复杂度 O(1)。
        """
        store = {}
        with open(meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                global_id = record.get("global_id")
                store[global_id] = record
        return store

    def _get_query_embedding(self, query_text: str) -> np.ndarray:
        """调用大模型 API 获取查询文本的稠密向量"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": EMBEDDING_MODEL,
            "input": query_text
        }
        
        response = requests.post(EMBEDDING_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 提取向量
        embedding_list = data.get("data", [])[0]["embedding"]
        return np.array([embedding_list], dtype=np.float32)

    def search(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        核心对外接口：执行语义检索
        
        :param query_text: Issue 描述、标题或日志等自然语言组合。
        :param top_k: 需要召回的数量 (推荐设定在 20-50 之间，供下游重排)。
        :return: 包含相似度得分和原始代码块的字典列表。
        """
        if not query_text.strip():
            return []

        # 1. 向量化 Query
        try:
            query_vector = self._get_query_embedding(query_text)
        except Exception as e:
            print(f"[Retriever Error] 向量化 Query 失败: {e}")
            return []

        # 2. L2 归一化 (因为离线阶段使用的是 IndexFlatIP，Query 也必须归一化才能得到正确的余弦相似度)
        faiss.normalize_L2(query_vector)

        # 3. FAISS 极速内积检索 (耗时仅为几毫秒)
        # D 为距离 (相似度得分), I 为全局向量 ID
        D, I = self.index.search(query_vector, top_k)

        # 4. 组装结果
        results = []
        for rank, (score, global_id) in enumerate(zip(D[0], I[0])):
            # FAISS 如果找不到足够多结果，会返回 -1
            if global_id == -1:
                continue
                
            meta_record = self.metadata_store.get(int(global_id))
            if meta_record:
                # 组合得分与元数据
                retrieved_item = {
                    "rank": rank + 1,
                    "score": float(score),
                    "project_id": meta_record["project_id"],
                    "bug_id": meta_record["bug_id"],
                    "file": meta_record["file"],
                    "header": meta_record["header"],
                    "body": meta_record["body"]
                }
                results.append(retrieved_item)

        return results

# -----------------------------
# 独立测试入口 (供您验证模块可用性)
# -----------------------------
if __name__ == "__main__":
    # 该模块可以直接运行以验证功能
    try:
        retriever = OnlinePatchRetriever()
        
        test_query = "修复内存泄漏问题，检查 free(ptr) 是否被调用"
        print(f"\n模拟查询: '{test_query}'")
        
        results = retriever.search(test_query, top_k=3)
        
        for res in results:
            print(f"\n[{res['project_id']}/{res['bug_id']}] Score: {res['score']:.4f} | File: {res['file']}")
            # 打印 Hunk 头部两行
            preview = "\n".join(res['body'].splitlines()[:2])
            print(f"{preview}\n...")
            
    except Exception as e:
        print(f"测试失败: {e}")