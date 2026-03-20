#!/usr/bin/env python3
"""
在线补丁语义精排模块 (Online Patch Reranker)

功能：
- 接收来自 Retriever 的 Top-K 粗筛候选代码块 (Hunks)。
- 调用 SiliconCloud Reranker API (交叉编码器) 获取深度语义相关性得分。
- 结合代码修改密度特征 (Density Bonus) 进行融合打分。
- 截断并返回最核心的 Top-N 代码块，供下游 LLM 分析。
"""

import os
import requests
from typing import List, Dict, Any

# -----------------------------
# Configuration
# -----------------------------
RERANK_API_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # BGE 多语言重排 SOTA 模型

class OnlinePatchReranker:
    """
    精排引擎：Cross-Encoder + Heuristic Density Hybrid Ranking
    """
    def __init__(self):
        self.api_key = os.getenv("SILICONCLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("启动失败: 未设置 SILICONCLOUD_API_KEY 环境变量")

    def _compute_density_bonus(self, hunk_body: str) -> float:
        """
        保留原始架构的优秀工程经验：修改密度奖励
        修改越密集的块，包含核心逻辑修复的概率越大。
        """
        add_cnt = hunk_body.count("\n+") + (1 if hunk_body.startswith("+") else 0)
        del_cnt = hunk_body.count("\n-") + (1 if hunk_body.startswith("-") else 0)
        
        # 限制最大奖励上限，防止超大替换块占据绝对统治地位
        # 假设最大奖励分为 0.06 (相当于 30 行有效修改)
        density_bonus = min(add_cnt + del_cnt, 30) * 0.002
        return density_bonus

    def _get_cross_encoder_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        调用 SiliconCloud 重排 API 获取交叉编码器的精准得分
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建标准重排 Payload
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False
        }
        
        try:
            response = requests.post(RERANK_API_URL, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # API 返回的结果可能是乱序的（按得分排序），我们需要将其映射回原索引顺序
            results = data.get("results", [])
            
            # 初始化一个与 documents 等长的 0 分数组
            scores = [0.0] * len(documents)
            for res in results:
                original_idx = res.get("index")
                score = res.get("relevance_score", 0.0)
                scores[original_idx] = score
                
            return scores
        
        except Exception as e:
            print(f"[Reranker Error] 重排 API 调用失败: {e}")
            # Fallback: 如果 API 失败，返回全 0 分，退化为仅依靠召回分和密度分
            return [0.0] * len(documents)

    def rerank(self, query_text: str, candidates: List[Dict[str, Any]], top_n: int = 4) -> List[Dict[str, Any]]:
        """
        核心精排接口
        
        :param query_text: 原始查询文本 (Issue 描述)
        :param candidates: 召回阶段返回的候选列表
        :param top_n: 最终保留的核心代码块数量 (供 LLM 阅读)
        :return: 重排后的精华代码块列表
        """
        if not candidates:
            return []

        # 1. 提取所有待重排的纯文本内容 (加上文件路径作为上下文)
        docs_to_rerank = [f"File: {c['file']}\n{c['body']}" for c in candidates]
        
        # 2. 调用 Cross-Encoder 获取语义打分
        ce_scores = self._get_cross_encoder_scores(query_text, docs_to_rerank)
        
        # 3. 融合打分 (Hybrid Scoring) 与重组
        reranked_results = []
        for i, candidate in enumerate(candidates):
            semantic_score = ce_scores[i]
            # 计算启发式密度奖励
            density_bonus = self._compute_density_bonus(candidate["body"])
            
            # 综合得分计算公式：交叉编码器得分(主导) + 密度奖励 + 召回阶段相似度微调(可选)
            # 因为重排模型的得分绝对值区间通常在 0~1 或负数到正数之间，直接相加是一个基础且有效的融合策略
            final_score = semantic_score + density_bonus
            
            # 将新得分写入候选对象
            candidate["rerank_score"] = float(final_score)
            candidate["ce_score"] = float(semantic_score)
            candidate["density_bonus"] = density_bonus
            
            reranked_results.append(candidate)
            
        # 4. 根据综合得分降序排列
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # 5. 截断输出 (防止大语言模型上下文溢出)
        return reranked_results[:top_n]

# -----------------------------
# 独立测试入口
# -----------------------------
if __name__ == "__main__":
    # 模拟测试数据
    test_query = "fix null pointer exception in config parser"
    mock_candidates = [
        {"file": "config.py", "body": " def load():\n+    if not path:\n+        return None\n     pass", "score": 0.8},
        {"file": "utils.py", "body": " def helper():\n-    x = 1\n+    x = 2\n     return x", "score": 0.82}, # 召回得分高，但语义不符
    ]
    
    try:
        reranker = OnlinePatchReranker()
        print("开始精排测试...")
        results = reranker.rerank(test_query, mock_candidates, top_n=2)
        
        for res in results:
            print(f"File: {res['file']} | 最终分: {res['rerank_score']:.4f} (语义:{res['ce_score']:.4f}, 密度:{res['density_bonus']:.4f})")
    except Exception as e:
        print(f"测试失败: {e}")