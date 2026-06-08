"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
import math
from typing import Optional


# =============================================================================
# Helper functions
# =============================================================================

def cosine_sim(a: list[float], b: list[float]) -> float:
    """
    Tính cosine similarity giữa 2 vector.

    Args:
        a: Vector thứ nhất
        b: Vector thứ hai

    Returns:
        Cosine similarity trong khoảng gần [-1, 1].
    """
    if not a or not b:
        return 0.0

    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def get_candidate_key(item: dict) -> str:
    """
    Tạo key ổn định để deduplicate candidates giữa nhiều ranker.

    Ưu tiên dùng metadata chunk_id/path/source/chunk_index.
    Nếu không có thì fallback sang content.
    """
    metadata = item.get("metadata", {}) or {}

    chunk_id = metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index", "")

    if source != "" and chunk_index != "":
        return f"{source}::{chunk_index}"

    path = metadata.get("path", "")
    if path != "" and chunk_index != "":
        return f"{path}::{chunk_index}"

    return item.get("content", "")


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    import requests

    query = query.strip()

    jina_api_key = os.getenv("JINA_API_KEY")

    if not jina_api_key:
        raise EnvironmentError(
            "JINA_API_KEY is not set. "
        )

    documents = [c.get("content", "") for c in candidates]

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {jina_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(candidates)),
        },
        timeout=60,
    )

    response.raise_for_status()
    data = response.json()

    reranked = data.get("results", [])

    results = []

    for r in reranked:
        idx = r["index"]
        relevance_score = float(r.get("relevance_score", 0.0))

        item = candidates[idx].copy()
        item["original_score"] = item.get("score", 0.0)
        item["score"] = relevance_score
        item["rerank_score"] = relevance_score
        item["rerank_method"] = "cross_encoder"

        results.append(item)

    results.sort(key=lambda x: x["rerank_score"], reverse=True)
    return results[:top_k]

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx: Optional[int] = None
        best_score = float("-inf")

        for idx in remaining:
            candidate_embedding = candidates[idx]["embedding"]

            # Relevance to query
            relevance = cosine_sim(query_embedding, candidate_embedding)

            # Max similarity to already selected docs
            max_sim_to_selected = 0.0

            for sel_idx in selected:
                selected_embedding = candidates[sel_idx]["embedding"]
                sim = cosine_sim(candidate_embedding, selected_embedding)
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = (
                lambda_param * relevance
                - (1.0 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)

        candidates[best_idx]["original_score"] = candidates[best_idx].get("score", 0.0)
        candidates[best_idx]["score"] = float(best_score)
        candidates[best_idx]["rerank_score"] = float(best_score)
        candidates[best_idx]["rerank_method"] = "mmr"

    return [candidates[i] for i in selected]

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    source_scores: dict[str, list[dict]] = {}

    for list_idx, ranked_list in enumerate(ranked_lists, start=1):
        for rank, item in enumerate(ranked_list, start=1):
            key = get_candidate_key(item)

            rrf_score = 1.0 / (k + rank)

            rrf_scores[key] = rrf_scores.get(key, 0.0) + rrf_score

            if key not in content_map:
                content_map[key] = item
            else:
                old_score = content_map[key].get("score", 0.0)
                new_score = item.get("score", 0.0)
                if new_score > old_score:
                    content_map[key] = item

            source_scores.setdefault(key, []).append({
                "ranked_list_index": list_idx,
                "rank": rank,
                "original_score": float(item.get("score", 0.0)),
                "rrf_contribution": rrf_score,
            })

    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for key, score in sorted_items[:top_k]:
        item = content_map[key].copy()

        item["original_score"] = item.get("score", 0.0)
        item["score"] = float(score)
        item["rerank_score"] = float(score)
        item["rerank_method"] = "rrf"
        item["rrf_details"] = source_scores.get(key, [])

        results.append(item)

    return results

# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    semantic_results = [
        {
            "content": "Nghệ sĩ X bị bắt vì sử dụng ma túy",
            "score": 0.90,
            "metadata": {"source": "article_01.md", "chunk_index": 4},
        },
        {
            "content": "Điều 249: Tội tàng trữ trái phép chất ma túy",
            "score": 0.80,
            "metadata": {"source": "bo_luat_hinh_su.md", "chunk_index": 1},
        },
        {
            "content": "Hình phạt tù từ 2-7 năm cho một số tội liên quan ma túy",
            "score": 0.60,
            "metadata": {"source": "bo_luat_hinh_su.md", "chunk_index": 8},
        },
    ]

    lexical_results = [
        {
            "content": "Điều 249: Tội tàng trữ trái phép chất ma túy",
            "score": 12.5,
            "metadata": {"source": "bo_luat_hinh_su.md", "chunk_index": 1},
        },
        {
            "content": "Hình phạt tù từ 2-7 năm cho một số tội liên quan ma túy",
            "score": 10.2,
            "metadata": {"source": "bo_luat_hinh_su.md", "chunk_index": 8},
        },
        {
            "content": "Nghệ sĩ X bị bắt vì sử dụng ma túy",
            "score": 8.4,
            "metadata": {"source": "article_01.md", "chunk_index": 4},
        },
    ]

    results = rerank_rrf([semantic_results, lexical_results], top_k=3)

    for r in results:
        print(f"[{r['score']:.5f}] {r['content']}")
