"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from task5_semantic_search import semantic_search
from task6_lexical_search import lexical_search
from task7_reranking import rerank, rerank_rrf
from task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.01   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def _safe_search(search_fn, query: str, top_k: int, name: str) -> list[dict]:
    """
    Chạy một retriever an toàn.
    Nếu một nhánh lỗi, pipeline vẫn tiếp tục với nhánh còn lại.
    """
    try:
        results = search_fn(query, top_k=top_k)
        print(f"  ✓ {name}: {len(results)} results")
        return results

    except Exception as e:
        print(f"  ⚠ {name} failed: {e}")
        return []


def _mark_retrieval_source(results: list[dict], source: str) -> list[dict]:
    """
    Thêm field source ở top-level để Task 10 biết context đến từ đâu.
    Giữ nguyên metadata bên trong.
    """
    marked = []

    for item in results:
        new_item = item.copy()
        new_item["source"] = source
        marked.append(new_item)

    return marked


def _should_fallback(results: list[dict], score_threshold: float) -> bool:
    """
    Quyết định có fallback PageIndex không.
    """
    if not results:
        return True

    best_score = float(results[0].get("score", 0.0))
    return best_score < score_threshold

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = query.strip()

    retrieval_top_k = top_k * 2

    print(f"Retrieving for query: {query}")
    print(f"  top_k={top_k}, retrieval_top_k={retrieval_top_k}")

    # Step 1: Chạy semantic + lexical.
    # Có thể gọi tuần tự. Comment gốc nói song song, nhưng với lab gọi tuần tự dễ debug hơn.
    dense_results = _safe_search(
        semantic_search,
        query=query,
        top_k=retrieval_top_k,
        name="Semantic Search",
    )

    sparse_results = _safe_search(
        lexical_search,
        query=query,
        top_k=retrieval_top_k,
        name="Lexical Search",
    )

    # Step 2: Merge bằng RRF.
    ranked_lists = []

    if dense_results:
        ranked_lists.append(dense_results)

    if sparse_results:
        ranked_lists.append(sparse_results)

    if ranked_lists:
        merged_results = rerank_rrf(
            ranked_lists=ranked_lists,
            top_k=retrieval_top_k,
        )
        merged_results = _mark_retrieval_source(merged_results, "hybrid")
        print(f"  ✓ RRF merged: {len(merged_results)} results")
    else:
        merged_results = []
        print("  ⚠ No results from semantic or lexical search")

    # Step 3: Rerank.
    # Với RRF, merge đã chính là rerank/fusion.
    # Nếu method='cross_encoder', có thể rerank lại lần nữa bằng Jina.
    if use_reranking and merged_results:
        if RERANK_METHOD == "rrf":
            final_results = merged_results[:top_k]

        elif RERANK_METHOD == "cross_encoder":
            final_results = rerank(
                query=query,
                candidates=merged_results,
                top_k=top_k,
                method="cross_encoder",
            )
            final_results = _mark_retrieval_source(final_results, "hybrid")

        elif RERANK_METHOD == "mmr":
            raise NotImplementedError(
                "MMR requires query embedding and candidate embeddings. "
                "Current Task 9 pipeline uses RRF."
            )

        else:
            raise ValueError(f"Unknown RERANK_METHOD: {RERANK_METHOD}")

    else:
        final_results = merged_results[:top_k]

    # Step 4: Check threshold → fallback PageIndex.
    if _should_fallback(final_results, score_threshold):
        best_score = final_results[0]["score"] if final_results else 0.0

        print(
            f"  ⚠ Hybrid score ({best_score:.4f}) < threshold ({score_threshold:.4f}). "
            "Fallback → PageIndex"
        )

        try:
            fallback_results = pageindex_search(query, top_k=top_k)
            fallback_results = _mark_retrieval_source(fallback_results, "pageindex")
            return fallback_results[:top_k]

        except Exception as e:
            print(f"  ⚠ PageIndex fallback failed: {e}")
            print("  Returning hybrid results instead.")
            return final_results[:top_k]

    # Step 5: Return top_k.
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:120]}...")
            metadata = r.get("metadata", {})
            print(
                f"     source_file={metadata.get('source') or metadata.get('filename')} "
                f"type={metadata.get('type')} "
                f"chunk_index={metadata.get('chunk_index')}"
            )