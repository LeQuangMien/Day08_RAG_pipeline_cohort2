"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None

# =============================================================================
# HELPERS
# =============================================================================

def strip_accents(text: str) -> str:
    """
    Bỏ dấu tiếng Việt để tăng recall cho query không dấu hoặc khác kiểu dấu.
    Ví dụ: 'ma túy' / 'ma tuý' / 'ma tuy' vẫn có cơ hội match nhau.
    """
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return text


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa nhẹ cho tiếng Việt:
    - lowercase
    - normalize unicode
    - sửa một số biến thể phổ biến như 'ma tuý' -> 'ma túy'
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    # Một số nguồn/user có thể dùng 'tuý' thay vì 'túy'.
    # Mapping này giúp query 'ma tuý' vẫn match tốt với văn bản 'ma túy'.
    text = text.replace("ma tuý", "ma túy")
    text = text.replace("chất ma tuý", "chất ma túy")

    return text


def tokenize(text: str) -> list[str]:
    """
    Tokenizer đơn giản cho BM25.

    Ghi chú:
    - Không dùng split() thuần vì sẽ dính dấu câu.
    - Thêm cả token bỏ dấu để tăng recall tiếng Việt.
    - Đây chưa phải tokenizer tiếng Việt hoàn hảo, nhưng đủ ổn cho lab.
    """
    text = normalize_text(text)

    # \w với re.UNICODE giữ được chữ tiếng Việt và số.
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)

    accentless_tokens = []
    for token in tokens:
        no_accent = strip_accents(token)
        if no_accent != token:
            accentless_tokens.append(no_accent)

    return tokens + accentless_tokens


def load_corpus_from_task4_chunks() -> list[dict]:
    """
    Tạo corpus BM25 bằng cách reuse load_documents() và chunk_documents()
    từ Task 4. Như vậy BM25 search và semantic search đều làm việc ở chunk level.
    """
    try:
        from .task4_chunking_indexing import load_documents, chunk_documents
    except ImportError:
        from task4_chunking_indexing import load_documents, chunk_documents

    documents = load_documents()
    chunks = chunk_documents(documents)

    return chunks

def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """

    # Tokenize - cho tiếng Việt nên dùng underthesea hoặc đơn giản split()
    tokenized_corpus = []

    for doc in corpus:
        content = doc.get('content', '')
        tokens = tokenize(content)

        if not tokens:
            tokens = ["__empty__"]
        
        tokenized_corpus.append(tokens)
    
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    
    global CORPUS, BM25_INDEX

    query = query.strip()

    # Lazy loading: lần đầu gọi lexical_search mới load corpus và build index.
    if not CORPUS:
        print("Loading corpus for BM25 from Task 4 chunks...")
        CORPUS = load_corpus_from_task4_chunks()
        print(f"✓ Loaded {len(CORPUS)} chunks for BM25")

    if BM25_INDEX is None:
        print("Building BM25 index...")
        BM25_INDEX = build_bm25_index(CORPUS)
        print("✓ BM25 index built")

    tokenized_query = tokenize(query)

    if not tokenized_query:
        return []

    scores = BM25_INDEX.get_scores(tokenized_query)

    # Get top_k indices sorted by score descending.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        score = float(scores[idx])

        if score <= 0:
            continue

        results.append({
            "content": CORPUS[idx]["content"],
            "score": score,
            "metadata": CORPUS[idx]["metadata"],
        })

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 sản xuất trái phép chất ma túy", top_k=5)
    
    print(f"Found {len(results)} results\n")

    for i, result in enumerate(results, 1):
        metadata = result["metadata"]

        print("=" * 80)
        print(f"Rank {i}")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {metadata.get('source')}")
        print(f"Type: {metadata.get('type')}")
        print(f"Chunk index: {metadata.get('chunk_index')}")
        print("-" * 80)
        print(result["content"][:700].replace("\n", " "))
        print()
