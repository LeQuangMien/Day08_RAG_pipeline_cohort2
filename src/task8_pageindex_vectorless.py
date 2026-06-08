"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import json
import time
import os
import re
from typing import Any
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_BASE_URL = os.getenv("PAGEINDEX_BASE_URL", "https://api.pageindex.ai").rstrip("/")

MAX_UPLOAD_FILES = 2
MAX_FILE_SIZE_MB = 5

PROJECT_ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
LANDING_DIR = PROJECT_ROOT / "data" / "landing"

PAGEINDEX_UPLOAD_DIR = LANDING_DIR / "legal"

MANIFEST_PATH = PROJECT_ROOT / "data" / "pageindex_manifest.json"

# =============================================================================
# Helpers
# =============================================================================


def clean_pageindex_response(text: str) -> str:
    """
    Xóa các JSON citation inline ở đầu response của PageIndex nếu có.
    Ví dụ:
    {"doc_name": "..."}{"doc_name": "...", "pages": "1-6"}## Nội dung
    -> ## Nội dung
    """
    if not text:
        return ""

    text = text.strip()

    # Xóa các object JSON đơn giản nằm liên tiếp ở đầu string.
    text = re.sub(r'^(?:\{[^{}]*\})+', '', text).strip()

    return text

def _check_api_key():
    if not PAGEINDEX_API_KEY:
        raise EnvironmentError(
            "PAGEINDEX_API_KEY is not set. "
            "Hãy thêm PAGEINDEX_API_KEY vào .env hoặc set trong terminal."
        )


def _headers() -> dict:
    _check_api_key()
    return {
        "Authorization": f"Bearer {PAGEINDEX_API_KEY}",
    }


def _json_headers() -> dict:
    _check_api_key()
    return {
        "Authorization": f"Bearer {PAGEINDEX_API_KEY}",
        "Content-Type": "application/json",
    }


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    return {
        "base_url": PAGEINDEX_BASE_URL,
        "documents": []
    }


def _save_manifest(manifest: dict):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _extract_document_id(response_json: dict) -> str:
    """
    PageIndex API response có thể thay đổi tên field theo version.
    Hàm này thử các key phổ biến.
    """
    candidates = [
        "document_id",
        "doc_id",
        "id",
        "file_id",
    ]

    for key in candidates:
        value = response_json.get(key)
        if value:
            return str(value)

    # Một số API có thể bọc trong data/document.
    data = response_json.get("data")
    if isinstance(data, dict):
        for key in candidates:
            value = data.get(key)
            if value:
                return str(value)

        document = data.get("document")
        if isinstance(document, dict):
            for key in candidates:
                value = document.get(key)
                if value:
                    return str(value)

    document = response_json.get("document")
    if isinstance(document, dict):
        for key in candidates:
            value = document.get(key)
            if value:
                return str(value)

    raise RuntimeError(
        "Cannot find document id in PageIndex upload response. "
        f"Response keys: {list(response_json.keys())}"
    )


def _extract_results(response_json: dict) -> list:
    """
    PageIndex query response cũng có thể khác version.
    Hàm này normalize các dạng phổ biến về list results.
    """
    for key in ["results", "matches", "chunks", "contexts", "data"]:
        value = response_json.get(key)
        if isinstance(value, list):
            return value

    data = response_json.get("data")
    if isinstance(data, dict):
        for key in ["results", "matches", "chunks", "contexts"]:
            value = data.get(key)
            if isinstance(value, list):
                return value

    return []


def _get_text_from_result(item: Any) -> str:
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        for key in ["content", "text", "answer", "snippet", "context", "markdown"]:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        # Có API trả section/node.
        node = item.get("node")
        if isinstance(node, dict):
            for key in ["content", "text", "summary"]:
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    return ""


def _get_score_from_result(item: Any) -> float:
    if not isinstance(item, dict):
        return 1.0

    for key in ["score", "relevance_score", "confidence"]:
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    # PageIndex reasoning-based retrieval có thể không trả score.
    # Dùng fallback 1.0 để vẫn tương thích Task 9.
    return 1.0


def _get_metadata_from_result(item: Any) -> dict:
    if not isinstance(item, dict):
        return {}

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        return metadata

    # Gom một số field hữu ích nếu có.
    meta = {}
    for key in ["document_id", "doc_id", "id", "filename", "page", "page_number", "section", "title"]:
        if key in item:
            meta[key] = item[key]

    return meta


def _request_with_fallback(method: str, endpoint_candidates: list[str], **kwargs) -> requests.Response:
    """
    Thử nhiều endpoint candidate vì PageIndex cloud API/SDK có thể thay đổi.
    Endpoint nào trả status khác 404 thì dùng endpoint đó.
    """
    last_response = None

    for endpoint in endpoint_candidates:
        url = f"{PAGEINDEX_BASE_URL}{endpoint}"

        if method.lower() == "post":
            response = requests.post(url, **kwargs)
        elif method.lower() == "get":
            response = requests.get(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")

        last_response = response

        # 404 có thể nghĩa là endpoint path không đúng version.
        # Các lỗi khác như 401/400/500 nên raise để debug thật.
        if response.status_code != 404:
            return response

    return last_response


def upload_documents():
    """
    Upload toàn bộ PDF documents lên PageIndex.

    Lưu ý:
        - Bản này ưu tiên PDF trong data/landing/legal/
        - Sau upload, lưu document_id vào data/pageindex_manifest.json
        - Nếu file đã có trong manifest thì bỏ qua để tránh upload lặp
    """
    _check_api_key()

    if not PAGEINDEX_UPLOAD_DIR.exists():
        raise FileNotFoundError(f"Upload directory not found: {PAGEINDEX_UPLOAD_DIR}")

    pdf_files = sorted(PAGEINDEX_UPLOAD_DIR.rglob("*.pdf"))

    # Ưu tiên file nhỏ trước để tránh dính quota PageIndex.
    pdf_files = sorted(pdf_files, key=lambda p: p.stat().st_size)

    filtered_pdf_files = []

    for pdf_file in pdf_files:
        size_mb = pdf_file.stat().st_size / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:
            print(f"  ⚠ Skipped large file: {pdf_file.name} ({size_mb:.2f} MB)")
            continue

        filtered_pdf_files.append(pdf_file)

    pdf_files = filtered_pdf_files[:MAX_UPLOAD_FILES]
    
    if not pdf_files:
        raise RuntimeError(
            f"No PDF files found in {PAGEINDEX_UPLOAD_DIR}. "
            "Bạn cần convert legal docx sang pdf trước khi upload PageIndex."
        )

    manifest = _load_manifest()
    existing_filenames = {
        doc.get("filename")
        for doc in manifest.get("documents", [])
    }

    uploaded_count = 0
    skipped_count = 0

    for pdf_file in pdf_files:
        if pdf_file.name in existing_filenames:
            print(f"  ↷ Skipped existing: {pdf_file.name}")
            skipped_count += 1
            continue

        print(f"Uploading: {pdf_file.name}")

        with pdf_file.open("rb") as f:
            files = {
                "file": (pdf_file.name, f, "application/pdf")
            }

            # Endpoint chính theo API style thường gặp.
            # Nếu PageIndex đổi path, code sẽ thử các endpoint candidate.
            response = _request_with_fallback(
                method="post",
                endpoint_candidates=[
                    "/doc/",
                    "/documents/",
                    "/upload/",
                    "/api/doc/",
                    "/api/documents/",
                ],
                headers=_headers(),
                files=files,
                timeout=300,
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            print(f"  ✗ Upload failed: {pdf_file.name}")
            print(f"  Status code: {response.status_code}")
            print(f"  Response: {response.text[:1000]}")

            if response.status_code == 403 and "LimitReached" in response.text:
                print("  ⚠ PageIndex quota/limit reached. Stop uploading more documents.")
                break

            raise e

        response_json = response.json()
        document_id = _extract_document_id(response_json)

        manifest["documents"].append({
            "document_id": document_id,
            "filename": pdf_file.name,
            "path": str(pdf_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "type": "legal",
            "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        _save_manifest(manifest)

        uploaded_count += 1
        print(f"  ✓ Uploaded: {pdf_file.name} -> document_id={document_id}")

    print()
    print(f"Upload finished. Uploaded={uploaded_count}, skipped={skipped_count}")
    print(f"Manifest saved to: {MANIFEST_PATH}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex Chat API.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Lưu ý:
        PageIndex hiện hỗ trợ reasoning trong từng document.
        Vì vậy ta query lần lượt từng doc_id trong manifest,
        rồi gom kết quả thành list[dict] để tương thích Task 9.
    """
    _check_api_key()

    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    manifest = _load_manifest()
    documents = manifest.get("documents", [])

    if not documents:
        raise RuntimeError(
            "No PageIndex documents found in manifest. "
            "Please run upload_documents() first."
        )

    from pageindex import PageIndexClient

    pi_client = PageIndexClient(api_key=PAGEINDEX_API_KEY)

    results = []

    for doc in documents:
        doc_id = doc.get("document_id")
        filename = doc.get("filename", "")
        doc_type = doc.get("type", "")

        if not doc_id:
            continue

        prompt = (
            "Trả lời ngắn gọn bằng tiếng Việt dựa trên tài liệu được cung cấp. "
            "Nếu tài liệu không có thông tin liên quan, hãy nói không tìm thấy thông tin phù hợp. "
            f"Câu hỏi: {query}"
        )

        try:
            response_text = ""

            # Theo docs PageIndex SDK, chat_completions hỗ trợ streaming.
            for chunk in pi_client.chat_completions(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                doc_id=doc_id,
                stream=True,
            ):
                response_text += str(chunk)

            response_text = clean_pageindex_response(response_text)
            
            if not response_text:
                continue

            # Bỏ qua kết quả không có thông tin hữu ích.
            lower_text = response_text.lower()
            no_info_markers = [
                "không tìm thấy",
                "không có thông tin",
                "không đề cập",
                "not found",
                "no relevant",
            ]

            if any(marker in lower_text for marker in no_info_markers):
                score = 0.2
            else:
                score = 1.0

            results.append({
                "content": response_text,
                "score": score,
                "metadata": {
                    "document_id": doc_id,
                    "filename": filename,
                    "type": doc_type,
                    "retrieval_mode": "pageindex_chat",
                },
                "source": "pageindex",
            })

        except Exception as e:
            print(f"  ⚠ PageIndex query failed for {filename}: {e}")
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        if not MANIFEST_PATH.exists():
            print("Uploading documents...")
            upload_documents()
        else:
            print(f"Found manifest: {MANIFEST_PATH}")
            print("Skip upload. Use existing PageIndex document ids.")

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma túy", top_k=3)

        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:300]}...")
            print(f"metadata={r['metadata']}")
            print()
