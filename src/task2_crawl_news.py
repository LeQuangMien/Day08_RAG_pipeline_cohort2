"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import re
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://tuoitre.vn/bat-nguoi-mau-an-tay-ca-si-chi-dan-co-tien-truc-phuong-do-lien-quan-ma-tuy-20241114114826655.htm",
    "https://tuoitre.vn/chuyen-an-vn10-truy-to-227-bi-can-trong-do-co-ca-si-chi-dan-an-tay-2026040308051239.htm",
    "https://tuoitre.vn/vien-kiem-sat-tp-hcm-nhieu-nghe-si-nguoi-noi-tieng-bi-khoi-to-do-lien-quan-ma-tuy-20251209142132042.htm",
    "https://tuoitre.vn/nha-thiet-ke-cong-tri-lien-quan-ma-tuy-nguoi-noi-tieng-cung-la-cong-dan-deu-bi-xu-ly-nghiem-20250724192919372.htm",
    "https://thanhnien.vn/chi-dan-huu-tin-va-loat-sao-viet-gay-on-ao-vi-dinh-toi-ma-tuy-185241110141122628.htm",
    "https://thanhnien.vn/ca-si-long-nhat-bi-bat-showbiz-viet-lien-tiep-chan-dong-vi-ma-tuy-18526052013032001.htm",
    "https://thanhnien.vn/nghe-si-dinh-ma-tuy-can-mot-lan-ranh-do-185260520134802695.htm",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
]


def clean_markdown(text: str) -> str:
    """
    Làm sạch markdown trả về từ crawler:
    - Chuẩn hóa newline
    - Xóa khoảng trắng thừa
    - Giảm số dòng trống liên tiếp
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_get_title(metadata: dict, url: str) -> str:
    """
    Lấy title từ metadata của crawler.
    Nếu không có thì fallback sang domain/path.
    """
    if metadata:
        for key in ["title", "og:title", "twitter:title"]:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    parsed = urlparse(url)
    return parsed.path.strip("/").split("/")[-1] or parsed.netloc


def extract_markdown_from_result(result) -> str:
    """
    Crawl4AI có thể trả markdown ở nhiều dạng tùy version.
    Hàm này giúp code đỡ phụ thuộc chặt vào version.
    """
    markdown = getattr(result, "markdown", "")

    if isinstance(markdown, str):
        return markdown

    for attr in ["fit_markdown", "raw_markdown", "markdown"]:
        value = getattr(markdown, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    return str(markdown) if markdown else ""

async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str,
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    # Một số version Crawl4AI có result.success
    success = getattr(result, "success", True)
    if success is False:
        error_message = getattr(result, "error_message", "Unknown crawl error")
        raise RuntimeError(f"Crawl failed for {url}: {error_message}")

    metadata = getattr(result, "metadata", {}) or {}
    title = safe_get_title(metadata, url)

    content_markdown = extract_markdown_from_result(result)
    content_markdown = clean_markdown(content_markdown)

    if not content_markdown:
        raise ValueError(f"Crawled content is empty for URL: {url}")

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    success_count = 0

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

        try:
            article = await crawl_article(url)

            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename

            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            success_count += 1
            print(f" ✓ Saved: {filepath}")

        except Exception as e:
            print(f" ✗ Failed: {url}")
            print(f"   Reason: {e}")

    print(f"\nDone. Crawled successfully: {success_count}/{len(ARTICLE_URLS)} articles.")

    if success_count < 5:
        print("⚠ Cảnh báo: Task 2 yêu cầu tối thiểu 5 bài báo crawl thành công.")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
