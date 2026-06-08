"""
RAG Evaluation Pipeline using DeepEval.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics:
        - Faithfulness
        - Answer Relevance
        - Context Recall
        - Context Precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md
"""

import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Paths
# =============================================================================

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

GOLDEN_DATASET_PATH = EVAL_DIR / "golden_dataset.json"
RESULTS_PATH = EVAL_DIR / "results.md"


# =============================================================================
# Config
# =============================================================================

TOP_K = int(os.getenv("EVAL_TOP_K", "5"))

# DeepEval dùng LLM-as-judge. Dùng gpt-4o-mini để tiết kiệm chi phí cho lab.
EVAL_MODEL = os.getenv("DEEPEVAL_MODEL", "gpt-4o-mini")

# Có thể set EVAL_LIMIT=3 để test nhanh trước khi chạy full.
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "0"))

METRIC_THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.7"))


# =============================================================================
# Load dataset
# =============================================================================

def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    if not GOLDEN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Golden dataset not found: {GOLDEN_DATASET_PATH}")

    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("golden_dataset.json must be a list of dicts.")

    if len(data) < 15:
        print(f"⚠ Golden dataset has {len(data)} items. Requirement is at least 15.")

    required_keys = {"question", "expected_answer", "expected_context"}

    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(f"Item {i} missing keys: {missing}")

    if EVAL_LIMIT > 0:
        data = data[:EVAL_LIMIT]
        print(f"⚠ EVAL_LIMIT={EVAL_LIMIT}. Only evaluating first {len(data)} cases.")

    return data


# =============================================================================
# RAG pipeline adapters
# =============================================================================

def _load_task10_helpers():
    """
    Import helpers từ Task 10.
    Đặt trong function để tránh import lỗi quá sớm khi chạy eval.
    """
    from src.task10_generation import (
        SYSTEM_PROMPT,
        TOP_P,
        TEMPERATURE,
        GENERATION_MODEL,
        reorder_for_llm,
        format_context,
    )

    return {
        "SYSTEM_PROMPT": SYSTEM_PROMPT,
        "TOP_P": TOP_P,
        "TEMPERATURE": TEMPERATURE,
        "GENERATION_MODEL": GENERATION_MODEL,
        "reorder_for_llm": reorder_for_llm,
        "format_context": format_context,
    }


def _build_sources_from_chunks(chunks: list[dict]) -> list[dict]:
    """
    Build source records cho eval.
    Khác Task 10 một chút: giữ cả full content để DeepEval có retrieval_context đầy đủ.
    """
    sources = []

    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {}) or {}

        source_name = (
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("path")
            or f"Source {i}"
        )

        doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"

        sources.append({
            "citation_label": f"S{i}",
            "source": source_name,
            "type": doc_type,
            "retrieval_source": chunk.get("source", "unknown"),
            "score": float(chunk.get("score", 0.0)),
            "chunk_index": metadata.get("chunk_index"),
            "metadata": metadata,
            "content": chunk.get("content", ""),
            "content_preview": chunk.get("content", "")[:300],
        })

    return sources


def _generate_answer_from_chunks(query: str, chunks: list[dict], config_name: str) -> dict:
    """
    Dùng cùng logic generation với Task 10 nhưng giữ full source content cho evaluation.
    """
    if not chunks:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": config_name,
        }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    helpers = _load_task10_helpers()

    reordered = helpers["reorder_for_llm"](chunks)
    context = helpers["format_context"](reordered)

    user_message = f"""Context:
{context}

---

Question:
{query}

Hãy trả lời bằng tiếng Việt. Nhớ cite bằng các nhãn [S1], [S2], ... tương ứng với Context."""

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=helpers["GENERATION_MODEL"],
        messages=[
            {"role": "system", "content": helpers["SYSTEM_PROMPT"]},
            {"role": "user", "content": user_message},
        ],
        temperature=helpers["TEMPERATURE"],
        top_p=helpers["TOP_P"],
    )

    answer = response.choices[0].message.content or ""

    return {
        "answer": answer,
        "sources": _build_sources_from_chunks(reordered),
        "retrieval_source": config_name,
    }


class HybridRerankPipeline:
    """
    Config A:
        Task 9 retrieve()
        = semantic search + BM25 lexical search + RRF reranking + PageIndex fallback.
    """

    name = "Config A: hybrid + RRF rerank"

    def generate_with_citation(self, question: str) -> dict:
        from src.task9_retrieval_pipeline import retrieve

        chunks = retrieve(
            question,
            top_k=TOP_K,
            use_reranking=True,
        )

        return _generate_answer_from_chunks(
            query=question,
            chunks=chunks,
            config_name="hybrid",
        )


class DenseOnlyPipeline:
    """
    Config B:
        Chỉ dùng semantic search từ Weaviate.
        Không dùng BM25, không RRF, không PageIndex fallback.
    """

    name = "Config B: dense-only"

    def generate_with_citation(self, question: str) -> dict:
        from src.task5_semantic_search import semantic_search

        chunks = semantic_search(question, top_k=TOP_K)

        for chunk in chunks:
            chunk["source"] = "dense_only"

        return _generate_answer_from_chunks(
            query=question,
            chunks=chunks,
            config_name="dense_only",
        )


# =============================================================================
# DeepEval
# =============================================================================

def _make_metric(metric_name: str):
    """
    Tạo metric object mới cho từng test case để tránh state bị ghi đè.
    """
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )

    if metric_name == "Faithfulness":
        return FaithfulnessMetric(
            threshold=METRIC_THRESHOLD,
            model=EVAL_MODEL,
            include_reason=True,
        )

    if metric_name == "Answer Relevance":
        return AnswerRelevancyMetric(
            threshold=METRIC_THRESHOLD,
            model=EVAL_MODEL,
            include_reason=True,
        )

    if metric_name == "Context Recall":
        return ContextualRecallMetric(
            threshold=METRIC_THRESHOLD,
            model=EVAL_MODEL,
            include_reason=True,
        )

    if metric_name == "Context Precision":
        return ContextualPrecisionMetric(
            threshold=METRIC_THRESHOLD,
            model=EVAL_MODEL,
            include_reason=True,
        )

    raise ValueError(f"Unknown metric: {metric_name}")


def _extract_retrieval_context(result: dict) -> list[str]:
    """
    DeepEval retrieval_context cần list[str].
    Ưu tiên full content; fallback sang content_preview nếu cần.
    """
    contexts = []

    for source in result.get("sources", []):
        text = source.get("content") or source.get("content_preview") or ""
        text = text.strip()

        if text:
            contexts.append(text)

    return contexts


def _safe_score(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    Returns:
        {
            "config_name": str,
            "overall": dict,
            "average": float,
            "cases": list[dict]
        }
    """
    from deepeval.test_case import LLMTestCase

    metric_names = [
        "Faithfulness",
        "Answer Relevance",
        "Context Recall",
        "Context Precision",
    ]

    case_results = []

    print(f"\n=== Evaluating {rag_pipeline.name} ===")
    print(f"Dataset size: {len(golden_dataset)}")
    print(f"Judge model: {EVAL_MODEL}")
    print(f"Threshold: {METRIC_THRESHOLD}")

    for idx, item in enumerate(golden_dataset, 1):
        question = item["question"]
        expected_answer = item["expected_answer"]

        print("\n" + "-" * 80)
        print(f"[{idx}/{len(golden_dataset)}] Question: {question}")

        rag_result = rag_pipeline.generate_with_citation(question)
        answer = rag_result.get("answer", "")
        retrieval_context = _extract_retrieval_context(rag_result)

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_answer,
            retrieval_context=retrieval_context,
        )

        metrics_result = {}

        for metric_name in metric_names:
            metric = _make_metric(metric_name)

            try:
                metric.measure(test_case)

                score = _safe_score(getattr(metric, "score", 0.0))
                reason = getattr(metric, "reason", "")

                metrics_result[metric_name] = {
                    "score": score,
                    "reason": reason,
                    "success": bool(score >= METRIC_THRESHOLD),
                }

                print(f"  {metric_name}: {score:.3f}")

            except Exception as e:
                metrics_result[metric_name] = {
                    "score": 0.0,
                    "reason": f"Metric failed: {e}",
                    "success": False,
                }

                print(f"  {metric_name}: ERROR — {e}")

        case_avg = mean([m["score"] for m in metrics_result.values()])

        case_results.append({
            "index": idx,
            "question": question,
            "expected_answer": expected_answer,
            "expected_context": item.get("expected_context", ""),
            "actual_answer": answer,
            "retrieval_context_count": len(retrieval_context),
            "sources": [
                {
                    "source": s.get("source"),
                    "type": s.get("type"),
                    "score": s.get("score"),
                    "chunk_index": s.get("chunk_index"),
                }
                for s in rag_result.get("sources", [])
            ],
            "metrics": metrics_result,
            "average": case_avg,
        })

        print(f"  Case Average: {case_avg:.3f}")

    overall = {}

    for metric_name in metric_names:
        overall[metric_name] = mean([
            case["metrics"][metric_name]["score"]
            for case in case_results
        ])

    overall_average = mean(list(overall.values()))

    print("\n=== Overall ===")
    for metric_name, score in overall.items():
        print(f"{metric_name}: {score:.3f}")
    print(f"Average: {overall_average:.3f}")

    return {
        "config_name": rag_pipeline.name,
        "overall": overall,
        "average": overall_average,
        "cases": case_results,
    }


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa 2 configs:
        - Config A: hybrid + RRF rerank
        - Config B: dense-only
    """
    configs = [
        HybridRerankPipeline(),
        DenseOnlyPipeline(),
    ]

    comparison = {}

    for config in configs:
        comparison[config.name] = evaluate_with_deepeval(config, golden_dataset)

    return comparison


# =============================================================================
# Export Results
# =============================================================================

def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _get_worst_performers(primary_result: dict, n: int = 3) -> list[dict]:
    return sorted(primary_result["cases"], key=lambda x: x["average"])[:n]


def _infer_failure_stage(case: dict) -> tuple[str, str]:
    """
    Heuristic để điền Worst Performers:
        - Recall/Precision thấp => retrieval issue
        - Faithfulness thấp => generation grounding issue
        - Relevance thấp => answer alignment issue
    """
    metrics = case["metrics"]

    scores = {
        name: info["score"]
        for name, info in metrics.items()
    }

    worst_metric = min(scores, key=scores.get)
    worst_score = scores[worst_metric]

    if worst_metric in {"Context Recall", "Context Precision"}:
        stage = "Retrieval"
    elif worst_metric == "Faithfulness":
        stage = "Generation / Grounding"
    elif worst_metric == "Answer Relevance":
        stage = "Generation / Relevance"
    else:
        stage = "Unknown"

    reason = metrics[worst_metric].get("reason") or f"Lowest metric: {worst_metric}={worst_score:.3f}"
    reason = reason.replace("\n", " ").strip()

    if len(reason) > 180:
        reason = reason[:177] + "..."

    return stage, reason


def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    config_names = list(comparison.keys())

    if len(config_names) < 2:
        raise ValueError("Need at least 2 configs for A/B comparison.")

    config_a = comparison[config_names[0]]
    config_b = comparison[config_names[1]]

    metrics = [
        "Faithfulness",
        "Answer Relevance",
        "Context Recall",
        "Context Precision",
    ]

    content = "# RAG Evaluation Results\n\n"

    content += "## Framework sử dụng\n\n"
    content += "- **Framework:** DeepEval\n"
    content += f"- **Judge model:** `{EVAL_MODEL}`\n"
    content += f"- **Threshold:** `{METRIC_THRESHOLD}`\n"
    content += f"- **Golden dataset size:** `{len(config_a['cases'])}`\n"
    content += "- **Metrics:** Faithfulness, Answer Relevance, Context Recall, Context Precision\n\n"

    content += "---\n\n"

    content += "## Overall Scores\n\n"
    content += "| Metric | Config A (hybrid + RRF rerank) | Config B (dense-only) | Δ |\n"
    content += "|--------|-------------------------------:|----------------------:|---:|\n"

    for metric in metrics:
        a = config_a["overall"][metric]
        b = config_b["overall"][metric]
        delta = a - b
        content += f"| {metric} | {_format_float(a)} | {_format_float(b)} | {_format_float(delta)} |\n"

    content += (
        f"| **Average** | **{_format_float(config_a['average'])}** "
        f"| **{_format_float(config_b['average'])}** "
        f"| **{_format_float(config_a['average'] - config_b['average'])}** |\n"
    )

    content += "\n---\n\n"

    content += "## A/B Comparison Analysis\n\n"

    content += "**Config A — hybrid + RRF rerank:**\n\n"
    content += (
        "Config A sử dụng retrieval pipeline đầy đủ: semantic search từ Weaviate, "
        "lexical search bằng BM25, merge bằng RRF và PageIndex fallback khi cần. "
        "Cấu hình này được kỳ vọng có context recall tốt hơn vì kết hợp cả dense retrieval "
        "và keyword matching.\n\n"
    )

    content += "**Config B — dense-only:**\n\n"
    content += (
        "Config B chỉ sử dụng semantic search từ Weaviate. Cấu hình này đơn giản hơn, "
        "nhưng có thể bỏ lỡ các câu hỏi cần keyword chính xác như số điều luật, tên tội danh, "
        "hoặc tên riêng trong bài báo.\n\n"
    )

    if config_a["average"] >= config_b["average"]:
        conclusion = (
            "Config A tốt hơn hoặc tương đương Config B theo điểm trung bình. "
            "Điều này cho thấy hybrid retrieval giúp pipeline lấy evidence ổn định hơn, "
            "đặc biệt với dữ liệu pháp luật có nhiều thuật ngữ và số điều khoản."
        )
    else:
        conclusion = (
            "Config B có điểm trung bình cao hơn Config A trong lần chạy này. "
            "Điều này có thể do BM25/RRF đưa thêm context nhiễu vào prompt. "
            "Cần kiểm tra worst performers để tinh chỉnh chunking hoặc reranking."
        )

    content += f"**Kết luận:**\n\n{conclusion}\n\n"

    content += "---\n\n"

    content += "## Worst Performers (Bottom 3 — Config A)\n\n"
    content += "| # | Question | Faithfulness | Relevance | Recall | Precision | Failure Stage | Root Cause |\n"
    content += "|---|----------|-------------:|----------:|-------:|----------:|---------------|------------|\n"

    worst_cases = _get_worst_performers(config_a, n=3)

    for i, case in enumerate(worst_cases, 1):
        stage, root_cause = _infer_failure_stage(case)

        q = case["question"].replace("|", "\\|")
        root_cause = root_cause.replace("|", "\\|")

        content += (
            f"| {i} | {q} "
            f"| {_format_float(case['metrics']['Faithfulness']['score'])} "
            f"| {_format_float(case['metrics']['Answer Relevance']['score'])} "
            f"| {_format_float(case['metrics']['Context Recall']['score'])} "
            f"| {_format_float(case['metrics']['Context Precision']['score'])} "
            f"| {stage} | {root_cause} |\n"
        )

    content += "\n---\n\n"

    content += "## Per-case Scores — Config A\n\n"
    content += "| # | Question | Faithfulness | Relevance | Recall | Precision | Average |\n"
    content += "|---|----------|-------------:|----------:|-------:|----------:|--------:|\n"

    for case in config_a["cases"]:
        q = case["question"].replace("|", "\\|")
        content += (
            f"| {case['index']} | {q} "
            f"| {_format_float(case['metrics']['Faithfulness']['score'])} "
            f"| {_format_float(case['metrics']['Answer Relevance']['score'])} "
            f"| {_format_float(case['metrics']['Context Recall']['score'])} "
            f"| {_format_float(case['metrics']['Context Precision']['score'])} "
            f"| {_format_float(case['average'])} |\n"
        )

    content += "\n---\n\n"

    content += "## Recommendations\n\n"

    content += "### Cải tiến 1 — Làm sạch context từ bài báo\n"
    content += "**Action:** Loại bỏ markdown ảnh, menu, navigation và footer khỏi các bài báo trước khi chunking.\n\n"
    content += "**Expected impact:** Tăng Context Precision và giảm khả năng LLM cite nhầm đoạn nhiễu.\n\n"

    content += "### Cải tiến 2 — Ưu tiên nguồn pháp luật cho câu hỏi pháp lý\n"
    content += "**Action:** Thêm rule hoặc metadata filter: nếu query chứa `Điều`, `hình phạt`, `Bộ luật`, `Luật`, ưu tiên `type=legal`.\n\n"
    content += "**Expected impact:** Tăng Faithfulness cho câu hỏi pháp luật, giảm nhiễu từ bài báo.\n\n"

    content += "### Cải tiến 3 — Reranking mạnh hơn cho top candidates\n"
    content += "**Action:** Thử cross-encoder reranker như Jina Reranker v2 cho top 20 candidates trước generation.\n\n"
    content += "**Expected impact:** Cải thiện Context Precision và Answer Relevance, đặc biệt với câu hỏi có nhiều thực thể như nghệ sĩ/người nổi tiếng.\n\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"✓ Exported results to: {RESULTS_PATH}")
    print("=" * 80)

    print("\nOverall Scores:")
    print("| Metric | Config A | Config B | Delta |")
    print("|--------|----------|----------|-------|")
    for metric in metrics:
        a = config_a["overall"][metric]
        b = config_b["overall"][metric]
        print(f"| {metric} | {_format_float(a)} | {_format_float(b)} | {_format_float(a - b)} |")

    print(
        f"| Average | {_format_float(config_a['average'])} "
        f"| {_format_float(config_b['average'])} "
        f"| {_format_float(config_a['average'] - config_b['average'])} |"
    )

    print("\nWorst performers:")
    for i, case in enumerate(worst_cases, 1):
        print(f"{i}. Avg={case['average']:.3f} — {case['question']}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    comparison = compare_configs(
        rag_pipeline=None,
        golden_dataset=golden_dataset,
    )

    # primary results = Config A
    first_config_name = list(comparison.keys())[0]
    results = comparison[first_config_name]

    export_results(results, comparison)