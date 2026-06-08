from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "services" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

from app.services.rag_eval_service import evaluate_rag_accuracy


SUMMARY_KEYS = [
    "total",
    "passed",
    "failed",
    "accuracy",
    "recallAt20",
    "recallAt4",
    "mrr",
    "factHitRate",
    "citationAccuracy",
    "useReranker",
    "requirePgvector",
    "retrievalSources",
    "pgvectorFallbackErrors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG QA baseline.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N cases.")
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Use the configured reranker to select the final Top4.",
    )
    parser.add_argument(
        "--require-pgvector",
        action="store_true",
        help="Fail instead of falling back to the local mixed index if pgvector/embedding is unavailable.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="Print failed cases in addition to the summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the full evaluation JSON to this path.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit with code 1 if accuracy is below this value, e.g. 0.9.",
    )
    return parser.parse_args()


def compact_failure(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "questionType": item["questionType"],
        "question": item["question"],
        "failureReasons": item["failureReasons"],
        "factHitRate": item["factHitRate"],
        "recallAt20Hit": item["recallAt20Hit"],
        "recallAt4Hit": item["recallAt4Hit"],
        "citationCorrect": item["citationCorrect"],
        "retrievalSource": item["retrievalSource"],
        "matchedFacts": item["matchedFacts"],
        "missingFacts": item["missingFacts"],
        "citations": item["citations"],
    }


def build_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {key: result[key] for key in SUMMARY_KEYS}
    failures = [item for item in result["results"] if not item["passed"]]
    summary["failureCount"] = len(failures)
    summary["failures"] = [compact_failure(item) for item in failures]
    return summary


async def run() -> int:
    args = parse_args()
    result = await evaluate_rag_accuracy(
        limit=args.limit,
        use_reranker=args.use_reranker,
        require_pgvector=args.require_pgvector,
    )
    summary = build_summary(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    printable = summary if args.show_failures else {key: summary[key] for key in SUMMARY_KEYS}
    print(json.dumps(printable, ensure_ascii=False, indent=2))

    if args.fail_under is not None and result["accuracy"] < args.fail_under:
        return 1
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(run()))


if __name__ == "__main__":
    main()
