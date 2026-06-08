from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.embedding_service import rerank_chunks
from app.services.pgvector_store import search_pgvector_semantic
from app.services.rag_service import answer_with_knowledge, retrieve_knowledge
from app.services.vector_index import search_index


TOPK_RECALL_K = 20
FINAL_RECALL_K = 4
FACT_THRESHOLD = 0.6
_PGVECTOR_AVAILABLE: bool | None = None
_PGVECTOR_ERROR: str | None = None


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    expected_answer_hint: str
    expected_chunk_ids: list[str]
    question_type: str
    key_facts: list[str]


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data" / "knowledge_base").exists():
            return parent
    return current.parents[4]


def _qa_file() -> Path:
    return _project_root() / "data" / "knowledge_base" / "qa_test_set.json"


def _chunk_file() -> Path:
    return _project_root() / "data" / "knowledge_base" / "chunks" / "chunks.jsonl"


def _normalize(text: str) -> str:
    normalized = re.sub(r"\s+", "", text or "").lower()
    typo_map = {
        "胜景": "胜境",
        "灵山梵官": "灵山梵宫",
        "灵山大昭壁": "灵山大照壁",
        "五名桥": "五明桥",
        "阿育王住": "阿育王柱",
        "祥福禅寺": "祥符禅寺",
        "九龍": "九龙",
        "吉详颂": "吉祥颂",
    }
    for wrong, right in typo_map.items():
        normalized = normalized.replace(wrong, right)
    return normalized


def _terms(text: str) -> set[str]:
    normalized = _normalize(text)
    terms: set[str] = set()
    for match in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized):
        token = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) > 1:
                terms.add(token)
            for size in (2, 3, 4):
                if len(token) >= size:
                    terms.update(token[index : index + size] for index in range(len(token) - size + 1))
        else:
            terms.add(token)
    return terms


def _load_cases(limit: int | None = None) -> list[EvaluationCase]:
    path = _qa_file()
    if not path.exists():
        return []
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        EvaluationCase(
            id=item.get("id", f"qa-{index:03d}"),
            question=item.get("question", ""),
            expected_answer_hint=item.get("expected_answer_hint", ""),
            expected_chunk_ids=item.get("expected_chunk_ids", []),
            question_type=item.get("question_type", "unknown"),
            key_facts=item.get("key_facts", []),
        )
        for index, item in enumerate(raw_cases, start=1)
    ]
    cases = [case for case in cases if case.question and case.expected_answer_hint]
    return cases[:limit] if limit else cases


def _load_chunks() -> list[dict[str, Any]]:
    path = _chunk_file()
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _score_answer(answer: str, expected_hint: str) -> tuple[float, list[str]]:
    answer_terms = _terms(answer)
    expected_terms = _terms(expected_hint)
    if not expected_terms:
        return 0.0, []

    matched = sorted(expected_terms & answer_terms, key=len, reverse=True)
    coverage = len(matched) / len(expected_terms)
    return round(coverage, 4), matched[:8]


def _fact_hit(answer: str, facts: list[str]) -> tuple[float, list[str], list[str]]:
    if not facts:
        return 1.0, [], []
    normalized_answer = _normalize(answer)
    matched: list[str] = []
    missing: list[str] = []
    for fact in facts:
        normalized_fact = _normalize(fact)
        if normalized_fact in normalized_answer:
            matched.append(fact)
        elif "没有" in normalized_fact and ("没有" in normalized_answer or "未提供" in normalized_answer or "不能" in normalized_answer):
            matched.append(fact)
        elif "未提供" in normalized_fact and ("没有" in normalized_answer or "未提供" in normalized_answer or "不能" in normalized_answer):
            matched.append(fact)
        elif "不能" in normalized_fact and ("不能" in normalized_answer or "不足以" in normalized_answer):
            matched.append(fact)
        else:
            missing.append(fact)
    return round(len(matched) / len(facts), 4), matched, missing


def _out_of_scope_intent_hit(question: str, answer: str) -> tuple[bool, str]:
    normalized_question = _normalize(question)
    normalized_answer = _normalize(answer)
    generic_decline = any(
        phrase in normalized_answer
        for phrase in ["现有知识库没有", "知识库没有", "没有实时数据", "不能可靠回答", "不能确认", "不足以"]
    )
    official_fallback = any(
        phrase in normalized_answer
        for phrase in ["官方小程序", "现场广播", "实时服务为准", "以景区官方"]
    )

    if any(word in normalized_question for word in ["天气"]):
        return (
            any(phrase in normalized_answer for phrase in ["没有实时数据", "没有实时天气", "不包含实时天气", "实时服务为准"]),
            "天气类越界拒答",
        )
    if any(word in normalized_question for word in ["酒店", "最便宜"]):
        return (
            any(phrase in normalized_answer for phrase in ["没有酒店", "外部服务数据", "不能可靠回答", "不包含酒店"]),
            "酒店类越界拒答",
        )
    if any(word in normalized_question for word in ["临时取消", "明天", "今晚", "准确时间"]):
        return (
            any(phrase in normalized_answer for phrase in ["没有实时数据", "临时调整", "官方小程序", "现场广播", "实时服务为准"])
            or official_fallback,
            "临时排期类越界拒答",
        )
    if any(word in normalized_question for word in ["客服电话", "电话"]):
        return (
            any(phrase in normalized_answer for phrase in ["没有客服电话", "外部服务数据", "不能可靠回答"]),
            "电话类越界拒答",
        )
    if any(word in normalized_question for word in ["停车", "优惠券", "排队", "客流", "打车"]):
        return (generic_decline, "实时服务类越界拒答")
    if any(word in normalized_question for word in ["帮我买", "预约", "预订", "代购"]):
        return (
            any(phrase in normalized_answer for phrase in ["不能代办", "不能代购", "不能可靠回答", "不能"]),
            "动作代办类越界拒答",
        )
    if any(word in normalized_question for word in ["预测", "明年", "未来"]):
        return (
            any(phrase in normalized_answer for phrase in ["不足以预测", "不能可靠回答", "不足以"]),
            "预测类越界拒答",
        )
    return (generic_decline or official_fallback, "通用越界拒答")


def _chunk_text(chunk: dict[str, Any]) -> str:
    return "\n".join(
        [
            chunk.get("title", ""),
            chunk.get("section", ""),
            " ".join(chunk.get("keywords", [])),
            chunk.get("content", ""),
        ]
    )


def _recall_item(
    chunk: dict[str, Any],
    rank: int,
    *,
    expected_hint: str,
    score: float | None = None,
) -> dict[str, Any]:
    expected_coverage, matched_terms = _score_answer(_chunk_text(chunk), expected_hint)
    return {
        "rank": rank,
        "chunkId": chunk.get("chunk_id", ""),
        "documentId": chunk.get("document_id", ""),
        "title": chunk.get("title", ""),
        "section": chunk.get("section", ""),
        "topic": (chunk.get("metadata") or {}).get("topic", ""),
        "citation": chunk.get("citation", ""),
        "vectorScore": round(float(score if score is not None else chunk.get("_vector_score", 0.0)), 6),
        "rerankScore": round(float(chunk.get("_rerank_score", 0.0)), 6),
        "expectedCoverage": expected_coverage,
        "matchedTerms": matched_terms,
    }


def _has_expected_hit(items: list[dict[str, Any]], expected_chunk_ids: list[str]) -> bool:
    if not expected_chunk_ids:
        return any(item.get("expectedCoverage", 0.0) >= 0.35 for item in items)
    item_ids = {item["chunkId"] for item in items}
    return bool(set(expected_chunk_ids) & item_ids)


def _mrr(items: list[dict[str, Any]], expected_chunk_ids: list[str]) -> float:
    if not expected_chunk_ids:
        for item in items:
            if item.get("expectedCoverage", 0.0) >= 0.35:
                return round(1.0 / item["rank"], 4)
        return 0.0
    expected = set(expected_chunk_ids)
    for item in items:
        if item["chunkId"] in expected:
            return round(1.0 / item["rank"], 4)
    return 0.0


def _citation_correct(citations: list[str], expected_chunk_ids: list[str]) -> bool:
    if not expected_chunk_ids:
        return bool(citations)
    return any(chunk_id in citation for chunk_id in expected_chunk_ids for citation in citations)


def _failure_reasons(
    *,
    question_type: str,
    recall20_hit: bool,
    recall4_hit: bool,
    fact_score: float,
    citation_correct: bool,
    expected_chunk_ids: list[str],
) -> list[str]:
    reasons: list[str] = []
    if question_type == "out_of_scope":
        if fact_score < FACT_THRESHOLD:
            reasons.append("越界拒答事实命中低")
        return reasons
    if expected_chunk_ids and not recall20_hit:
        reasons.append("Recall@20 未命中预期 chunk")
    if expected_chunk_ids and recall20_hit and not recall4_hit:
        reasons.append("Recall@4 未命中预期 chunk")
    if fact_score < FACT_THRESHOLD:
        reasons.append("关键事实命中低")
    if not citation_correct:
        reasons.append("引用未指向预期证据")
    return reasons


async def _retrieve_for_eval(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    require_pgvector: bool = False,
) -> tuple[str, list[tuple[float, dict[str, Any]]], str | None]:
    global _PGVECTOR_AVAILABLE, _PGVECTOR_ERROR
    if _PGVECTOR_AVAILABLE is not False:
        try:
            pgvector_results = await search_pgvector_semantic(question, top_k=TOPK_RECALL_K)
            if pgvector_results:
                _PGVECTOR_AVAILABLE = True
                return "pgvector", pgvector_results, None
            _PGVECTOR_AVAILABLE = False
            _PGVECTOR_ERROR = "pgvector returned no rows"
        except Exception as error:
            _PGVECTOR_AVAILABLE = False
            _PGVECTOR_ERROR = str(error)

    if require_pgvector:
        raise RuntimeError(_PGVECTOR_ERROR or "pgvector retrieval is required but unavailable")

    mixed_chunks = await retrieve_knowledge(question, [], limit=TOPK_RECALL_K)
    if mixed_chunks:
        return "local_mixed", [
            (float(chunk.get("_vector_score", 0.0)) + float(chunk.get("_keyword_score", 0.0)) + float(chunk.get("_type_boost", 0.0)), chunk)
            for chunk in mixed_chunks
        ], _PGVECTOR_ERROR

    local_results = search_index(question, chunks, top_k=TOPK_RECALL_K)
    return "local_file", local_results, _PGVECTOR_ERROR


async def evaluate_rag_accuracy(
    limit: int | None = None,
    use_reranker: bool = False,
    require_pgvector: bool = False,
) -> dict[str, Any]:
    cases = _load_cases(limit)
    chunks = _load_chunks()
    results: list[dict[str, Any]] = []
    passed = 0
    citation_hits = 0
    recall20_hits = 0
    recall4_hits = 0
    fact_scores: list[float] = []
    mrr_scores: list[float] = []
    retrieval_sources: dict[str, int] = {}
    pgvector_errors: list[str] = []

    for case in cases:
        retrieval_source, recall_results, pgvector_error = await _retrieve_for_eval(
            case.question,
            chunks,
            require_pgvector=require_pgvector,
        )
        retrieval_sources[retrieval_source] = retrieval_sources.get(retrieval_source, 0) + 1
        if pgvector_error and pgvector_error not in pgvector_errors:
            pgvector_errors.append(pgvector_error)

        recalled_chunks = [chunk for _, chunk in recall_results]
        top20_items = [
            _recall_item(chunk, rank=index, expected_hint=case.expected_answer_hint, score=score)
            for index, (score, chunk) in enumerate(recall_results, start=1)
        ]

        if use_reranker:
            reranked = await rerank_chunks(case.question, recalled_chunks, top_n=FINAL_RECALL_K)
            final_items = []
            for index, item in enumerate(reranked, start=1):
                item.chunk["_rerank_score"] = item.score
                final_items.append(
                    _recall_item(item.chunk, rank=index, expected_hint=case.expected_answer_hint)
                )
            if not final_items:
                final_items = top20_items[:FINAL_RECALL_K]
        else:
            final_items = top20_items[:FINAL_RECALL_K]

        rag = await answer_with_knowledge(case.question, [])
        coverage, matched_terms = _score_answer(rag.answer, case.expected_answer_hint)
        fact_score, matched_facts, missing_facts = _fact_hit(rag.answer, case.key_facts)
        if case.question_type == "out_of_scope":
            intent_hit, intent_label = _out_of_scope_intent_hit(case.question, rag.answer)
            if intent_hit:
                fact_score = 1.0
                matched_facts = sorted(set(matched_facts + [intent_label]))
                missing_facts = []
        recall20_hit = _has_expected_hit(top20_items, case.expected_chunk_ids)
        recall4_hit = _has_expected_hit(final_items, case.expected_chunk_ids)
        mrr = _mrr(top20_items, case.expected_chunk_ids)
        citation_correct = _citation_correct(rag.citations, case.expected_chunk_ids)
        failure_reasons = _failure_reasons(
            question_type=case.question_type,
            recall20_hit=recall20_hit,
            recall4_hit=recall4_hit,
            fact_score=fact_score,
            citation_correct=citation_correct,
            expected_chunk_ids=case.expected_chunk_ids,
        )
        is_passed = not failure_reasons

        passed += 1 if is_passed else 0
        citation_hits += 1 if citation_correct else 0
        recall20_hits += 1 if recall20_hit else 0
        recall4_hits += 1 if recall4_hit else 0
        fact_scores.append(fact_score)
        mrr_scores.append(mrr)

        results.append(
            {
                "id": case.id,
                "questionType": case.question_type,
                "question": case.question,
                "expectedHint": case.expected_answer_hint,
                "expectedChunkIds": case.expected_chunk_ids,
                "keyFacts": case.key_facts,
                "answerPreview": rag.answer[:260],
                "coverage": coverage,
                "factHitRate": fact_score,
                "passed": is_passed,
                "failureReasons": failure_reasons,
                "matchedTerms": matched_terms,
                "matchedFacts": matched_facts,
                "missingFacts": missing_facts,
                "citations": rag.citations[:4],
                "citationCorrect": citation_correct,
                "recallAt20Hit": recall20_hit,
                "recallAt4Hit": recall4_hit,
                "mrr": mrr,
                "retrievalSource": retrieval_source,
                "top20": top20_items[:TOPK_RECALL_K],
                "top4": final_items[:FINAL_RECALL_K],
            }
        )

    total = len(cases)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "recallAt20": round(recall20_hits / total, 4) if total else 0.0,
        "recallAt4": round(recall4_hits / total, 4) if total else 0.0,
        "mrr": round(sum(mrr_scores) / total, 4) if total else 0.0,
        "factHitRate": round(sum(fact_scores) / total, 4) if total else 0.0,
        "citationAccuracy": round(citation_hits / total, 4) if total else 0.0,
        "factThreshold": FACT_THRESHOLD,
        "useReranker": use_reranker,
        "requirePgvector": require_pgvector,
        "retrievalSources": retrieval_sources,
        "pgvectorFallbackErrors": pgvector_errors[:3],
        "results": results,
    }
