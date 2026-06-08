from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.services.embedding_service import rerank_chunks, semantic_search
from app.services.pgvector_store import search_pgvector_semantic
from app.services.vector_index import search_index

# 尝试导入 pgvector 相关依赖（可选）
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass(frozen=True)
class KnowledgeAnswer:
    answer: str
    route_suggestion: list[str]
    citations: list[str]
    raw_snippets: list[dict] = field(default_factory=list)


_CHUNK_CACHE: dict[str, Any] = {"mtime": None, "chunks": []}
_PGVECTOR_AVAILABLE: bool | None = None

_DOMAIN_TERMS = [
    "灵山胜境",
    "灵山大佛",
    "祥符禅寺",
    "九龙灌浴",
    "灵山梵宫",
    "五印坛城",
    "曼飞龙塔",
    "大照壁",
    "五明桥",
    "佛足坛",
    "五智门",
    "菩提大道",
    "阿育王柱",
    "百子戏弥勒",
    "佛教文化博览馆",
    "玄奘",
    "赵朴初",
    "拈花湾",
    "拈花广场",
    "梵天花海",
    "香月花街",
    "五灯湖",
    "鹿鸣谷",
    "游客行为",
    "满意度",
    "停留时长",
    "消费",
    "路线",
    "亲子",
    "文化",
    "拍照",
]

_STOPWORDS = {
    "什么",
    "怎么",
    "如何",
    "一下",
    "介绍",
    "推荐",
    "适合",
    "可以",
    "哪些",
    "哪里",
    "为什么",
    "多少",
}

_DEFAULT_ROUTE = ["灵山大照壁", "五明桥", "九龙灌浴", "祥符禅寺", "灵山大佛", "灵山梵宫"]

_ROUTE_PRESETS = {
    "文化": ["灵山大照壁", "祥符禅寺", "佛教文化博览馆", "灵山梵宫", "五印坛城"],
    "历史": ["灵山大照壁", "祥符禅寺", "灵山大佛", "灵山梵宫", "五印坛城"],
    "亲子": ["九龙灌浴", "百子戏弥勒", "佛教文化博览馆", "灵山大佛", "灵山梵宫"],
    "拍照": ["灵山大照壁", "五明桥", "九龙灌浴", "灵山大佛", "灵山梵宫"],
    "拈花": ["拈花广场", "梵天花海", "香月花街", "五灯湖", "鹿鸣谷"],
}

_QUESTION_TYPE_RULES = {
    "behavior": ["行为数据", "游客行为", "样例数据", "满意度", "停留时长", "消费", "团队", "attraction_name", "正文弱相关"],
    "route_photo": ["拍照", "打卡", "取景"],
    "route_family": ["亲子", "带孩子", "带娃", "家庭"],
    "route": ["路线", "怎么走", "安排", "游览", "先看", "去哪几个"],
    "opening_hours": ["开放", "闭馆", "几点关门", "几点开放"],
    "performance_time": ["演出", "表演", "几点", "吉祥颂", "九龙灌浴"],
    "price": ["票价", "门票", "多少钱", "收费", "免费", "费用"],
}


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "data" / "knowledge_base").exists():
            return parent
    return current.parents[4]


def _chunk_file() -> Path:
    return _project_root() / "data" / "knowledge_base" / "chunks" / "chunks.jsonl"


def _normalize(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
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
        normalized = normalized.replace(wrong.lower(), right.lower())
    return normalized


def _tokenize(text: str) -> set[str]:
    normalized = _normalize(text)
    tokens: set[str] = set()
    for match in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized):
        term = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", term):
            if 1 < len(term) <= 6:
                tokens.add(term)
            for size in (2, 3, 4):
                if len(term) >= size:
                    tokens.update(term[index : index + size] for index in range(len(term) - size + 1))
        else:
            tokens.add(term)

    for term in _DOMAIN_TERMS:
        if term.lower() in normalized:
            tokens.add(term.lower())

    return {token for token in tokens if token and token not in _STOPWORDS}


def _detect_question_type(query: str) -> str:
    for question_type, keywords in _QUESTION_TYPE_RULES.items():
        if any(keyword in query for keyword in keywords):
            return question_type
    for term in _DOMAIN_TERMS:
        if term in query:
            return "single_spot"
    return "general"


def _matched_spot_names(query: str, chunks: list[dict[str, Any]]) -> set[str]:
    normalized_query = _normalize(query)
    names: set[str] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        spot_name = str(metadata.get("spot_name", "")).strip()
        if spot_name and _normalize(spot_name) in normalized_query:
            names.add(spot_name)
    aliases = {
        "大照壁": "灵山大照壁",
        "梵宫": "灵山梵宫",
        "大佛": "灵山大佛",
        "博览馆": "佛教文化博览馆",
        "祥福禅寺": "祥符禅寺",
        "灵山梵官": "灵山梵宫",
        "灵山大昭壁": "灵山大照壁",
        "五名桥": "五明桥",
        "阿育王住": "阿育王柱",
        "吉详颂": "灵山梵宫",
    }
    for alias, spot_name in aliases.items():
        if alias in query:
            names.add(spot_name)
    return names


def _preferred_document_ids(question_type: str) -> set[str]:
    if question_type == "behavior":
        return {"tourism_behavior_summary", "ling_shan_curated_facts"}
    return set()


def _boost_chunk_for_question_type(
    chunk: dict[str, Any],
    question_type: str,
    matched_spots: set[str],
) -> float:
    metadata = chunk.get("metadata", {})
    topic = str(metadata.get("topic", ""))
    document_id = chunk.get("document_id", "")
    spot_name = str(metadata.get("spot_name", ""))
    boost = 0.0

    preferred_documents = _preferred_document_ids(question_type)
    if preferred_documents and document_id in preferred_documents:
        boost += 18.0
    elif preferred_documents:
        boost -= 12.0

    if matched_spots:
        if spot_name in matched_spots or chunk.get("title") in matched_spots:
            boost += 16.0
        elif spot_name:
            boost -= 8.0

    topic_boosts = {
        "route_photo": {"route": 14.0, "photo": 10.0},
        "route_family": {"route": 14.0, "family": 10.0, "audience": 6.0},
        "route": {"route": 12.0},
        "opening_hours": {"opening_hours": 16.0, "schedule": 6.0},
        "performance_time": {"performance_time": 16.0, "performance": 10.0},
        "price": {"price": 18.0},
        "behavior": {"behavior": 18.0},
    }
    boost += topic_boosts.get(question_type, {}).get(topic, 0.0)
    return boost


def _postprocess_candidates(
    candidates: list[tuple[float, dict[str, Any]]],
    *,
    question_type: str,
    matched_spots: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    scored: list[tuple[float, dict[str, Any]]] = []
    for base_score, chunk in candidates:
        boost = _boost_chunk_for_question_type(chunk, question_type, matched_spots)
        keyword_score = float(chunk.get("_keyword_score", 0.0))
        chunk["_type_boost"] = round(boost, 3)
        scored.append((base_score + keyword_score + boost, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]


def _merge_candidates(candidates: list[tuple[float, dict[str, Any]]]) -> list[tuple[float, dict[str, Any]]]:
    merged: dict[str, tuple[float, dict[str, Any]]] = {}
    for score, chunk in candidates:
        chunk_id = chunk.get("chunk_id", "")
        if not chunk_id:
            continue
        previous = merged.get(chunk_id)
        if previous is None or score > previous[0]:
            merged[chunk_id] = (score, chunk)
    return list(merged.values())


def _load_chunks() -> list[dict[str, Any]]:
    path = _chunk_file()
    if not path.exists():
        _CHUNK_CACHE["mtime"] = None
        _CHUNK_CACHE["chunks"] = []
        return []

    mtime = path.stat().st_mtime
    if _CHUNK_CACHE["mtime"] == mtime:
        return _CHUNK_CACHE["chunks"]

    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            chunk = json.loads(line)
            search_text = "\n".join(
                [
                    chunk.get("title", ""),
                    chunk.get("section", ""),
                    " ".join(chunk.get("keywords", [])),
                    chunk.get("content", ""),
                ]
            )
            chunk["_search_text"] = _normalize(search_text)
            chunk["_tokens"] = _tokenize(search_text)
            chunks.append(chunk)

    _CHUNK_CACHE["mtime"] = mtime
    _CHUNK_CACHE["chunks"] = chunks
    return chunks


def _score_chunk(
    chunk: dict[str, Any],
    query: str,
    query_terms: set[str],
    interest_terms: set[str],
) -> float:
    search_text = chunk.get("_search_text", "")
    chunk_terms = chunk.get("_tokens", set())
    if not search_text or not chunk_terms:
        return 0.0

    score = 0.0
    overlap = query_terms & chunk_terms
    score += len(overlap) * 2.0

    title_terms = _tokenize(chunk.get("title", ""))
    section_terms = _tokenize(chunk.get("section", ""))
    score += len(query_terms & title_terms) * 4.0
    score += len(query_terms & section_terms) * 2.5
    score += len(interest_terms & chunk_terms) * 1.5

    normalized_query = _normalize(query)
    if normalized_query and normalized_query in search_text:
        score += 8.0

    for keyword in chunk.get("keywords", []):
        if keyword and keyword.lower() in normalized_query:
            score += 5.0

    metadata = chunk.get("metadata", {})
    spot_name = _normalize(str(metadata.get("spot_name", "")))
    if spot_name and spot_name in normalized_query:
        score += 8.0
    for term in _DOMAIN_TERMS:
        normalized_term = term.lower()
        if normalized_term in normalized_query and normalized_term in search_text:
            if normalized_term in _normalize(chunk.get("title", "")) or normalized_term in spot_name:
                score += 6.0
            elif spot_name:
                score -= 6.0

    return score


async def retrieve_knowledge(query: str, interests: Optional[list[str]] = None, limit: int = 4) -> list[dict[str, Any]]:
    """
    混合检索策略：
    1. 优先使用语义检索（bge-m3 embedding + bge-reranker）
    2. 如果语义检索失败或无可用API，降级为关键词检索
    3. 结合兴趣标签进行个性化排序
    """
    chunks = _load_chunks()
    if not chunks:
        return []

    query_terms = _tokenize(query)
    interest_terms = _tokenize(" ".join(interests or []))
    query_with_interests = f"{query} {' '.join(interests or [])}".strip()
    question_type = _detect_question_type(query)
    matched_spots = _matched_spot_names(query, chunks)
    recall_top_k = 24 if question_type in {"route", "route_photo", "route_family", "behavior"} else 20

    # 优先使用 PostgreSQL/pgvector 持久化索引。
    global _PGVECTOR_AVAILABLE
    pgvector_results = []
    if _PGVECTOR_AVAILABLE is not False:
        try:
            pgvector_results = await search_pgvector_semantic(query_with_interests, top_k=recall_top_k)
            _PGVECTOR_AVAILABLE = bool(pgvector_results)
        except Exception:
            _PGVECTOR_AVAILABLE = False
    if pgvector_results:
        candidate_chunks: list[dict[str, Any]] = []
        for vector_score, chunk in pgvector_results:
            search_text = "\n".join(
                [
                    chunk.get("title", ""),
                    chunk.get("section", ""),
                    " ".join(chunk.get("keywords", [])),
                    chunk.get("content", ""),
                ]
            )
            chunk["_search_text"] = _normalize(search_text)
            chunk["_tokens"] = _tokenize(search_text)
            keyword_score = _score_chunk(chunk, query, query_terms, interest_terms)
            chunk["_vector_source"] = "pgvector"
            chunk["_vector_score"] = round(vector_score, 6)
            chunk["_keyword_score"] = round(keyword_score, 3)
            candidate_chunks.append(chunk)

        pre_ranked = _postprocess_candidates(
            [(float(chunk.get("_vector_score", 0.0)) * 10.0, chunk) for chunk in candidate_chunks],
            question_type=question_type,
            matched_spots=matched_spots,
            limit=min(len(candidate_chunks), 12),
        )
        reranked_results = await rerank_chunks(query, pre_ranked, top_n=limit)
        if reranked_results:
            for result in reranked_results:
                result.chunk["_rerank_score"] = round(float(result.score), 6)
            return [result.chunk for result in reranked_results[:limit]]

        return pre_ranked[:limit]

    # 其次使用已构建的本地向量索引，避免每次查询都重新向量化全量文档。
    vector_results = search_index(query_with_interests, chunks, top_k=recall_top_k)
    if vector_results:
        candidates: list[tuple[float, dict[str, Any]]] = []
        for vector_score, chunk in vector_results:
            keyword_score = _score_chunk(chunk, query, query_terms, interest_terms)
            chunk["_vector_source"] = "local_file"
            chunk["_vector_score"] = round(vector_score, 6)
            chunk["_keyword_score"] = round(keyword_score, 3)
            candidates.append((vector_score * 10.0, chunk))
        for chunk in chunks:
            keyword_score = _score_chunk(chunk, query, query_terms, interest_terms)
            type_boost = _boost_chunk_for_question_type(chunk, question_type, matched_spots)
            if keyword_score > 0 or type_boost > 0:
                chunk["_vector_source"] = "local_file"
                chunk["_vector_score"] = round(float(chunk.get("_vector_score", 0.0)), 6)
                chunk["_keyword_score"] = round(keyword_score, 3)
                candidates.append((keyword_score + type_boost, chunk))
        candidates = _merge_candidates(candidates)
        return _postprocess_candidates(
            candidates,
            question_type=question_type,
            matched_spots=matched_spots,
            limit=limit,
        )

    # 优先尝试在线语义检索
    try:
        semantic_results = await semantic_search(query, chunks, interests, initial_top_k=12, final_top_k=limit)
        if semantic_results:
            # 对语义检索结果进行关键词增强
            reranked: list[tuple[float, dict[str, Any]]] = []
            for chunk in semantic_results:
                keyword_score = _score_chunk(chunk, query, query_terms, interest_terms)
                # 结合语义相似度（隐式）和关键词得分
                combined_score = keyword_score + 5.0  # 基础分保证语义结果优先
                reranked.append((combined_score, chunk))

            reranked.sort(key=lambda x: x[0], reverse=True)
            return [chunk for _, chunk in reranked[:limit]]
    except Exception:
        pass

    # 降级：纯关键词检索
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        score = _score_chunk(chunk, query, query_terms, interest_terms)
        score += _boost_chunk_for_question_type(chunk, question_type, matched_spots)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return []
    minimum_score = max(2.0, scored[0][0] * 0.35)
    return [chunk for score, chunk in scored if score >= minimum_score][:limit]


def _truncate(text: str, limit: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip("，,；;。 ") + "…"


def _clean_title_prefix(text: str) -> str:
    return re.sub(r"【[^】]*】\s*", "", text)


def _best_snippet(content: str, query: str) -> str:
    query_terms = _tokenize(query)
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    lines = [line for line in lines if not (line.startswith("【") and "】" in line)]
    field_preferences = {
        "位置": ["具体位置"],
        "在哪": ["具体位置"],
        "开放": ["演艺/开放信息"],
        "时间": ["演艺/开放信息"],
        "亮点": ["游玩亮点"],
        "玩": ["游玩亮点"],
        "文化": ["文化内涵"],
        "历史": ["文化内涵", "详细介绍"],
        "意义": ["文化内涵"],
        "消费": ["平均总消费", "费用均值"],
        "满意度": ["平均满意度"],
        "停留": ["平均停留时长"],
    }
    for query_word, fields in field_preferences.items():
        if query_word in query:
            for field in fields:
                for line in lines:
                    if line.startswith(field):
                        return _truncate(line, 220)

    if not lines:
        return ""

    scored: list[tuple[int, str]] = []
    for line in lines:
        line_terms = _tokenize(line)
        score = len(query_terms & line_terms)
        if line.startswith(("景区名称", "景点ID", "景点名称")):
            score -= 5
        if line.startswith(("具体位置", "文化内涵", "详细介绍", "游玩亮点", "演艺/开放信息")):
            score += 1
        scored.append((score, line))
    scored.sort(key=lambda item: item[0], reverse=True)

    if scored and scored[0][0] > 0:
        return _truncate(scored[0][1], 220)

    sentences = [item.strip() for item in re.split(r"(?<=[。！？；;])", content) if item.strip()]
    sentences = [s for s in sentences if not (s.startswith("【") and "】" in s)]
    return _truncate(sentences[0] if sentences else content)


def _route_from_chunks(chunks: list[dict[str, Any]], query: str, interests: list[str]) -> list[str]:
    route: list[str] = []
    route_intent = any(word in query for word in ["路线", "怎么走", "游览", "推荐", "安排", "玩", "打卡"])
    preference_text = f"{query} {' '.join(interests)}"
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        spot_name = metadata.get("spot_name")
        scenic_area = metadata.get("scenic_area", "")
        should_include = route_intent or (spot_name and spot_name in preference_text)
        if spot_name and scenic_area == "灵山胜境" and should_include and spot_name not in route:
            route.append(spot_name)

    for keyword, preset in _ROUTE_PRESETS.items():
        if keyword in preference_text:
            for spot in preset:
                if spot not in route:
                    route.append(spot)

    for spot in _DEFAULT_ROUTE:
        if spot not in route:
            route.append(spot)

    return route[:6]


def _citations(chunks: list[dict[str, Any]]) -> list[str]:
    citations: list[str] = []
    for chunk in chunks:
        citation = chunk.get("citation")
        if citation and citation not in citations:
            citations.append(citation)
    return citations


def _should_decline(query: str, chunks: list[dict[str, Any]]) -> str:
    realtime_words = ["今天", "现在", "实时", "明天", "最新", "临时", "排队", "客流", "天气", "停车", "优惠券", "今晚", "准确时间"]
    action_words = ["帮我买", "代购", "预约", "预订"]
    prediction_words = ["预测", "明年", "未来"]
    price_words = ["成人票", "儿童票", "最新价格", "价目表"]
    unsupported_words = ["酒店", "客服电话", "电话", "打车"]
    if any(word in query for word in unsupported_words):
        return "现有知识库没有酒店、客服电话或实时交通价格等外部服务数据，不能可靠回答。"
    if any(word in query for word in action_words):
        return "现有知识库只能提供导览资料，不能代办购买、预约或预订。"
    if any(word in query for word in prediction_words):
        return "现有知识库不足以预测未来客流或经营数据。"
    if any(word in query for word in realtime_words):
        has_realtime_evidence = any((chunk.get("metadata") or {}).get("data_scope") == "realtime" for chunk in chunks)
        if not has_realtime_evidence:
            return "现有知识库没有实时数据；请以景区官方小程序、现场广播或实时服务为准。"
    if any(word in query for word in price_words):
        has_full_price_table = any(
            (chunk.get("metadata") or {}).get("topic") == "price"
            and (chunk.get("metadata") or {}).get("data_scope") == "complete"
            for chunk in chunks
        )
        if not has_full_price_table:
            return "现有知识库没有完整最新票价表，不能确认成人票、儿童票等实时票价。"
    return ""


async def answer_with_knowledge(query: str, interests: list[str]) -> KnowledgeAnswer:
    chunks = await retrieve_knowledge(query, interests)
    if not chunks:
        return KnowledgeAnswer(
            answer=(
                "现有知识库没有足够证据回答这个问题。请补充景区资料或接入实时服务后再查询："
                f"{query}"
            ),
            route_suggestion=_DEFAULT_ROUTE,
            citations=[],
            raw_snippets=[],
        )

    decline = _should_decline(query, chunks)
    if decline:
        return KnowledgeAnswer(
            answer=decline,
            route_suggestion=_route_from_chunks(chunks, query, interests),
            citations=_citations(chunks),
            raw_snippets=[],
        )

    snippets = [(chunk, _best_snippet(chunk.get("content", ""), query)) for chunk in chunks]
    if re.search(r"什么时候|哪年|日期|时间", query):
        snippets.sort(key=lambda item: bool(re.search(r"\d{4}年|\d{4}-\d{2}-\d{2}", item[1])), reverse=True)

    ordered_chunks = [chunk for chunk, _ in snippets]
    raw_snippets = [
        {
            "title": chunk.get("title") or chunk.get("section") or "知识片段",
            "content": _clean_title_prefix(snippet),
            "citation": chunk.get("citation", ""),
        }
        for chunk, snippet in snippets
    ]

    knowledge_context = "\n".join([s["content"] for s in raw_snippets])

    return KnowledgeAnswer(
        answer=knowledge_context,
        route_suggestion=_route_from_chunks(ordered_chunks, query, interests),
        citations=_citations(ordered_chunks),
        raw_snippets=raw_snippets,
    )
