from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

from app.core.config import settings
from app.services.embedding_service import embed_texts
from app.services.vector_index import (
    INDEX_VERSION,
    chunk_search_text,
    text_embedding,
)


SCHEMA_VERSION = 1


def psycopg_database_url(database_url: str | None = None) -> str:
    url = database_url or settings.database_url
    return re.sub(r"^postgresql\+psycopg://", "postgresql://", url)


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def embedding_dimensions() -> int:
    return settings.embedding_dimensions or 1024


def normalize_embedding(values: list[float], dimensions: int | None = None) -> list[float]:
    target = dimensions or embedding_dimensions()
    if len(values) == target:
        return values
    if len(values) > target:
        return values[:target]
    return values + [0.0] * (target - len(values))


def get_connection() -> psycopg.Connection:
    return psycopg.connect(psycopg_database_url(), row_factory=dict_row, connect_timeout=1)


def _current_embedding_dimensions(cursor: psycopg.Cursor) -> int | None:
    cursor.execute(
        """
        SELECT a.atttypmod - 4 AS dimensions
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        WHERE c.relname = 'rag_chunks'
          AND a.attname = 'embedding'
          AND NOT a.attisdropped
        """
    )
    row = cursor.fetchone()
    if not row:
        return None
    dimensions = row["dimensions"]
    return int(dimensions) if dimensions and dimensions > 0 else None


def ensure_schema(
    connection: psycopg.Connection,
    *,
    dimensions: int | None = None,
    recreate_on_dimension_mismatch: bool = False,
) -> None:
    target_dimensions = dimensions or embedding_dimensions()
    with connection.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        current_dimensions = _current_embedding_dimensions(cursor)
        if (
            current_dimensions is not None
            and current_dimensions != target_dimensions
            and recreate_on_dimension_mismatch
        ):
            cursor.execute("DROP INDEX IF EXISTS rag_chunks_embedding_hnsw_idx")
            cursor.execute("DROP TABLE IF EXISTS rag_chunks")
            cursor.execute("DROP TABLE IF EXISTS rag_index_manifest")
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                section TEXT NOT NULL DEFAULT '',
                source_file TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL,
                keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                citation TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL,
                embedding vector({target_dimensions}) NOT NULL,
                index_version INTEGER NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_index_manifest (
                id INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
                schema_version INTEGER NOT NULL,
                index_version INTEGER NOT NULL,
                provider TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                vector_count INTEGER NOT NULL,
                chunks_file TEXT NOT NULL DEFAULT '',
                updated_at TIMESTAMPTZ NOT NULL
            )
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS rag_chunks_document_id_idx ON rag_chunks(document_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS rag_chunks_metadata_idx ON rag_chunks USING gin(metadata)"
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
            ON rag_chunks USING hnsw (embedding vector_cosine_ops)
            """
        )
    connection.commit()


def chunk_to_row(chunk: dict[str, Any]) -> dict[str, Any]:
    return chunk_to_row_with_embedding(chunk)


def chunk_to_row_with_embedding(
    chunk: dict[str, Any],
    embedding: list[float] | None = None,
) -> dict[str, Any]:
    search_text = chunk_search_text(chunk)
    vector = normalize_embedding(embedding or text_embedding(search_text))
    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk.get("document_id", ""),
        "title": chunk.get("title", ""),
        "section": chunk.get("section", ""),
        "source_file": chunk.get("source_file", ""),
        "source_type": chunk.get("source_type", ""),
        "content": chunk.get("content", ""),
        "keywords": json.dumps(chunk.get("keywords", []), ensure_ascii=False),
        "metadata": json.dumps(chunk.get("metadata", {}), ensure_ascii=False),
        "citation": chunk.get("citation", ""),
        "search_text": search_text,
        "embedding": vector_literal(vector),
        "index_version": INDEX_VERSION,
    }


async def chunks_to_rows(
    chunks: list[dict[str, Any]],
    *,
    require_embedding_api: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_size = max(settings.embedding_batch_size or 16, 1)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        search_texts = [chunk_search_text(chunk) for chunk in batch]
        embeddings = await embed_texts(search_texts, allow_fallback=not require_embedding_api)
        if len(embeddings) != len(batch):
            embeddings = []
        for index, chunk in enumerate(batch):
            embedding = embeddings[index].embedding if embeddings else None
            rows.append(chunk_to_row_with_embedding(chunk, embedding))
    return rows


def upsert_chunks(
    chunks: list[dict[str, Any]],
    *,
    chunks_file: str = "",
    prune_missing: bool = True,
) -> dict[str, Any]:
    rows = [chunk_to_row(chunk) for chunk in chunks]
    return upsert_rows(rows, chunks, chunks_file=chunks_file, prune_missing=prune_missing)


async def upsert_chunks_with_embeddings(
    chunks: list[dict[str, Any]],
    *,
    chunks_file: str = "",
    prune_missing: bool = True,
    require_embedding_api: bool = False,
) -> dict[str, Any]:
    rows = await chunks_to_rows(chunks, require_embedding_api=require_embedding_api)
    return upsert_rows(rows, chunks, chunks_file=chunks_file, prune_missing=prune_missing)


def upsert_rows(
    rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    chunks_file: str = "",
    prune_missing: bool = True,
) -> dict[str, Any]:
    with get_connection() as connection:
        ensure_schema(connection, recreate_on_dimension_mismatch=True)

        with connection.cursor() as cursor:
            if prune_missing:
                chunk_ids = [row["chunk_id"] for row in rows]
                if chunk_ids:
                    cursor.execute("DELETE FROM rag_chunks WHERE NOT (chunk_id = ANY(%s))", (chunk_ids,))
                else:
                    cursor.execute("DELETE FROM rag_chunks")

            cursor.executemany(
                """
                INSERT INTO rag_chunks (
                    chunk_id, document_id, title, section, source_file, source_type,
                    content, keywords, metadata, citation, search_text, embedding, index_version, updated_at
                )
                VALUES (
                    %(chunk_id)s, %(document_id)s, %(title)s, %(section)s, %(source_file)s, %(source_type)s,
                    %(content)s, %(keywords)s::jsonb, %(metadata)s::jsonb, %(citation)s, %(search_text)s,
                    %(embedding)s::vector, %(index_version)s, now()
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id,
                    title = EXCLUDED.title,
                    section = EXCLUDED.section,
                    source_file = EXCLUDED.source_file,
                    source_type = EXCLUDED.source_type,
                    content = EXCLUDED.content,
                    keywords = EXCLUDED.keywords,
                    metadata = EXCLUDED.metadata,
                    citation = EXCLUDED.citation,
                    search_text = EXCLUDED.search_text,
                    embedding = EXCLUDED.embedding,
                    index_version = EXCLUDED.index_version,
                    updated_at = EXCLUDED.updated_at
                """,
                rows,
            )
            cursor.execute(
                """
                INSERT INTO rag_index_manifest (
                    id, schema_version, index_version, provider, dimensions,
                    chunk_count, vector_count, chunks_file, updated_at
                )
                VALUES (1, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    schema_version = EXCLUDED.schema_version,
                    index_version = EXCLUDED.index_version,
                    provider = EXCLUDED.provider,
                    dimensions = EXCLUDED.dimensions,
                    chunk_count = EXCLUDED.chunk_count,
                    vector_count = EXCLUDED.vector_count,
                    chunks_file = EXCLUDED.chunks_file,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    SCHEMA_VERSION,
                    INDEX_VERSION,
                    f"postgresql_pgvector_{settings.embedding_model_name}",
                    embedding_dimensions(),
                    len(chunks),
                    len(rows),
                    chunks_file,
                    datetime.now(timezone.utc),
                ),
            )
        connection.commit()

    return {
        "schema_version": SCHEMA_VERSION,
        "index_version": INDEX_VERSION,
        "provider": f"postgresql_pgvector_{settings.embedding_model_name}",
        "dimensions": embedding_dimensions(),
        "chunk_count": len(chunks),
        "vector_count": len(chunks),
        "chunks_file": chunks_file,
    }


def row_to_chunk(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": row["chunk_id"],
        "document_id": row["document_id"],
        "title": row["title"],
        "section": row["section"],
        "source_file": row["source_file"],
        "source_type": row["source_type"],
        "content": row["content"],
        "keywords": row["keywords"] or [],
        "metadata": row["metadata"] or {},
        "citation": row["citation"],
        "_pgvector_score": round(float(row["score"]), 6),
    }


def search_pgvector(query: str, *, top_k: int = 12) -> list[tuple[float, dict[str, Any]]]:
    query_embedding = vector_literal(text_embedding(query))
    with get_connection() as connection:
        ensure_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_id, document_id, title, section, source_file, source_type,
                    content, keywords, metadata, citation,
                    1 - (embedding <=> %s::vector) AS score
                FROM rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cursor.fetchall()

    results: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        score = float(row["score"])
        if score > 0:
            results.append((score, row_to_chunk(row)))
    return results


async def search_pgvector_semantic(
    query: str,
    *,
    top_k: int = 12,
    allow_query_fallback: bool = False,
) -> list[tuple[float, dict[str, Any]]]:
    query_embeddings = await embed_texts([query], allow_fallback=allow_query_fallback)
    if not query_embeddings:
        if not allow_query_fallback:
            raise RuntimeError("Embedding API returned no query vector")
        return search_pgvector(query, top_k=top_k)

    query_embedding = vector_literal(normalize_embedding(query_embeddings[0].embedding))
    with get_connection() as connection:
        ensure_schema(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    chunk_id, document_id, title, section, source_file, source_type,
                    content, keywords, metadata, citation,
                    1 - (embedding <=> %s::vector) AS score
                FROM rag_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cursor.fetchall()

    results: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        score = float(row["score"])
        if score > 0:
            results.append((score, row_to_chunk(row)))
    return results


def read_manifest() -> dict[str, Any]:
    try:
        with get_connection() as connection:
            ensure_schema(connection)
            with connection.cursor() as cursor:
                cursor.execute("SELECT * FROM rag_index_manifest WHERE id = 1")
                row = cursor.fetchone()
    except Exception:
        return {}

    if not row:
        return {}
    return dict(row)
