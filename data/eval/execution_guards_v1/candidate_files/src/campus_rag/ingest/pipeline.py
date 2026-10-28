from __future__ import annotations

import uuid
import json
from functools import lru_cache
from pathlib import Path

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.ingest.chunker import TokenChunker, load_bge_tokenizer
from campus_rag.ingest.parsers import parse_file
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Chunk
from campus_rag.ingest.cleaning import content_key


def ingest_path(
    path: str | Path,
    settings: Settings,
    source_url: str = "",
    department: str = "",
    published_at: str | None = None,
    source_title: str = "",
) -> int:
    parsed = parse_file(path, source_url, department, published_at)
    if source_title:
        parsed.source.title = source_title
    return ingest_document(parsed, settings)


@lru_cache(maxsize=2)
def _tokenizer(path):
    return load_bge_tokenizer(path)


def ingest_document(parsed, settings, sources=None):
    key = content_key(parsed)
    encode, decode = _tokenizer(str(settings.embedding_model))
    chunker = TokenChunker(
        encode, decode, settings.chunk_target_min, settings.chunk_target_max,
        settings.chunk_hard_max, settings.chunk_overlap,
    )
    with Repository(settings.database_path) as repository:
        repository.initialize()
        with repository.transaction():
            existing = repository.connection.execute('SELECT doc_id FROM ingestion_state WHERE content_key=?', (key,)).fetchone()
            if existing:
                doc_id = existing['doc_id']
            else:
                if repository.connection.execute('SELECT 1 FROM documents WHERE source_url=? AND sha256=?',
                    (parsed.source.source_url, parsed.source.sha256)).fetchone():
                    raise ValueError('发现旧版未标记的导入记录，请先备份并使用新数据库重建，避免重复表格')
                doc_id = repository.add_document(parsed.source)
                for warning in parsed.warnings:
                    repository.add_issue(doc_id, 'parse_warning', warning)
                if not parsed.source.published_at:
                    repository.add_issue(doc_id, 'missing_date', '发布日期未知，不得以抓取时间代替')
                chunks = []
                for heading, text, count in chunker.split_sections(parsed.sections):
                    chunks.append(Chunk(str(uuid.uuid4()), doc_id, text, heading, count))
                for table in parsed.tables:
                    normalized = [[normalize_cell(value) for value in row] for row in table.rows]
                    table_id = repository.add_table(doc_id, table, normalized)
                    description = _table_description(table)
                    # All rows contribute to retrieval; no first-eight-row truncation.
                    for heading, text, count in chunker.split_sections([(table.name, description)]):
                        chunks.append(Chunk(str(uuid.uuid4()), doc_id, text, heading, count, 'table', table_id))
                if not chunks:
                    raise ValueError('没有可入库的正文或表格')
                if any(chunk.token_count > settings.chunk_hard_max for chunk in chunks):
                    raise ValueError('分块超出 token 硬上限')
                repository.add_chunks(chunks)
                repository.connection.execute('INSERT INTO ingestion_state(doc_id,content_key) VALUES (?,?)', (doc_id,key))
            for source in sources or [{'url': parsed.source.source_url}]:
                url = source.get('url') or source.get('final_url')
                if url:
                    repository.connection.execute('INSERT OR REPLACE INTO document_sources(doc_id,source_url,metadata_json) VALUES (?,?,?)',
                        (doc_id, url, json.dumps(source, ensure_ascii=False)))
    return doc_id


def _table_description(table) -> str:
    samples = [" | ".join(f'{header}={value}' for header, value in zip(table.headers, row)) for row in table.rows]
    return "\n".join([
        f"表格：{table.name}",
        f"字段：{' | '.join(table.headers)}",
        '\n\n'.join(samples),
        f"位置：{table.locator}",
    ])

