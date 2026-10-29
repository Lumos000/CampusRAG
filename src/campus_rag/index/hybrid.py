from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from campus_rag.index.bm25 import BM25Index
from campus_rag.types import Evidence
from campus_rag.tools.temporal import matches_period, question_periods
from campus_rag.ingest.navigation import navigation_chunk


def validate_records(records, connection):
    fields = ('text', 'kind', 'doc_id', 'table_id', 'heading_path', 'title',
              'source_url', 'department', 'published_at')
    chunks = {row[0]: tuple(row[1:]) for row in connection.execute(
        'SELECT c.id,c.text,c.kind,c.doc_id,c.table_id,c.heading_path,'
        'd.title,d.source_url,d.department,d.published_at '
        'FROM chunks c JOIN documents d ON d.id=c.doc_id')}
    if (len(chunks) != len(records) or {r['id'] for r in records} != set(chunks)
            or any(tuple(r.get(k) for k in fields) != chunks[r['id']] for r in records)):
        raise ValueError('索引与数据库分块不一致，请使用配套数据库和索引')


class HybridIndex:
    def __init__(self, records: list[dict], vectors: np.ndarray | None = None, embedder=None, reranker=None):
        self.records = records
        self.bm25 = BM25Index([record["text"] for record in records])
        self.vectors = _normalize(vectors) if vectors is not None else None
        self.embedder = embedder
        self.reranker = reranker
        self.navigation_tables = {r.get('table_id') for r in records if r.get('table_id') is not None and navigation_chunk(r['text'])}

    @classmethod
    def load(cls, index_dir: str | Path, embedding_model: str | Path, reranker_model: str | Path | None = None,
             database_path: str | Path | None = None):
        directory = Path(index_dir)
        with (directory / 'records.jsonl').open(encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        vectors = np.load(directory / "vectors.npy")
        if vectors.shape != (len(records), 512) or not np.isfinite(vectors).all():
            raise ValueError('索引记录与 BGE 向量不匹配，请重建索引')
        if database_path is not None:
            with closing(sqlite3.connect(Path(database_path).resolve().as_uri() + '?mode=ro', uri=True)) as connection:
                validate_records(records, connection)
        embedder = LocalEmbedder(embedding_model)
        reranker = LocalReranker(reranker_model) if reranker_model and Path(reranker_model).exists() else None
        return cls(records, vectors, embedder, reranker)

    def search(
        self, query: str, bm25_top_k: int = 40, dense_top_k: int = 40,
        fusion_top_k: int = 25, final_top_k: int = 8, mode: str = "rerank", time_range=None,
    ) -> list[Evidence]:
        if mode not in {"bm25", "dense", "hybrid", "rerank"}:
            raise ValueError(f"不支持的检索模式：{mode}")
        periods = question_periods(query) if time_range is None else time_range
        anchors = [term for term in ['招生章程','综合评价','教学质量报告'] if term in query]
        title_term = anchors[0] if len(anchors) == 1 else ''
        allowed = {i for i,r in enumerate(self.records)
                   if r.get('table_id') not in self.navigation_tables
                   and (not title_term or title_term in r.get('title',''))
                   and matches_period(r.get('title',''), r.get('published_at'), periods)}
        if not allowed:
            return []
        bm25 = [(i,s) for i,s in self.bm25.search(query, len(self.records)) if i in allowed][:bm25_top_k]
        dense = self._dense_search(query, dense_top_k, allowed)
        if mode == "bm25":
            return [self._evidence(index, score) for index, score in bm25[:final_top_k]]
        if mode == "dense":
            return [self._evidence(index, score) for index, score in dense[:final_top_k]]
        fused = reciprocal_rank_fusion([bm25, dense], k=60)[:fusion_top_k]
        candidates = [(index, self.records[index]["text"]) for index, _ in fused]
        if mode == "rerank" and self.reranker and candidates:
            scores = self.reranker.score(query, [text for _, text in candidates])
            ordered = sorted(zip(candidates, scores), key=lambda item: item[1], reverse=True)
            ranked = [(candidate[0], float(score)) for candidate, score in ordered[:final_top_k]]
        else:  # hybrid mode, or reranker intentionally unavailable
            score_map = dict(fused)
            ranked = [(index, score_map[index]) for index, _ in candidates[:final_top_k]]
        return [self._evidence(index, score) for index, score in ranked]

    def _dense_search(self, query: str, top_k: int, allowed=None) -> list[tuple[int, float]]:
        if self.embedder is None or self.vectors is None or not len(self.vectors):
            return []
        vector = _normalize(self.embedder.encode_queries([query]))[0]
        scores = self.vectors @ vector
        if allowed is not None:
            mask = np.zeros(len(scores), dtype=bool)
            mask[list(allowed)] = True
            scores[~mask] = -np.inf
        indexes = np.argsort(-scores)[:top_k]
        return [(int(index), float(scores[index])) for index in indexes if np.isfinite(scores[index])]

    def _evidence(self, index: int, score: float) -> Evidence:
        record = self.records[index]
        return Evidence(
            evidence_id=str(record["id"]), text=record["text"], score=score,
            doc_id=int(record["doc_id"]), title=record["title"],
            source_url=record["source_url"], department=record.get("department", ""),
            published_at=record.get("published_at"), chunk_id=str(record["id"]),
            table_id=record.get("table_id"),
        )


def reciprocal_rank_fusion(result_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for results in result_lists:
        for rank, (index, _) in enumerate(results, 1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class LocalEmbedder:
    def __init__(self, model_path: str | Path):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(model_path), local_files_only=True)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True))

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        instructed = [f"为这个句子生成表示以用于检索相关文章：{query}" for query in queries]
        return np.asarray(self.model.encode(instructed, normalize_embeddings=True, show_progress_bar=False))


class LocalReranker:
    def __init__(self, model_path: str | Path):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(str(model_path), local_files_only=True, max_length=512)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        return np.asarray(self.model.predict([(query, passage) for passage in passages], batch_size=4))


def build_index(records: list[dict], index_dir: str | Path, embedding_model: str | Path) -> None:
    directory = Path(index_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    embedder = LocalEmbedder(embedding_model)
    vectors = embedder.encode_documents([record["text"] for record in records])
    np.save(directory / "vectors.npy", vectors)


def _normalize(vectors: np.ndarray | None) -> np.ndarray | None:
    if vectors is None:
        return None
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return array / norms
