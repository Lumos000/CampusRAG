from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_path: Path
    index_dir: Path
    embedding_model: Path
    reranker_model: Path
    generator_model: Path
    chunk_target_min: int = 280
    chunk_target_max: int = 420
    chunk_hard_max: int = 480
    chunk_overlap: int = 48
    bm25_top_k: int = 40
    dense_top_k: int = 40
    fusion_top_k: int = 25
    rerank_top_k: int = 8
    max_context_tokens: int = 7168
    max_new_tokens: int = 512
    numeric_shortcuts: bool = True


def _path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_settings(path: str | Path | None = None) -> Settings:
    config_path = Path(path) if path else ROOT / "config" / "settings.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("database_path", "index_dir", "embedding_model", "reranker_model", "generator_model"):
        data[key] = _path(data[key])
    return Settings(**data)

