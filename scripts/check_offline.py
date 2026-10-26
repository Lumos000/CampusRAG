from __future__ import annotations

import json
import os
import sqlite3
import sys

import _bootstrap  # noqa: F401

from campus_rag.config import load_settings


def main() -> None:
    settings = load_settings()
    checks: list[tuple[str, bool, str]] = []
    for name, path in (("Qwen", settings.generator_model), ("Embedding", settings.embedding_model),
                       ("Reranker", settings.reranker_model)):
        weights = ([path / "model.safetensors", path / "pytorch_model.bin"]
                   + list(path.glob("*.gguf")) + list(path.glob("model-*.safetensors"))) if path.exists() else []
        complete = path.is_dir() and (path / "config.json").is_file() and any(item.is_file() and item.stat().st_size > 0 for item in weights)
        checks.append((name, complete, str(path)))
    checks.extend([
        ("SQLite", settings.database_path.is_file(), str(settings.database_path)),
        ("Index records", (settings.index_dir / "records.jsonl").is_file(), str(settings.index_dir / "records.jsonl")),
        ("Index vectors", (settings.index_dir / "vectors.npy").is_file(), str(settings.index_dir / "vectors.npy")),
    ])
    if settings.database_path.exists():
        connection = sqlite3.connect(settings.database_path)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            checks.append(("Database integrity", integrity and count > 0, f"chunks={count}"))
        finally:
            connection.close()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(json.dumps({name: {"ok": ok, "detail": detail} for name, ok, detail in checks}, ensure_ascii=False, indent=2))
    if not all(ok for _, ok, _ in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
