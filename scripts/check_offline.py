from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import importlib.util
from pathlib import Path

import _bootstrap  # noqa: F401

from campus_rag.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description='检查离线模型、数据库和索引准备情况')
    parser.add_argument('--config', type=Path, help='指定配置，默认使用校园公开资料库')
    args = parser.parse_args()
    settings = load_settings(args.config)
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
    if settings.planner_adapter_path:
        checks.append(('LoRA dependency', importlib.util.find_spec('peft') is not None, 'peft'))
        for name in ('adapter_config.json', 'adapter_model.safetensors'):
            path = settings.planner_adapter_path / name
            checks.append(('Planner adapter '+name, path.is_file() and path.stat().st_size > 0, str(path)))
    if settings.database_path.exists():
        connection = sqlite3.connect(settings.database_path.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            count = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
            checks.append(("Database integrity", integrity and count > 0, f"chunks={count}"))
            try:
                from campus_rag.index.hybrid import validate_records
                with (settings.index_dir / 'records.jsonl').open(encoding='utf-8') as handle:
                    records = [json.loads(line) for line in handle if line.strip()]
                validate_records(records, connection)
                import numpy as np
                vectors = np.load(settings.index_dir / 'vectors.npy', mmap_mode='r')
                if vectors.shape != (len(records), 512) or not np.isfinite(vectors).all():
                    raise ValueError('索引向量与记录不一致')
                checks.append(('Database/index consistency', True, f'records={len(records)}'))
            except (OSError, ValueError, KeyError) as exc:
                checks.append(('Database/index consistency', False, str(exc)))
        finally:
            connection.close()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(json.dumps({name: {"ok": ok, "detail": detail} for name, ok, detail in checks}, ensure_ascii=False, indent=2))
    if not all(ok for _, ok, _ in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
