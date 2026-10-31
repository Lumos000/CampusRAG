from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from campus_rag.config import load_settings
from campus_rag.database import Repository
from campus_rag.index.hybrid import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description='为指定数据库构建检索索引')
    parser.add_argument('--config', type=Path, help='指定数据库和索引配置')
    args = parser.parse_args()
    settings = load_settings(args.config)
    with Repository(settings.database_path) as repository:
        repository.initialize()
        records = [dict(row) for row in repository.index_records()]
    if not records:
        raise SystemExit("数据库中没有可索引的分块，请先导入资料。")
    build_index(records, settings.index_dir, settings.embedding_model)
    print(f"索引构建完成：{len(records)} 条记录。")


if __name__ == "__main__":
    main()
