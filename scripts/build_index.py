from __future__ import annotations

import _bootstrap  # noqa: F401

from campus_rag.config import load_settings
from campus_rag.database import Repository
from campus_rag.index.hybrid import build_index


def main() -> None:
    settings = load_settings()
    with Repository(settings.database_path) as repository:
        repository.initialize()
        records = [dict(row) for row in repository.index_records()]
    if not records:
        raise SystemExit("数据库中没有可索引的分块，请先导入资料。")
    build_index(records, settings.index_dir, settings.embedding_model)
    print(f"索引构建完成：{len(records)} 条记录。")


if __name__ == "__main__":
    main()
