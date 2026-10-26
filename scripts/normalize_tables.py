"""Offline migration of derived metadata, retaining raw values and original IDs."""
import argparse
import sqlite3
from pathlib import Path

import _bootstrap  # noqa: F401
from campus_rag.config import load_settings
from campus_rag.database import Repository
from campus_rag.database.normalization import normalize_existing_tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database',type=Path,default=load_settings().database_path)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error('数据库不存在')
    backup = args.database.with_suffix('.before-normalization.db')
    if not backup.exists():
        with sqlite3.connect(args.database) as original, sqlite3.connect(backup) as saved:
            original.backup(saved)
    with Repository(args.database) as repo:
        repo.initialize()
        with repo.transaction():
            normalize_existing_tables(repo.connection)
        print('已离线规范化表格：',repo.connection.execute('SELECT count(*) FROM table_metadata').fetchone()[0])


if __name__ == '__main__':
    main()
