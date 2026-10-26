from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from campus_rag.config import load_settings


def main():
    parser = argparse.ArgumentParser(description='只读检查入库计数、外键、token上限及导入完成标记')
    parser.add_argument('--database', type=Path, default=load_settings().database_path)
    args = parser.parse_args()
    if not args.database.exists():
        raise SystemExit('数据库不存在，请先运行清洗和导入')
    with sqlite3.connect(args.database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        counts = {table: connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
            for table in ['documents', 'tables', 'table_rows', 'table_cells', 'chunks', 'document_sources', 'ingestion_state']}
        checks = {'integrity': connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok',
            'foreign_keys': not connection.execute('PRAGMA foreign_key_check').fetchall(),
            'token_limit': connection.execute('SELECT count(*) FROM chunks WHERE token_count > ?',
                (load_settings().chunk_hard_max,)).fetchone()[0] == 0,
            'completed_documents': counts['documents'] == counts['ingestion_state'],
            'nonempty': counts['documents'] > 0 and counts['chunks'] > 0}
        report = {'counts': counts, 'checks': checks,
            'max_chunk_tokens': connection.execute('SELECT max(token_count) FROM chunks').fetchone()[0],
            'missing_dates': connection.execute('SELECT count(*) FROM documents WHERE published_at IS NULL OR published_at=""').fetchone()[0]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
