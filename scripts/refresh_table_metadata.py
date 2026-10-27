"""Refresh derived table metadata with a byte-preserving backup and audit."""
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import _bootstrap  # noqa: F401
from campus_rag.config import ROOT, load_settings
from campus_rag.database.normalization import normalize_existing_tables


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_digest(connection):
    checks = {}
    for name,sql in {
        'tables':'SELECT id,doc_id,name,unit,locator FROM tables ORDER BY id',
        'columns':'SELECT id,table_id,column_index,original_name FROM table_columns ORDER BY id',
        'rows':'SELECT * FROM table_rows ORDER BY id',
        'cells':'SELECT id,row_id,column_id,raw_value,locator FROM table_cells ORDER BY id',
        'chunks':'SELECT * FROM chunks ORDER BY id',
    }.items():
        h=hashlib.sha256()
        for row in connection.execute(sql):
            h.update(json.dumps(list(row),ensure_ascii=False).encode('utf-8'))
            h.update(b'\n')
        checks[name]=h.hexdigest()
    return checks


def main():
    settings=load_settings()
    backup=ROOT/'.tmp/baseline_campus_before_stage3.db'
    audit_path=ROOT/'data/eval/stage3_normalization_audit.json'
    if backup.exists() or audit_path.exists():
        raise SystemExit('Backup or audit already exists; refusing to overwrite.')
    backup.parent.mkdir(exist_ok=True)
    shutil.copyfile(settings.database_path,backup)
    with sqlite3.connect(settings.database_path) as connection:
        connection.row_factory=sqlite3.Row
        before=raw_digest(connection)
        with connection:
            normalize_existing_tables(connection)
            after=raw_digest(connection)
            if before!=after:
                raise ValueError('Raw records changed; rolling back derived refresh')
        assert connection.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
        assert not connection.execute('PRAGMA foreign_key_check').fetchall()
    audit={'backup':str(backup.relative_to(ROOT)), 'before_database_sha256':digest(backup),
        'after_database_sha256':digest(settings.database_path),'before_raw_sha256':before,
        'after_raw_sha256':after,'integrity_check':'ok','foreign_key_check':'ok',
        'index_sha256':{p.name:digest(p) for p in settings.index_dir.iterdir() if p.is_file()},
        'purpose':'Derived normalization only; original IDs, raw cells and chunks retained; no index rebuild.'}
    audit_path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Raw records unchanged; derived table metadata refreshed. Audit:',audit_path)


if __name__=='__main__':
    main()
