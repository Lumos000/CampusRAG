from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

from campus_rag.types import Chunk, ParsedTable, SourceDocument


class Repository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._in_transaction = False

    def _commit(self):
        if not self._in_transaction:
            self.connection.commit()

    @contextmanager
    def transaction(self):
        self._in_transaction = True
        try:
            self.connection.execute('BEGIN')
            yield
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        finally:
            self._in_transaction = False

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Repository":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        self.connection.executescript(schema)
        self.connection.commit()

    def add_document(self, document: SourceDocument) -> int:
        cursor = self.connection.execute(
            """INSERT OR IGNORE INTO documents
            (title, source_url, department, published_at, crawled_at, local_path,
             content_type, sha256, version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                document.title, document.source_url, document.department,
                document.published_at, document.crawled_at, document.local_path,
                document.content_type, document.sha256, document.version,
            ),
        )
        if cursor.rowcount == 1:
            doc_id = int(cursor.lastrowid)
        else:
            row = self.connection.execute(
                "SELECT id FROM documents WHERE source_url = ? AND sha256 = ?",
                (document.source_url, document.sha256),
            ).fetchone()
            if row is None:
                raise RuntimeError("文档写入失败")
            doc_id = int(row["id"])
        self._commit()
        return doc_id

    def add_table(self, doc_id: int, table: ParsedTable, normalized: list[list[dict[str, str | None]]]) -> int:
        search_text = " ".join([table.name, *table.headers, *(" ".join(row) for row in table.rows[:30])])
        cursor = self.connection.execute(
            """INSERT INTO tables
            (doc_id, name, locator, header_levels, unit, notes, search_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (doc_id, table.name, table.locator, table.header_levels, table.unit, table.notes, search_text),
        )
        table_id = int(cursor.lastrowid)
        column_ids: list[int] = []
        for index, header in enumerate(table.headers):
            sample = [row[index] for row in normalized if index < len(row)]
            numeric = sum(cell.get("numeric_value") is not None for cell in sample)
            data_type = "number" if sample and numeric >= max(1, len(sample) // 2) else "text"
            unit = next((str(cell.get("unit")) for cell in sample if cell.get("unit")), "")
            cur = self.connection.execute(
                """INSERT INTO table_columns
                (table_id, column_index, original_name, normalized_name, data_type, unit)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (table_id, index, header, _normalize_header(header), data_type, unit),
            )
            column_ids.append(int(cur.lastrowid))
        for row_index, row in enumerate(normalized):
            label = str(row[0].get("raw_value", "")) if row else ""
            cur = self.connection.execute(
                "INSERT INTO table_rows (table_id, row_index, row_label) VALUES (?, ?, ?)",
                (table_id, row_index, label),
            )
            row_id = int(cur.lastrowid)
            for column_index, cell in enumerate(row[: len(column_ids)]):
                self.connection.execute(
                    """INSERT INTO table_cells
                    (row_id, column_id, raw_value, normalized_value, numeric_value, unit, locator)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row_id, column_ids[column_index], cell.get("raw_value", ""),
                        cell.get("normalized_value", ""), cell.get("numeric_value"),
                        cell.get("unit", ""), f"{table.locator}:R{row_index + 1}C{column_index + 1}",
                    ),
                )
        from campus_rag.database.normalization import normalize_table
        normalize_table(self.connection, table_id)
        self._commit()
        return table_id

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            digest = hashlib.sha256(f'{chunk.doc_id}:{chunk.table_id}:{chunk.heading_path}:{chunk.text}'.encode('utf-8')).hexdigest()
            self.connection.execute(
                """INSERT OR IGNORE INTO chunks
                (id, doc_id, table_id, kind, heading_path, text, token_count, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk.chunk_id, chunk.doc_id, chunk.table_id, chunk.kind,
                    chunk.heading_path, chunk.text, chunk.token_count, digest,
                ),
            )
        self._commit()

    def add_issue(self, doc_id: int | None, code: str, message: str, severity: str = "warning") -> None:
        self.connection.execute(
            "INSERT INTO audit_issues (doc_id, severity, code, message) VALUES (?, ?, ?, ?)",
            (doc_id, severity, code, message),
        )
        self._commit()

    def index_records(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            """SELECT c.id, c.text, c.kind, c.doc_id, c.table_id, c.heading_path,
                      d.title, d.source_url, d.department, d.published_at
               FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"""
        ).fetchall()

    def get_document(self, doc_id: int) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()


def _normalize_header(value: str) -> str:
    return "".join(value.lower().split()).replace("（", "(").replace("）", ")")

