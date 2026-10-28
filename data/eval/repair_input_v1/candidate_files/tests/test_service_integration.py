import json
import sqlite3
from pathlib import Path

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence, ParsedTable, SourceDocument


class FakeGenerator:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({
                "question_type": "aggregate", "time_range": ["2026"],
                "keywords": ["招生计划"], "entities": [], "fields": ["计划数"],
                "filters": [{"field": "首选科目", "operator": "eq", "value": "物理"}],
                "group_by": [], "unit": "人", "needs_text": False, "needs_table": True,
                "calculation": {"operation": "sum", "operands": [
                    {"field": "计划数", "aggregate": "sum", "filters": []}
                ], "multiplier": None, "output_unit": "人"},
            }, ensure_ascii=False)
        return "筛选首选科目为物理的记录后，计划合计为30人。"


class FakeIndex:
    def __init__(self, evidence):
        self.evidence = evidence

    def search(self, *_args, **_kwargs):
        return [self.evidence]


def test_end_to_end_system_owned_reference(tmp_path):
    db_path = tmp_path / "service.db"
    with Repository(db_path) as repository:
        repository.initialize()
        doc_id = repository.add_document(SourceDocument(
            "2026招生计划", "https://example.edu/official", "招生办公室", "2026-06-01", sha256="svc"
        ))
        table = ParsedTable("招生计划", ["专业", "首选科目", "计划数"], [
            ["通信工程", "物理", "10人"], ["计算机", "物理", "20人"], ["中文", "历史", "5人"],
        ])
        table_id = repository.add_table(doc_id, table, [[normalize_cell(v) for v in row] for row in table.rows])
    connection = sqlite3.connect(db_path)
    try:
        evidence = Evidence(
            "table-chunk", "招生计划表", 1.0, doc_id, "2026招生计划",
            "https://example.edu/official", "招生办公室", "2026-06-01", table_id=table_id,
        )
        dummy = Path(tmp_path)
        settings = Settings(dummy / "db", dummy, dummy, dummy, dummy)
        result = RAGService(settings, FakeIndex(evidence), FakeGenerator(), connection).answer("物理类计划多少人？")
    finally:
        connection.close()
    assert "30人" in result.answer
    assert result.calculation and result.calculation.ok
    assert result.references == ["2026招生计划｜招生办公室｜2026-06-01｜https://example.edu/official"]

