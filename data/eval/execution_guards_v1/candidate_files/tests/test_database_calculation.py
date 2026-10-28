from decimal import Decimal

from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import CalculationSpec, FilterSpec, OperandSpec, ParsedTable, QueryPlan, SourceDocument


def test_parameterized_filter_and_sum(tmp_path):
    database = tmp_path / "test.db"
    with Repository(database) as repository:
        repository.initialize()
        doc_id = repository.add_document(SourceDocument("招生计划", "https://example.edu/plan", sha256="abc"))
        table = ParsedTable(
            "2026招生计划", ["专业", "首选科目", "再选科目", "计划数"],
            [["通信工程", "物理", "政治、地理", "10人"], ["计算机科学", "物理", "政治、地理", "20人"],
             ["英语", "历史", "不限", "30人"]],
        )
        normalized = [[normalize_cell(value) for value in row] for row in table.rows]
        table_id = repository.add_table(doc_id, table, normalized)
        plan = QueryPlan(
            question_type="aggregate", keywords=["招生计划"], needs_text=False, needs_table=True,
            filters=[FilterSpec("首选科目", "eq", "物理"), FilterSpec("再选科目", "contains", "政治")],
            calculation=CalculationSpec("sum", [OperandSpec("计划数", "sum")], output_unit="人"),
        )
        result = StructuredQueryEngine(repository.connection).execute(plan, [table_id])
    assert result.ok
    assert result.value == Decimal("30")
    assert result.unit == "人"
    assert len(result.details) == 2
    assert result.evidence[0].source_url == "https://example.edu/plan"


def test_ambiguous_value_is_rejected(tmp_path):
    database = tmp_path / "test.db"
    with Repository(database) as repository:
        repository.initialize()
        doc_id = repository.add_document(SourceDocument("费用", "https://example.edu/fee", sha256="def"))
        table = ParsedTable("费用", ["类别", "金额"], [["学费", "5000元"], ["住宿费", "1000元"]])
        table_id = repository.add_table(doc_id, table, [[normalize_cell(value) for value in row] for row in table.rows])
        plan = QueryPlan(
            question_type="fee", needs_text=False, needs_table=True,
            calculation=CalculationSpec("sum", [OperandSpec("金额", "value")]),
        )
        result = StructuredQueryEngine(repository.connection).execute(plan, [table_id])
    assert not result.ok
    assert "增加筛选条件" in result.error

