from campus_rag.rag.citations import build_references
from campus_rag.rag.planner import parse_plan
from campus_rag.types import Evidence


def test_plan_schema_and_system_citations():
    plan = parse_plan('''前缀 {"question_type":"ratio","time_range":["2024"],"keywords":["面积"],
      "entities":[],"fields":["面积"],"filters":[],"group_by":[],"unit":"%",
      "needs_text":false,"needs_table":true,"calculation":{"operation":"ratio",
      "operands":[{"field":"面积","aggregate":"sum","filters":[]},{"field":"总面积","aggregate":"value","filters":[]}],
      "multiplier":null,"output_unit":"%"}} 后缀''')
    assert plan.needs_table
    assert plan.calculation.operation == "ratio"
    evidence = [
        Evidence("1", "x", 1, 1, "报告", "https://example.edu/a", "教务处", "2024-01-01"),
        Evidence("2", "y", 1, 1, "报告", "https://example.edu/a", "教务处", "2024-01-01"),
    ]
    references = build_references(evidence)
    assert references == ["报告｜教务处｜2024-01-01｜https://example.edu/a"]

