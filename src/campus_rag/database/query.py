from __future__ import annotations

import sqlite3
from decimal import Decimal

from campus_rag.tools.numbers import convert_unit
from campus_rag.types import CalculationResult, Evidence, FilterSpec, OperandSpec, QueryPlan
from campus_rag.tools.temporal import matches_period
from campus_rag.ingest.navigation import navigation_table_ids


class StructuredQueryEngine:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self.navigation_tables = navigation_table_ids(connection)

    def execute(self, plan: QueryPlan, table_ids: list[int] | None = None) -> CalculationResult:
        if not plan.needs_table or plan.calculation.operation == "none":
            return CalculationResult(ok=False, error="该问题不需要结构化计算")
        if not plan.calculation.operands:
            return CalculationResult(ok=False, error='没有有效的计算操作数，拒绝返回空集合合计')
        tables = self._candidate_tables(plan, table_ids)
        if not tables:
            return CalculationResult(ok=False, error="未找到与问题匹配的结构化表格")
        all_details: list[dict] = []
        operand_values: list[tuple[Decimal, str, list[dict]]] = []
        for operand in plan.calculation.operands:
            try:
                matches = self._collect_operand(tables, operand, plan.filters)
            except ValueError as exc:
                return CalculationResult(ok=False,error=str(exc))
            if not matches:
                return CalculationResult(ok=False, error=f"字段“{operand.field}”没有满足条件的数据")
            try:
                if len({item['table_id'] for item in matches}) > 1:
                    raise ValueError('同一操作数匹配多张表，需明确范围，拒绝跨表重复累计')
                value, unit = self._aggregate_operand(operand, matches)
            except ValueError as exc:
                return CalculationResult(ok=False, error=str(exc))
            operand_values.append((value, unit, matches))
            all_details.extend(matches)
        try:
            result, expression, unit = self._calculate(plan, operand_values)
        except (ValueError, ArithmeticError) as exc:
            return CalculationResult(ok=False, error=str(exc), details=all_details)
        evidence = self._evidence_from_details(all_details)
        return CalculationResult(True, result, unit, expression, all_details, evidence=evidence)

    def _candidate_tables(self, plan: QueryPlan, table_ids: list[int] | None) -> list[sqlite3.Row]:
        if table_ids is not None:
            if not table_ids:
                return []
            placeholders = ",".join("?" for _ in table_ids)
            rows = self.connection.execute(
                f"SELECT t.*, d.title, d.source_url, d.department, d.published_at "
                f"FROM tables t JOIN documents d ON d.id=t.doc_id WHERE t.id IN ({placeholders})",
                tuple(table_ids),
            ).fetchall()
            return [r for r in rows if r['id'] not in self.navigation_tables and matches_period(r['title'], r['published_at'], plan.time_range)]
        terms = [*plan.keywords, *plan.entities, *plan.fields]
        rows = self.connection.execute(
            "SELECT t.*, d.title, d.source_url, d.department, d.published_at "
            "FROM tables t JOIN documents d ON d.id=t.doc_id"
        ).fetchall()
        scored = []
        for row in rows:
            if row['id'] in self.navigation_tables or not matches_period(row['title'], row['published_at'], plan.time_range):
                continue
            haystack = f"{row['name']} {row['search_text']}".lower()
            score = sum(1 for term in terms if term and term.lower() in haystack)
            if score:
                scored.append((score, row))
        return [row for _, row in sorted(scored, key=lambda item: item[0], reverse=True)[:10]]

    def _collect_operand(self, tables: list[sqlite3.Row], operand: OperandSpec, common_filters: list[FilterSpec]) -> list[dict]:
        matches: list[dict] = []
        for table in tables:
            columns = self.connection.execute(
                "SELECT * FROM table_columns WHERE table_id=? ORDER BY column_index", (table["id"],)
            ).fetchall()
            target = _match_column(operand.field, columns)
            if target is None:
                continue
            rows = self.connection.execute(
                "SELECT * FROM table_rows WHERE table_id=? ORDER BY row_index", (table["id"],)
            ).fetchall()
            for row in rows:
                cells = self.connection.execute(
                    """SELECT tc.*, col.original_name, col.normalized_name
                       FROM table_cells tc JOIN table_columns col ON col.id=tc.column_id
                       WHERE tc.row_id=?""", (row["id"],)
                ).fetchall()
                if not self._row_matches(cells, [*common_filters, *operand.filters]):
                    continue
                cell = next((cell for cell in cells if cell["column_id"] == target["id"]), None)
                if cell is None or cell["numeric_value"] is None:
                    raise ValueError(f'符合条件的字段“{operand.field}”存在缺失或非数值单元格')
                matches.append({
                    "table_id": int(table["id"]), "table_name": table["name"],
                    "row_id": int(row["id"]), "row_index": int(row["row_index"]),
                    "row_label": row["row_label"], "field": target["original_name"],
                    "raw_value": cell["raw_value"], "numeric_value": cell["numeric_value"],
                    "unit": cell["unit"] or target["unit"] or table["unit"],
                    "doc_id": int(table["doc_id"]), "title": table["title"],
                    "source_url": table["source_url"], "department": table["department"],
                    "published_at": table["published_at"],
                })
        return matches

    @staticmethod
    def _row_matches(cells: list[sqlite3.Row], filters: list[FilterSpec]) -> bool:
        for item in filters:
            column = _match_column(item.field, cells)
            if column is None:
                return False
            cell = next((candidate for candidate in cells if candidate["column_id"] == column["column_id"]), None)
            if cell is None or not _compare(cell, item):
                return False
        return True

    @staticmethod
    def _aggregate_operand(operand: OperandSpec, matches: list[dict]) -> tuple[Decimal, str]:
        units = {item["unit"] for item in matches if item["unit"]}
        if len(units) > 1:
            raise ValueError(f"字段“{operand.field}”存在不一致单位：{sorted(units)}")
        values = [Decimal(item["numeric_value"]) for item in matches]
        if operand.aggregate == "count":
            return Decimal(len(matches)), "项"
        if operand.aggregate == "sum":
            return sum(values, Decimal(0)), next(iter(units), "")
        if operand.aggregate == "avg":
            return sum(values, Decimal(0)) / Decimal(len(values)), next(iter(units), "")
        if len(values) != 1:
            raise ValueError(f"字段“{operand.field}”匹配到 {len(values)} 个值，请增加筛选条件")
        return values[0], next(iter(units), "")

    @staticmethod
    def _calculate(plan: QueryPlan, operands: list[tuple[Decimal, str, list[dict]]]) -> tuple[Decimal, str, str]:
        operation = plan.calculation.operation
        values = [item[0] for item in operands]
        units = [item[1] for item in operands]
        if operation in {"sum", "count", "average"}:
            if operation == "count":
                value, unit = Decimal(sum(int(value) for value in values)), "项"
            elif operation == "average":
                value, unit = sum(values, Decimal(0)) / Decimal(len(values)), units[0]
            else:
                if len({unit for unit in units if unit}) > 1:
                    raise ValueError("求和字段单位不一致")
                value, unit = sum(values, Decimal(0)), units[0] if units else ""
            expression = f" {operation} ".join(format(value, "f") for value in values)
        elif operation == "ratio":
            if len(values) != 2 or values[1] == 0:
                raise ValueError("比例计算需要两个操作数，且分母不能为零")
            value, unit = values[0] / values[1], plan.calculation.output_unit or "%"
            if unit == "%":
                value *= Decimal(100)
            expression = f"{values[0]} / {values[1]}" + (" * 100" if unit == "%" else "")
        elif operation == "difference":
            if len(values) != 2:
                raise ValueError("差值计算需要两个操作数")
            if units[0] != units[1]:
                raise ValueError('差值计算的单位不一致')
            value, unit = values[0] - values[1], units[0]
            expression = f"{values[0]} - {values[1]}"
        elif operation == "multiply":
            if not values:
                raise ValueError("乘法计算缺少操作数")
            value = Decimal(1)
            for number in values:
                value *= number
            if plan.calculation.multiplier is not None:
                value *= plan.calculation.multiplier
                values.append(plan.calculation.multiplier)
            unit = plan.calculation.output_unit or next((unit for unit in units if unit), "")
            expression = " * ".join(format(number, "f") for number in values)
        else:
            raise ValueError(f"不支持的计算操作：{operation}")
        if plan.calculation.output_unit and unit and unit != plan.calculation.output_unit and operation != "ratio":
            value = convert_unit(value, unit, plan.calculation.output_unit)
            unit = plan.calculation.output_unit
        return value, expression, unit

    @staticmethod
    def _evidence_from_details(details: list[dict]) -> list[Evidence]:
        grouped: dict[tuple[int, int], dict] = {}
        for detail in details:
            key = (detail["doc_id"], detail["table_id"])
            grouped.setdefault(key, {**detail, "rows": []})["rows"].append(detail["row_id"])
        return [Evidence(
            evidence_id=f"table:{table_id}",
            text=f"{item['table_name']}，使用行：{sorted(set(item['rows']))}", score=1.0,
            doc_id=doc_id, title=item["title"], source_url=item["source_url"],
            department=item["department"], published_at=item["published_at"],
            table_id=table_id, row_ids=sorted(set(item["rows"])),
        ) for (doc_id, table_id), item in grouped.items()]


def _match_column(field: str, columns: list[sqlite3.Row]) -> sqlite3.Row | None:
    needle = "".join(field.lower().split())
    exact = [column for column in columns if needle in {column["normalized_name"], "".join(column["original_name"].lower().split())}]
    if len(exact) == 1:
        return exact[0]
    aliases = [column for column in columns if needle == column['normalized_name'].split('/')[-1]]
    if len(aliases) == 1:
        return aliases[0]
    partial = [column for column in columns if needle in column["normalized_name"] or column["normalized_name"] in needle]
    if len(partial) == 1:
        return partial[0]
    aliases = [column for column in columns if column['normalized_name'].split('/')[-1] in needle]
    return aliases[0] if len(aliases) == 1 else None


def _compare(cell: sqlite3.Row, item: FilterSpec) -> bool:
    raw = str(cell["normalized_value"]).lower()
    expected = str(item.value).strip().lower()
    if item.operator == "contains":
        return expected in raw
    if item.operator == "eq":
        return expected == raw
    if cell["numeric_value"] is None:
        return False
    try:
        left, right = Decimal(cell["numeric_value"]), Decimal(expected)
    except Exception:
        return False
    return {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[item.operator]

