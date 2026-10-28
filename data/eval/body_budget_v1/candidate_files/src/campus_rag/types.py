from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal


@dataclass
class SourceDocument:
    title: str
    source_url: str
    department: str = ""
    published_at: str | None = None
    crawled_at: str | None = None
    local_path: str = ""
    content_type: str = ""
    sha256: str = ""
    version: str = ""


@dataclass
class ParsedTable:
    name: str
    headers: list[str]
    rows: list[list[str]]
    locator: str = ""
    header_levels: int = 1
    unit: str = ""
    notes: str = ""


@dataclass
class ParsedDocument:
    source: SourceDocument
    sections: list[tuple[str, str]] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: int
    text: str
    heading_path: str
    token_count: int
    kind: Literal["text", "table"] = "text"
    table_id: int | None = None


@dataclass
class Evidence:
    evidence_id: str
    text: str
    score: float
    doc_id: int
    title: str
    source_url: str
    department: str = ""
    published_at: str | None = None
    chunk_id: str | None = None
    table_id: int | None = None
    row_ids: list[int] = field(default_factory=list)


@dataclass
class FilterSpec:
    field: str
    operator: Literal["contains", "eq", "gt", "gte", "lt", "lte"] = "contains"
    value: str = ""


@dataclass
class OperandSpec:
    field: str
    aggregate: Literal["value", "sum", "count", "avg"] = "value"
    filters: list[FilterSpec] = field(default_factory=list)


@dataclass
class CalculationSpec:
    operation: Literal["none", "sum", "count", "ratio", "difference", "multiply", "average"] = "none"
    operands: list[OperandSpec] = field(default_factory=list)
    multiplier: Decimal | None = None
    output_unit: str = ""


@dataclass
class QueryPlan:
    question_type: Literal["fact", "list", "aggregate", "compare", "ratio", "fee", "unanswerable"] = "fact"
    time_range: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    filters: list[FilterSpec] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    unit: str = ""
    needs_text: bool = True
    needs_table: bool = False
    calculation: CalculationSpec = field(default_factory=CalculationSpec)
    table_id: int | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    output: str = ""


@dataclass
class CalculationResult:
    ok: bool
    value: Decimal | None = None
    unit: str = ""
    expression: str = ""
    details: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    repairable: bool = False


@dataclass
class AnswerResult:
    question: str
    answer: str
    references: list[str]
    evidence: list[Evidence] = field(default_factory=list)
    calculation: CalculationResult | None = None
    numeric_verification: dict[str, Any] | None = None

