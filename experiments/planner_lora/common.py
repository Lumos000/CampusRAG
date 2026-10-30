"""Reference scoring for the numeric planner adapter."""
import json
from decimal import Decimal
from pathlib import Path

from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.tools.temporal import question_periods
from campus_rag.types import Evidence


def read_cases(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def original_score(raw,case,connection):
    record={'syntax_ok':False,'execution_ok':False,'correct':False,'raw':raw}
    try:
        plan=parse_plan(raw,require_final=True);plan.time_range=question_periods(case['question'])
        record['syntax_ok']=True
        facts={f['fact_id']:{**f,'evidence':Evidence(**case['evidence_map'][f['evidence_id']])} for f in case['context']['facts']}
        result=StructuredQueryEngine(connection).execute_steps(plan,case['table_ids'],case['question'],facts)
        sources={d['doc_id'] for d in result.details}
        fields={d['field_id'] for d in result.details if d.get('field_id') is not None}
        expected_fields=set(case['reference']['measure_field_ids'])
        correct=(not plan.steps) if case['expected'] is None else bool(result.ok and result.evidence
            and result.unit==case['unit'] and abs(result.value-Decimal(case['expected']))<Decimal('0.000001')
            and sources==set(case['source_doc_ids']) and (not expected_fields or fields<=expected_fields))
        record.update(execution_ok=result.ok,correct=correct,error=result.error,value=str(result.value),
            unit=result.unit,expression=result.expression,source_ids=sorted(sources),field_ids=sorted(fields))
    except (ValueError,KeyError,TypeError) as exc:record['error']=str(exc)
    return record
