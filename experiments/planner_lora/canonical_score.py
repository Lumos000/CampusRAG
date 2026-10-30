"""Accept verified full-detail/total equivalence, preserving operand provenance."""
from collections import Counter
from decimal import Decimal
import json
import os
from pathlib import Path
import sys

ROOT=Path(os.environ.get('CAMPUSRAG_EXPERIMENT_ROOT',Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import original_score
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.tools.temporal import question_periods
from campus_rag.types import Evidence


def sources(engine,details,periods):
    result=set();tables={}
    for detail in details:
        if 'fact_id' in detail:
            result.add(('fact',detail['fact_id']));continue
        tid,rid,fid=detail['table_id'],detail['row_id'],detail.get('field_id')
        if tid not in tables:tables[tid]=engine._step_table(tid,periods)[2]
        rows=tables[tid]
        row=next(r for r in rows if r['id']==rid)
        totals=[r for r in rows if r['is_total']]
        domain=[r for r in rows if not r['is_total']]
        if fid is not None and row['is_total'] and len(totals)==1 and domain:
            cells=[r['cells'][fid] for r in domain]
            total=row['cells'][fid]
            if total['numeric_value'] is not None and all(c['numeric_value'] is not None and c['unit']==total['unit'] for c in cells):
                if sum(Decimal(c['numeric_value']) for c in cells)==Decimal(total['numeric_value']):
                    result.update(('cell',tid,r['id'],fid) for r in domain);continue
        result.add(('cell',tid,rid,fid))
    return frozenset(result)


def ordered_score(raw,case,connection):
    record=original_score(raw,case,connection)
    if not record['correct'] or case['expected'] is None:return record
    facts={f['fact_id']:{**f,'evidence':Evidence(**case['evidence_map'][f['evidence_id']])} for f in case['context']['facts']}
    actual=parse_plan(raw,require_final=True); reference=parse_plan(json.dumps(case['plan'],ensure_ascii=False),require_final=True)
    periods=question_periods(case['question']);actual.time_range=reference.time_range=periods
    engine=StructuredQueryEngine(connection)
    actual_result=engine.execute_steps(actual,case['table_ids'],case['question'],facts)
    reference_result=engine.execute_steps(reference,case['table_ids'],case['question'],facts)
    assert reference_result.ok
    if sources(engine,actual_result.details,periods)!=sources(engine,reference_result.details,periods):
        record.update(correct=False,provenance_ok=False);return record
    nodes=lambda p:{s['id']:s for s in p.steps}
    target=nodes(reference)[reference.output];final=nodes(actual)[actual.output]
    if target['op'] not in {'subtract','ratio'}:return record
    if final['op']!=target['op']:
        record.update(correct=False,ordered_operands_ok=False);return record
    def sides(plan,execution,output):
        by_id=nodes(plan);trace={s['id']:s for s in execution.steps}
        def literals(item):
            node=by_id[item] if isinstance(item,str) else item
            if 'literal' in node:return Counter({(str(Decimal(node['literal']).normalize()),'人' if node['unit'] in {'名','位'} else node['unit']):1})
            counts=Counter()
            for child in node.get('inputs',[]):counts.update(literals(child))
            return counts
        return [(sources(engine,trace[item]['details'],periods),literals(item),Decimal(trace[item]['value']),trace[item]['unit'])
            if isinstance(item,str) else (frozenset(),literals(item),Decimal(item['literal']),item['unit']) for item in output['inputs']]
    record.update(correct=sides(actual,actual_result,final)==sides(reference,reference_result,target),ordered_operands_ok=True)
    record['ordered_operands_ok']=record['correct']
    return record
