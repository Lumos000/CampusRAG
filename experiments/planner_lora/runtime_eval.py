"""Evaluate actual initial -> validation -> repair -> validation, not teacher repairs."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

from campus_rag.config import Settings
from campus_rag.rag.service import RAGService
from campus_rag.types import Evidence
from canonical_score import ordered_score


def summarize(rows):
    repaired=[r for r in rows if r['was_repaired']]
    return {'total':len(rows),'first_correct':sum(r['first_correct'] for r in rows),
        'final_correct':sum(r['correct'] for r in rows),'repair_attempted':len(repaired),
        'repair_succeeded':sum(r['correct'] and not r['first_correct'] for r in repaired),
        'repair_introduced_wrong':[r['id'] for r in repaired if not r['first_execution_ok'] and r['execution_ok'] and not r['correct']],
        'wrong_executable':[r['id'] for r in rows if r['execution_ok'] and not r['correct']]}


def evaluate_case(generator,case,connection,mode,root):
    evidence=[Evidence(**e) for e in case['evidence_map'].values()]
    class Index:
        def search(self,*args,**kwargs):return list(evidence)
    class Router:
        def generate(self,*args,**kwargs):
            with generator.model.disable_adapter():return generator.generate(*args,**kwargs)
        def planning_generate(self,*args,**kwargs):
            if mode=='base':
                with generator.model.disable_adapter():return generator.generate(*args,**kwargs)
            return generator.generate(*args,**kwargs)
    router=Router()
    service=RAGService(Settings(root,root,root,root,root,numeric_shortcuts=False),Index(),router,connection)
    service.planner.generator=SimpleNamespace(generate=router.planning_generate)
    answer=service.answer(case['question'])
    attempts=service.last_diagnostics.get('attempts',[])
    first=ordered_score(attempts[0].get('raw_plan',''),case,connection) if attempts else {'correct':False,'execution_ok':False}
    calculation=answer.calculation
    bound=json.dumps({'table_id':service.last_plan.table_id,'steps':service.last_plan.steps},ensure_ascii=False)
    final=ordered_score(bound,case,connection) if calculation and calculation.ok else {'correct':False,'execution_ok':False}
    if case['expected'] is None:
        final['correct']=not answer.references and not (calculation and calculation.ok)
    return {'id':case['id'],'family':case['family'],'correct':bool(final['correct']),
        'execution_ok':bool(calculation and calculation.ok),'first_correct':bool(first['correct']),
        'first_execution_ok':bool(first['execution_ok']),'was_repaired':len(attempts)>1,
        'answer':answer.answer,'raw':bound,'value':str(calculation.value) if calculation else None,
        'unit':calculation.unit if calculation else None,'error':calculation.error if calculation else answer.answer,
        'attempts':[{k:a.get(k) for k in ['kind','feedback','raw_plan','syntax_ok','execution_ok','error','failure_stage']} for a in attempts],
        'final_score':final,'first_score':first}
