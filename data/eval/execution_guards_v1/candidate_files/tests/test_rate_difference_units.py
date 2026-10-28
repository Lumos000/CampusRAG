import json


import pytest


from campus_rag.config import Settings


from campus_rag.database import Repository


from campus_rag.database.query import StructuredQueryEngine


from campus_rag.rag.numeric import NumericEngine


from campus_rag.rag.planner import QueryPlanner,parse_plan,step_example,text_plan_context


from campus_rag.rag.service import RAGService


from campus_rag.types import Evidence


@pytest.fixture
def service(tmp_path):
    with Repository(tmp_path/'scope.db') as repo:
        repo.initialize()
        instance=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),None,None,repo.connection)
        instance.last_diagnostics={}
        yield instance


@pytest.mark.parametrize('question,ok',[
    ('技术组每人标准比研习组高多少元？',True),
    ('技术组每人奖励比研习组每人奖励高多少元？',True),
    ('技术组与研习组预算相差多少元？每人标准见原文。',False),
    ('技术组与研习组总额相差多少元？每人标准见原文。',False)])
def test_per_person_difference_is_not_a_monetary_total(service,question,ok):
    facts=NumericEngine.text_facts([Evidence('fee','技术组：4名，补助标准为137元/人；研习组：6名，补助标准为89元/人。',
        1,1,'补助规定','https://example.edu/fees')])
    ids={f['subject']:fid for fid,f in facts.items() if f['unit']=='元/人'}
    steps=[{'id':sid,'op':'lookup','fact_id':ids[name]} for sid,name in [('a','技术组'),('b','研习组')]]
    steps.append({'id':'result','op':'subtract','inputs':['a','b']})
    result=StructuredQueryEngine(service.structured.connection).execute_steps(parse_plan(json.dumps({'steps':steps})),[],question,facts)
    assert result.ok==ok
    if ok:assert result.value==48 and result.unit=='元/人'
