import json
from decimal import Decimal
from pathlib import Path

import pytest

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.database.query import PlanError, StructuredQueryEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence, ParsedTable, SourceDocument


@pytest.fixture
def table_repo(tmp_path):
    with Repository(tmp_path / 'fixture.db') as repo:
        repo.initialize()
        yield repo


def add_table(repo, marker='共计', missing=False):
    did=repo.add_document(SourceDocument('课程核算办法','https://example.edu/courses'))
    table=ParsedTable('课程明细',['课程名称','学分'],[['课程甲','2'],['课程乙','缺失' if missing else '3'],[marker,'5']])
    tid=repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
    field=repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? AND original_name=?',(tid,'学分')).fetchone()[0]
    return did,tid,field


@pytest.mark.parametrize('marker',['共计','合计','总计','小计','总额','总数','总面积'])
def test_prompt_and_executor_share_normalized_total_rows(table_repo,marker):
    _,tid,field=add_table(table_repo,marker)
    engine=StructuredQueryEngine(table_repo.connection)
    schema=engine.schema_context([tid],[], '学分合计多少学分？')
    assert schema[0]['total_row_count']==1
    for op,extra in [('sum',{}),('lookup',{'scope':'total'})]:
        plan=parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'result','op':op,'field_id':field,**extra}]}))
        result=engine.execute_steps(plan,[tid],'学分合计多少学分？')
        assert result.ok and result.value==Decimal(5),result.error


def test_multiple_lookup_is_repairable_without_guessing_operation(table_repo):
    _,tid,field=add_table(table_repo)
    plan=parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'total','op':'lookup','field_id':field}]}))
    result=StructuredQueryEngine(table_repo.connection).execute_steps(plan,[tid],'学分合计多少学分？')
    assert not result.ok and result.repairable
    assert 'lookup匹配多个数值' in result.error


def test_service_repairs_multiple_lookup_once(table_repo):
    did,tid,field=add_table(table_repo)
    evidence=Evidence('courses','课程安排见表。',1.,did,'课程核算办法','https://example.edu/courses',table_id=tid)
    class Index:
        def search(self,*args,**kwargs): return [evidence]
    class Generator:
        def __init__(self): self.prompts=[]
        def generate(self,prompt,**kwargs):
            self.prompts.append(prompt)
            return json.dumps({'table_id':tid,'steps':[{'id':'total','op':'lookup' if len(self.prompts)==1 else 'sum','field_id':field}]})
    root=Path('.'); generator=Generator()
    service=RAGService(Settings(root,root,root,root,root,numeric_shortcuts=False),Index(),generator,table_repo.connection)
    answer=service.answer('按《课程核算办法》，学分合计多少学分？')
    assert answer.calculation and answer.calculation.ok and answer.calculation.value==5
    assert len(generator.prompts)==2 and 'lookup匹配多个数值' in generator.prompts[1]


def test_missing_measurements_remain_nonrepairable(table_repo):
    _,tid,field=add_table(table_repo,missing=True)
    plan=parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'result','op':'sum','field_id':field}]}))
    result=StructuredQueryEngine(table_repo.connection).execute_steps(plan,[tid],'学分合计多少学分？')
    assert not result.ok and not result.repairable
    assert '缺失' in result.error


def test_known_incompatible_operands_are_a_plan_error():
    operands=[{'value':Decimal(3),'unit':'人'}, {'value':Decimal(10),'unit':'元/人'}]
    with pytest.raises(PlanError,match='操作数单位不一致'):
        StructuredQueryEngine._step_arithmetic('add',operands)


def test_wrong_period_fact_is_repairable_without_implicit_conversion():
    operands=[{'value':Decimal(40),'unit':'小时','per_unit':'月','expression':'40'},
              {'value':Decimal(4),'unit':'周','expression':'4'}]
    with pytest.raises(PlanError,match='周期单位不一致'):
        StructuredQueryEngine._step_arithmetic('multiply',operands)
