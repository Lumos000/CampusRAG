import json
from decimal import Decimal
from pathlib import Path

import pytest

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.database.query import PlanError,StructuredQueryEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell,question_operand_roles
from campus_rag.types import Evidence,ParsedTable,SourceDocument


@pytest.mark.parametrize('question,roles',[
    ('总计10学分，已完成4学分，计划修读2学分，还差多少学分？',['total','completed','planned']),
    ('已完成4学分，总计10学分，还剩6学分。',['completed','total','remaining']),
    ('总计10学分，4学分已完成，还需多少学分？',['total','completed']),
    ('总计10学分，已完成10学分，还差多少学分？',['total','completed']),
    ('计划完成4学分，还需多少学分？',['planned']),
    ('预计已完成4学分，还需多少学分？',['planned']),
])
def test_roles_preserve_each_quantity_span(question,roles):
    quantities=question_operand_roles(question,'学分')
    assert [q['role'] for q in quantities]==roles
    assert all(question[slice(*q['span'])]==q['quote'] for q in quantities)


def execute_literals(question,total,done):
    plan=parse_plan(json.dumps({'table_id':None,'steps':[
        {'id':'total','op':'lookup','literal':str(total),'unit':'学分'},
        {'id':'done','op':'lookup','literal':str(done),'unit':'学分'},
        {'id':'result','op':'subtract','inputs':['total','done']}]}))
    with Repository(':memory:') as repo:
        repo.initialize()
        return StructuredQueryEngine(repo.connection).execute_steps(plan,[],question)


@pytest.mark.parametrize('done,ok',[(4,True),(10,False),(2,False)])
def test_subtrahend_must_be_completed_quantity(done,ok):
    result=execute_literals('总计10学分，已完成4学分，计划修读2学分，还差多少学分？',10,done)
    assert result.ok==ok,result.error
    if not ok:assert result.repairable and '已完成量4学分' in result.error


@pytest.mark.parametrize('value',[0,10])
def test_zero_remaining_and_repeated_values_are_valid(value):
    result=execute_literals(f'总计{value}学分，已完成{value}学分，还差多少学分？',value,value)
    assert result.ok and result.value==0,result.error


def test_ambiguous_multiple_completed_quantities_are_not_guessed():
    result=execute_literals('总计10学分，甲已完成4学分，乙已完成2学分，还差多少学分？',10,4)
    assert not result.ok and '多个对象' in result.error


def test_period_feedback_identifies_fact_and_quantity():
    operands=[{'value':Decimal(40),'unit':'小时','per_unit':'月','expression':'40','details':[{'fact_id':'f2'}]},
              {'value':Decimal(4),'unit':'周','expression':'4'}]
    with pytest.raises(PlanError,match='f2为每月用量，但乘数为4周'):
        StructuredQueryEngine._step_arithmetic('multiply',operands)


def test_repair_receives_range_and_role_errors_and_rechecks_all_steps(tmp_path):
    with Repository(tmp_path/'db.sqlite') as repo:
        repo.initialize();did=repo.add_document(SourceDocument('课程规定','https://example.edu/courses'))
        table=ParsedTable('明细',['课程名称','学分'],[['甲','2'],['乙','3'],['合计','5']])
        tid=repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
        field=repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? AND original_name=?',(tid,'学分')).fetchone()[0]
        evidence=Evidence('table','课程总计5学分。',1,did,'课程规定','https://example.edu/courses',table_id=tid)
        class Index:
            def search(self,*args,**kwargs):return [evidence]
        class Generator:
            def __init__(self):self.prompts=[]
            def generate(self,prompt,**kwargs):
                self.prompts.append(prompt)
                return json.dumps({'table_id':tid,'steps':[
                    {'id':'total','op':'lookup' if len(self.prompts)==1 else 'sum','field_id':field},
                    {'id':'done','op':'lookup','literal':'5' if len(self.prompts)==1 else '2','unit':'学分'},
                    {'id':'result','op':'subtract','inputs':['total','done']}]})
        root=Path('.');generator=Generator()
        service=RAGService(Settings(root,root,root,root,root,numeric_shortcuts=False),Index(),generator,repo.connection)
        result=service.answer('按《课程规定》，总计5学分，已完成2学分，还差多少学分？')
        assert result.calculation and result.calculation.ok and result.calculation.value==3,result.answer
        assert len(generator.prompts)==2
        assert 'lookup匹配多个数值' in generator.prompts[1] and '已完成量2学分' in generator.prompts[1]
        assert '重新核对所有lookup' in generator.prompts[1]
