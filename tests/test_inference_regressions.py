import json
from decimal import Decimal
from pathlib import Path

import pytest

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.numeric import NumericEngine, render_calculation
from campus_rag.rag.planner import operation_hint, parse_plan, step_example
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence, ParsedTable, SourceDocument


def table_fixture(repo, unit='学分', missing=False, include_total=True):
    did = repo.add_document(SourceDocument('项目核算规则','https://example.edu/rules'))
    rows = [['项目甲','缺失' if missing else '3'],['项目乙','缺失' if missing else '7']]
    if include_total:
        rows.append(['合计','缺失' if missing else '10'])
    table = ParsedTable('项目明细',['项目名称',f'数量（{unit}）'],rows)
    tid = repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
    columns = {r['original_name']:r['id'] for r in repo.connection.execute(
        'SELECT id,original_name FROM table_columns WHERE table_id=?',(tid,))}
    return did,tid,columns


@pytest.mark.parametrize('op',['sum','average'])
def test_text_field_selection_is_repairable_but_missing_values_are_not(tmp_path,op):
    with Repository(tmp_path/'types.db') as repo:
        repo.initialize()
        _,tid,columns = table_fixture(repo,missing=True)
        engine = StructuredQueryEngine(repo.connection)
        for name,repairable in [('项目名称',True),('数量（学分）',False)]:
            plan = parse_plan(json.dumps({'table_id':tid,'steps':[
                {'id':'result','op':op,'field_id':columns[name]}]}))
            result = engine.execute_steps(plan,[tid],'数量合计多少学分？')
            assert not result.ok and result.repairable == repairable


def test_service_can_repair_a_text_field_without_changing_object_filters(tmp_path):
    with Repository(tmp_path/'repair.db') as repo:
        repo.initialize()
        did,tid,columns = table_fixture(repo)
        evidence = Evidence('table','项目明细见表。',1,did,'项目核算规则','https://example.edu/rules',table_id=tid)
        class Index:
            def search(self,*a,**kw): return [evidence]
        class Generator:
            def __init__(self): self.prompts=[]
            def generate(self,prompt,**kw):
                self.prompts.append(prompt)
                field = columns['项目名称'] if len(self.prompts)==1 else columns['数量（学分）']
                return json.dumps({'table_id':tid,'steps':[{'id':'total','op':'sum','field_id':field,
                    'filters':[{'field_id':columns['项目名称'],'op':'in','value':['项目甲','项目乙']}]}]})
        generator = Generator()
        root = Path('.')
        service = RAGService(Settings(root,root,root,root,root),Index(),generator,repo.connection)
        result = service.answer('项目甲和项目乙的数量合计多少学分？')
        assert result.calculation.ok and result.calculation.value==10
        assert len(generator.prompts)==2 and '文字字段不可求和' in generator.prompts[1]
        assert result.evidence and len(result.calculation.details)==2


@pytest.mark.parametrize('unit',['学分','学时','小时','元','台'])
@pytest.mark.parametrize('include_total',[False,True])
def test_table_remaining_example_uses_the_completed_quantity_and_full_details(tmp_path,unit,include_total):
    with Repository(tmp_path/(unit+'.db')) as repo:
        repo.initialize()
        _,tid,_ = table_fixture(repo,unit,include_total=include_total)
        engine = StructuredQueryEngine(repo.connection)
        question = f'总计10{unit}，已完成4{unit}，计划完成2{unit}，还差多少{unit}？'
        schema = engine.schema_context([tid],[],question)
        plan = parse_plan(step_example(operation_hint(question),json.dumps({'tables':schema}),question))
        result = engine.execute_steps(plan,[tid],question)
        assert result.ok and result.value==6 and result.unit==unit, result.error
        assert [s['op'] for s in plan.steps]==['lookup' if include_total else 'sum','lookup','subtract']
        if include_total:
            assert plan.steps[0]['scope']=='total'
        assert plan.steps[1]['literal']=='4'
        assert len({d['row_id'] for d in result.details})==(1 if include_total else 2)


def test_a_subset_remaining_example_does_not_use_the_whole_table_total(tmp_path):
    with Repository(tmp_path/'subset.db') as repo:
        repo.initialize()
        _,tid,_ = table_fixture(repo)
        engine = StructuredQueryEngine(repo.connection)
        question = '项目甲总计3学分，已完成1学分，还差多少学分？'
        schema = engine.schema_context([tid],[],question)
        plan = parse_plan(step_example('subtract',json.dumps({'tables':schema}),question))
        result = engine.execute_steps(plan,[tid],question)
        assert result.ok and result.value==2, result.error
        assert plan.steps[0]['op']=='sum' and plan.steps[0]['filters']


@pytest.mark.parametrize('text',['计划完成4学分，还差多少学分？',
    '甲已完成4学分，乙已完成2学分，还差多少学分？','已完成至少4学分，还差多少学分？'])
def test_remaining_example_does_not_choose_a_planned_or_ambiguous_quantity(tmp_path,text):
    with Repository(tmp_path/'ambiguous.db') as repo:
        repo.initialize()
        _,tid,_ = table_fixture(repo)
        schema = StructuredQueryEngine(repo.connection).schema_context([tid],[],text)
        plan = parse_plan(step_example('subtract',json.dumps({'tables':schema}),text))
        assert not any('literal' in step for step in plan.steps)


def test_report_names_come_from_actual_operands_and_keep_the_ratio_scale(tmp_path):
    with Repository(tmp_path/'report.db') as repo:
        repo.initialize()
        evidence = Evidence('source','原始人员23人，附加人员16人，加权人员总数为31，服务人数62.1人。',
            1,1,'核算规则','https://example.edu/rules')
        facts = NumericEngine.text_facts([evidence])
        steps = [{'id':name,'op':'lookup','fact_id':fid} for name,fid in zip(
            ['base','extra','weighted','people'],facts)]
        steps += [{'id':'delta','op':'subtract','inputs':['weighted','base']},
            {'id':'coefficient','op':'divide','inputs':['delta','extra']},
            {'id':'ratio','op':'divide','inputs':['people','weighted']},
            {'id':'report','op':'report','inputs':['coefficient','ratio']}]
        result = StructuredQueryEngine(repo.connection).execute_steps(
            parse_plan(json.dumps({'steps':steps})),[],'反推系数，再复算比值',facts)
        assert result.ok, result.error
        answer = render_calculation(result)
        assert '结果1' not in answer and '结果2' not in answer
        assert all(word in answer for word in ['原始人员','附加人员','加权人员','服务人数'])
        assert '约2:1' in answer and '0.5' in answer
        assert result.outputs[0]['value']==Decimal('0.5')
        assert result.outputs[1]['value']==Decimal('62.1')/31


@pytest.mark.parametrize('named_source',[False,True])
def test_question_constants_do_not_substitute_for_an_explicit_source(tmp_path,named_source):
    question = ('按《项目核算规则》，' if named_source else '')+'总计10学分，已完成4学分，还差多少学分？'
    plan = parse_plan(json.dumps({'steps':[{'id':'total','op':'lookup','literal':'10','unit':'学分'},
        {'id':'done','op':'lookup','literal':'4','unit':'学分'},
        {'id':'result','op':'subtract','inputs':['total','done']}]}))
    with Repository(tmp_path/'source.db') as repo:
        repo.initialize()
        result = StructuredQueryEngine(repo.connection).execute_steps(plan,[],question)
        assert result.ok == (not named_source)
        if named_source:
            assert result.repairable and '资料来源' in result.error
        else:
            assert result.value==6
