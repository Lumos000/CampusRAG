import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.database.normalization import normalize_existing_tables
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence, ParsedTable, SourceDocument


@pytest.fixture
def table_db(tmp_path):
    with Repository(tmp_path/'steps.db') as repo:
        repo.initialize()
        doc = repo.add_document(SourceDocument('2026测试统计','https://example.edu/source',published_at='2026-01-01',sha256='steps'))
        table = ParsedTable('用房统计',['场馆','面积（平方米）'],[
            ['东馆','12'],['西馆','8'],['南馆','5'],['北馆','15'],['合 计','40']],locator='page-2-table-1')
        tid = repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
        columns = repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,)).fetchall()
        yield repo,tid,columns[0]['id'],columns[1]['id']


def plan(tid,steps,year='2026'):
    result = parse_plan(json.dumps({'table_id':tid,'steps':steps,'output':steps[-1]['id']}))
    result.time_range = [year] if year else []
    return result


def test_schema_samples_are_not_aggregate_domain(table_db):
    repo,tid,name,area = table_db
    engine = StructuredQueryEngine(repo.connection)
    schema = engine.schema_context([tid],['2026'])
    assert schema[0]['row_count']==5 and len(schema[0]['examples'])==3
    assert schema[0]['columns'][1]['unit']=='平方米'
    result = engine.execute_steps(plan(tid,[{'id':'a','op':'sum','field_id':area}]),[tid],'所有场馆合计多少？')
    assert result.ok and result.value==40
    assert len(result.details)==4
    assert all(d['cell_id'] and d['locator'] and d['raw_value'] for d in result.details)
    assert result.evidence[0].source_url=='https://example.edu/source'


def test_filter_sum_total_ratio_and_value_changes(table_db):
    repo,tid,name,area = table_db
    steps = [
        {'id':'r','op':'filter','filters':[{'field_id':name,'op':'in','value':['东馆','西馆']}]},
        {'id':'a','op':'sum','field_id':area,'rows':'r'},
        {'id':'b','op':'lookup','field_id':area,'scope':'total'},
        {'id':'c','op':'ratio','inputs':['a','b']}]
    engine = StructuredQueryEngine(repo.connection)
    result = engine.execute_steps(plan(tid,steps),[tid],'东馆和西馆面积之和占表内合计多少？')
    assert result.ok and result.value==50 and result.unit=='%'
    assert len(result.steps)==4 and result.steps[-1]['operands'][0]['value']=='20'
    repo.connection.execute("UPDATE table_cells SET raw_value='16' WHERE column_id=? AND raw_value='8'",(area,))
    normalize_existing_tables(repo.connection)
    changed = engine.execute_steps(plan(tid,steps),[tid],'东馆和西馆面积之和占表内合计多少？')
    assert changed.ok and changed.value==70


def test_named_subset_cannot_silently_aggregate_whole_table(table_db):
    repo,tid,name,area=table_db
    engine=StructuredQueryEngine(repo.connection)
    result=engine.execute_steps(plan(tid,[{'id':'a','op':'sum','field_id':area}]),[tid],'东馆和西馆面积合计多少？')
    assert not result.ok and result.repairable and '范围之外' in result.error
    steps=[{'id':'a','op':'sum','field_id':area,'filters':[{'field_id':name,'op':'in','value':['东馆','西馆']}]},
        {'id':'b','op':'sum','field_id':area},{'id':'c','op':'ratio','inputs':['a','b']}]
    result=engine.execute_steps(plan(tid,steps),[tid],'东馆和西馆面积占全部明细面积多少？')
    assert result.ok and result.value==50


@pytest.mark.parametrize('op,expected',[('add','20'),('subtract','4'),('multiply','96'),('divide','1.5'),('ratio','150'),('average','10')])
def test_arithmetic_operators(table_db,op,expected):
    repo,tid,name,area = table_db
    # Multiplying two physical areas is deliberately unsupported by the unit checker.
    steps = [{'id':'a','op':'lookup','field_id':area,'filters':[{'field_id':name,'op':'eq','value':'东馆'}]},
             {'id':'b','op':'lookup','field_id':area,'filters':[{'field_id':name,'op':'eq','value':'西馆'}]},
             {'id':'c','op':op,'inputs':['a','b']}]
    result = StructuredQueryEngine(repo.connection).execute_steps(plan(tid,steps),[tid],'东馆和西馆比较计算多少？')
    if op=='multiply':
        assert not result.ok and '单位' in result.error
    else:
        assert result.ok and result.value==Decimal(expected)


def test_count_threshold_and_missing_values(table_db):
    repo,tid,name,area = table_db
    steps = [{'id':'a','op':'count','filters':[{'field_id':area,'op':'gte','value':'8'}]}]
    engine = StructuredQueryEngine(repo.connection)
    result = engine.execute_steps(plan(tid,steps),[tid],'面积大于等于8平方米的场馆有几个？')
    assert result.ok and result.value==3
    repo.connection.execute('UPDATE table_cells SET numeric_value=NULL WHERE column_id=? AND raw_value=?',(area,'5'))
    result = engine.execute_steps(plan(tid,steps),[tid],'面积大于等于8平方米的场馆有几个？')
    assert not result.ok and not result.repairable and '缺失' in result.error


@pytest.mark.parametrize('failure',['field','table','year','unknown_object','omitted_object','duplicate','missing','unit','count','zero'])
def test_fail_closed_validations(table_db,failure):
    repo,tid,name,area = table_db
    steps = [{'id':'a','op':'sum','field_id':area,'filters':[{'field_id':name,'op':'in','value':['东馆','西馆']}]}]
    p = plan(tid,steps)
    allowed = [tid]
    if failure=='field': steps[0]['field_id']=99999
    if failure=='table': allowed=[]
    if failure=='year': p.time_range=['2030']
    if failure=='unknown_object': steps[0]['filters'][0]['value']=['不存在的馆']
    if failure=='omitted_object': steps[0]['filters'][0]['value']=['东馆']
    if failure=='duplicate': repo.connection.execute("UPDATE table_cells SET normalized_value='东馆' WHERE column_id=? AND raw_value='南馆'",(name,))
    if failure=='missing': repo.connection.execute("UPDATE table_cells SET numeric_value=NULL WHERE column_id=? AND raw_value='8'",(area,))
    if failure=='unit': repo.connection.execute("UPDATE table_cells SET unit='元' WHERE column_id=? AND raw_value='8'",(area,))
    if failure=='count': steps[0]['expected_count']=3
    if failure=='zero':
        repo.connection.execute("UPDATE table_cells SET numeric_value='0' WHERE column_id=? AND raw_value='40'",(area,))
        steps.extend([{'id':'b','op':'lookup','field_id':area,'scope':'total'}, {'id':'c','op':'ratio','inputs':['a','b']}])
        p.output='c'
    p.steps=steps
    result = StructuredQueryEngine(repo.connection).execute_steps(p,allowed,'东馆和西馆面积之和占合计多少？')
    assert not result.ok, failure


def test_offline_units_years_and_original_values(tmp_path):
    with Repository(tmp_path/'units.db') as repo:
        repo.initialize()
        doc = repo.add_document(SourceDocument('2026预算统计','https://example.edu/b',published_at='2026-01-01'))
        table = ParsedTable('预算',['年度','部门','金额（万元）'],[['2025','甲部','1'],['2026','甲部','2'],['2026','乙部','30000元'],['2026','合计','5']])
        tid = repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
        cols = repo.connection.execute('SELECT * FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,)).fetchall()
        result = StructuredQueryEngine(repo.connection).execute_steps(plan(tid,[{'id':'a','op':'sum','field_id':cols[2]['id']}]),[tid],'2026年度预算合计多少？')
        assert result.ok and result.value==50000 and result.unit=='元'
        assert {d['raw_value'] for d in result.details}=={'2','30000元'}
        assert cols[0]['data_type']=='year'


def test_verified_text_facts_and_no_invented_numbers(table_db):
    repo,tid,_,_ = table_db
    source = Evidence('text-1','第40页。甲项目预算为120元；乙项目预算为80元。',1,1,'2026项目预算','https://example.edu/t')
    facts = NumericEngine.text_facts([source])
    assert len(facts)==2 and all(f['value']!='40' for f in facts.values())
    p = parse_plan(json.dumps({'steps':[{'id':'a','op':'lookup','fact_id':'f1'},
        {'id':'b','op':'lookup','fact_id':'f2'}, {'id':'c','op':'subtract','inputs':['a','b']}],'output':'c'}))
    result = StructuredQueryEngine(repo.connection).execute_steps(p,[],'甲项目和乙项目预算相差多少？',facts)
    assert result.ok and result.value==40 and result.evidence==[source]
    p.steps[-1]['inputs']=['a',{'literal':'999','unit':'元'}]
    assert not StructuredQueryEngine(repo.connection).execute_steps(p,[],'预算相差多少？',facts).ok


def test_service_retrieves_first_repairs_once_and_skips_final_llm(table_db):
    repo,tid,name,area = table_db
    calls=[]
    evidence = Evidence('table-chunk','表格',1,1,'2026测试统计','https://example.edu/source',table_id=tid)
    class Index:
        def search(self,q,*args,**kwargs):
            calls.append('search')
            assert q=='东馆面积合计多少？'
            return [evidence]
    class Generator:
        def generate(self,prompt,**kwargs):
            calls.append('generate')
            assert 'row_count' in prompt and 'field_id' in prompt
            return json.dumps({'table_id':tid,'steps':[{'id':'a','op':'sum','field_id':999 if len(calls)==2 else area,
                'filters':[{'field_id':name,'op':'eq','value':'东馆'}]}],'output':'a'})
    dummy = Path('.')
    settings = Settings(dummy,dummy,dummy,dummy,dummy,numeric_shortcuts=False)
    result = RAGService(settings,Index(),Generator(),repo.connection).answer('东馆面积合计多少？')
    assert result.calculation.ok and result.calculation.value==12
    assert calls==['search','generate','generate']
    assert 'field_id' in result.numeric_verification['steps'][0]['details'][0]


def test_direct_facts_use_only_one_generation(table_db):
    repo,_,_,_ = table_db
    calls=[]
    class Index:
        def search(self,*a,**k):
            calls.append('search')
            return [Evidence('text','凭校园卡入馆。',1,1,'入馆规定','https://example.edu/rule')]
    class Generator:
        def generate(self,*a,**k):
            calls.append('answer')
            return '凭校园卡入馆。'
    p = Path('.')
    result = RAGService(Settings(p,p,p,p,p),Index(),Generator(),repo.connection).answer('如何办理入馆？')
    assert result.calculation is None and calls==['search','answer']


def test_plan_repair_budget_is_shared_between_parse_and_execution(table_db):
    repo,tid,_,area = table_db
    calls=[]
    class Index:
        def search(self,*a,**k):
            return [Evidence('t','表',1,1,'2026测试统计','url',table_id=tid)]
    class Generator:
        def generate(self,*a,**k):
            calls.append(1)
            return 'invalid JSON' if len(calls)==1 else json.dumps({'table_id':tid,'steps':[{'id':'a','op':'sum','field_id':999}],'output':'a'})
    p=Path('.')
    result=RAGService(Settings(p,p,p,p,p,numeric_shortcuts=False),Index(),Generator(),repo.connection).answer('面积合计多少？')
    assert len(calls)==2 and not result.calculation.ok and not result.references


def test_missing_data_does_not_trigger_repair(table_db):
    repo,tid,name,area=table_db
    repo.connection.execute("UPDATE table_cells SET numeric_value=NULL WHERE column_id=? AND raw_value='12'",(area,))
    calls=[]
    class Index:
        def search(self,*a,**k):
            return [Evidence('t','表',1,1,'2026测试统计','url',table_id=tid)]
    class Generator:
        def generate(self,*a,**k):
            calls.append(1)
            return json.dumps({'table_id':tid,'steps':[{'id':'a','op':'sum','field_id':area}],'output':'a'})
    p=Path('.')
    result=RAGService(Settings(p,p,p,p,p),Index(),Generator(),repo.connection).answer('面积合计多少？')
    assert len(calls)==1 and not result.references and not result.calculation.ok


def test_competing_same_schema_tables_refuse(table_db):
    repo,tid,name,area=table_db
    doc=repo.add_document(SourceDocument('另一份2026统计','https://example.edu/other',sha256='other'))
    table=ParsedTable('另一用房表',['场馆','面积（平方米）'],[['东馆','100'],['西馆','80']])
    other=repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
    result=StructuredQueryEngine(repo.connection).execute_steps(plan(tid,[{'id':'a','op':'sum','field_id':area}]),[tid,other],'所有面积合计多少？')
    assert not result.ok and not result.repairable and '范围' in result.error


def test_normalization_is_idempotent_and_rejects_ranges(table_db):
    repo,tid,name,area=table_db
    repo.connection.execute("UPDATE table_cells SET raw_value='约12平方米' WHERE column_id=? AND raw_value='12'",(area,))
    normalize_existing_tables(repo.connection)
    first=[tuple(r) for r in repo.connection.execute('SELECT * FROM table_cells ORDER BY id')]
    normalize_existing_tables(repo.connection)
    assert first==[tuple(r) for r in repo.connection.execute('SELECT * FROM table_cells ORDER BY id')]
    assert repo.connection.execute("SELECT numeric_value FROM table_cells WHERE raw_value='约12平方米'").fetchone()[0] is None


def test_question_constants_are_grounded_and_audited(table_db):
    repo,tid,name,area=table_db
    source=Evidence('t','单次租金为30元。',1,1,'2026租金说明','https://example.edu/rent')
    facts=NumericEngine.text_facts([source])
    p=parse_plan(json.dumps({'steps':[{'id':'a','op':'lookup','fact_id':'f1'},
        {'id':'b','op':'multiply','inputs':['a',{'literal':'3','unit':'次'}]}],'output':'b'}))
    result=StructuredQueryEngine(repo.connection).execute_steps(p,[],'三次租金合计多少？',facts)
    assert result.ok and result.value==90 and result.steps[-1]['operands'][1]['value']=='3'
    p.steps[-1]['inputs'][1]['unit']='年'
    assert not StructuredQueryEngine(repo.connection).execute_steps(p,[],'三次租金合计多少？',facts).ok


def test_output_dependency_path_and_total_ambiguity(table_db):
    repo,tid,name,area=table_db
    p=parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'a','op':'sum','field_id':area},
        {'id':'b','op':'lookup','field_id':name}],'output':'a'}))
    assert len(p.steps)==1
    result=StructuredQueryEngine(repo.connection).execute_steps(p,[tid],'所有明细面积合计多少？')
    assert result.ok and result.value==40
    repo.connection.execute("UPDATE table_cells SET normalized_value='小计' WHERE column_id=? AND raw_value='北馆'",(name,))
    repo.connection.execute("UPDATE table_cells SET raw_value='小计' WHERE column_id=? AND raw_value='北馆'",(name,))
    normalize_existing_tables(repo.connection)
    p=plan(tid,[{'id':'a','op':'sum','field_id':area,'scope':'total'}])
    result=StructuredQueryEngine(repo.connection).execute_steps(p,[tid],'表内合计面积多少？')
    assert not result.ok and '小计' in result.error


def test_model_answer_number_is_never_used_as_operand(table_db):
    repo,tid,name,area=table_db
    p=parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'a','op':'sum','field_id':area,'output':'a'}],'output':'999999'}))
    result=StructuredQueryEngine(repo.connection).execute_steps(p,[tid],'所有场馆面积合计多少？')
    assert result.ok and result.value==40 and '999999' not in result.expression


def test_identical_tables_are_deduplicated_only_in_prompt(table_db):
    repo,tid,_,_=table_db
    doc=repo.add_document(SourceDocument('资料附件','https://example.edu/alias',published_at='2026-01-01',sha256='alias'))
    table=ParsedTable('用房统计',['场馆','面积（平方米）'],[
        ['东馆','12'],['西馆','8'],['南馆','5'],['北馆','15'],['合 计','40']])
    other=repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
    schema=StructuredQueryEngine(repo.connection).schema_context([other,tid],[],'面积合计多少？')
    assert len(schema)==1 and schema[0]['table_id']==tid
    assert repo.connection.execute('SELECT count(*) FROM tables').fetchone()[0]==2


def test_merged_html_and_multilevel_header_normalize_offline(tmp_path):
    from campus_rag.ingest.parsers import parse_file
    path=tmp_path/'table.html'
    path.write_text('<article><table><tr><th rowspan="2">部门</th><th colspan="2">预算（元）</th></tr><tr><th>收入</th><th>支出</th></tr><tr><td rowspan="2">甲部</td><td>20</td><td>10</td></tr><tr><td>30</td><td>15</td></tr></table></article>',encoding='utf-8')
    parsed=parse_file(path)
    assert parsed.tables[0].rows==[['甲部','20','10'],['甲部','30','15']]
    with Repository(tmp_path/'merged.db') as repo:
        repo.initialize()
        doc=repo.add_document(parsed.source)
        table=parsed.tables[0]
        tid=repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
        cols=repo.connection.execute('SELECT unit FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,)).fetchall()
        assert [c['unit'] for c in cols]==['','元','元']


@pytest.mark.parametrize('steps',[
    [{'id':'a','op':'exec','inputs':[]}],
    [{'id':'a','op':'add','inputs':['a','b']}],
    [{'id':'a','op':'lookup','field_id':True}],
    [{'id':'a','op':'lookup','field_id':1,'sql':'DROP TABLE documents'}],
    [{'id':'a','op':'filter','filters':[{'field_id':1,'op':'in','value':'东馆'}]}],
])
def test_reject_invalid_or_executable_plan(steps):
    with pytest.raises(ValueError):
        parse_plan(json.dumps({'table_id':1,'steps':steps,'output':'a'}))
