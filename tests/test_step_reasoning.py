import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine, _field_role, _object_mentioned
from campus_rag.database.normalization import normalize_existing_tables
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.rag.planner import operation_hint, step_example, repair_json_delimiter
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


def test_schema_roles_and_examples_bind_real_question_objects(table_db):
    repo,_,_,_ = table_db
    doc = repo.add_document(SourceDocument('2028产品对照表','https://example.edu/products',sha256='products'))
    table = ParsedTable('产品对照',['序号','部门','产品名称','件数（个）'],[
        ['1','管理部门','探测器','4'],['2','管理部门','探测器','7'],['3','生产部门','连接器','9']])
    tid = repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
    q = '产品对照表中产品名称为探测器的对应记录合计多少条？'
    engine = StructuredQueryEngine(repo.connection)
    schema = engine.schema_context([tid],[],q)
    cols = schema[0]['columns']
    assert [c['role'] for c in cols]==['identifier','category','category','measure']
    assert schema[0]['question_matches']==[{'field_id':cols[2]['field_id'],'values':['探测器']}]
    raw = step_example(operation_hint(q),json.dumps({'tables':schema}),q)
    example = parse_plan(raw)
    assert example.steps[0]['op']=='count'
    assert example.steps[0]['filters']==[{'field_id':cols[2]['field_id'],'op':'eq','value':'探测器'}]
    result = engine.execute_steps(example,[tid],q)
    assert result.ok and result.value==2
    invalid = plan(tid,[{'id':'a','op':'sum','field_id':cols[0]['field_id']}],year='')
    result = engine.execute_steps(invalid,[tid],'所有序号之和多少？')
    assert not result.ok and result.repairable and '标识' in result.error


def test_examples_do_not_invent_tables_fields_or_thresholds(table_db):
    repo,tid,_,_ = table_db
    schema = StructuredQueryEngine(repo.connection).schema_context([tid],[],'场馆共有几个？')
    raw = json.loads(step_example('count',json.dumps({'tables':schema}),'场馆共有几个？'))
    assert raw['table_id']==tid and raw['steps']==[{'id':'a','op':'count'}]
    context = json.dumps({'facts':[{'fact_id':'f8','unit':'元'},{'fact_id':'f9','unit':'元'}]})
    raw = json.loads(step_example('subtract',context,'两个预算相差多少元？'))
    assert raw['table_id'] is None and [s.get('fact_id') for s in raw['steps'][:2]]==['f8','f9']
    assert all('field_id' not in s for s in raw['steps'])
    assert operation_hint('一般申请人与运动员的平均学分绩点门槛相差多少？')=='subtract'
    assert operation_hint('所有场馆平均面积多少？')=='average'


def test_schema_dedup_uses_full_contents_and_business_scope(table_db):
    repo,tid,_,_ = table_db
    for name,last in [('另一份用房统计','16'),('用房安排','15')]:
        doc = repo.add_document(SourceDocument(name,'https://example.edu/'+last,sha256=name))
        table = ParsedTable(name,['场馆','面积（平方米）'],[
            ['东馆','12'],['西馆','8'],['南馆','5'],['北馆',last],['合 计','40']])
        repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
    ids = [r['id'] for r in repo.connection.execute('SELECT id FROM tables')]
    schema = StructuredQueryEngine(repo.connection).schema_context(ids,[],'面积多少？')
    assert len(schema)==3


def test_duration_measure_is_distinct_from_calendar_time():
    assert _field_role({'original_name':'工作时间（小时）','data_type':'number','unit':'小时'})=='measure'
    assert _field_role({'original_name':'办理时间','data_type':'text','unit':''})=='time'
    assert _field_role({'original_name':'年度','data_type':'year','unit':''})=='time'
    assert _field_role({'original_name':'年度预算（元）','data_type':'number','unit':'元'})=='measure'


@pytest.mark.parametrize('quantity',['9小时','9 小时'])
def test_question_constant_lookup_uses_verified_question_span(table_db,quantity):
    repo,_,_,_ = table_db
    source = Evidence('wage','岗位酬金不低于每小时17元。',1,1,'计酬规则','https://example.edu/wage')
    facts = NumericEngine.text_facts([source])
    p = parse_plan(json.dumps({'steps':[{'id':'a','op':'lookup','fact_id':'f1'},
        {'id':'b','op':'lookup','literal':'9','unit':'小时'},
        {'id':'c','op':'multiply','inputs':['a','b']}]}))
    q = f'按最低标准计算{quantity}酬金合计多少元？'
    result = StructuredQueryEngine(repo.connection).execute_steps(p,[],q,facts)
    assert result.ok and result.value==153 and result.unit=='元'
    assert result.steps[1]['source']['kind']=='question'
    a,b = result.steps[1]['source']['span']
    assert q[a:b]==result.steps[1]['source']['quote']==quantity
    p.steps[1]['literal']='99'
    assert not StructuredQueryEngine(repo.connection).execute_steps(p,[],q,facts).ok


def test_comparison_cannot_succeed_with_lookup_or_wrong_direction(table_db):
    repo,_,_,_ = table_db
    source = Evidence('budget','甲项目预算为90元；乙项目预算为150元。',1,1,'预算','https://example.edu/b')
    facts = NumericEngine.text_facts([source]); engine=StructuredQueryEngine(repo.connection)
    steps = [{'id':'a','op':'lookup','fact_id':'f1'},{'id':'b','op':'lookup','fact_id':'f2'}]
    p = parse_plan(json.dumps({'steps':steps}))
    result = engine.execute_steps(p,[],'乙项目比甲项目高多少元？',facts)
    assert not result.ok and result.repairable and 'subtract' in result.error
    p = parse_plan(json.dumps({'steps':steps+[{'id':'c','op':'subtract','inputs':['a','b']}]}))
    result = engine.execute_steps(p,[],'乙项目比甲项目高多少元？',facts)
    assert not result.ok and result.repairable and '方向' in result.error


def test_period_multiplier_cannot_be_omitted(table_db):
    repo,_,_,_ = table_db
    source=Evidence('period','学生每周不超过6小时，每月不超过30小时。',1,1,'工时','https://example.edu/h')
    facts=NumericEngine.text_facts([source])
    p=parse_plan(json.dumps({'steps':[{'id':'a','op':'lookup','fact_id':'f2'},
        {'id':'b','op':'lookup','fact_id':'f1'},{'id':'c','op':'subtract','inputs':['a','b']}]}))
    result=StructuredQueryEngine(repo.connection).execute_steps(p,[],
        '连续三周每周工作6小时，与每月上限相比还剩多少小时？',facts)
    assert not result.ok and result.repairable and '周期数' in result.error


def test_explicit_missing_personal_inputs_do_not_use_public_examples(table_db):
    repo,_,_,_ = table_db
    calls=[]
    class Index:
        def search(self,*args,**kwargs):
            calls.append('search')
            return [Evidence('example','某合同还款期限为16年。',1,1,'还款示例','https://example.edu/example')]
    class Generator:
        def generate(self,*args,**kwargs):
            raise AssertionError('Explicitly missing personal inputs must not be planned')
    p=Path('.')
    service=RAGService(Settings(p,p,p,p,p),Index(),Generator(),repo.connection)
    result=service.answer('请计算我的应付利息；我没有提供借款金额、适用利率和计息期限。')
    assert calls==['search'] and not result.references and result.numeric_verification['verified'] is False
    assert '借款金额、适用利率和计息期限' in result.answer
    assert service.last_diagnostics['route']=='missing_question_inputs'


def test_public_standard_calculation_does_not_require_personal_identity(table_db):
    repo,_,_,_ = table_db
    class Index:
        def search(self,*args,**kwargs):
            return [Evidence('rule','岗位酬金不低于每小时17元。',1,1,'计酬规则','https://example.edu/rule')]
    class Generator:
        def generate(self,*args,**kwargs):
            return json.dumps({'steps':[{'id':'a','op':'lookup','fact_id':'f1'},
                {'id':'b','op':'lookup','literal':'9','unit':'小时'},
                {'id':'c','op':'multiply','inputs':['a','b']}]})
    p=Path('.')
    result=RAGService(Settings(p,p,p,p,p),Index(),Generator(),repo.connection).answer(
        '我没有提供个人身份，但只按公开最低标准，计算9小时酬金合计多少元？')
    assert result.calculation.ok and result.calculation.value==153 and result.references


def test_self_reference_feedback_identifies_legal_predecessors():
    raw={'steps':[{'id':'price','op':'lookup','fact_id':'f1'},
        {'id':'result','op':'multiply','inputs':['price','result']}]}
    with pytest.raises(ValueError,match='合法前序id：price'):
        parse_plan(json.dumps(raw))


def test_model_output_must_not_discard_final_arithmetic():
    raw=json.dumps({'steps':[{'id':'left','op':'lookup','fact_id':'f1'},
        {'id':'right','op':'lookup','fact_id':'f2'},
        {'id':'result','op':'multiply','inputs':['left','right']}],'output':'left'})
    model_plan = parse_plan(raw,require_final=True)
    assert model_plan.output=='result' and len(model_plan.steps)==3
    assert len(parse_plan(raw).steps)==1  # Existing explicit API selection remains compatible.


def test_missing_table_binds_only_unique_schema_field_ids(table_db):
    repo,tid,name,area = table_db
    doc=repo.add_document(SourceDocument('2026设备预算','https://example.edu/other',sha256='budget-bind'))
    table=ParsedTable('设备预算',['设备','预算（元）'],[['网关','50']])
    other=repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
    class Index:
        def search(self,*args,**kwargs):
            return [Evidence('room','用房表',1,1,'2026测试统计','https://example.edu/source',table_id=tid),
                Evidence('budget','预算表',1,doc,'2026设备预算','https://example.edu/other',table_id=other)]
    class Generator:
        def generate(self,*args,**kwargs):
            return json.dumps({'table_id':None,'steps':[{'id':'a','op':'lookup','field_id':area,
                'filters':[{'field_id':name,'op':'eq','value':'东馆'}]}]})
    p=Path('.')
    service=RAGService(Settings(p,p,p,p,p),Index(),Generator(),repo.connection)
    result=service.answer('东馆面积多少平方米？')
    assert len(service.last_diagnostics['schema'])==2
    assert result.calculation.ok and result.calculation.value==12 and service.last_plan.table_id==tid


def test_examples_preserve_question_order_without_shortening_class_names(table_db):
    repo,tid,name,area=table_db
    q='北馆比东馆面积多多少平方米？'
    schema=StructuredQueryEngine(repo.connection).schema_context([tid],[],q)
    assert schema[0]['question_matches'][0]['values']==['北馆','东馆']
    example=json.loads(step_example('subtract',json.dumps({'tables':schema}),q))
    assert 'output' not in example
    assert [s['filters'][0]['value'] for s in example['steps'][:2]]==['北馆','东馆']
    assert not _object_mentioned('管理学院','战略管理对应哪个学院？')


@pytest.mark.parametrize('filtered',[False,True])
def test_single_delimiter_repair_consumes_shared_repair_budget(table_db,filtered):
    repo,tid,name,area=table_db
    calls=[]
    step={'id':'a','op':'lookup','field_id':area}
    if filtered:step['filters']=[{'field_id':name,'op':'eq','value':'东馆'}]
    valid=json.dumps({'table_id':tid,'steps':[step]})
    malformed=valid[:-2]+valid[-1]  # Remove only the steps closing bracket.
    assert repair_json_delimiter(malformed)==valid
    class Index:
        def search(self,*args,**kwargs):
            return [Evidence('t','表格',1,1,'2026测试统计','https://example.edu/source',table_id=tid)]
    class Generator:
        def generate(self,*args,**kwargs):
            calls.append(1)
            return malformed
    p=Path('.')
    service=RAGService(Settings(p,p,p,p,p),Index(),Generator(),repo.connection)
    result=service.answer('东馆面积多少平方米？')
    assert len(calls)==1 and result.calculation.ok==filtered
    assert len(service.last_diagnostics['attempts'])==2
    assert service.last_diagnostics['attempts'][1]['kind']=='syntax_repair'
    assert service.planner.last_raw==malformed
    if filtered:assert result.calculation.value==12


def test_format_repair_never_supplies_values_or_changes_valid_json():
    assert repair_json_delimiter('{"steps":[{"id":"a","op":"lookup","literal":,"unit":"元"}]}') is None
    assert repair_json_delimiter('{"steps":[{"id":"a","op":"lookup","fact_id":"f1"}]}') is None


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


def test_text_facts_preserve_bounds_scope_and_original_offsets():
    text = '第七条 学生每周不超过7小时，每月不超过28小时。岗位酬金不低于每小时19\n元。'
    source = Evidence('bounds',text,1,1,'办法','https://example.edu/rule')
    facts = list(NumericEngine.text_facts([source]).values())
    assert [(f['value'],f['unit'],f['comparison']) for f in facts]==[
        ('7','小时','lte'),('28','小时','lte'),('19','元/小时','gte')]
    assert [f['scope']['period'] for f in facts]==['每周','每月','每小时']
    for fact in facts:
        a,b = fact['span']
        assert fact['quote']==text[a:b] and fact['subject']
        assert fact['scope']['clause']=='第七条'
        assert fact['source_url']==source.source_url


def test_text_facts_dimensionless_gpa_and_subjects():
    source = Evidence('gpa','第二条 普通申请人平均学分绩点为2.3及以上。运动员平均学分绩点要求为1.9及以上。',
        1,1,'学位办法','https://example.edu/degree')
    facts = list(NumericEngine.text_facts([source]).values())
    assert len(facts)==2 and {f['subject'] for f in facts}=={'普通申请人','运动员'}
    assert all(f['unit']=='绩点' and f['comparison']=='gte' for f in facts)


def test_text_facts_distinguish_upper_lower_and_reject_unknown_numbers():
    source = Evidence('limits','甲类贷款额度不超过18000元，不低于900元；乙类贷款额度不超过23000元。'
        '第36页。编号为2026123。预算约为500元。缺失补助金额。',1,1,'贷款','https://example.edu/loan')
    facts = list(NumericEngine.text_facts([source]).values())
    assert [(f['value'],f['comparison']) for f in facts]==[('18000','lte'),('900','gte'),('23000','lte')]
    assert '甲类' in facts[0]['subject'] and '乙类' in facts[2]['subject']


def test_conflicting_same_scope_facts_are_not_silently_selected():
    source = Evidence('conflict','甲项目预算为110元。甲项目预算为130元。乙项目预算为90元。',
        1,1,'预算','https://example.edu/budget')
    facts = list(NumericEngine.text_facts([source]).values())
    assert len(facts)==1 and facts[0]['value']=='90'


def test_text_facts_keep_post_value_applicability_and_dimensionless_interval():
    source = Evidence('scope','补贴金额为140元（仅限已备案人员），未备案者不适用。'
        '平均学分绩点低于2.6但高于2.1。',1,1,'申请规则','https://example.edu/scope')
    facts = list(NumericEngine.text_facts([source]).values())
    assert '未备案者不适用' in facts[0]['scope']['context']
    assert [(f['value'],f['unit'],f['comparison']) for f in facts[1:]]==[
        ('2.6','绩点','lt'),('2.1','绩点','gt')]


@pytest.mark.parametrize('reverse',[False,True])
def test_compound_unit_decimal_calculation_from_text(table_db,reverse):
    repo,_,_,_ = table_db
    source = Evidence('rate','临时岗位酬金不低于每小时19.25元。',1,1,'报酬规则','https://example.edu/pay')
    facts = NumericEngine.text_facts([source])
    operands = ['a',{'literal':'6','unit':'小时'}]
    p = parse_plan(json.dumps({'steps':[{'id':'a','op':'lookup','fact_id':'f1'},
        {'id':'b','op':'multiply','inputs':operands[::-1] if reverse else operands}]}))
    result = StructuredQueryEngine(repo.connection).execute_steps(p,[],'按最低标准计算6小时酬金合计多少？',facts)
    assert result.ok and result.value==Decimal('115.50') and result.unit=='元'
    assert result.details[0]['comparison']=='gte' and result.evidence==[source]
    p.steps[-1]['inputs'] = ['a',{'literal':'6','unit':'天'}]
    assert not StructuredQueryEngine(repo.connection).execute_steps(p,[],'计算6小时酬金',facts).ok


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
    service = RAGService(settings,Index(),Generator(),repo.connection)
    result = service.answer('东馆面积合计多少？')
    assert result.calculation.ok and result.calculation.value==12
    assert calls==['search','generate','generate']
    assert 'field_id' in result.numeric_verification['steps'][0]['details'][0]
    diagnostic = service.last_diagnostics
    assert diagnostic['semantic_correct'] is None
    assert diagnostic['retrieved']==['table-chunk']
    first,repair = diagnostic['attempts']
    assert first['kind']=='initial' and first['syntax_ok'] and not first['execution_ok']
    assert repair['kind']=='repair' and repair['feedback'] and repair['execution_ok']
    assert first['raw_plan']!=repair['raw_plan']
    assert repair['field_bindings'] and repair['execution']['steps']
    assert diagnostic['citations']==result.references
    service.answer('东馆面积合计多少？')
    assert len(service.last_diagnostics['attempts'])==1  # Never retain a previous request.


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
