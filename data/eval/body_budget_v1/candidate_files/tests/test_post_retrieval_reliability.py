import json
from decimal import Decimal

import pytest

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine, _object_mentioned
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import operation_hint, parse_plan, step_example
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence, ParsedTable, SourceDocument


def evidence(text):
    return Evidence('rule',text,1,1,'合成管理规则','https://example.edu/rule')


@pytest.fixture
def repo(tmp_path):
    with Repository(tmp_path/'case.db') as active:
        active.initialize()
        yield active


def execute_text(repo,text,question,steps):
    facts=NumericEngine.text_facts([evidence(text)])
    return StructuredQueryEngine(repo.connection).execute_steps(
        parse_plan(json.dumps({'steps':steps})),[],question,facts)


def test_generic_amounts_belong_to_their_real_heading():
    text='资料维护费（完成登记后发放）\n审核后，奖励金额为17元。'
    text+='2.外业补贴（完成核实后发放）\n登记后，奖励金额为29元。'
    facts=list(NumericEngine.text_facts([evidence(text)]).values())
    assert {(f['subject'],f['value']) for f in facts}=={('资料维护费','17'),('外业补贴','29')}
    assert all(f['quote']==text[slice(*f['span'])] for f in facts)


def test_real_same_subject_conflict_is_still_rejected():
    facts=NumericEngine.text_facts([evidence('甲项目预算为27元。甲项目预算为39元。乙项目预算为11元。')])
    assert [f['value'] for f in facts.values()]==['11']


@pytest.mark.parametrize('text,expected',[
    ('2.服务类。每学年能够完成93小时的服务时长。',[('93','小时','eq')]),
    ('辅修课程共6学期。',[('6','学期','eq')]),
    ('专业一般设置7至12门主要课程。',[('7','门','gte'),('12','门','lte')]),
    ('专业一般设置7-12门主要课程。',[('7','门','gte'),('12','门','lte')]),
])
def test_explicit_completion_total_and_interval_facts(text,expected):
    facts=list(NumericEngine.text_facts([evidence(text)]).values())
    assert [(f['value'],f['unit'],f['comparison']) for f in facts]==expected
    assert all(f['quote']==text[slice(*f['span'])] for f in facts)


def test_calendar_date_is_not_a_duration_operand():
    facts=NumericEngine.text_facts([evidence('公示期为6月12日到6月15日。培训时长为9小时。')])
    assert [(f['value'],f['unit']) for f in facts.values()]==[('9','小时')]


def test_person_rate_and_explicit_total_cannot_degrade_to_lookup(repo):
    text='生均补助金额为17元。'
    q='按生均补助标准，若有9名参与者，合计需要多少元？'
    facts=NumericEngine.text_facts([evidence(text)])
    assert facts['f1']['unit']=='元/人'
    wrong=execute_text(repo,text,q,[{'id':'a','op':'lookup','fact_id':'f1'}])
    assert not wrong.ok and wrong.repairable
    good=execute_text(repo,text,q,[{'id':'a','op':'lookup','fact_id':'f1'},
        {'id':'n','op':'lookup','literal':'9','unit':'人'},
        {'id':'total','op':'multiply','inputs':['a','n']}])
    assert good.ok and good.value==153 and good.unit=='元'
    assert good.steps[1]['source']['quote']=='9名'


def test_year_is_not_an_amount_operand(repo):
    q='按2031年生均标准，若有7人，合计需要多少元？'
    result=execute_text(repo,'人均补助金额为19元。',q,[
        {'id':'a','op':'lookup','fact_id':'f1'},{'id':'n','op':'lookup','literal':'7','unit':'人'},
        {'id':'total','op':'multiply','inputs':['a','n']}])
    assert result.ok and result.value==133


def test_annual_completed_amount_has_to_share_an_explicit_period(repo):
    text='2.服务类。每学年能够完成93小时的服务时长。'
    steps=[{'id':'cap','op':'lookup','fact_id':'f1'},
        {'id':'done','op':'lookup','literal':'68','unit':'小时'},
        {'id':'rest','op':'subtract','inputs':['cap','done']}]
    good=execute_text(repo,text,'按每学年的服务要求，已完成68小时，还差多少小时？',steps)
    assert good.ok and good.value==25
    bad=execute_text(repo,text,'按每月的服务要求，已完成68小时，还差多少小时？',steps)
    assert not bad.ok


@pytest.fixture
def awards(repo):
    doc=repo.add_document(SourceDocument('合成资助标准','https://example.edu/awards',sha256='awards'))
    table=ParsedTable('奖励标准',['奖项','金额（元）'],[
        ['一等奖学金','260'],['二等奖学金','170'],['三等奖学金','90'],['专项支持','40～70']])
    tid=repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
    fields=[r['id'] for r in repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,))]
    return StructuredQueryEngine(repo.connection),tid,fields


def test_short_award_names_bind_to_real_values_without_extra_range_row(awards):
    engine,tid,(name,amount)=awards
    q='综合一、二、三等奖各发一份，奖励金额合计多少元？'
    schema=engine.schema_context([tid],[],q)
    labels=schema[0]['question_matches'][0]['values']
    assert labels==['一等奖学金','二等奖学金','三等奖学金']
    step={'id':'a','op':'sum','field_id':amount,
        'filters':[{'field_id':name,'op':'in','value':labels}]}
    result=engine.execute_steps(parse_plan(json.dumps({'table_id':tid,'steps':[step]})),[tid],q)
    assert result.ok and result.value==520
    del step['filters']
    assert not engine.execute_steps(parse_plan(json.dumps({'table_id':tid,'steps':[step]})),[tid],q).ok
    assert not _object_mentioned('优秀一等奖学金','综合一等奖的奖励金额')


def test_short_name_exclusion_cannot_be_reversed(awards):
    engine,tid,(name,amount)=awards
    q='除一等奖之外，共有多少条记录？'
    def run(op,value):
        plan=parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'a','op':'count',
            'filters':[{'field_id':name,'op':op,'value':value}]}]}))
        return engine.execute_steps(plan,[tid],q)
    assert not run('eq','一等奖学金').ok
    good=run('not_in',['一等奖学金'])
    assert good.ok and good.value==3


def test_total_accepts_added_group_amounts_but_not_their_difference(repo):
    text='人均补助金额为17元。'
    q='两组分别有9名和7名参与者，补助金额合计多少元？'
    steps=[{'id':'rate','op':'lookup','fact_id':'f1'},
        {'id':'n1','op':'lookup','literal':'9','unit':'人'},
        {'id':'n2','op':'lookup','literal':'7','unit':'人'},
        {'id':'a','op':'multiply','inputs':['rate','n1']},
        {'id':'b','op':'multiply','inputs':['rate','n2']},
        {'id':'total','op':'add','inputs':['a','b']}]
    good=execute_text(repo,text,q,steps)
    assert good.ok and good.value==272
    steps[-1]['op']='subtract'
    assert not execute_text(repo,text,q,steps).ok


def test_uncertain_people_range_is_not_silently_replaced_by_upper_bound(repo):
    result=execute_text(repo,'生均补助金额为17元。','若有7—9名参与者，合计多少元？',[
        {'id':'a','op':'lookup','fact_id':'f1'},
        {'id':'n','op':'lookup','literal':'9','unit':'人'},
        {'id':'total','op':'multiply','inputs':['a','n']}])
    assert not result.ok


def test_examples_use_question_constants_and_bounds_in_correct_order():
    q='按人均标准，若有9名参与者，共需多少元？'
    facts={'facts':[{'fact_id':'f4','subject':'人均补助','value':'17','unit':'元/人','comparison':'eq'}]}
    assert operation_hint(q)=='multiply'
    plan=json.loads(step_example('multiply',json.dumps(facts),q))
    assert plan['steps'][1]['literal']=='9' and plan['steps'][1]['unit']=='人'
    bounds={'facts':[{'fact_id':'low','subject':'课程','value':'7','unit':'门','comparison':'gte'},
        {'fact_id':'high','subject':'课程','value':'12','unit':'门','comparison':'lte'}]}
    plan=json.loads(step_example('subtract',json.dumps(bounds),'课程上限比下限多多少门？'))
    assert [s['fact_id'] for s in plan['steps'][:2]]==['high','low']


@pytest.mark.parametrize('question',[
    '辅修成绩是否计入平均学分绩点？','这些补贴会计入平均工资吗？'])
def test_membership_question_is_text_not_average_calculation(repo,tmp_path,question):
    class Index:
        def search(self,*a,**kw):return [evidence('这些成绩和补贴不计入相关平均值。')]
    class Generator:
        def generate(self,*a,**kw):return json.dumps({'sentence_ids':['S1']})
    service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection)
    result=service.answer(question)
    assert service.last_diagnostics['route']=='text' and result.references


@pytest.mark.parametrize('question',[
    '我今年能否获得这项补助，最终会到账多少元？','按评选办法，我是否会获得这个奖？'])
def test_personal_outcome_uses_qualification_boundary_not_unit_error(repo,tmp_path,question):
    class Index:
        def search(self,*a,**kw):return [evidence('人均奖励金额为19元。')]
    class Generator:
        def generate(self,*a,**kw):raise AssertionError('No personal assessment outcome is public evidence')
    result=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection).answer(question)
    assert not result.references and '个人资格' in result.answer and '评审' in result.answer


@pytest.mark.parametrize('question',[
    '我能否获得补助，需要满足哪些条件？',
    '我是否能获得补助，申请条件是什么？',
])
def test_public_application_conditions_remain_answerable(repo,tmp_path,question):
    class Index:
        def search(self,*a,**kw):return [evidence('申请条件是已完成登记且审核通过。')]
    class Generator:
        def generate(self,*a,**kw):return json.dumps({'sentence_ids':['S1']})
    service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection)
    result=service.answer(question)
    assert service.last_diagnostics['route']=='text' and result.references


def test_only_following_public_rules_cannot_predict_personal_outcome(repo,tmp_path):
    class Index:
        def search(self,*a,**kw):return [evidence('人均奖励金额为19元。')]
    class Generator:
        def generate(self,*a,**kw):raise AssertionError('Public rules cannot determine personal outcomes')
    service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection)
    result=service.answer('只按评选办法，我是否会获得这个奖，最终实际到账多少元？')
    assert service.last_diagnostics['route']=='unsupported_personal_outcome' and not result.references


def test_equal_sized_groups_cannot_be_counted_once(repo):
    q='两组分别有9名和9名参与者，补助金额合计多少元？'
    steps=[{'id':'rate','op':'lookup','fact_id':'f1'},
        {'id':'n','op':'lookup','literal':'9','unit':'人'},
        {'id':'a','op':'multiply','inputs':['rate','n']}]
    assert not execute_text(repo,'人均补助金额为17元。',q,steps).ok
    steps.append({'id':'total','op':'add','inputs':['a','a']})
    result=execute_text(repo,'人均补助金额为17元。',q,steps)
    assert result.ok and result.value==306


def test_named_ratio_example_uses_the_requested_denominator(awards):
    engine,tid,(name,amount)=awards
    def example(question):
        context=json.dumps({'tables':engine.schema_context([tid],[],question),'facts':[]},ensure_ascii=False)
        return json.loads(step_example('ratio',context,question))['steps']
    pair=example('二等奖金额占一等奖金额的百分比是多少？')
    assert pair[0]['filters']==[{'field_id':name,'op':'eq','value':'二等奖学金'}]
    assert pair[1]['filters']==[{'field_id':name,'op':'eq','value':'一等奖学金'}]
    total=example('一等奖和二等奖金额之和占全表金额的百分比是多少？')
    assert len(total[0]['filters'][0]['value'])==2 and not total[1].get('filters')
