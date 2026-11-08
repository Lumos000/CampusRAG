import json
from decimal import Decimal

import pytest

from campus_rag.config import ROOT, load_settings
from campus_rag.crawl.crawler import CampusCrawler
from campus_rag.database import Repository
from campus_rag.database.normalization import header_unit
from campus_rag.database.query import StructuredQueryEngine, _requirement_filters
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import QueryPlanner, parse_plan, step_example, _adjustment_example, _inverse_weight_example
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence, ParsedTable, SourceDocument


def facts_for(text):
    return NumericEngine.text_facts([Evidence('source',text,1,1,'服务规则','https://example.edu/rules')])


@pytest.mark.parametrize('header',['招生专业及计划 / 计划','人员安排 / 计划','设备采购 / 计划'])
def test_business_plan_names_do_not_invent_units(header):
    assert header_unit(header)==''


@pytest.mark.parametrize('configured,expected',[(None,'portal.example.edu'),('示例大学','示例大学')])
def test_crawler_fallback_identity_comes_from_configuration(tmp_path,configured,expected):
    data={'allowed_domains':['example.edu'],'seed_urls':[]}
    if configured is not None:
        data['default_department']=configured
    path=tmp_path/'sites.json';path.write_text(json.dumps(data),encoding='utf-8')
    crawler=CampusCrawler(path,tmp_path/'raw')
    try:
        assert crawler._department('https://portal.example.edu/article')==expected
    finally:
        crawler.session.close()


@pytest.mark.parametrize('unit',['人','元','台'])
def test_dataset_unit_annotations_are_explicit_and_do_not_rewrite_data(tmp_path,unit):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        did=repo.add_document(SourceDocument('公开安排','https://example.edu/plan'))
        table=ParsedTable('安排',['项目','安排 / 计划'],[['甲','11'],['乙','17']])
        tid=repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
        engine=StructuredQueryEngine(repo.connection,{'安排 / 计划':unit})
        schema=engine.schema_context([tid],[])
        fid=schema[0]['columns'][1]['field_id']
        assert schema[0]['columns'][1]['unit']==unit
        plan=parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'total','op':'sum','field_id':fid}]}))
        result=engine.execute_steps(plan,[tid],'全部计划合计多少'+unit+'？')
        assert result.ok, result.error
        assert result.value==Decimal(28) and result.unit==unit
        assert repo.connection.execute('SELECT unit FROM table_columns WHERE id=?',(fid,)).fetchone()[0]==''


@pytest.mark.parametrize('bad',[[],{'计划':None},{'': '人'}])
def test_invalid_dataset_unit_configuration_is_rejected(tmp_path,bad):
    data=json.loads((ROOT/'config/settings.json').read_text(encoding='utf-8'))
    data['field_units']=bad
    path=tmp_path/'settings.json';path.write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='field_units'):
        load_settings(path)


def test_dataset_fallback_does_not_override_an_explicit_table_unit(tmp_path):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        did=repo.add_document(SourceDocument('公开计划','https://example.edu/plan'))
        table=ParsedTable('计划',['项目','计划'],[['甲','11'],['乙','17']],unit='元')
        tid=repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
        engine=StructuredQueryEngine(repo.connection,{'计划':'人'})
        schema=engine.schema_context([tid],[])[0]['columns']
        assert schema[0]['unit']==''  # A table's currency unit does not type an item name.
        column=schema[1]
        assert column['unit']=='元'
        plan=parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'total','op':'sum','field_id':column['field_id']}]}))
        result=engine.execute_steps(plan,[tid],'全部计划合计多少元？')
        assert result.ok and result.value==28 and result.unit=='元'


@pytest.mark.parametrize('extra,question,expected',[
    ('附加服务每月90元。','基础增强服务上浮后加附加服务，两个月合计多少元？','1815.2'),
    ('','基础增强服务上浮后，两个月合计多少元？','1635.2'),
    ('附加服务每月90元。','基础增强服务上浮后，不含附加服务，两个月合计多少元？','1635.2')])
def test_period_adjustments_support_months_and_optional_or_excluded_costs(tmp_path,extra,question,expected):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        facts=facts_for('基础服务每月730元，增强服务在上述标准基础上上浮12%。'+extra)
        context=json.dumps({'facts':list(facts.values())},default=str,ensure_ascii=False)
        plan=parse_plan(step_example('sum',context,question))
        result=StructuredQueryEngine(repo.connection).execute_steps(plan,[],question,facts)
        assert result.ok, result.error
        assert result.value==Decimal(expected) and result.unit=='元'
        assert any(s.get('literal')=='2' and s.get('unit')=='月' for s in plan.steps)


def test_negated_adjustment_does_not_narrow_facts_to_an_increase_template():
    facts=list(facts_for('基础服务每月730元，增强服务在上述标准基础上上浮12%。').values())
    assert _adjustment_example(facts,'假设基础服务不上浮，两个月合计多少？') is None


def test_coefficient_only_does_not_require_or_invent_another_ratio(tmp_path):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        facts=facts_for('基础人员23人，附加人员16人，加权人员总数为31。')
        question='反推附加折算系数。'
        plan=parse_plan(step_example('lookup',json.dumps({'facts':list(facts.values())},default=str),question))
        assert all(s['op']!='report' for s in plan.steps)
        result=StructuredQueryEngine(repo.connection).execute_steps(plan,[],question,facts)
        assert result.ok and result.value==Decimal('0.5')


def test_weighted_mean_does_not_use_a_weighted_total_equation():
    facts=list(facts_for('基础人员23人，附加人员16人，加权平均人员数为31。').values())
    assert _inverse_weight_example(facts,'反推附加折算系数。') is None


def test_inverse_word_alone_does_not_hide_available_tables():
    context={'tables':[{'table_id':7,'name':'统计','document':'统计','columns':[
        {'field_id':10,'name':'数量','role':'measure','unit':'人'}],
        'question_matches':[],'total_row_count':0}],
        'facts':[{'fact_id':'f1','field':'无关人数','subject':'无关事项','value':'3','unit':'人'}]}
    class Generator:
        def generate(self,prompt,**kwargs):
            assert '"table_id": 7' in prompt
            return '{"table_id":7,"steps":[{"id":"value","op":"sum","field_id":10}]}'
    plan=QueryPlanner(Generator()).step_plan('按表反推一个系数',json.dumps(context))
    assert plan.table_id==7


@pytest.mark.parametrize('separator',['和','、','，',','])
def test_selected_options_are_not_truncated_at_list_punctuation(separator):
    columns={1:{'original_name':'首选科目','data_type':'text'},
             2:{'original_name':'再选科目','data_type':'text'},
             3:{'original_name':'计划（人）','data_type':'number'}}
    filters=_requirement_filters(columns,f'首选物理、再选政治{separator}地理的考生，计划合计多少？')
    assert filters==[{'field_id':1,'op':'satisfies','value':['物理']},
                     {'field_id':2,'op':'satisfies','value':['政治','地理']}]


def test_alternative_question_options_are_not_treated_as_a_definite_set():
    columns={1:{'original_name':'许可条件','data_type':'text'}}
    with pytest.raises(ValueError,match='明确'):
        _requirement_filters(columns,'许可条件：甲或乙，计划多少？')


@pytest.mark.parametrize('query',['许可条件是什么？','许可条件有哪些？','许可条件有多少项？'])
def test_asking_about_a_requirement_does_not_assert_having_those_options(query):
    columns={1:{'original_name':'许可条件','data_type':'text'}}
    assert _requirement_filters(columns,query)==[]
