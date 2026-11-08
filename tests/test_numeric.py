from decimal import Decimal
import json

import pytest

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.numeric import NumericEngine, check_draft
from campus_rag.rag.planner import QueryPlanner, parse_plan, repair_json_delimiter, step_example
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell
from campus_rag.tools.requirements import satisfies
from campus_rag.types import Evidence, ParsedTable, SourceDocument


def evidence(text):
    return Evidence('source',text,1,1,'公开标准','https://example.edu/rule')


def test_labelled_quantities_preserve_scope_and_skip_page_numbers():
    source = evidence('40\n专任教师2318人，外聘教师864人，折合教师总数为2750，折合在校生人数40863.2，生师比为14.86：1。')
    facts = NumericEngine.text_facts([source])
    assert [(f['value'],f['unit']) for f in facts.values()] == [
        ('2318','人'),('864','人'),('2750','人'),('40863.2','人'),('14.86','ratio')]
    assert all(source.text[slice(*f['span'])] == f['quote'] for f in facts.values())
    assert not NumericEngine.text_facts([evidence('40\n数据总数为2750。')])


@pytest.mark.parametrize('base,rate,lodging,years,expected', [
    ('5800','10','1500','4','31520'),('6000','5','1200','3','22500')])
def test_adjusted_recurring_costs_use_the_shared_service(tmp_path,base,rate,lodging,years,expected):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        source = evidence(f'甲类服务每人每年{base}元，增强服务在上述标准基础上上浮{rate}%。附加服务暂按{lodging}元/人·年预收。')
        steps = [
            {'id':'base','op':'lookup','fact_id':'f1'},
            {'id':'rate','op':'lookup','fact_id':'f2'},
            {'id':'adjusted','op':'increase','inputs':['base','rate']},
            {'id':'extra','op':'lookup','fact_id':'f3'},
            {'id':'annual','op':'add','inputs':['adjusted','extra']},
            {'id':'years','op':'lookup','literal':years,'unit':'年'},
            {'id':'total','op':'multiply','inputs':['annual','years']},
            {'id':'results','op':'report','inputs':['adjusted','extra','total']}]
        class Index:
            def search(self,*a,**kw): return [source]
        class Generator:
            def __init__(self): self.calls = 0
            def planning_generate(self,prompt,**kw):
                self.calls += 1
                assert 'increase' in prompt
                return json.dumps({'steps':steps})
        generator = Generator()
        service = RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),generator,repo.connection)
        result = service.answer(f'甲类增强服务上浮后加附加服务，标准{years}年不变，预收合计多少？')
        assert result.calculation.ok, result.calculation.error
        assert result.calculation.value == Decimal(expected)
        assert result.calculation.unit == '元'
        assert generator.calls == 1 and service.last_diagnostics['route'] == 'numeric_plan'
        assert result.evidence == [source]


def test_inverse_weight_and_ratio_can_both_be_reported(tmp_path):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        source = evidence('原始人员2000人，附加人员1000人，折合人员总数为2500，服务人数40000。')
        steps = [{'id':name,'op':'lookup','fact_id':'f'+str(i)} for i,name in enumerate(['base','extra','weighted','people'],1)]
        steps += [
            {'id':'delta','op':'subtract','inputs':['weighted','base']},
            {'id':'coefficient','op':'divide','inputs':['delta','extra']},
            {'id':'ratio','op':'divide','inputs':['people','weighted']},
            {'id':'results','op':'report','inputs':['coefficient','ratio']}]
        result = StructuredQueryEngine(repo.connection).execute_steps(
            parse_plan(json.dumps({'steps':steps})),[],'反推系数，再复算比值',NumericEngine.text_facts([source]))
        assert result.ok, result.error
        assert result.value == Decimal(16)
        assert '0.5' in result.expression
        assert result.evidence == [source]


def test_requirement_logic_is_independent_of_business_names():
    assert satisfies('不限',{'甲','乙'})
    assert satisfies('甲或丙',{'甲','乙'})
    assert satisfies('甲和乙',{'甲','乙'})
    assert not satisfies('甲和丙',{'甲','乙'})
    assert not satisfies('丙',{'甲','乙'})
    with pytest.raises(ValueError): satisfies('甲、丙',{'甲','乙'})
    with pytest.raises(ValueError): satisfies('优先甲',{'甲','乙'})


@pytest.mark.parametrize('requirement,selected,expected',[
    ('甲、乙任选其一',{'甲'},True),('甲、乙任选其一',{'丙'},False),
    ('甲和乙均须选考',{'甲','乙'},True),('甲和乙均须选考',{'甲'},False),
    ('甲、乙均须满足',{'甲','乙'},True),('同时甲和乙',{'甲','乙'},True),
    ('甲和乙任选其一',{'甲'},True),('甲和乙任选其一',{'丙'},False),
])
def test_requirement_modifiers_are_parsed_instead_of_becoming_option_names(requirement,selected,expected):
    assert satisfies(requirement,selected) is expected


@pytest.mark.parametrize('requirement',['甲或乙和丙','甲至少两项','甲或乙和丙任选其一'])
def test_unsupported_requirement_grammar_refuses_instead_of_dropping_rows(requirement):
    with pytest.raises(ValueError):
        satisfies(requirement,{'甲','乙','丙'})


def test_requirement_sum_reads_full_table_and_rejects_omitted_condition(tmp_path):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        did = repo.add_document(SourceDocument('公开计划','https://example.edu/plan'))
        table = ParsedTable('计划',['首选科目','再选科目','专业名称','计划（人）'],[
            ['物理','不限','专业甲','11'],['物理','不限','专业乙','17'],
            ['物理','化学','专业丙','20'],['历史','不限','专业丁','30'],['合计','合计','合计','78']])
        tid = repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
        engine = StructuredQueryEngine(repo.connection)
        question = '首选物理，再选政治和地理的考生，可报计划合计多少人？'
        columns = repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,)).fetchall()
        filters = [{'field_id':columns[0]['id'],'op':'satisfies','value':['物理']},
                   {'field_id':columns[1]['id'],'op':'satisfies','value':['政治','地理']}]
        plan = parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'total','op':'sum','field_id':columns[3]['id'],'filters':filters}]}))
        result = engine.execute_steps(plan,[tid],question)
        assert result.ok, result.error
        assert result.value == Decimal(28)
        assert len(result.details) == 2
        plan.steps[0]['filters'] = []
        assert not engine.execute_steps(plan,[tid],question).ok


def test_checked_answer_rejects_number_changes_label_swaps_and_new_claims():
    canonical = '专任教师为2318人；外聘教师为864人。'
    for draft in ['专任教师为40人；外聘教师为864人。','专任教师为864人；外聘教师为2318人。',canonical+'共计9999人。']:
        answer, audit = check_draft(draft,canonical)
        assert answer == canonical and audit['used_template']
    assert not check_draft(canonical,canonical)[1]['used_template']


@pytest.mark.parametrize('text,question,operation,expected',[
    ('专任教师2000人，外聘教师1000人。','专任教师和外聘教师合计多少人？','sum','3000'),
    ('工科类专业每生每年6000元，理科类专业每生每年5500元。','工科和理科学费相差多少？','subtract','500')])
def test_neighboring_questions_are_no_longer_intercepted(tmp_path,text,question,operation,expected):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        steps = [{'id':'left','op':'lookup','fact_id':'f1'},
                 {'id':'right','op':'lookup','fact_id':'f2'},
                 {'id':'result','op':operation,'inputs':['left','right']}]
        result = StructuredQueryEngine(repo.connection).execute_steps(
            parse_plan(json.dumps({'steps':steps})),[],question,NumericEngine.text_facts([evidence(text)]))
        assert result.ok, result.error
        assert result.value == Decimal(expected)


def test_conflicts_and_missing_counts_do_not_become_facts():
    assert not NumericEngine.text_facts([evidence('专任教师10人。专任教师20人。')])
    assert not NumericEngine.text_facts([evidence('40\n教师队伍结构合理。')])


def test_incomplete_nominal_phrases_do_not_create_extra_operands():
    facts = NumericEngine.text_facts([evidence('课程体系由5门核心课程构成，共10学分。')],academic_units=True)
    assert [(f['value'],f['unit']) for f in facts.values()] == [('10','学分')]
    facts = NumericEngine.text_facts([evidence('平均学分绩点为3.0及以上或者专业排名前5%。')])
    assert not any(f['unit']=='ratio' for f in facts.values())


def test_difference_cannot_repeat_one_subject(tmp_path):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        facts = NumericEngine.text_facts([evidence('甲组人数20人，乙组人数30人。')])
        steps = [{'id':'left','op':'lookup','fact_id':'f1'},
                 {'id':'right','op':'lookup','fact_id':'f1'},
                 {'id':'result','op':'subtract','inputs':['left','right']}]
        engine = StructuredQueryEngine(repo.connection)
        question = '甲组与乙组人数相差多少人？'
        result = engine.execute_steps(parse_plan(json.dumps({'steps':steps})),[],question,facts)
        assert not result.ok and '乙组人数' in result.error
        steps[0]['fact_id'] = 'f2'
        result = engine.execute_steps(parse_plan(json.dumps({'steps':steps})),[],question,facts)
        assert result.ok and result.value == Decimal(10)


@pytest.mark.parametrize('steps',[
    [{'id':'r','op':'report','inputs':[]}],
    [{'id':'a','op':'lookup','literal':'1','unit':''},
     {'id':'r','op':'report','inputs':['a','a']},
     {'id':'x','op':'add','inputs':['r','a']}],
])
def test_invalid_reports_cannot_be_used_as_arithmetic(steps):
    with pytest.raises(ValueError,match='report'):
        parse_plan(json.dumps({'steps':steps}))


@pytest.mark.parametrize('hidden',[False,True])
def test_prompt_budget_keeps_related_facts_and_rejects_unseen_ids(hidden):
    facts = [
        {'fact_id':'f1','field':'甲组人数','subject':'甲组','value':'4','unit':'人',
         'scope':{'context':'甲组人数为4人，每人标准为137元/人。'}},
        {'fact_id':'f2','field':'人均标准','subject':'标准','value':'137','unit':'元/人',
         'scope':{'context':'甲组人数为4人，每人标准为137元/人。'}}]
    facts += [{'fact_id':'f'+str(i),'field':'其他事项','subject':'其他事项','value':'9','unit':'元',
               'scope':{'context':'完全无关的规定。'*100}} for i in range(3,21)]
    class Generator:
        max_context_tokens = 2000
        count_tokens = staticmethod(len)
        def generate(self,prompt,**kwargs):
            assert len(prompt)<=self.max_context_tokens
            assert '"fact_id": "f1"' in prompt and '"fact_id": "f2"' in prompt
            assert '"fact_id": "f20"' not in prompt
            return json.dumps({'steps':[{'id':'value','op':'lookup','fact_id':'f20' if hidden else 'f1'}]})
    planner = QueryPlanner(Generator())
    context = json.dumps({'facts':facts},ensure_ascii=False)
    if hidden:
        with pytest.raises(ValueError,match='未提供'):
            planner.step_plan('甲组人数与每人标准相乘是多少？',context)
    else:
        assert planner.step_plan('甲组人数与每人标准相乘是多少？',context).steps[0]['fact_id']=='f1'


def test_weighted_example_works_for_unrelated_object_labels(tmp_path):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        source = evidence('基础人员23人，附加人员16人，加权人员总数为31，服务人数62人。')
        facts = NumericEngine.text_facts([source])
        question = '反推附加折算系数，再复算服务人数与加权人员的比值。'
        raw = step_example('lookup',json.dumps({'facts':list(facts.values())},default=str),question)
        plan = parse_plan(raw)
        result = StructuredQueryEngine(repo.connection).execute_steps(plan,[],question,facts)
        assert result.ok, result.error
        assert [v['value'] for v in result.outputs[-2:]] == [Decimal('0.5'),Decimal('2')]


def test_syntax_repair_preserves_compact_fact_bindings(tmp_path):
    with Repository(tmp_path/'db') as repo:
        repo.initialize()
        source = evidence('甲类服务每人每年730元，增强服务在上述标准基础上上浮12%。附加服务暂按90元/人·学年预收，按实际情况结算。')
        extracted = list(NumericEngine.text_facts([source]).values())
        facts = {'f'+str(i):{**f,'fact_id':'f'+str(i)} for i,f in zip((9,20,21),extracted)}
        question = '甲类增强服务上浮后加附加服务，标准两年不变，预收合计多少？'
        class Generator:
            def generate(self,prompt,**kwargs):
                context = {'facts':[{**f,'fact_id':'f'+str(i)} for i,f in enumerate(extracted,1)]}
                raw = step_example('sum',json.dumps(context,default=str),question)
                return raw[:-2]+raw[-1]  # Missing only the steps-list closer.
        planner = QueryPlanner(Generator())
        with pytest.raises(ValueError):
            planner.step_plan(question,json.dumps({'facts':list(facts.values())},default=str))
        repaired = parse_plan(repair_json_delimiter(planner.last_raw))
        repaired = planner.bind_visible_facts(repaired)
        assert {s['fact_id'] for s in repaired.steps if 'fact_id' in s} == set(facts)
        result = StructuredQueryEngine(repo.connection).execute_steps(repaired,[],question,facts)
        assert result.ok, result.error
        assert result.value == Decimal('1815.2')
        assert result.outputs[1]['period_label']=='学年'
        assert '按实际情况结算' in result.outputs[1]['note']
