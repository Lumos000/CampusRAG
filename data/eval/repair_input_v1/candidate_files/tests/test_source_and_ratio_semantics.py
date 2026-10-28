import json

import pytest

from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence,ParsedTable,SourceDocument
from campus_rag.config import Settings
from campus_rag.rag.service import RAGService


@pytest.fixture
def engine(tmp_path):
    with Repository(tmp_path/'fixture.db') as repo:
        repo.initialize()
        yield repo,StructuredQueryEngine(repo.connection)


def text_cases():
    return NumericEngine.text_facts([
        Evidence('pine','本科生人数为17人。研究生人数为29人。',1,1,'“林岳资助”评选办法','https://example.edu/pine'),
        Evidence('river','研究生人数为43人。',1,2,'“溪谷资助”评选办法','https://example.edu/river')])


def execute_facts(engine,facts,question,values,operation='add'):
    ids=[next(fid for fid,f in facts.items() if f['value']==str(value)) for value in values]
    plan=parse_plan(json.dumps({'steps':[{'id':'a','op':'lookup','fact_id':ids[0]},
        {'id':'b','op':'lookup','fact_id':ids[1]},
        {'id':'result','op':operation,'inputs':['a','b']}]}))
    return engine.execute_steps(plan,[],question,facts)


def test_requested_policy_cannot_mix_another_policy(engine):
    _,query=engine;facts=text_cases()
    q='按“林岳资助”办法，本科生和研究生人数合计多少人？'
    good=execute_facts(query,facts,q,[17,29])
    assert good.ok and good.value==46
    wrong=execute_facts(query,facts,q,[17,43])
    assert not wrong.ok and wrong.repairable


def test_explicit_two_policies_may_be_combined(engine):
    _,query=engine;facts=text_cases()
    q='“林岳资助”和“溪谷资助”两份办法各取研究生人数，合计多少人？'
    good=execute_facts(query,facts,q,[29,43])
    assert good.ok and good.value==72
    assert not execute_facts(query,facts,q,[17,29]).ok


def test_policy_sections_in_one_handbook_remain_distinct(engine):
    _,query=engine
    facts=NumericEngine.text_facts([
        Evidence('chapter-a','《林岳资助办法》\n本科生人数为17人。',1,1,'公开学生手册','https://example.edu/book'),
        Evidence('chapter-b','《溪谷资助办法》\n研究生人数为43人。',1,1,'公开学生手册','https://example.edu/book')])
    assert not execute_facts(query,facts,'按《林岳资助办法》，本科生和研究生人数之和是多少人？',[17,43]).ok


def test_handbook_heading_does_not_change_the_fact_subject(engine):
    _,query=engine
    facts=NumericEngine.text_facts([
        Evidence('book','《林岳资助办法》\n青石项目预算为17元。白湖项目预算为29元。',1,1,'公开汇编','https://example.edu/book'),
        Evidence('other','青石项目预算为43元。白湖项目预算为47元。',1,2,'溪谷资助办法','https://example.edu/other')])
    q='按《林岳资助办法》，青石项目和白湖项目预算合计多少元？'
    result=execute_facts(query,facts,q,[17,29])
    assert result.ok and result.value==46
    assert facts[next(fid for fid,f in facts.items() if f['value']=='17')]['subject']=='青石项目'


@pytest.mark.parametrize('left,right',[(19,47),(47,19),(19,19)])
def test_named_table_ratio_keeps_direction_even_above_one_hundred(engine,left,right):
    repo,query=engine
    did=repo.add_document(SourceDocument('合成经费清单','https://example.edu/budget',sha256='ratio'))
    table=ParsedTable('经费清单',['项目','金额（元）'],[['青石项目',str(left)],['白湖项目',str(right)],['其他项目','11']])
    tid=repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
    name,amount=[r['id'] for r in repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,))]
    def run(reverse=False):
        labels=['白湖项目','青石项目'] if reverse else ['青石项目','白湖项目']
        steps=[{'id':sid,'op':'lookup','field_id':amount,
            'filters':[{'field_id':name,'op':'eq','value':label}]} for sid,label in zip(['a','b'],labels)]
        steps.append({'id':'result','op':'ratio','inputs':['a','b']})
        return query.execute_steps(parse_plan(json.dumps({'table_id':tid,'steps':steps})),[tid],
            '青石项目金额占白湖项目金额的百分比是多少？')
    good=run();assert good.ok
    assert not run(True).ok


def test_named_text_ratio_keeps_subject_direction(engine):
    _,query=engine
    facts=NumericEngine.text_facts([Evidence('facts','青石项目预算为19元。白湖项目预算为47元。',1,1,'经费规则','https://example.edu/rules')])
    q='青石项目预算占白湖项目预算的百分比是多少？'
    assert execute_facts(query,facts,q,[19,47],'ratio').ok
    assert not execute_facts(query,facts,q,[47,19],'ratio').ok


def test_table_ratio_does_not_have_to_consume_unrelated_text_facts(engine):
    repo,query=engine
    did=repo.add_document(SourceDocument('经费清单','https://example.edu/table',sha256='table-text'))
    table=ParsedTable('经费清单',['项目','金额（元）'],[['青石项目','19'],['白湖项目','47']])
    tid=repo.add_table(did,table,[[normalize_cell(v) for v in row] for row in table.rows])
    name,amount=[r['id'] for r in repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,))]
    facts=NumericEngine.text_facts([Evidence('other','青石项目预算为19元。白湖项目预算为47元。',1,2,'另一份规定','https://example.edu/other')])
    steps=[{'id':sid,'op':'lookup','field_id':amount,'filters':[{'field_id':name,'op':'eq','value':label}]}
        for sid,label in [('a','青石项目'),('b','白湖项目')]]
    steps.append({'id':'result','op':'ratio','inputs':['a','b']})
    q='按《经费清单》，青石项目金额占白湖项目金额的百分比是多少？'
    assert query.execute_steps(parse_plan(json.dumps({'table_id':tid,'steps':steps})),[tid],q,facts).ok
    steps[-1]['inputs'].reverse()
    assert not query.execute_steps(parse_plan(json.dumps({'table_id':tid,'steps':steps})),[tid],q,facts).ok


@pytest.mark.parametrize('named',[False,True])
def test_only_explicit_source_questions_add_source_labels_to_prompt(engine,tmp_path,named):
    repo,_=engine
    class Index:
        def search(self,*args,**kwargs):
            return [Evidence('budget','甲项目预算为17元。',1,1,'林岳资助办法','https://example.edu/policy')]
    class Generator:
        def generate(self,prompt,**kwargs):
            self.prompt=prompt
            return '{"table_id":null,"steps":[{"id":"result","op":"lookup","fact_id":"f1"}]}'
    g=Generator();service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),g,repo.connection)
    result=service.answer(('按《林岳资助办法》，' if named else '')+'甲项目预算是多少元？')
    assert result.calculation.ok
    assert ('"source_title"' in g.prompt)==named
