import json

import pytest

from campus_rag.database import Repository
from campus_rag.database.normalization import header_unit
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import ParsedTable, SourceDocument
from campus_rag.types import Evidence, Chunk
from campus_rag.config import Settings
from campus_rag.rag.service import RAGService


@pytest.fixture
def projects(tmp_path):
    with Repository(tmp_path/'projects.db') as repo:
        repo.initialize()
        doc=repo.add_document(SourceDocument('项目统计','https://example.edu/projects',sha256='projects'))
        table=ParsedTable('项目统计',['类别','项目','金额（元）'],[
            ['研究类','甲项目','7'],['研究类','乙项目','11'],['服务类','丙项目','13']])
        tid=repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
        fields=[r['id'] for r in repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,))]
        yield StructuredQueryEngine(repo.connection),tid,fields


def execute(projects,question,steps):
    engine,tid,_=projects
    return engine.execute_steps(parse_plan(json.dumps({'table_id':tid,'steps':steps})),[tid],question)


def test_compound_filter_preserves_complete_matching_domain(projects):
    _,_,(category,item,amount)=projects
    q='研究类中金额大于10元的记录有多少条？'
    filters=[{'field_id':category,'op':'eq','value':'研究类'},{'field_id':amount,'op':'gt','value':'10'}]
    good=execute(projects,q,[{'id':'a','op':'count','filters':filters}])
    assert good.ok and good.value==1
    extra=execute(projects,q,[{'id':'a','op':'count','filters':filters[:1]}])
    assert not extra.ok
    missing=execute(projects,'研究类和服务类中金额大于10元的记录有多少条？',[
        {'id':'a','op':'count','filters':filters}])
    assert not missing.ok
    chained=execute(projects,q,[{'id':'a','op':'filter','filters':filters},
        {'id':'b','op':'count','rows':'a'}])
    assert chained.ok and chained.value==1


def test_record_count_rejects_money_but_accepts_added_counts(projects):
    _,_,(category,item,amount)=projects
    q='全部项目有多少条？'
    wrong=execute(projects,q,[{'id':'a','op':'sum','field_id':amount}])
    assert not wrong.ok and wrong.repairable
    right=execute(projects,q,[{'id':'a','op':'count'}])
    assert right.ok and right.value==3
    added=execute(projects,'研究类和服务类合计多少条？',[
        {'id':'a','op':'count','filters':[{'field_id':category,'op':'eq','value':'研究类'}]},
        {'id':'b','op':'count','filters':[{'field_id':category,'op':'eq','value':'服务类'}]},
        {'id':'c','op':'add','inputs':['a','b']}])
    assert added.ok and added.value==3
    duplicate=execute(projects,q,[{'id':'a','op':'count'},{'id':'b','op':'count'},
        {'id':'c','op':'add','inputs':['a','b']}])
    assert not duplicate.ok


@pytest.mark.parametrize('operation,question,value',[
    ('sum','所有项目金额合计多少元？',31),('average','所有项目平均金额多少元？',31/3)])
def test_sum_and_average_are_distinct(projects,operation,question,value):
    amount=projects[2][2]
    good=execute(projects,question,[{'id':'a','op':operation,'field_id':amount}])
    assert good.ok and abs(float(good.value)-value)<1e-8
    wrong=execute(projects,question,[{'id':'a','op':'average' if operation=='sum' else 'sum','field_id':amount}])
    assert not wrong.ok


def test_average_allows_equivalent_sum_divided_by_count(projects):
    amount=projects[2][2]
    result=execute(projects,'所有项目平均金额多少元？',[
        {'id':'a','op':'sum','field_id':amount},{'id':'b','op':'count'},
        {'id':'c','op':'divide','inputs':['a','b']}])
    assert result.ok and float(result.value)==pytest.approx(31/3) and result.unit=='元'


def test_exclusion_cannot_be_executed_as_positive_filter(projects):
    category=projects[2][0]
    q='除研究类之外的项目有多少条？'
    wrong=execute(projects,q,[{'id':'a','op':'count','filters':[{'field_id':category,'op':'eq','value':'研究类'}]}])
    assert not wrong.ok
    right=execute(projects,q,[{'id':'a','op':'count','filters':[{'field_id':category,'op':'not_in','value':['研究类']}]}])
    assert right.ok and right.value==1
    wrong=execute(projects,'研究类项目有多少条？',[{'id':'a','op':'count','filters':[{'field_id':category,'op':'not_in','value':['研究类']}]}])
    assert not wrong.ok


def test_ratio_denominator_must_cover_requested_whole_table(projects):
    category=projects[2][0]
    q='研究类记录占全部记录的百分比多少？'
    steps=[{'id':'a','op':'count','filters':[{'field_id':category,'op':'eq','value':'研究类'}]},
        {'id':'b','op':'count'},{'id':'c','op':'ratio','inputs':['a','b']}]
    good=execute(projects,q,steps)
    assert good.ok and float(good.value)==pytest.approx(200/3)
    steps[1]['filters']=[{'field_id':category,'op':'eq','value':'研究类'}]
    bad=execute(projects,q,steps)
    assert not bad.ok and '分母' in bad.error


def test_leaf_unit_ignores_parent_version_parentheses():
    assert header_unit('场馆统计（2026版） / 面积（平方米）')=='平方米'
    assert header_unit('预算（万元） / 项目')=='万元'
    assert header_unit('统计（2026版） / 编号')==''


def test_text_citations_only_include_selected_real_evidence(tmp_path):
    evidence=[Evidence('source-one','捐赠不得附带条件。',1,1,'捐赠规定','https://example.edu/gift'),
        Evidence('source-two','借阅期限另行规定。',.5,2,'借阅规定','https://example.edu/loan')]
    class Index:
        def search(self,*a,**k): return evidence
    class Generator:
        def generate(self,prompt,**k):
            return json.dumps({'sentence_ids':['S1']})
    with Repository(tmp_path/'text.db') as repo:
        repo.initialize()
        result=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection).answer('捐赠可以附带条件吗？')
    assert result.answer=='捐赠不得附带条件。'
    assert len(result.references)==1 and 'https://example.edu/gift' in result.references[0]


def test_text_invalid_id_cannot_add_model_source(tmp_path):
    class Index:
        def search(self,*a,**k):return [Evidence('real','凭校园卡入馆。',1,1,'规定','https://example.edu/real')]
    class Generator:
        def generate(self,*a,**k):return json.dumps({'answer':'伪造来源 https://fake.example','evidence_ids':['invented']})
    with Repository(tmp_path/'text.db') as repo:
        repo.initialize()
        result=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection).answer('如何入馆？')
    assert 'fake.example' not in result.answer
    assert not any('fake.example' in r for r in result.references)
    assert '凭校园卡入馆' in result.answer


def test_requested_clause_can_continue_in_adjacent_chunk(tmp_path):
    class Index:
        def search(self,*a,**k):return [Evidence('first','第九条 可放宽要求。\n第 十 条 不得申请的情形：',1,doc,'申请办法','https://example.edu/rule',chunk_id='first')]
    class Generator:
        def generate(self,prompt,**k):
            assert '受到处分' in prompt and '可放宽要求' not in prompt
            return json.dumps({'answer':'受到处分的申请人不得申请。','evidence_ids':['E1']},ensure_ascii=False)
    with Repository(tmp_path/'text.db') as repo:
        repo.initialize()
        doc=repo.add_document(SourceDocument('申请办法','https://example.edu/rule',sha256='clauses'))
        repo.add_chunks([Chunk('first',doc,'第九条 可放宽要求。\n第 十 条 不得申请的情形：','',20),
            Chunk('next',doc,'（一）受到处分的申请人；（二）已取消资格的申请人。\n第十一条 其他程序。','',40)])
        service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection)
        result=service.answer('申请办法第十条规定哪些人不得申请？')
        assert result.references and service.last_diagnostics['text_context']['expanded_chunks']['first']==['first','next']


def test_clause_cross_reference_is_not_a_boundary_or_another_policy(tmp_path):
    evidence=[Evidence('handbook','第七条 国家奖学金评选规定。',1,1,'学生手册','https://example.edu/book'),
        Evidence('budget','第七条 对本办法第三条所述预算，经本人申请，满足（一）项目通过；（二）经费落实，可批准。',.9,2,'预算申请办法','https://example.edu/budget')]
    class Index:
        def search(self,*a,**k):return evidence
    class Generator:
        def generate(self,prompt,**k):
            assert '经费落实' in prompt and '本人申请' in prompt
            assert '国家奖学金' not in prompt
            return json.dumps({'answer':'经本人申请，项目通过且经费落实。','evidence_ids':['E1']},ensure_ascii=False)
    with Repository(tmp_path/'text.db') as repo:
        repo.initialize()
        result=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection).answer('预算申请办法第七条列出了哪些条件？')
        assert len(result.references)==1 and 'https://example.edu/budget' in result.references[0]


def test_excerpt_fallback_keeps_all_relevant_table_records(tmp_path):
    class Index:
        def search(self,*a,**k):return [Evidence('rows','设备=探测器 | 编号=A17\n\n设备=探测器 | 编号=B23\n\n设备=探测器 | 编号=C39',1,1,'设备对照表','https://example.edu/devices',table_id=1)]
    class Generator:
        def generate(self,*a,**k):return '格式不合法'
    with Repository(tmp_path/'text.db') as repo:
        repo.initialize()
        result=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection).answer('设备对照表中探测器对应的三个编号是什么？')
        assert all(code in result.answer for code in ['A17','B23','C39'])
        assert len(result.references)==1


def test_valid_citation_id_cannot_authorize_reversing_a_rule(tmp_path):
    class Index:
        def search(self,*a,**k):return [Evidence('real','访客拍摄必须经过同意。',1,1,'访客规则','https://example.edu/photo')]
    class Generator:
        def generate(self,*a,**k):return json.dumps({'answer':'访客可以自行拍摄，无需同意。','evidence_ids':['E1']},ensure_ascii=False)
    with Repository(tmp_path/'text.db') as repo:
        repo.initialize()
        service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection)
        result=service.answer('访客可以自行拍摄吗？')
        assert '必须经过同意' in result.answer and '无需同意' not in result.answer
        assert service.last_diagnostics['text_answer']['excerpt_fallback']


def test_excerpt_cannot_drop_a_negative_prefix(tmp_path):
    class Index:
        def search(self,*a,**k):return [Evidence('real','不允许附带任何条件。',1,1,'规则','https://example.edu/rule')]
    class Generator:
        def generate(self,*a,**k):return json.dumps({'answer':'允许附带任何条件。','evidence_ids':['E1']},ensure_ascii=False)
    with Repository(tmp_path/'text.db') as repo:
        repo.initialize()
        result=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection).answer('可以附带条件吗？')
        assert result.answer.startswith('相关原文：不允许')


@pytest.mark.parametrize('question',['请查询我的学业成绩。','请查看我本人的借阅记录。'])
def test_private_record_request_is_not_answered_by_public_process(tmp_path,question):
    class Index:
        def search(self,*a,**k):return [Evidence('real','学生可以在业务系统查看个人记录。',1,1,'办理指南','https://example.edu/guide')]
    class Generator:
        def generate(self,*a,**k):raise AssertionError('Public documents do not provide the current user record')
    with Repository(tmp_path/'text.db') as repo:
        repo.initialize()
        service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection)
        result=service.answer(question)
        assert '无法查询' in result.answer and not result.references


def test_personal_process_question_can_still_use_public_guidance(tmp_path):
    class Index:
        def search(self,*a,**k):return [Evidence('real','在个人业务系统查看成绩。',1,1,'办理指南','https://example.edu/guide')]
    class Generator:
        def generate(self,*a,**k):return json.dumps({'sentence_ids':['S1']})
    with Repository(tmp_path/'text.db') as repo:
        repo.initialize()
        result=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection).answer('如何查询我的学业成绩？')
        assert result.answer=='在个人业务系统查看成绩。' and result.references
