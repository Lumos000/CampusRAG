import json

import pytest

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.rag.service import RAGService
from campus_rag.types import Evidence, Chunk, SourceDocument


def answer(tmp_path, text, question, ids):
    class Index:
        def search(self, *args, **kwargs):
            return [Evidence('real',text,1,1,'公开规则','https://example.edu/rules')]
    class Generator:
        def generate(self, prompt, **kwargs):
            return json.dumps({'sentence_ids':ids})
    with Repository(tmp_path/'rules.db') as repo:
        repo.initialize()
        service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection)
        result=service.answer(question)
        return result,service.last_diagnostics


def test_select_original_sentence_without_unrelated_extension(tmp_path):
    result,diag=answer(tmp_path,'访客须登记后入内。拍摄必须事先获准。借阅规则另行公布。',
        '拍摄需要什么条件？',['S2'])
    assert result.answer=='拍摄必须事先获准。'
    assert not diag['text_answer']['excerpt_fallback']
    assert len(result.references)==1


@pytest.mark.parametrize('text',[
    '外部人员可以申请。\n但涉及保密区域的申请不得批准。',
    '员工可以借用。除损坏未赔偿者外，学生亦可申请。',
    '科研人员：\n经主管审核后可以申请。',
])
def test_conditions_and_exceptions_cannot_be_detached(tmp_path,text):
    result,_=answer(tmp_path,text,'申请条件是什么？',['S1'])
    assert result.answer==text


def test_list_keeps_every_item_and_intro(tmp_path):
    text='下列人员不得申请：\n（一）材料缺失者；（二）尚在停用期者；（三）未完成培训者；（四）违规者。'
    result,_=answer(tmp_path,text,'哪些人员不得申请？请列全。',['S1'])
    assert result.answer==text


def test_selected_ids_are_deduplicated_in_source_order(tmp_path):
    result,_=answer(tmp_path,'入场须登记。拍摄须批准。退场须签退。','入场和退场要求是什么？',['S3','S1','S3'])
    assert result.answer=='入场须登记。\n退场须签退。'


@pytest.mark.parametrize('ids',[['invented'],['S1',3],[], 'S1'])
def test_invalid_or_empty_selection_uses_safe_existing_fallback(tmp_path,ids):
    text='未经批准不得拍摄。'
    result,diag=answer(tmp_path,text,'能否拍摄？',ids)
    assert result.answer=='相关原文：'+text
    assert diag['text_answer']['excerpt_fallback']
    assert len(result.references)==1


def test_partial_table_selection_cannot_omit_other_retrieved_records(tmp_path):
    class Index:
        def search(self,*args,**kwargs):
            return [Evidence('row'+str(i),'设备='+name+' | 编号='+code,1,1,
                '设备对照表','https://example.edu/devices',table_id=1,row_ids=[i])
                for i,(name,code) in enumerate([('光谱仪','A11'),('光谱仪','B22'),
                    ('光谱仪','C33'),('光谱仪','D44'),('光谱仪附件','EXTRA')],1)]
    class Generator:
        def generate(self,*args,**kwargs):
            return json.dumps({'sentence_ids':['S1']})
    with Repository(tmp_path/'table.db') as repo:
        repo.initialize()
        service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection)
        result=service.answer('“光谱仪”对应的编号是什么？')
        assert all(code in result.answer for code in ['A11','B22','C33','D44'])
        assert 'EXTRA' not in result.answer


def test_selected_chunks_follow_document_order_not_retrieval_rank(tmp_path):
    class Index:
        def search(self,*args,**kwargs):
            return [Evidence('second','申请手续必须经过登记。',1,doc,'规则','https://example.edu/rules',chunk_id='second'),
                Evidence('first','申请手续必须经过核验。',.9,doc,'规则','https://example.edu/rules',chunk_id='first')]
    class Generator:
        def generate(self,*args,**kwargs):return json.dumps({'sentence_ids':['S1','S2']})
    with Repository(tmp_path/'ordered.db') as repo:
        repo.initialize()
        doc=repo.add_document(SourceDocument('规则','https://example.edu/rules',sha256='ordered'))
        repo.add_chunks([Chunk('first',doc,'申请手续必须经过核验。','',8),Chunk('second',doc,'申请手续必须经过登记。','',8)])
        service=RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),Index(),Generator(),repo.connection)
        result=service.answer('申请的条件是什么？')
        assert result.answer=='申请手续必须经过核验。\n申请手续必须经过登记。'


def test_top_level_list_cannot_lose_its_shared_condition(tmp_path):
    text='仅完成培训的正式工作人员可按以下条件申请：\n一、提交完整材料；\n二、获得负责人同意。'
    result,_=answer(tmp_path,text,'申请条件是什么？',['S1'])
    assert result.answer==text


def test_numbered_exception_cannot_be_selected_away(tmp_path):
    text='一、工作人员完成登记后可以借用。\n二、但停用账户不得借用。'
    result,_=answer(tmp_path,text,'工作人员可以借用吗？',['S1'])
    assert result.answer==text


@pytest.mark.parametrize('question', [
    '《青石办法》和《白湖办法》分别要求哪些申请材料？',
    '青石办法与白湖办法的申请材料是什么？',
])
def test_cross_document_materials_reach_sentence_selection(tmp_path,question):
    evidence = [Evidence(f'e{i}', f'申请须提交材料{label}。', 1, i,
        title, f'https://example.edu/{i}') for i, (label, title) in enumerate([
            ('甲', '青石办法'), ('乙', '白湖办法')], 1)]
    class Index:
        def search(self, *args, **kwargs): return evidence
    class Generator:
        def generate(self, prompt, **kwargs):
            assert '材料甲' in prompt and '材料乙' in prompt
            return json.dumps({'sentence_ids': ['S1', 'S2']})
    with Repository(tmp_path/'multi.db') as repo:
        repo.initialize()
        service = RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),
            Index(), Generator(), repo.connection)
        result = service.answer(question)
        assert '材料甲' in result.answer and '材料乙' in result.answer
        assert len(result.references) == 2


def test_separate_requirements_in_one_named_policy_do_not_expand_to_other_sources(tmp_path):
    with Repository(tmp_path/'single_named.db') as repo:
        repo.initialize()
        service = RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),None,None,repo.connection)
        service.last_diagnostics = {}
        evidence = [Evidence('a','申请文件须采用PDF格式且不得加密。',1,1,'青石办法','https://example.edu/a'),
            Evidence('b','另一办法要求提交完整的纸本文件。',.9,2,'白湖办法','https://example.edu/b')]
        selected = service._text_evidence('《青石办法》对文件格式和加密分别有什么要求？',evidence)
        assert {e.doc_id for e in selected} == {1}


def test_generic_portal_title_is_not_a_named_policy_constraint(tmp_path):
    with Repository(tmp_path/'portal.db') as repo:
        repo.initialize()
        service = RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),None,None,repo.connection)
        service.last_diagnostics = {}
        evidence = [Evidence('a','委托他人办理时须提交签名委托书和双方证件复印件。',1,1,
            '本科成绩办理规定（修订）','https://example.edu/a'),
            Evidence('b','请登录栏目入口提交线上申请。',.9,2,'课程申请','https://example.edu/b')]
        selected = service._text_evidence('按本科成绩办理规定，课程申请需要什么委托材料？',evidence)
        assert {e.doc_id for e in selected} == {1}


def test_requested_clauses_are_not_limited_to_three(tmp_path):
    text = '\n'.join(f'第{i}条 申请须提交材料{i}。' for i in range(1, 6))
    result, _ = answer(tmp_path, text, '第1条、第2条、第3条、第4条、第5条分别要求什么材料？',
        ['S1', 'S2', 'S3', 'S4', 'S5'])
    assert all(f'材料{i}' in result.answer for i in range(1, 6))


def test_identical_clauses_from_different_sources_keep_both_identities(tmp_path):
    with Repository(tmp_path/'identical.db') as repo:
        repo.initialize()
        service = RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),None,None,repo.connection)
        service.last_diagnostics = {}
        evidence = [Evidence(str(i), '申请须提交完整的登记材料。', 1, i,
            title, f'https://example.edu/{i}') for i,title in enumerate(['青石办法','白湖办法'],1)]
        selected = service._text_evidence('《青石办法》和《白湖办法》分别要求什么？', evidence)
        assert {e.doc_id for e in selected} == {1, 2}
