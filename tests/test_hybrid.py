import numpy as np
import pytest

from campus_rag.index.hybrid import HybridIndex, reciprocal_rank_fusion


class FakeEmbedder:
    def encode_queries(self, queries):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_rrf_and_hybrid_search_without_reranker():
    assert reciprocal_rank_fusion([[(0, 2.0), (1, 1.0)], [(1, 3.0)]])[0][0] == 1
    records = [
        {"id": "a", "text": "招生 计划 人数", "doc_id": 1, "title": "招生", "source_url": "u1", "table_id": None},
        {"id": "b", "text": "图书馆 开放 时间", "doc_id": 2, "title": "图书馆", "source_url": "u2", "table_id": None},
    ]
    index = HybridIndex(records, np.array([[1.0, 0.0], [0.0, 1.0]]), FakeEmbedder())
    results = index.search("招生人数", bm25_top_k=2, dense_top_k=2, final_top_k=1)
    assert results[0].evidence_id == "a"
    assert index.search("图书馆", final_top_k=1, mode="bm25")[0].evidence_id == "b"


@pytest.mark.parametrize('topic',['招生章程','综合评价','教学质量报告','预算管理'])
def test_topic_words_do_not_exclude_documents_with_other_titles(topic):
    records = [
        {'id':'a','text':topic+' 具体申请办法','doc_id':1,'title':'公共事务办事指南','source_url':'u1','table_id':None},
        {'id':'b','text':'其他说明','doc_id':2,'title':topic,'source_url':'u2','table_id':None}]
    result = HybridIndex(records).search(topic+'具体申请办法',mode='bm25')
    assert 'a' in {e.evidence_id for e in result}


def test_quoted_sources_are_generic_and_multiple_sources_form_a_union():
    records = [{'id':str(i),'text':'补助 标准 规则','doc_id':i,'title':name,'source_url':'u','table_id':None}
               for i,name in enumerate(['甲项目补助办法','乙项目补助办法','其他标准'],1)]
    index = HybridIndex(records)
    assert {e.doc_id for e in index.search('按《甲项目补助办法》，补助标准是什么？',mode='bm25')} == {1}
    assert {e.doc_id for e in index.search('比较《甲项目补助办法》和《乙项目补助办法》的补助标准',mode='bm25')} == {1,2}
    # An unresolved source may be a section inside a document. Do not silently
    # narrow a multi-source request to whichever title happened to resolve.
    assert {e.doc_id for e in index.search('《甲项目补助办法》和《丙规则》的补助标准',mode='bm25')} == {1,2,3}


def test_outdated_title_does_not_hide_an_eligible_document_section():
    records=[{'id':'old','text':'补助 标准','doc_id':1,'title':'2025年补助办法','source_url':'u','table_id':None},
             {'id':'current','text':'《补助办法》补助 标准','doc_id':2,'title':'2026年事务手册','source_url':'u','table_id':None}]
    result=HybridIndex(records).search('2026年《补助办法》的补助标准',mode='bm25')
    assert [e.doc_id for e in result]==[2]
