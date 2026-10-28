import json

import numpy as np
import pytest
from bs4 import BeautifulSoup

from campus_rag.index.hybrid import HybridIndex
from campus_rag.ingest.navigation import navigation_values, navigation_chunk
from campus_rag.tools.temporal import question_periods, matches_period


def test_academic_year_precedes_publication_date():
    assert question_periods('根据2023—2024学年报告') == ['2023-2024']
    assert matches_period('2023—2024学年本科教学质量报告', '2025-12-29', ['2023-2024'])
    assert not matches_period('2024—2025学年报告', '2025-12-29', ['2023-2024'])
    assert not matches_period('2025招生章程', '2026-01-01', ['2026'])
    assert not matches_period('收费标准', None, ['2026'])


def test_filters_before_topk_and_no_wrong_year_fallback():
    rows = [{'id': str(year), 'text': '招生计划 物理', 'doc_id': year, 'title': f'{year}招生简章',
             'source_url': 'local'} for year in [2023,2024,2026]]
    index = HybridIndex(rows)
    assert [r.doc_id for r in index.search('2026招生计划', final_top_k=1)] == [2026]
    assert index.search('2038招生计划') == []


def test_navigation_not_confused_with_policy_or_numeric_table():
    labels = ['入馆须知','部门设置','服务流程','馆藏分布','借阅规则','电子资源']
    assert navigation_values(labels)
    assert not navigation_values(['专业','人数','通信工程','20'])
    assert not navigation_values(['本科生','借阅20本','研究生','借阅30本'])
    text = '\n'.join('列1=' + v for v in labels)
    assert navigation_chunk(text)
    rows = [{'id':'a','text':text,'doc_id':1,'title':'新读者向导','source_url':'u','table_id':7}]
    assert HybridIndex(rows).search('借阅规则') == []


def test_jsonl_loader_keeps_unicode_newline(tmp_path, monkeypatch):
    import campus_rag.index.hybrid as hybrid
    row = {'id':'a','text':'正文\u0085继续正文','doc_id':1,'title':'标题','source_url':'u'}
    (tmp_path/'records.jsonl').write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf-8')
    np.save(tmp_path/'vectors.npy', np.ones((1,512),dtype=np.float32))
    monkeypatch.setattr(hybrid,'LocalEmbedder', lambda _: None)
    index = HybridIndex.load(tmp_path, tmp_path/'model')
    assert len(index.records) == 1
    assert index.records[0]['text'] == row['text']


@pytest.mark.parametrize('data', [
    {'filters':['首选物理']}, {'calculation':{'operands':['计划数']}},
    {'calculation':{'multiplier':'invalid'}}, {'calculation':{'operation':[]}},
])
def test_invalid_model_plan_rejected_as_validation_error(data):
    from campus_rag.rag.planner import parse_plan
    with pytest.raises(ValueError):
        parse_plan(json.dumps(data))


def test_database_filters_year_and_navigation_on_existing_tables(tmp_path):
    from campus_rag.database import Repository
    from campus_rag.database.query import StructuredQueryEngine
    from campus_rag.types import SourceDocument, ParsedTable, QueryPlan
    from campus_rag.tools.numbers import normalize_cell
    with Repository(tmp_path/'test.db') as repo:
        repo.initialize()
        old = repo.add_document(SourceDocument('2024招生简章','old',content_type='html',sha256='1'))
        current = repo.add_document(SourceDocument('2026招生简章','current',content_type='html',sha256='2'))
        table = ParsedTable('计划',['专业','计划'],[['计算机','10']])
        normalized = [[normalize_cell(v) for v in row] for row in table.rows]
        tid_old = repo.add_table(old,table,normalized)
        tid_current = repo.add_table(current,table,normalized)
        menu = ParsedTable('导航',['入口1','入口2'],[['入馆须知','部门设置'],['服务流程','馆藏分布'],['借阅规则','电子资源']])
        tid_menu = repo.add_table(current,menu,[[normalize_cell(v) for v in row] for row in menu.rows])
        engine = StructuredQueryEngine(repo.connection)
        plan = QueryPlan(time_range=['2026'])
        assert [r['id'] for r in engine._candidate_tables(plan,[tid_old,tid_current,tid_menu])] == [tid_current]
        assert engine._candidate_tables(plan,[]) == []
