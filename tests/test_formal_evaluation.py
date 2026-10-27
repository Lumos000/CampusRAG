import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def formal(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('evaluate_formal')


def test_hit_is_not_recall_and_required_groups_are_separate(formal):
    case = {'evidence_groups': [{'chunk_ids': ['a', 'b']}, {'chunk_ids': ['c']}]}
    result = formal.retrieval_metrics(['noise', 'a'], case)
    assert result['hit@5'] == 1
    assert result['recall@5'] == pytest.approx(1 / 3)
    assert result['evidence_group_recall@5'] == 0.5
    assert result['all_evidence_hit@5'] == 0
    assert result['rr@20'] == 0.5


def test_duplicate_hits_do_not_inflate_recall(formal):
    case = {'evidence_groups': [{'chunk_ids': ['a', 'b']}]}
    result = formal.retrieval_metrics(['a', 'a', 'a'], case)
    assert result['recall@5'] == 0.5
    assert result['evidence_group_recall@5'] == 1


def test_wrapper_preserves_default_ranking_without_gold(formal):
    class Index:
        def __init__(self):
            self.calls = []

        def search(self, query, bm25, dense, fusion, count, mode, **kwargs):
            self.calls.append((query, bm25, dense, fusion, count, mode, kwargs))
            return [(mode, i) for i in range(count)]

    index = Index()
    wrapper = formal.EvaluationIndex(index)
    result = wrapper.search('原始问题', 40, 40, 25, 8, time_range=['2026'])
    assert result == [('rerank', i) for i in range(8)]
    assert [call[5] for call in index.calls] == formal.MODES
    assert all(call[:5] == ('原始问题', 40, 40, 25, 20) for call in index.calls)


def test_freeze_rejects_changed_input(formal, tmp_path, monkeypatch):
    monkeypatch.setattr(formal, 'ROOT', tmp_path)
    source = tmp_path / 'source.txt'
    source.write_text('frozen', encoding='utf-8')
    formal.save(tmp_path / 'manifest.json', {'files': {'source.txt': formal.digest(source)}})
    formal.verify(tmp_path)
    source.write_text('changed', encoding='utf-8')
    with pytest.raises(SystemExit, match='Frozen file changed'):
        formal.verify(tmp_path)


def test_raw_gold_sheet_preserves_merged_values_and_zero(formal, tmp_path):
    from openpyxl import Workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(['对象', '值'])
    sheet.append(['名称', '数量'])
    sheet.append(['项目甲', 0])
    sheet.append([None, 2])
    sheet.merge_cells('A3:A4')
    path = tmp_path / 'source.xlsx'
    workbook.save(path)
    workbook.close()
    assert formal.raw_sheet(path) == [['项目甲', '0'], ['项目甲', '2']]


def test_numeric_scoring_requires_verified_value_and_unit(formal):
    scoring = importlib.import_module('score_formal')
    case = {'category': 'calculation', 'unit': '项', 'expected': '3'}
    assert scoring.numeric_correct(case, {'calculation': {'ok': True, 'value': '3', 'unit': '条'}})
    assert not scoring.numeric_correct(case, {'calculation': {'ok': True, 'value': '3', 'unit': '元'}})
    assert not scoring.numeric_correct(case, {'calculation': {'ok': True, 'value': 'NaN', 'unit': '项'}})
    assert not scoring.numeric_correct(case, {'answer': '3条'})


def test_semantic_scoring_does_not_count_partial_or_inadequate_refusal(formal):
    scoring = importlib.import_module('score_formal')
    cases = [
        {'id': 'a', 'category': 'fact', 'points': ['point']},
        {'id': 'b', 'category': 'calculation', 'points': ['point'], 'unit': '元', 'expected': '2'},
        {'id': 'r', 'category': 'refusal', 'points': ['reason', 'no guess'], 'refuse': True}]
    retrieval = {mode: {'metrics': {'hit@5': 1, 'rr@20': 1}} for mode in formal.MODES}
    results = [{'id': c['id'], 'result': {'references': []}, 'retrieval': retrieval} for c in cases]
    reviews = {
        'a': {'verdict': 'partial', 'points_met': [False], 'reason': 'missing point', 'citations': []},
        'b': {'verdict': 'correct', 'points_met': [True], 'reason': 'correct text answer', 'citations': []},
        'r': {'verdict': 'refusal', 'points_met': [False, True], 'refusal_correct': False, 'reason': 'wrong explanation', 'citations': []}}
    result = scoring.summarize(cases, results, reviews)
    assert result['overall']['correct'] == 1
    assert result['numeric_execution_accuracy'] == 0
    assert result['refusal_accuracy'] == 0


def test_reference_review_must_cover_every_actual_reference(formal):
    scoring = importlib.import_module('score_formal')
    case = {'id': 'a', 'category': 'fact', 'points': ['point']}
    review = {'a': {'verdict': 'correct', 'points_met': [True], 'reason': 'correct', 'citations': []}}
    with pytest.raises(ValueError, match='Every actual reference'):
        scoring.summarize([case], [{'id': 'a', 'result': {'references': ['source']}}], review)
