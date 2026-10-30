import json

import pytest

from campus_rag.database import Repository
from campus_rag.config import Settings
from campus_rag.database.normalization import header_unit
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Chunk, Evidence, ParsedTable, SourceDocument


@pytest.mark.parametrize('header,expected', [('学分', '学分'), ('学时', '学时'),
    ('实践教学学分', '学分'), ('课程信息 / 总学时', '学时'),
    ('学时/学分', ''), ('学分绩点', ''), ('学分比例（%）', 'ratio')])
def test_academic_header_units_are_unambiguous(header, expected):
    assert header_unit(header) == expected


@pytest.fixture
def academic_table(tmp_path):
    with Repository(tmp_path / 'academic.db') as repo:
        repo.initialize()
        doc = repo.add_document(SourceDocument('课程安排', 'https://example.edu/curriculum'))
        table = ParsedTable('课程安排', ['课程名称', '学分', '学时'],
                            [['课程甲', '2', '32'], ['课程乙', '3', '48'], ['共计', '5', '80']])
        tid = repo.add_table(doc, table, [[normalize_cell(value) for value in row] for row in table.rows])
        columns = {r['original_name']: r['id'] for r in repo.connection.execute('SELECT * FROM table_columns WHERE table_id=?', (tid,))}
        yield repo, tid, columns


@pytest.mark.parametrize('unit,expected', [('学分', 5), ('学时', 80)])
def test_academic_sum_excludes_total_row(academic_table, unit, expected):
    repo, tid, columns = academic_table
    plan = parse_plan(json.dumps({'table_id': tid, 'steps': [
        {'id': 'result', 'op': 'sum', 'field_id': columns[unit]}]}))
    result = StructuredQueryEngine(repo.connection).execute_steps(plan, [tid], f'课程甲与课程乙的{unit}合计多少{unit}？')
    assert result.ok, result.error
    assert result.value == expected and result.unit == unit
    assert len(result.details) == 2


@pytest.mark.parametrize('unit,completed,expected', [('学分', '2', 3), ('学时', '32', 48)])
def test_academic_question_constants_can_be_subtracted(academic_table, unit, completed, expected):
    repo, tid, columns = academic_table
    plan = parse_plan(json.dumps({'table_id': tid, 'steps': [
        {'id': 'total', 'op': 'sum', 'field_id': columns[unit]},
        {'id': 'done', 'op': 'lookup', 'literal': completed, 'unit': unit},
        {'id': 'result', 'op': 'subtract', 'inputs': ['total', 'done']}]}))
    result = StructuredQueryEngine(repo.connection).execute_steps(
        plan, [tid], f'课程所需{unit}合计后，已完成{completed}{unit}，还差多少{unit}？')
    assert result.ok, result.error
    assert result.value == expected and result.unit == unit


@pytest.mark.parametrize('threshold_unit,valid', [('学分', True), ('学时', False)])
def test_academic_numeric_filters_reject_mixed_units(academic_table, threshold_unit, valid):
    repo, tid, columns = academic_table
    plan = parse_plan(json.dumps({'table_id': tid, 'steps': [{'id': 'result', 'op': 'sum',
        'field_id': columns['学分'], 'filters': [{'field_id': columns['学分'], 'op': 'gt', 'value': '2'}]}]}))
    result = StructuredQueryEngine(repo.connection).execute_steps(
        plan, [tid], f'课程学分大于2{threshold_unit}的记录，学分合计多少学分？')
    assert result.ok == valid, result.error
    if valid:
        assert result.value == 3
    else:
        assert '单位' in result.error


@pytest.mark.parametrize('unit,expected', [('学分', 5), ('学时', 80)])
def test_named_academic_document_supplies_its_missing_table(academic_table, unit, expected):
    repo, tid, columns = academic_table
    doc_id = repo.connection.execute('SELECT doc_id FROM tables WHERE id=?', (tid,)).fetchone()[0]
    repo.add_chunks([Chunk('academic-table', doc_id, '课程甲：2学分、32学时；课程乙：3学分、48学时。', '', 30,
                           kind='table', table_id=tid)])
    body = Evidence('academic-body', '下面列出课程安排。', 1., doc_id, '课程安排', 'https://example.edu/curriculum')
    class Index:
        def search(self, *args, **kwargs):
            return [body]
    class Generator:
        def generate(self, prompt, **kwargs):
            return json.dumps({'table_id': tid, 'steps': [{'id': 'result', 'op': 'sum', 'field_id': columns[unit]}]})
    from pathlib import Path
    root = Path('.')
    service = RAGService(Settings(root, root, root, root, root, numeric_shortcuts=False), Index(), Generator(), repo.connection)
    result = service.answer(f'按《课程安排》的课程表，课程甲与课程乙的{unit}合计多少{unit}？')
    assert result.calculation and result.calculation.ok, result.answer
    assert result.calculation.value == expected
    assert service.last_diagnostics['academic_table_expansions'] == [
        {'doc_id': doc_id, 'table_id': tid, 'chunk_id': 'academic-table'}]


@pytest.mark.parametrize('question', ['课程学分合计多少学分？', '按《其他办法》，课程学分合计多少学分？',
                                    '按《课程安排》，平均学分绩点相差多少？'])
def test_academic_expansion_does_not_guess_document_or_gpa_unit(academic_table, question):
    repo, tid, columns = academic_table
    doc_id = repo.connection.execute('SELECT doc_id FROM tables WHERE id=?', (tid,)).fetchone()[0]
    repo.add_chunks([Chunk('academic-table', doc_id, '课程甲：2学分、32学时。', '', 15, kind='table', table_id=tid)])
    body = Evidence('academic-body', '下面列出课程安排。', 1., doc_id, '课程安排', 'https://example.edu/curriculum')
    class Index:
        def search(self, *args, **kwargs):
            return [body]
    class Generator:
        def generate(self, *args, **kwargs):
            return '{"table_id":null,"steps":[]}'
    from pathlib import Path
    root = Path('.')
    service = RAGService(Settings(root, root, root, root, root, numeric_shortcuts=False), Index(), Generator(), repo.connection)
    service.answer(question)
    assert not service.last_diagnostics.get('academic_table_expansions')


def test_academic_expansion_retains_both_requested_documents(academic_table):
    repo, tid, _ = academic_table
    first_id = repo.connection.execute('SELECT doc_id FROM tables WHERE id=?', (tid,)).fetchone()[0]
    other_id = repo.add_document(SourceDocument('其他课程', 'https://example.edu/other-curriculum'))
    table = ParsedTable('其他课程', ['课程名称', '学分'], [['课程丙', '4']])
    other_tid = repo.add_table(other_id, table, [[normalize_cell(value) for value in row] for row in table.rows])
    repo.add_chunks([Chunk('first-table', first_id, '课程安排表', '', 8, kind='table', table_id=tid),
                     Chunk('other-table', other_id, '其他课程表', '', 8, kind='table', table_id=other_tid)])
    original = [Evidence('first-body', '课程安排正文。', 1., first_id, '课程安排', 'https://example.edu/curriculum'),
                Evidence('other-body', '其他课程正文。', 1., other_id, '其他课程', 'https://example.edu/other-curriculum')]
    from pathlib import Path
    root = Path('.')
    service = RAGService(Settings(root, root, root, root, root), None, None, repo.connection)
    service.last_diagnostics = {}
    expanded = service._academic_table_evidence('比较《课程安排》和《其他课程》的学分。', original)
    assert expanded[:2] == original
    assert {item.table_id for item in expanded if item.table_id is not None} == {tid, other_tid}
