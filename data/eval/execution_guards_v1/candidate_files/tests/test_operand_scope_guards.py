import json

import pytest

from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence, ParsedTable, SourceDocument


@pytest.fixture
def engine(tmp_path):
    with Repository(tmp_path/'scope.db') as repo:
        repo.initialize()
        yield repo, StructuredQueryEngine(repo.connection)


@pytest.mark.parametrize('crossed', [False, True])
@pytest.mark.parametrize('reverse', [False, True])
def test_each_headcount_uses_its_own_rate(engine, crossed, reverse):
    _, query = engine
    facts = NumericEngine.text_facts([Evidence('policy',
        '技术组：4名，137元/人；研习组：6名，89元/人。',
        1, 1, '公开补助办法', 'https://example.edu/policy')])
    ids = {(f['subject'], f['unit']): fid for fid, f in facts.items()}
    steps = []
    for i, subject in enumerate(['技术组', '研习组']):
        rate_subject = ['研习组', '技术组'][i] if crossed else subject
        inputs = [f'n{i}', f'r{i}']
        steps.extend([{'id': inputs[0], 'op': 'lookup', 'fact_id': ids[subject, '人']},
            {'id': inputs[1], 'op': 'lookup', 'fact_id': ids[rate_subject, '元/人']},
            {'id': f't{i}', 'op': 'multiply', 'inputs': inputs[::-1] if reverse else inputs}])
    steps.append({'id': 'result', 'op': 'add', 'inputs': ['t0', 't1']})
    result = query.execute_steps(parse_plan(json.dumps({'steps': steps})), [],
        '技术组和研习组按各自人数和人均标准计算，预算合计多少元？', facts)
    assert result.ok is not crossed
    if not crossed:
        assert result.value == 1082 and result.unit == '元'
    else:
        assert result.repairable and '配对' in result.error


@pytest.mark.parametrize('same_source', [False, True])
def test_headcount_rate_pair_keeps_policy_scope(engine, same_source):
    _, query = engine
    facts = NumericEngine.text_facts([
        Evidence('a', '技术组人数为4人。', 1, 1, '青石补助办法', 'https://example.edu/a'),
        Evidence('b', '技术组标准为137元/人。', 1, 1 if same_source else 2,
            '青石补助办法' if same_source else '白湖补助办法', 'https://example.edu/a' if same_source else 'https://example.edu/b')])
    steps = [{'id': f['fact_id'], 'op': 'lookup', 'fact_id': f['fact_id']} for f in facts.values()]
    steps.append({'id': 'result', 'op': 'multiply', 'inputs': list(facts)})
    result = query.execute_steps(parse_plan(json.dumps({'steps': steps})), [],
        '技术组按人数和人均标准计算，预算合计多少元？', facts)
    assert result.ok is same_source


def test_question_headcount_can_use_a_public_rate(engine):
    _, query = engine
    facts = NumericEngine.text_facts([Evidence('a', '技术组标准为137元/人。',
        1, 1, '公开补助办法', 'https://example.edu/a')])
    plan = {'steps': [{'id': 'n', 'op': 'lookup', 'literal': '4', 'unit': '人'},
        {'id': 'rate', 'op': 'lookup', 'fact_id': next(iter(facts))},
        {'id': 'result', 'op': 'multiply', 'inputs': ['n', 'rate']}]}
    result = query.execute_steps(parse_plan(json.dumps(plan)), [],
        '按技术组标准，4人共需多少元？', facts)
    assert result.ok and result.value == 548


@pytest.mark.parametrize('shared', [False, True])
def test_explicit_shared_standard_is_not_an_accidental_cross_pair(engine, shared):
    _, query = engine
    facts = NumericEngine.text_facts([Evidence('a', '技术组人数为4人。研习组标准为89元/人。',
        1, 1, '公开补助办法', 'https://example.edu/a')])
    steps = [{'id': fid, 'op': 'lookup', 'fact_id': fid} for fid in facts]
    steps.append({'id': 'result', 'op': 'multiply', 'inputs': list(facts)})
    question = ('假设技术组按研习组标准计算，预算合计多少元？' if shared else
        '技术组按各自人数和标准计算，预算合计多少元？')
    result = query.execute_steps(parse_plan(json.dumps({'steps': steps})), [], question, facts)
    assert result.ok is shared


def test_shared_body_rate_remains_usable(engine):
    _, query = engine
    facts = NumericEngine.text_facts([Evidence('a', '技术组人数为4人。统一标准为89元/人。',
        1, 1, '公开补助办法', 'https://example.edu/a')])
    steps = [{'id': fid, 'op': 'lookup', 'fact_id': fid} for fid in facts]
    steps.append({'id': 'result', 'op': 'multiply', 'inputs': list(facts)})
    result = query.execute_steps(parse_plan(json.dumps({'steps': steps})), [],
        '技术组按统一标准计算，预算合计多少元？', facts)
    assert result.ok and result.value == 356


@pytest.mark.parametrize('correct_source', [False, True])
def test_single_table_must_match_explicit_policy(engine, correct_source):
    repo, query = engine
    title = '青石补助办法' if correct_source else '白湖补助办法'
    did = repo.add_document(SourceDocument(title, 'https://example.edu/policy', sha256=title))
    table = ParsedTable('补助标准', ['类别', '金额（元）'], [['技术组', '137']])
    tid = repo.add_table(did, table, [[normalize_cell(v) for v in row] for row in table.rows])
    name, amount = [r['id'] for r in repo.connection.execute(
        'SELECT id FROM table_columns WHERE table_id=? ORDER BY column_index', (tid,))]
    plan = {'table_id': tid, 'steps': [{'id': 'result', 'op': 'lookup', 'field_id': amount,
        'filters': [{'field_id': name, 'op': 'eq', 'value': '技术组'}]}]}
    result = query.execute_steps(parse_plan(json.dumps(plan)), [tid],
        '按《青石补助办法》，技术组金额是多少元？')
    assert result.ok is correct_source
    if not correct_source:
        assert result.repairable and '来源' in result.error
