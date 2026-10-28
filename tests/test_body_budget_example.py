import json

import pytest

from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import QueryPlanner, operation_hint, parse_plan, step_example
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence, ParsedTable, SourceDocument


@pytest.fixture
def budget(tmp_path):
    with Repository(tmp_path/'budget.db') as repo:
        repo.initialize()
        did = repo.add_document(SourceDocument('青石补助办法', 'https://example.edu/body', sha256='body'))
        other = repo.add_document(SourceDocument('普通奖励名单', 'https://example.edu/list', sha256='table'))
        table = ParsedTable('奖励名单', ['类别', '金额（元）'], [['其他项目', '71']])
        tid = repo.add_table(other, table, [[normalize_cell(v) for v in row] for row in table.rows])
        engine = StructuredQueryEngine(repo.connection)
        schema = engine.schema_context([tid], [])
        yield engine, did, schema


@pytest.mark.parametrize('groups', [1, 2])
def test_body_budget_example_does_not_use_an_unrelated_money_table(budget, groups):
    engine, did, schema = budget
    text = '技术组：4名，137元/人。' + ('研习组：6名，89元/人。' if groups == 2 else '')
    facts = NumericEngine.text_facts([Evidence('body', text, 1, did,
        '青石补助办法', 'https://example.edu/body')])
    payload = {'tables': schema, 'facts': [{k: v for k, v in f.items() if k != 'evidence'} for f in facts.values()]}
    question = '按《青石补助办法》所列名额及每人标准，预算合计多少元？'
    plan = parse_plan(step_example(operation_hint(question), json.dumps(payload), question))
    result = engine.execute_steps(plan, [schema[0]['table_id']], question, facts)
    assert result.ok and result.value == (548 if groups == 1 else 1082)
    assert plan.table_id is None
    assert all('field_id' not in s for s in plan.steps)


def test_body_budget_example_preserves_a_matching_user_quantity(budget):
    engine, did, schema = budget
    facts = NumericEngine.text_facts([Evidence('body', '技术组：4名，137元/人。', 1, did,
        '青石补助办法', 'https://example.edu/body')])
    payload = {'tables': schema, 'facts': [{k: v for k, v in f.items() if k != 'evidence'} for f in facts.values()]}
    question = '按《青石补助办法》列出的4人名额及每人标准，预算合计多少元？'
    raw = step_example(operation_hint(question), json.dumps(payload), question)
    result = engine.execute_steps(parse_plan(raw), [schema[0]['table_id']], question, facts)
    assert result.ok and result.value == 548
    assert any(s.get('literal') == '4' and s.get('unit') == '人' for s in json.loads(raw)['steps'])


def test_competition_example_keeps_all_model_evidence(budget):
    _, did, schema = budget
    facts = NumericEngine.text_facts([Evidence('body', '技术组：4名，137元/人。', 1, did,
        '青石补助办法', 'https://example.edu/body')])
    payload = {'tables': schema, 'facts': [{k: v for k, v in f.items() if k != 'evidence'} for f in facts.values()]}
    class Generator:
        def generate(self, prompt, **kwargs):
            self.prompt = prompt
            return '{"steps":[{"id":"result","op":"lookup","fact_id":"f1"}]}'
    generator = Generator()
    QueryPlanner(generator).step_plan('按《青石补助办法》所列名额及每人标准，预算合计多少元？', json.dumps(payload))
    visible = generator.prompt.split('真实Schema和正文facts：', 1)[1].split('\n问题：', 1)[0]
    assert json.loads(visible) == json.loads(json.dumps(payload))


@pytest.mark.parametrize('missing_rate', [False, True])
def test_incomplete_or_mismatched_headcounts_are_not_filled_in(budget, missing_rate):
    engine, did, schema = budget
    text = '技术组：4名，137元/人。研习组人数为6人。' if missing_rate else '技术组：4名，137元/人。'
    facts = NumericEngine.text_facts([Evidence('body', text, 1, did,
        '青石补助办法', 'https://example.edu/body')])
    payload = {'tables': schema, 'facts': [{k: v for k, v in f.items() if k != 'evidence'} for f in facts.values()]}
    question = ('按《青石补助办法》所列两类名额及每人标准，预算合计多少元？' if missing_rate else
        '按《青石补助办法》列出的6人名额及每人标准，预算合计多少元？')
    plan = parse_plan(step_example(operation_hint(question), json.dumps(payload), question))
    result = engine.execute_steps(plan, [schema[0]['table_id']], question, facts)
    assert not result.ok


def test_a_direct_table_total_keeps_the_table_path(budget):
    engine, did, schema = budget
    tid = schema[0]['table_id']
    engine.connection.execute('UPDATE documents SET title=? WHERE id=(SELECT doc_id FROM tables WHERE id=?)',
        ('青石补助办法', tid))
    schema = engine.schema_context([tid], [], '其他项目')
    facts = NumericEngine.text_facts([Evidence('body', '技术组：4名，137元/人。', 1, did,
        '青石补助办法', 'https://example.edu/body')])
    payload = {'tables': schema, 'facts': [{k: v for k, v in f.items() if k != 'evidence'} for f in facts.values()]}
    question = '按《青石补助办法》，其他项目金额合计多少元？'
    plan = parse_plan(step_example(operation_hint(question), json.dumps(payload), question))
    result = engine.execute_steps(plan, [tid], question, facts)
    assert plan.table_id == tid and result.ok and result.value == 71
