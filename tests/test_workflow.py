import json
from types import SimpleNamespace

import pytest

from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.rag.generator import ContextBudgetError
from campus_rag.rag.service import RAGService
from campus_rag.types import Evidence


@pytest.fixture
def service_factory(tmp_path, monkeypatch):
    visited = []
    for name in ('retrieve', 'route', 'text', 'prepare_numeric', 'plan', 'execute'):
        method = getattr(RAGService, '_'+name+'_node')

        def record(self, state, method=method, name=name):
            visited.append(name)
            return method(self, state)

        monkeypatch.setattr(RAGService, '_'+name+'_node', record)

    with Repository(tmp_path/'workflow.db') as repository:
        repository.initialize()

        def create(outputs, evidence=None):
            responses = iter(outputs)
            calls = []

            def generate(prompt, **kwargs):
                calls.append(prompt)
                response = next(responses)
                if isinstance(response, Exception):
                    raise response
                return response

            index = SimpleNamespace(search=lambda *args, **kwargs: evidence if evidence is not None else [
                Evidence('e1', '学费为5000元/年。', 1, 1, '学费规定', 'https://example.edu/fees')])
            settings = Settings(tmp_path/'workflow.db', tmp_path, tmp_path, tmp_path, tmp_path)
            service = RAGService(settings, index, SimpleNamespace(generate=generate), repository.connection)
            return service, visited, calls

        yield create


def test_text_workflow_and_request_reset(service_factory):
    service, visited, calls = service_factory(['{"sentence_ids":["S1"]}']*2)
    for _ in range(2):
        result = service.answer('学费规定是什么？')
        assert result.answer == '学费为5000元/年。'
        assert result.references == ['学费规定｜https://example.edu/fees']
        assert service.last_diagnostics['attempts'] == []
    assert visited == ['retrieve', 'route', 'text']*2
    assert len(calls) == 2


def test_no_evidence_stops_before_model_call(service_factory):
    service, visited, calls = service_factory([], evidence=[])
    result = service.answer('学费是多少？')
    assert not result.references and '未找到' in result.answer
    assert visited == ['retrieve'] and not calls


def test_invalid_plan_stops_after_one_repair(service_factory):
    service, visited, calls = service_factory(['invalid', 'invalid'])
    result = service.answer('学费是多少？')
    assert not result.references and '计划校验失败' in result.answer
    assert visited == ['retrieve', 'route', 'prepare_numeric', 'plan', 'plan']
    assert len(calls) == len(service.last_diagnostics['attempts']) == 2


@pytest.mark.parametrize('repair', [False, True])
def test_numeric_workflow_executes_grounded_plan(service_factory, repair):
    plan = {'table_id':None, 'steps':[{'id':'fee', 'op':'lookup', 'fact_id':'f1'}], 'output':'fee'}
    invalid = {'table_id':None, 'steps':[{'id':'fee', 'op':'lookup', 'literal':'5001', 'unit':'元'}]}
    outputs = ([json.dumps(invalid)] if repair else [])+[json.dumps(plan)]
    service, visited, calls = service_factory(outputs)
    result = service.answer('学费是多少？')
    assert result.calculation.ok and result.calculation.value == 5000
    assert result.references == ['学费规定｜https://example.edu/fees']
    assert visited == ['retrieve', 'route', 'prepare_numeric']+['plan', 'execute']*(2 if repair else 1)
    assert len(calls) == len(service.last_diagnostics['attempts']) == (2 if repair else 1)


def test_graph_context_error_keeps_public_refusal(service_factory):
    service, visited, calls = service_factory([ContextBudgetError('budget')])
    result = service.answer('学费规定是什么？')
    assert '上下文预算' in result.answer and not result.references
    assert result.numeric_verification == {'verified':False, 'reason':'budget'}
    assert visited == ['retrieve', 'route', 'text'] and len(calls) == 1
