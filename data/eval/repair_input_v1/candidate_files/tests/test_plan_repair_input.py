import json

import pytest

from campus_rag.rag.planner import QueryPlanner


class Generator:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return '{"steps":[{"id":"result","op":"lookup","fact_id":"f11"}]}'


@pytest.mark.parametrize('as_string', [True, False])
def test_repair_plan_is_an_object_not_a_json_encoded_string(as_string):
    previous = {'steps': [{'id': 'left', 'op': 'lookup', 'fact_id': 'f11'},
        {'id': 'result', 'op': 'multiply', 'inputs': ['left', 'right']}]}
    raw = json.dumps(previous, ensure_ascii=False)
    generator = Generator()
    planner = QueryPlanner(generator)
    planner.step_plan('青石预算是多少元？', json.dumps({'facts': [
        {'fact_id': 'f11', 'unit': '元', 'subject': '青石'}]}),
        'right未定义', raw if as_string else previous)
    encoded = generator.calls[0].split('上次完整计划：', 1)[1]
    assert encoded.startswith(raw)
    assert len(generator.calls) == 1


def test_initial_plan_does_not_receive_repair_material():
    generator = Generator()
    QueryPlanner(generator).step_plan('青石预算是多少元？', json.dumps({'facts': [
        {'fact_id': 'f11', 'unit': '元', 'subject': '青石'}]}))
    assert '上次完整计划：' not in generator.calls[0]
    assert len(generator.calls) == 1
