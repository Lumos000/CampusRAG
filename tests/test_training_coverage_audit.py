import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('training_coverage', Path(__file__).resolve().parents[1] / 'scripts/audit_planner_training.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_distractor_units_do_not_count_as_supervised_coverage():
    row = {'context': {'facts': [{'fact_id': 'credit', 'unit': '学分'}, {'fact_id': 'distractor', 'unit': '学时'}]},
           'plan': {'steps': [{'id': 'result', 'op': 'lookup', 'fact_id': 'credit'}]}}
    result = audit.summarize([row], ['学分', '学时'])
    assert result['supervised_unit_examples'] == {'学分': 1}
    assert result['missing_required_units'] == ['学时'] and not result['passed']


def test_composed_retention_requires_multiply_connected_to_subtraction():
    steps = [{'id': 'rate', 'op': 'lookup', 'literal': '2', 'unit': '学时'},
             {'id': 'weeks', 'op': 'lookup', 'literal': '3', 'unit': '周'},
             {'id': 'used', 'op': 'multiply', 'inputs': ['rate', 'weeks']},
             {'id': 'cap', 'op': 'lookup', 'literal': '10', 'unit': '学时'},
             {'id': 'result', 'op': 'subtract', 'inputs': ['cap', 'used']}]
    result = audit.summarize([{'plan': {'steps': steps}}], ['学时'], True)
    assert result['passed'] and result['multiply_then_subtract_examples'] == 1
    steps[-1]['inputs'] = ['cap', 'rate']
    assert not audit.summarize([{'plan': {'steps': steps}}], ['学时'], True)['passed']


def test_refusal_labels_do_not_invent_unit_coverage():
    result = audit.summarize([{'unit': '学分', 'plan': {'steps': []}}], ['学分'])
    assert not result['passed'] and result['missing_required_units'] == ['学分']


def test_inline_question_operands_count_as_supervised_units():
    row = {'plan': {'steps': [{'id': 'result', 'op': 'add', 'inputs': [
        {'literal': '2', 'unit': '学分'}, {'literal': '3', 'unit': '学分'}]}]}}
    result = audit.summarize([row], ['学分'])
    assert result['passed'] and result['supervised_unit_examples'] == {'学分': 1}
