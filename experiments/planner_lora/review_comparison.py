"""Summarize exposed paired regressions with numeric values, units and sources."""
import argparse
from decimal import Decimal
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory
    report = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['network_attempts'] == 0
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    gold = {case['id']: case for case in json.loads((Path(__file__).with_name('fixtures') / 'dev36_gold.json').read_text(encoding='utf-8'))['cases']}
    rows = {(row['id'], row['mode']): row for row in report['results']}
    judged, pending = {}, []
    for case in protocol['cases']:
        for mode in ['base', 'adapter']:
            row = rows[(case['id'], mode)]
            answer = row['result']
            calculation = answer.get('calculation') or {}
            label = case if case['kind'] == 'new_numeric' else gold.get(case['id'], {})
            numeric = case['kind'] == 'new_numeric' or label.get('category') == 'calculation'
            unchanged_release = case['kind'] != 'new_numeric' and all(
                answer.get(key) == case.get(key) for key in ['answer', 'references'])
            if unchanged_release:
                # Current-corpus review may legitimately cite the same policy
                # inside its handbook, rather than an old gold document ID.
                correct = case['correct']
            elif numeric:
                sources = {detail['doc_id'] for detail in calculation.get('details', []) if detail.get('doc_id') is not None}
                required = {501} if case['kind'] == 'new_numeric' else {source['doc_id'] for source in label['sources']}
                correct = bool(calculation.get('ok') and answer['references'] and sources == required
                    and calculation['unit'] == label['unit']
                    and abs(Decimal(calculation['value']) - Decimal(label['expected'])) <= Decimal('0.000001'))
                if case['id'] == 'C01' and correct:
                    correct = '下限' in answer['answer'] or '最低' in answer['answer']
                if case['id'] == 'C08' and correct:
                    operands = calculation['steps'][-1]['operands']
                    correct = [Decimal(item['value']) for item in operands] == [Decimal(6), Decimal(254)]
            elif all(answer.get(key) == case.get(key) for key in ['answer', 'references']):
                correct = case['correct']
            else:
                correct = None
                pending.append({'id': case['id'], 'mode': mode, 'old_answer': case.get('answer'), 'answer': answer['answer']})
            judged[(case['id'], mode)] = {'correct': correct, 'numeric': numeric,
                'wrong_numeric': numeric and bool(calculation.get('ok')) and not correct}
    summary = {'groups': {}, 'gains': [], 'regressions': [], 'pending_review': pending,
        'method': 'Exposed regression only. Unchanged answers and references inherit the current-corpus frozen review; changed numeric cases check value, unit, sources and key qualifiers. Inspect changed plans before acceptance.',
        'adapter_enabled': False}
    for kind in sorted({case['kind'] for case in protocol['cases']}):
        cases = [case for case in protocol['cases'] if case['kind'] == kind]
        summary['groups'][kind] = {mode: {'correct': sum(judged[(case['id'], mode)]['correct'] is True for case in cases),
            'total': len(cases)} for mode in ['base', 'adapter']}
    for case in protocol['cases']:
        before, after = [judged[(case['id'], mode)]['correct'] for mode in ['base', 'adapter']]
        if before is False and after is True:
            summary['gains'].append(case['id'])
        if before is True and after is False:
            summary['regressions'].append(case['id'])
    summary['wrong_numeric'] = {mode: [case['id'] for case in protocol['cases'] if judged[(case['id'], mode)]['wrong_numeric']]
                                for mode in ['base', 'adapter']}
    summary['adapter_used_cases'] = [row['id'] for row in report['results'] if row['mode'] == 'adapter' and row['adapter_calls']]
    summary['passed'] = not pending and not summary['regressions'] and bool(summary['gains']) and set(summary['wrong_numeric']['adapter']) <= set(summary['wrong_numeric']['base'])
    (directory / 'comparison_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
