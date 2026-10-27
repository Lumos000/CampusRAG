"""Aggregate frozen outputs and explicit per-question semantic reviews."""
import argparse
import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean, median

from evaluate_formal import DEFAULT, MODES, digest, save, verify


def numeric_correct(case, result):
    if case['category'] != 'calculation':
        return None
    calc = result.get('calculation') or {}
    if not calc.get('ok') or calc.get('value') is None:
        return False
    count_units = {'项', '条', '个'}
    actual_unit, expected_unit = calc.get('unit'), case['unit']
    if actual_unit != expected_unit and not ({actual_unit, expected_unit} <= count_units):
        return False
    try:
        actual = Decimal(str(calc['value']))
        return actual.is_finite() and abs(actual - Decimal(case['expected'])) <= Decimal(case.get('tolerance', '0.000001'))
    except (InvalidOperation, ValueError):
        return False


def summarize(cases, results, reviews):
    labels = {case['id']: case for case in cases}
    if set(labels) != {r['id'] for r in results} or len(results) != len(labels):
        raise ValueError('Results must contain exactly one record for every frozen question')
    if set(labels) != set(reviews):
        raise ValueError('A semantic review is required for every question')
    counts, by_category, passed, citation_total, citation_relevant = Counter(), {}, 0, 0, 0
    rows = []
    for row in results:
        case, review = labels[row['id']], reviews[row['id']]
        verdict = review['verdict']
        if verdict not in {'correct', 'partial', 'incorrect', 'refusal', 'error'}:
            raise ValueError('Invalid verdict: ' + row['id'])
        if not review.get('reason') or len(review['points_met']) != len(case['points']) or any(type(v) is not bool for v in review['points_met']):
            raise ValueError('Missing reason or point review: ' + row['id'])
        if verdict == 'correct' and not all(review['points_met']):
            raise ValueError('Correct verdict has missing core points: ' + row['id'])
        result = row.get('result', {})
        references = result.get('references', [])
        judgments = review.get('citations', [])
        if [j['reference'] for j in judgments] != references or any(type(j['relevant']) is not bool or not j.get('reason') for j in judgments):
            raise ValueError('Every actual reference must be individually reviewed: ' + row['id'])
        correct = verdict == 'correct'
        if case.get('refuse'):
            correct = verdict == 'refusal' and review.get('refusal_correct') is True and all(review['points_met'])
        passed += int(correct)
        counts[verdict] += 1
        category = by_category.setdefault(case['category'], {'correct': 0, 'total': 0})
        category['correct'] += int(correct)
        category['total'] += 1
        if not case.get('refuse'):
            citation_total += len(judgments)
            citation_relevant += sum(j['relevant'] for j in judgments)
        rows.append({'id': case['id'], 'category': case['category'], 'correct': correct,
            'verdict': verdict, 'numeric_correct': numeric_correct(case, result),
            'citation_complete': bool(review.get('citation_complete')) and bool(references),
            'reason': review['reason'], 'failure_stage': review.get('failure_stage', '')})
    for category in by_category.values():
        category['accuracy'] = category['correct'] / category['total']
    answerable = [row for row in rows if row['category'] != 'refusal']
    numeric = [row for row in rows if row['category'] == 'calculation']
    actual_refusals = [row for row in rows if row['verdict'] == 'refusal']
    retrieval = {}
    for mode in MODES:
        metrics = [row['retrieval'][mode]['metrics'] for row in results if not labels[row['id']].get('refuse')]
        retrieval[mode] = {key: mean(m[key] for m in metrics if m[key] is not None) for key in metrics[0]}
    return {'overall': {'correct': passed, 'total': len(cases), 'accuracy': passed / len(cases)},
        'answerable_accuracy': sum(row['correct'] for row in answerable) / len(answerable),
        'categories': by_category, 'verdict_counts': dict(counts),
        'numeric_execution_accuracy': sum(row['numeric_correct'] for row in numeric) / len(numeric),
        'refusal_accuracy': by_category['refusal']['accuracy'],
        'false_refusal_rate': sum(row['verdict'] == 'refusal' for row in answerable) / len(answerable),
        'refusal_precision': sum(row['correct'] for row in actual_refusals) / len(actual_refusals) if actual_refusals else None,
        'citation_completeness': sum(row['citation_complete'] for row in answerable) / len(answerable),
        'citation_precision': citation_relevant / citation_total if citation_total else None,
        'citation_counts': {'relevant': citation_relevant, 'total': citation_total},
        'retrieval': retrieval, 'cases': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    directory = args.directory
    manifest = verify(directory)
    gold = json.loads((directory / 'gold.json').read_text(encoding='utf-8'))
    run = directory / 'run1'
    raw = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    if raw['manifest_sha256'] != digest(directory / 'manifest.json'):
        raise SystemExit('Results and frozen manifest differ')
    reviewed = json.loads((run / 'reviews.json').read_text(encoding='utf-8'))
    summary = summarize(gold['cases'], raw['results'], reviewed['cases'])
    durations = [r['service_seconds_estimate'] for r in raw['results']]
    summary['baseline_commit'] = manifest['baseline_commit']
    summary['manifest_sha256'] = raw['manifest_sha256']
    summary['reviewer'] = reviewed['reviewer']
    summary['artifact_hashes'] = {'gold_sha256': digest(directory / 'gold.json'),
        'results_sha256': digest(run / 'results.json'), 'reviews_sha256': digest(run / 'reviews.json'),
        'scoring_script_sha256': digest(Path(__file__))}
    summary['annotation_caveats'] = reviewed.get('annotation_caveats', {})
    refusal_cases = [row for row in raw['results'] if next(c for c in gold['cases'] if c['id'] == row['id']).get('refuse')]
    summary['safe_refusal_count'] = sum(reviewed['cases'][row['id']].get('safe_abstention') is True for row in refusal_cases)
    summary['safe_refusal_total'] = len(refusal_cases)
    summary['model_init_seconds'] = raw['model_init_seconds']
    summary['device'] = raw.get('device')
    summary['network_attempt_count'] = raw['network_attempt_count']
    summary['performance'] = {'mean_service_seconds_estimate': mean(durations), 'median_service_seconds_estimate': median(durations),
        'max_service_seconds_estimate': max(durations),
        'process_peak_working_set_bytes': max((r.get('memory') or {}).get('process_peak_working_set_bytes', 0) for r in raw['results']),
        'gpu_peak_allocated_bytes': max(r.get('gpu_peak_allocated_bytes', 0) for r in raw['results']),
        'gpu_peak_reserved_bytes': max(r.get('gpu_peak_reserved_bytes', 0) for r in raw['results'])}
    save(run / 'summary.json', summary)
    with (run / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary['cases'][0]))
        writer.writeheader()
        writer.writerows(summary['cases'])
    print(json.dumps({k: v for k, v in summary.items() if k not in {'cases', 'retrieval'}}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
