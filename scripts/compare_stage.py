"""Reuse a semantic judgment only when the answer and actual citations are identical."""
import argparse
import json
from pathlib import Path

from evaluate_formal import DEFAULT, ROOT, digest, save, verify
from score_formal import numeric_correct, summarize


def compare(directory):
    verify(directory)
    cases = json.loads((directory / 'gold.json').read_text(encoding='utf-8'))['cases']
    original = json.loads((DEFAULT / 'run1/results.json').read_text(encoding='utf-8'))
    current = json.loads((directory / 'run1/results.json').read_text(encoding='utf-8'))
    if current['manifest_sha256'] != digest(directory / 'manifest.json'):
        raise ValueError('Stage results do not match snapshot')
    before = {r['id']:r for r in original['results']}
    after = {r['id']:r for r in current['results']}
    if set(after)!={c['id'] for c in cases}:
        raise ValueError('All diagnostic cases must finish before comparison')
    old_reviews = json.loads((DEFAULT / 'run1/reviews.json').read_text(encoding='utf-8'))
    reviews, rows, pending = {}, [], []
    for case in cases:
        key = case['id']
        old, new = before[key].get('result',{}), after[key].get('result',{})
        same = not before[key].get('error') and not after[key].get('error') and all(
            old.get(field)==new.get(field) for field in ['answer','references','calculation'])
        if same:
            reviews[key] = old_reviews['cases'][key]
        else:
            pending.append(key)
        rows.append({'id':key,'identical_answer_citations_calculation':same,
            'before_answer':old.get('answer'), 'after_answer':new.get('answer'),
            'before_numeric_correct':numeric_correct(case,old),
            'after_numeric_correct':numeric_correct(case,new),
            'before_verdict':old_reviews['cases'][key]['verdict'],
            'after_verdict':reviews.get(key,{}).get('verdict','pending_semantic_review')})
    save(directory / 'comparison.json', {'cases':rows,'pending_semantic_review':pending,
        'artifact_hashes':{'script':digest(Path(__file__)),
            'baseline_results':digest(DEFAULT / 'run1/results.json'),
            'baseline_reviews':digest(DEFAULT / 'run1/reviews.json'),
            'stage_results':digest(directory / 'run1/results.json')},
        'rule':'Reuse only identical answer, citations and calculation; execution is not semantic correctness.'})
    if not pending:
        save(directory / 'run1/reviews.json', {**old_reviews,'cases':reviews,
            'method':'Exact answer/citations/calculation equality with reviewed baseline; no new semantic inference.'})
        save(directory / 'run1/summary.json', summarize(cases,current['results'],reviews))
    print(json.dumps({'compared':len(rows),'unchanged':len(reviews),'pending':pending},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, required=True)
    compare(parser.parse_args().directory)
