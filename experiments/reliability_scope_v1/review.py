"""Development-only semantic reviews; labels never enter service inputs."""
import argparse,copy,csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_formal import save
from score_formal import summarize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args();p=args.directory;b=args.baseline
    changes=[];comparisons=[];acceptance={}
    notes=json.loads((p/'manual_changes.json').read_text(encoding='utf-8')) if args.finalize else {}
    for group in ['dev36','exposed_v2','old_campus']:
        d=p/group;old=b/group
        report=json.loads((d/'run1/results.json').read_text(encoding='utf-8'))
        assert report.get('finished_at') and report['network_attempt_count']==0
        gold=json.loads((d/'gold.json').read_text(encoding='utf-8'))['cases'];labels={r['id']:r for r in gold}
        before={r['id']:r['result'] for r in json.loads((old/'run1/results.json').read_text(encoding='utf-8'))['results']}
        original=json.loads((old/'run1/reviews.json').read_text(encoding='utf-8'))['cases'];reviews=copy.deepcopy(original)
        unsafe=[];group_rows=[]
        for r in report['results']:
            key=r['id'];a=r['result'];same=all(a.get(k)==before[key].get(k) for k in ['answer','references'])
            if not same:
                changes.append({'id':key,'group':group,'question':r['question'],'gold':labels[key],
                    'before':before[key]['answer'],'after':a['answer'],'calculation':a.get('calculation'),
                    'plan':r.get('plan'),'references':a['references']})
                if args.finalize:
                    n=notes[key];passed=n['correct'];assert n['reason']
                    assert len(n['citation_relevant'])==len(a['references'])
                    if not passed and (a.get('calculation') or {}).get('ok'):unsafe.append(key)
                    reviews[key].update(verdict=n.get('verdict','correct' if passed else 'refusal' if not a['references'] else 'incorrect'),
                        points_met=[passed]*len(labels[key]['points']),citation_complete=passed and bool(a['references']),
                        citations=[{'reference':ref,'relevant':relevant,'reason':n['reason']} for ref,relevant in zip(a['references'],n['citation_relevant'])],
                        reason=n['reason'],failure_stage='' if passed else n.get('failure_stage','plan_generation'))
            record={'group':group,'id':key,'question':r['question'],'baseline_correct':all(original[key]['points_met']),
                'candidate_correct':all(reviews[key]['points_met']),'baseline_answer':before[key]['answer'],
                'candidate_answer':a['answer'],'reason':reviews[key]['reason']}
            comparisons.append(record);group_rows.append(record)
        if args.finalize:
            save(d/'run1/reviews.json',{'reviewer':'Codex; unchanged answers/references inherit frozen source reviews. Every changed answer receives explicit operand/source/scope review. Not external blind grading.','cases':reviews})
            s=summarize(gold,report['results'],reviews);save(d/'run1/summary.json',s)
            acceptance[group]={'overall':s['overall'],'numeric':s['categories']['calculation'],'citations':s['citation_counts'],
                'literal_numeric_match_count':sum(r['numeric_correct'] is True for r in s['cases']),
                'regressions':[r['id'] for r in group_rows if r['baseline_correct'] and not r['candidate_correct']],
                'gains':[r['id'] for r in group_rows if not r['baseline_correct'] and r['candidate_correct']],
                'new_wrong_answers':unsafe}
    if not args.finalize:
        assert not (p/'changes.json').exists();save(p/'changes.json',changes)
    else:
        assert set(notes)=={r['id'] for r in changes}
        numeric=json.loads((p/'numeric_results.json').read_text(encoding='utf-8'))
        no_regression=all(not a['regressions'] and not a['new_wrong_answers'] for a in acceptance.values()) and numeric['all']['correct']==12
        acceptance.update(numeric_regression=numeric['all'],taskbook=numeric['campus'],no_known_correct_regressions=no_regression,
            holdout_opened=False,adapter_enabled=False)
        save(p/'acceptance.json',acceptance)
        with (p/'same_question_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=comparisons[0].keys());w.writeheader();w.writerows(comparisons)
        print(json.dumps(acceptance,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
