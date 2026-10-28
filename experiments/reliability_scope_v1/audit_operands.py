"""Expose all numeric operands, even when rendered answer did not change."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_formal import save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',required=True);args=parser.parse_args()
    p=ROOT/'data/eval'/args.experiment;audit=[]
    for group in ['dev36','exposed_v2','old_campus']:
        d=p/group
        gold={r['id']:r for r in json.loads((d/'gold.json').read_text(encoding='utf-8'))['cases']}
        for r in json.loads((d/'run1/results.json').read_text(encoding='utf-8'))['results']:
            if r['category']!='calculation':continue
            calc=r['result'].get('calculation') or {};details=calc.get('details',[])
            keys=['doc_id','table_id','row_id','row_label','field','field_id','subject','value','numeric_value','unit','comparison','scope']
            unique={json.dumps({k:d.get(k) for k in keys},ensure_ascii=False,sort_keys=True) for d in details}
            actual={d['row_id'] for d in details if 'row_id' in d}
            expected={i for s in gold[r['id']]['sources'] for i in s.get('row_ids',[])}
            branches={step['id']:sorted({d['row_id'] for d in step.get('details',[]) if 'row_id' in d}) for step in calc.get('steps',[])}
            row={'id':r['id'],'execution_ok':calc.get('ok',False),'answer':r['result']['answer'],
                'plan':r.get('plan'),'actual_row_ids':sorted(actual),'gold_row_ids':sorted(expected),
                'row_sets_match':actual==expected if expected else None,'step_row_ids':branches,
                'operands':[json.loads(s) for s in sorted(unique)]}
            if calc.get('ok') and expected:
                assert actual==expected or (r['plan']['steps'][-1]['op']=='ratio' and sorted(expected) in branches.values()),r['id']
            audit.append(row)
    destination=p/'all_numeric_operand_audit.json';assert not destination.exists();save(destination,audit)


if __name__=='__main__':main()
