"""Review references and token budgets, then freeze one local run."""
from collections import Counter,defaultdict
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'data/eval/planner_lora_v9'
sys.path[:0]=[str(ROOT/'src')]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from campus_rag.config import load_settings
from common import read_cases
from canonical_score import ordered_score
from transformers import AutoTokenizer


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    assert not (WORK/'protocol.json').exists(),'Never replace a frozen experiment.'
    baseline=json.loads((WORK/'interface_baseline/results.json').read_text(encoding='utf-8'))
    assert baseline['complete'],'Finish the frozen V8 + new guards comparison before preparing training.'
    settings=load_settings()
    tokenizer=AutoTokenizer.from_pretrained(settings.generator_model,local_files_only=True)
    connection=sqlite3.connect(WORK/'data/fixtures.db'); connection.row_factory=sqlite3.Row
    spec=importlib.util.spec_from_file_location('coverage_audit',ROOT/'scripts/audit_planner_training.py')
    audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    training=read_cases(WORK/'data/train.jsonl')
    coverage=audit.summarize(training,['学分','学时'],True)
    assert coverage['passed']
    assert coverage['supervised_unit_examples'].get('绩点',0)>=60
    semantic_repairs=[r for r in training if r['family'] in {'course_count','course_ratio','credits_remaining','hours_remaining','grade_difference','cycle_remaining','wage_remaining','role_competition'} and r['repair_previous']]
    assert len(semantic_repairs)>=200
    assert all(r['repair_feedback'] and r['repair_previous']!='{broken-json' for r in semantic_repairs)
    assert sum(any(s.get('scope')=='total' for s in r['plan']['steps']) for r in training)>=60
    academic=[r for r in training if r['family'] in {'credits_remaining','hours_remaining'}]
    role_audit={'academic_remaining':len(academic),
        'same_unit_body_competition':sum(any(f['unit']==r['unit'] for f in r['context']['facts']) for r in academic),
        'zero_remaining':sum(r['expected'] is not None and Decimal(r['expected'])==0 for r in academic),
        'wrong_completed_repairs':sum(bool(r['repair_previous']) and '已完成量' in (r['repair_feedback'] or '') for r in academic),
        'multi_error_repairs':sum('同时发现' in (r['repair_feedback'] or '') for r in academic),
        'denominator_first_ratio':sum(r['kind']=='ratio' and '分母取' in r['question'] for r in training)}
    assert role_audit['same_unit_body_competition']==120 and role_audit['zero_remaining']>=25
    assert role_audit['wrong_completed_repairs']>=40 and role_audit['multi_error_repairs']>=20 and role_audit['denominator_first_ratio']>=8
    save(WORK/'role_coverage.json',role_audit)
    save(WORK/'coverage.json',coverage)
    budgets={}; reviews=[]; split_rows={}
    for split in ['train','validation','test']:
        cases=read_cases(WORK/f'data/{split}.jsonl'); split_rows[split]=cases
        assert len(cases)==(1200 if split=='train' else 240)
        lengths=[]
        for case in cases:
            score=ordered_score(json.dumps(case['plan'],ensure_ascii=False),case,connection)
            assert score['correct'],(case['id'],score)
            prefix=tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],enable_thinking=False,add_generation_prompt=True,tokenize=True,return_dict=False)
            target=tokenizer.encode(json.dumps(case['plan'],ensure_ascii=False),add_special_tokens=False)
            length=len(prefix)+len(target)+1
            assert length<=6144,(case['id'],length)
            lengths.append(length)
        ordered=sorted(lengths)
        budgets[split]={'examples':len(cases),'max_tokens':max(lengths),'median_tokens':ordered[len(ordered)//2],
            'p95_tokens':ordered[int(len(ordered)*.95)],'at_least_4000_tokens':sum(n>=4000 for n in lengths),
            'repairs':sum(bool(c['repair_previous']) for c in cases)}
        # A small, legible stratified audit is saved before training, not just totals.
        for family in sorted({c['family'] for c in cases}):
            sample=next(c for c in cases if c['family']==family)
            reviews.append({k:sample[k] for k in ['id','family','question','plan','expected','unit','reference']})
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for key in ['template_family','source_family','prompt']:
            assert not {c[key] for c in split_rows[a]} & {c[key] for c in split_rows[b]}
    connection.close()
    save(WORK/'data_review_samples.json',reviews)
    save(WORK/'data_review.json',{'reference_checks_passed':True,'reference_count':1680,'coverage':coverage,'token_budgets':budgets,
        'scope':'All plans executed against fresh normalized fixture DB; ordered operands, values, units, exact rows/facts/documents checked. Disjoint synthetic grammar/source families; real transfer still unproven.'})
    snapshot=WORK/'snapshot'; snapshot.mkdir(exist_ok=False)
    for source in (ROOT/'src').rglob('*.py'):
        dest=snapshot/source.relative_to(ROOT); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dest)
    schema=ROOT/'src/campus_rag/database/schema.sql'
    shutil.copyfile(schema,snapshot/'src/campus_rag/database/schema.sql')
    for source in Path(__file__).resolve().parent.rglob('*'):
        if source.is_file() and source.suffix in {'.py','.json','.txt'} and '__pycache__' not in source.parts:
            dest=snapshot/source.relative_to(ROOT)
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,dest)
    audit_source=ROOT/'scripts/audit_planner_training.py'
    audit_dest=snapshot/'scripts/audit_planner_training.py'
    audit_dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(audit_source,audit_dest)
    files=[*(p for p in snapshot.rglob('*') if p.is_file()),*sorted((ROOT/'src').rglob('*.py')),
        ROOT/'config/settings.json',settings.database_path, settings.index_dir/'records.jsonl',settings.index_dir/'vectors.npy',
        *sorted(settings.generator_model.glob('*.safetensors')),*sorted(settings.generator_model.glob('*.json')),
        *sorted((WORK/'data').glob('*')),WORK/'data_review.json',WORK/'coverage.json',
        ROOT/'experiments/planner_lora_v9/run_comparison.py',ROOT/'experiments/planner_lora_v9/review_comparison.py',
        ROOT/'data/eval/campus_public/verified_release.json',ROOT/'experiments/planner_lora_v9/fixtures/dev36_gold.json',
        WORK/'interface_baseline/results.json',WORK/'interface_baseline/comparison_summary.json',WORK/'role_coverage.json',
        ROOT/'models/adapters/planner_lora_v8/step_300/adapter_model.safetensors',ROOT/'models/adapters/planner_lora_v8/step_300/adapter_config.json']
    protocol={'training':{'seed':9909,'r':8,'alpha':16,'dropout':.05,'target_modules':'all-linear','learning_rate':2.5e-5,
        'epochs':2,'micro_batch_size':1,'gradient_accumulation':8,'max_length':6144,'updates':300,
        'full_state_every_updates':25,'adapter_every_updates':75,'loss':'completion JSON and assistant end only'},
        'model_path':settings.generator_model.relative_to(ROOT).as_posix(),
        'selection':'Validation only, checkpoints 150/300: minimize base regressions/new errors, then V8 regressions/new errors, maximize correctness, earlier on ties. Test only after selection.',
        'evaluation':'All synthetic modes run actual RAGService initial -> validation -> own repair -> validation. Compare base/current guards, frozen V8/current guards, and new adapter/current guards.',
        'gate':'No base or V8 correct-case regression, gains over both, no new wrong executable outputs or repair-introduced wrong answers, and current 54-case comparison passes. Remain disabled pending review.',
        'files':{p.relative_to(ROOT).as_posix():digest(p) for p in files if p.is_file()}}
    save(WORK/'protocol.json',protocol)
    print(json.dumps({'coverage':coverage,'token_budgets':budgets,'frozen_files':len(protocol['files'])}),flush=True)


if __name__=='__main__':main()
