"""Offline, resumable completion-only SFT and unattended paired evaluation."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time

ROOT=Path(os.environ.get('CAMPUSRAG_EXPERIMENT_ROOT',Path(__file__).resolve().parents[2]))
WORK=ROOT/'data/eval/planner_lora_v9'
SNAPSHOT=WORK/'snapshot'
sys.path[:0]=[str(SNAPSHOT/'src'),str(SNAPSHOT/'experiments/planner_lora_v9')]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import campus_rag.config as campus_config
campus_config.ROOT=ROOT
from campus_rag.rag.generator import TransformersGenerator
from canonical_score import ordered_score
from common import read_cases
from runtime_eval import evaluate_case,summarize


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):
    path=Path(path); tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8'); tmp.replace(path)


def verify(protocol):
    for name,checksum in protocol['files'].items():
        assert digest(ROOT/name)==checksum,('Frozen input changed',name)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--preflight',action='store_true'); args=parser.parse_args()
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    protocol=json.loads((WORK/'protocol.json').read_text(encoding='utf-8'))
    protocol_hash=digest(WORK/'protocol.json'); verify(protocol)
    cfg=protocol['training']; seed=cfg['seed']
    torch.set_num_threads(4); random.seed(seed); torch.manual_seed(seed)
    assert torch.cuda.is_available()
    generator=TransformersGenerator(ROOT/protocol['model_path'])
    model=get_peft_model(generator.model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules='all-linear',bias='none',task_type='CAUSAL_LM'))
    generator.model=model; model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.enable_input_require_grads()
    parameters=[p for p in model.parameters() if p.requires_grad]
    assert all('lora_' in name for name,p in model.named_parameters() if p.requires_grad)
    tokenizer=generator.tokenizer; eos=tokenizer.convert_tokens_to_ids('<|im_end|>')
    samples=[]
    for row in read_cases(WORK/'data/train.jsonl'):
        prefix=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],enable_thinking=False,tokenize=True,add_generation_prompt=True,return_dict=False)
        target=tokenizer.encode(json.dumps(row['plan'],ensure_ascii=False),add_special_tokens=False)+[eos]
        assert len(prefix)+len(target)<=cfg['max_length']
        samples.append((prefix+target,len(prefix),row['id']))
    optimizer=torch.optim.AdamW(parameters,lr=cfg['learning_rate'],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:(step+1)/30 if step<30 else .5*(1+math.cos(math.pi*(step-30)/270)))
    device=next(model.parameters()).device
    def loss_for(ids,prefix):
        inputs=torch.tensor([ids],device=device)
        positions=torch.arange(prefix-1,len(ids)-1,device=device)
        output=model(input_ids=inputs,attention_mask=torch.ones_like(inputs),use_cache=False,logits_to_keep=positions)
        return F.cross_entropy(output.logits.float().reshape(-1,output.logits.shape[-1]),inputs[:,prefix:].reshape(-1))
    if args.preflight:
        ids,prefix,key=max(samples,key=lambda row:len(row[0])); checks=[]
        for length in sorted({len(ids),cfg['max_length']}):
            torch.cuda.reset_peak_memory_stats(); optimizer.zero_grad(set_to_none=True); model.train()
            padding=length-len(ids); filler=tokenizer.encode('背景材料与当前计算无关。',add_special_tokens=False)
            padded=(filler*math.ceil(padding/len(filler)))[:padding]+ids
            loss=loss_for(padded,prefix+padding); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters,1.); optimizer.step(); torch.cuda.synchronize()
            checks.append({'tokens':length,'all_attention_positions_active':True,'loss':float(loss.detach()),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        model.load_adapter(ROOT/'models/adapters/planner_lora_v8/step_300',adapter_name='v8_reference',is_trainable=False)
        model.eval();model.config.use_cache=True
        smoke_connection=sqlite3.connect((WORK/'data/fixtures.db').resolve().as_uri()+'?mode=ro',uri=True);smoke_connection.row_factory=sqlite3.Row
        smoke_case=min([r for r in read_cases(WORK/'data/train.jsonl') if r['family']=='credits_remaining'],key=lambda r:len(r['prompt']))
        smoke=[]
        for mode in ['base','v8']:
            model.set_adapter('v8_reference' if mode=='v8' else 'default')
            row=evaluate_case(generator,smoke_case,smoke_connection,mode,ROOT)
            smoke.append({'mode':mode,'case_id':smoke_case['id'],'api_completed':True,'correct':row['correct'],'was_repaired':row['was_repaired']})
        smoke_connection.close()
        report={'passed':True,'case_id':key,'checks':checks,'protocol_sha256':protocol_hash,'formal_training_uses_fresh_model':True,'runtime_api_smoke':smoke}
        save(WORK/'preflight.json',report); print(json.dumps(report),flush=True); return
    preflight=json.loads((WORK/'preflight.json').read_text(encoding='utf-8'))
    assert preflight['passed'] and preflight['protocol_sha256']==protocol_hash
    report={'step':0,'steps':[],'adapter_enabled':False,'protocol_sha256':protocol_hash,'complete':False}
    epoch=0; position=0; order=list(range(len(samples))); random.shuffle(order)
    pointer=WORK/'latest_state.json'
    if pointer.exists():
        meta=json.loads(pointer.read_text(encoding='utf-8')); state_path=WORK/meta['file']
        assert digest(state_path)==meta['sha256']
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        assert state['protocol_sha256']==protocol_hash
        set_peft_model_state_dict(model,state['adapter']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        random.setstate(state['python_rng']); torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        report=state['report']; epoch=state['epoch']; position=state['position']; order=state['order']; del state
    started=time.perf_counter(); start_step=report['step']; losses=[]
    def status(phase,**extra):
        save(WORK/'status.json',{'pid':os.getpid(),'phase':phase,'step':report['step'],'total_updates':300,'adapter_enabled':False,**extra})
    def persist():
        payload={'protocol_sha256':protocol_hash,'adapter':get_peft_model_state_dict(model),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all(),'report':report,'epoch':epoch,'position':position,'order':order}
        path=WORK/f'state_{report["step"]}.pt'; tmp=path.with_suffix('.tmp'); torch.save(payload,tmp); tmp.replace(path)
        save(pointer,{'file':path.name,'sha256':digest(path),'step':report['step']}); save(WORK/'training.json',report)
    optimizer.zero_grad(set_to_none=True); status('training')
    while report['step']<300:
        if position==len(order):
            epoch+=1; position=0; order=list(range(len(samples))); random.shuffle(order)
        model.train(); model.config.use_cache=False
        ids,prefix,key=samples[order[position]]
        loss=loss_for(ids,prefix); assert torch.isfinite(loss),(key,'nonfinite loss')
        (loss/8).backward(); losses.append(float(loss.detach())); position+=1
        if position%8: continue
        torch.nn.utils.clip_grad_norm_(parameters,1.); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        report['step']+=1
        elapsed=time.perf_counter()-started
        metric={'step':report['step'],'loss':sum(losses[-8:])/8,'seconds_this_process':round(elapsed,2),
            'estimated_remaining_training_minutes':round(elapsed/max(1,report['step']-start_step)*(300-report['step'])/60,1)}
        report['steps'].append(metric)
        if report['step']==1 or report['step']%5==0:
            print(json.dumps(metric),flush=True); save(WORK/'training.json',report); status('training',**{k:v for k,v in metric.items() if k!='step'})
        if report['step']%25==0: persist()
        if report['step']%75==0:
            checkpoint=ROOT/'models/adapters/planner_lora_v9'/f'step_{report["step"]}'
            model.save_pretrained(checkpoint,safe_serialization=True)
    report['training_complete']=True; save(WORK/'training.json',report)
    model.gradient_checkpointing_disable(); model.eval(); model.config.use_cache=True
    connection=sqlite3.connect((WORK/'data/fixtures.db').resolve().as_uri()+'?mode=ro',uri=True); connection.row_factory=sqlite3.Row
    model.load_adapter(ROOT/'models/adapters/planner_lora_v8/step_300',adapter_name='v8_reference',is_trainable=False)
    def evaluate(split,mode):
        path=WORK/f'{split}_{mode}.json'
        results=json.loads(path.read_text(encoding='utf-8'))['results'] if path.exists() else []
        cases=read_cases(WORK/f'data/{split}.jsonl'); done={r['id'] for r in results}
        status('evaluation',split=split,mode=mode,completed=len(results),total=len(cases))
        model.set_adapter('v8_reference' if mode=='v8' else 'default');model.eval()
        for case in cases:
            if case['id'] in done: continue
            results.append(evaluate_case(generator,case,connection,mode,ROOT))
            metrics=summarize(results)
            save(path,{'results':results,'metrics':metrics,'complete':len(results)==len(cases),
                'evaluation_scope':'Actual runtime initial plan, own validation feedback, at most one repair; no teacher previous plan supplied.'})
            if len(results)%8==0:
                status('evaluation',split=split,mode=mode,completed=len(results),total=len(cases))
                print(json.dumps({'split':split,'mode':mode,'completed':len(results),'correct':metrics['final_correct'],'first_correct':metrics['first_correct']}),flush=True)
        return {r['id']:r for r in results}
    def compare(base,adapted):
        regressions=[key for key in base if base[key]['correct'] and not adapted[key]['correct']]
        gains=[key for key in base if not base[key]['correct'] and adapted[key]['correct']]
        wrong=lambda rows:[key for key,r in rows.items() if r['execution_ok'] and not r['correct']]
        new_wrong=sorted(set(wrong(adapted))-set(wrong(base)))
        result={'total':len(base),'base_correct':sum(r['correct'] for r in base.values()),'adapter_correct':sum(r['correct'] for r in adapted.values()),
            'gains':gains,'regressions':regressions,'wrong_numeric_base':wrong(base),'wrong_numeric_adapter':wrong(adapted),'new_wrong_executable':new_wrong}
        result['passed']=not regressions and bool(gains) and not new_wrong and len(wrong(adapted))<=len(wrong(base))
        return result
    from safetensors.torch import load_file
    base=evaluate('validation','base');v8=evaluate('validation','v8'); candidates=[]
    for step in [150,300]:
        checkpoint=ROOT/'models/adapters/planner_lora_v9'/f'step_{step}'
        set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter_model.safetensors')))
        adapted=evaluate('validation',f'adapter_step_{step}')
        candidates.append({'step':step,**compare(base,adapted),'v8_comparison':compare(v8,adapted),
            'runtime_metrics':summarize(list(adapted.values()))})
        save(WORK/'checkpoint_comparison.json',candidates)
    selected=min(candidates,key=lambda c:(len(c['regressions']),len(c['new_wrong_executable']),
        len(c['v8_comparison']['regressions']),len(c['v8_comparison']['new_wrong_executable']),-c['adapter_correct'],c['step']))
    checkpoint=ROOT/'models/adapters/planner_lora_v9'/f'step_{selected["step"]}'
    set_peft_model_state_dict(model,load_file(str(checkpoint/'adapter_model.safetensors')))
    report['selected_step']=selected['step'];save(WORK/'training.json',report)
    save(WORK/'selected_adapter.json',{'step':selected['step'],'path':checkpoint.relative_to(ROOT).as_posix(),
        'selection':'Validation only, steps 150/300: minimize base regressions/new errors, then V8 regressions/new errors, then maximize correctness; earlier on ties.',
        'adapter_enabled':False})
    summaries={'validation':selected};save(WORK/'synthetic_comparison.json',summaries)
    test_base=evaluate('test','base');test_v8=evaluate('test','v8');test_new=evaluate('test','adapter')
    summaries['test']={**compare(test_base,test_new),'v8_comparison':compare(test_v8,test_new),'runtime_metrics':summarize(list(test_new.values()))}
    save(WORK/'synthetic_comparison.json',summaries)
    connection.close(); del generator.model; del model; del optimizer; del scheduler; del parameters
    import gc
    gc.collect(); torch.cuda.empty_cache()
    # A separate process releases the training allocator before the full RAG check.
    status('current_corpus_evaluation')
    verify(protocol)
    subprocess.run([sys.executable,str(ROOT/'experiments/planner_lora_v9/run_comparison.py'),'--output',str(WORK/'current_corpus'),'--adapter',str(checkpoint)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'experiments/planner_lora_v9/review_comparison.py'),'--directory',str(WORK/'current_corpus')],cwd=ROOT,check=True)
    current=json.loads((WORK/'current_corpus/comparison_summary.json').read_text(encoding='utf-8'))
    previous=json.loads((WORK/'interface_baseline/comparison_summary.json').read_text(encoding='utf-8'))
    old_rows=json.loads((WORK/'interface_baseline/results.json').read_text(encoding='utf-8'))['results']
    new_rows=json.loads((WORK/'current_corpus/results.json').read_text(encoding='utf-8'))['results']
    old_base={r['id']:{k:r['result'][k] for k in ['answer','references']} for r in old_rows if r['mode']=='base'}
    new_base={r['id']:{k:r['result'][k] for k in ['answer','references']} for r in new_rows if r['mode']=='base'}
    gains=sorted((set(current['gains'])-set(previous['gains']))|(set(previous['regressions'])-set(current['regressions'])))
    regressions=sorted((set(previous['gains'])-set(current['gains']))|(set(current['regressions'])-set(previous['regressions'])))
    attribution={'base_outputs_identical':old_base==new_base,'gains_over_v8':gains,'regressions_from_v8':regressions,
        'v8_correct':sum(v['adapter']['correct'] for v in previous['groups'].values()),
        'new_correct':sum(v['adapter']['correct'] for v in current['groups'].values()),
        'new_wrong_over_v8':sorted(set(current['wrong_numeric']['adapter'])-set(previous['wrong_numeric']['adapter']))}
    save(WORK/'attribution.json',attribution)
    passed=all(r['passed'] and r['v8_comparison']['passed'] and not r['runtime_metrics']['repair_introduced_wrong'] for r in summaries.values())
    passed=passed and current['passed'] and attribution['base_outputs_identical'] and not regressions and not attribution['new_wrong_over_v8']
    save(WORK/'acceptance.json',{'passed':passed,'synthetic':summaries,'current_corpus':current,'attribution':attribution,
        'adapter_enabled':False,'requires_review_before_activation':True})
    report['complete']=True; save(WORK/'training.json',report); status('complete')


if __name__=='__main__':
    try: main()
    except Exception as error:
        save(WORK/'failure.json',{'type':type(error).__name__,'message':str(error),'adapter_enabled':False})
        save(WORK/'status.json',{'phase':'failed','pid':os.getpid(),'error_type':type(error).__name__,'adapter_enabled':False})
        raise
