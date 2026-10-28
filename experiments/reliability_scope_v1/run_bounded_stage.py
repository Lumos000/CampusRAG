import os,sys,json,subprocess,shutil
from pathlib import Path
ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evaluate_reliability_trial import prepare
from evaluate_formal import run,save,digest
stage=sys.argv[1];baseline=ROOT/sys.argv[2];p=ROOT/'data/eval'/stage
p.mkdir(exist_ok=False)
env={**os.environ,'PYTHONIOENCODING':'utf-8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
with (p/'automatic_tests.txt').open('w',encoding='utf-8') as f:
 subprocess.run([sys.executable,'-m','pytest','-q'],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
sources={}
for directory in ['src','tests']:
 for source in (ROOT/directory).rglob('*.py'):
  target=p/'candidate_files'/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
  sources[source.relative_to(ROOT).as_posix()]={'sha256':digest(source),'archive':target.relative_to(ROOT).as_posix()}
save(p/'frozen_sources.json',sources)
save(p/'protocol.json',{'baseline':baseline.relative_to(ROOT).as_posix(),'base_commit':'3ce4b7f','training':False,'adapter':None,'sealed_tests_opened':False,'source_sha256':{k:v['sha256'] for k,v in sources.items()},'purpose':'One frozen complete run, development regression only'})
for group in ['dev36','exposed_v2','old_campus']:
 prepare(p/group,baseline/group)
print('Prepared',stage,flush=True)
subprocess.run([sys.executable,'scripts/evaluate_numeric.py','--output',str(p/'numeric_results.json')],check=True,env=env)
print('Numeric regression complete',flush=True)
for group in ['dev36','exposed_v2','old_campus']:
 print('Running',stage,group,flush=True);run(p/group)
print('All model runs finished',stage,flush=True)
