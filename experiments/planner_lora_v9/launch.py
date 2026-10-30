"""Start the reviewed run independently of this terminal, without a visible window."""
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/'data/eval/planner_lora_v9'
script=WORK/'snapshot/experiments/planner_lora_v9/run.py'
assert os.name=='nt'
assert json.loads((WORK/'preflight.json').read_text(encoding='utf-8'))['passed']
assert not (WORK/'acceptance.json').exists(),'This experiment already finished.'
for process in psutil.process_iter(['cmdline']):
    if str(script) in (process.info['cmdline'] or []):
        raise SystemExit(f'Experiment already running, PID {process.pid}')
environment=dict(os.environ,CAMPUSRAG_EXPERIMENT_ROOT=str(ROOT),PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (WORK/'run.log').open('ab',buffering=0) as log:
    flags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        process=subprocess.Popen([sys.executable,'-u',str(script)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            close_fds=True,env=environment,creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB)
        breakaway=True
    except PermissionError:
        process=subprocess.Popen([sys.executable,'-u',str(script)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
            close_fds=True,env=environment,creationflags=flags)
        breakaway=False
result={'pid':process.pid,'detached':True,'breakaway_from_job':breakaway,'log':str(WORK/'run.log'),
    'status':str(WORK/'status.json'),'automatic_evaluation':True,'adapter_enabled':False}
(WORK/'launch.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result),flush=True)
