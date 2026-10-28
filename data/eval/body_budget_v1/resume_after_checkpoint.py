import sys,csv,json
from pathlib import Path
ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from evaluate_formal import run
p=ROOT/'data/eval/body_budget_v1'
run(p/'exposed_v2',resume=True)
# The interrupted last-case save preceded CSV writing; regenerate the same rows.
r=json.loads((p/'exposed_v2/run1/results.json').read_text(encoding='utf-8'))
with (p/'exposed_v2/run1/results.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['问题','答案','参考资料']);w.writeheader()
 for row in r['results']:
  a=row.get('result',{});w.writerow({'问题':row['question'],'答案':a.get('answer','运行异常：'+row.get('error','')),'参考资料':'\n'.join(a.get('references',[]))})
run(p/'old_campus')
print('Frozen candidate completed after checkpoint recovery',flush=True)
