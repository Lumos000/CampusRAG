"""Current runtime prompts, structural transfer and genuine semantic repairs."""
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Chunk,Evidence,ParsedTable,SourceDocument

spec=importlib.util.spec_from_file_location('v9_legacy_data',Path(__file__).with_name('legacy_data.py'))
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
ORIGINAL_WORDS={k:list(v) for k,v in legacy.WORDS.items()}
TEST_WORDS = {
    'lookup': '单独查询{a}这一行：{field}数值为多少{unit}？',
    'sum': '将整表所有明细的{field}累加，结果为多少{unit}？',
    'subset_sum': '仅纳入{b}及{a}，两者{field}累计为多少{unit}？',
    'table_total_ratio': '把{b}和{a}的{field}相加，相对于表内合计的百分比是多少？',
    'average': '逐条统计所有明细的{field}，平均为多少{unit}？',
    'count': '逐行数出名称是“{a}”的记录，一共几条？',
    'subset_count': '逐行计数，名称是“{b}”或“{a}”的记录一共几条？',
    'ratio': '以{b}的{field}作分母、{a}的{field}作分子，百分比为多少？',
    'full_ratio': '分母取全部行数，分子取“{a}”的行数，计算百分比。',
    'difference': '从{a}的{field}中减掉{b}对应的数额，余下多少{unit}？',
    'product': '将{a}对应的数量乘以其单价，金额为多少元？',
    'hours': '用小时标准核算{a}的{n}小时工作，总额为多少元？',
    'completed': '{a}的规定时长扣除已完成的{n}小时后，余量为多少小时？',
    'cycle': '把{a}的每周时长累加{n}周，结果为多少小时？',
    'bounds': '从{a}课程数量上限中扣除下限，两界限差几门？',
    'budget': '{b}安排{m}人、{a}安排{n}人，两组各按本组人均标准核算，共多少元？',
    'body_budget': '先分别用{b}与{a}的原文人数乘本组人均标准，再相加，总计多少元？',
    'missing': '未给出实际小时数，能否计算{a}工作的报酬总额？',
}
LEGACY={'retention':168,'fields':96,'source':48,'constants':72,'budget':96,'unanswerable':48,'repair':72}
KINDS=['course_count','course_ratio','credits_remaining','hours_remaining','grade_difference','cycle_remaining','wage_remaining','role_competition']
KIND_COUNTS={'course_count':120,'course_ratio':60,'credits_remaining':60,'hours_remaining':60,'grade_difference':120,'cycle_remaining':120,'wage_remaining':30,'role_competition':30}
TEST_WORDS['ratio']='比例核算：分子取{a}的{field}，分母取{b}的{field}，结果为多少百分比？'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(repo,split,index,kind):
    rng=random.Random(990000+index+100000*['train','validation','test'].index(split))
    prefix={'train':'澜洲','validation':'墨榆','test':'晴桦'}[split]+str(index)
    policy=prefix+('课程映射细则' if kind.startswith('course') else '学业核算规定')
    a,b=prefix+'综合实践',prefix+'应用设计'
    es=[];raw_sources=[];tables=[];required_fields=[];required_rows=[]
    def source(title,text,tid=None):
        url=f'https://example.edu/synthetic-v9/{split}/{index}/{len(es)}'
        did=repo.add_document(SourceDocument(title,url,sha256=hashlib.sha256((url+text).encode()).hexdigest()))
        eid=f'v9-{split}-{index}-{len(es)}'
        repo.add_chunks([Chunk(eid,did,text,'',len(text),table_id=tid)])
        es.append(Evidence(eid,text,1,did,title,url,chunk_id=eid,table_id=tid))
        raw_sources.append({'doc_id':did,'title':title,'text':text})
        return did
    def table(headers,rows):
        rng.shuffle(rows)
        permutation=list(range(len(headers)));rng.shuffle(permutation)
        headers=[headers[i] for i in permutation];rows=[[r[i] for i in permutation] for r in rows]
        did=source(policy,f'课程要求共{total}{unit}。本表列出完整明细。' if kind in {'credits_remaining','hours_remaining'} else '本表列出课程明细；统计范围以全部数据行为准。')
        tid=repo.add_table(did,ParsedTable(policy,headers,rows),[[normalize_cell(v) for v in row] for row in rows])
        es[-1].table_id=tid;tables.append(tid)
        raw_sources[-1].update(table_id=tid,headers=headers,rows=rows)
        cols={r['original_name']:r['id'] for r in repo.connection.execute('SELECT * FROM table_columns WHERE table_id=?',(tid,))}
        ids=[r[0] for r in repo.connection.execute('SELECT id FROM table_rows WHERE table_id=? ORDER BY row_index',(tid,))]
        return did,tid,cols,headers,rows,ids
    variant=rng.randrange(4) if split=='train' else 4 if split=='validation' else 5
    unit='元';expected=None;plan={'table_id':None,'steps':[]};broken=None
    if kind.startswith('course'):
        source_name=['慕课名称','网络课程 / 慕课名称','课程对照 / 慕课名称'][rng.randrange(3)]
        target_name=['校内课程 / 课程名称','课程映射 / 校内课程名称','可替代课程 / 课程名称'][rng.randrange(3)]
        code_name=['校内课程 / 课程代码','课程映射 / 课程代码'][rng.randrange(2)]
        n,m=rng.randint(2,8),rng.randint(2,8)
        rest=rng.choice({'train':[8,16,40,96],'validation':[14,28,66],'test':[20,48,84]}[split])
        names=[a]*n+[b]*m+[prefix+f'基础课程{k}' for k in range(rest)]
        rows=[[name,(a if j%4==0 else prefix+f'校内课{j%9}'),('P'+str(100+j)),str(j+1),prefix+'学院','以原始行计数'] for j,name in enumerate(names)]
        did,tid,cols,headers,rows,row_ids=table([source_name,target_name,code_name,'序号','开课学院','补充说明'],rows)
        source_field=cols[source_name]
        chosen=[a] if kind=='course_count' else [a,b]
        filt=[{'field_id':source_field,'op':'in','value':chosen}]
        plan={'table_id':tid,'steps':[{'id':'part','op':'count','filters':filt}]}
        required_rows=[rid for row,rid in zip(rows,row_ids) if row[headers.index(source_name)] in chosen]
        expected=n;unit='项'
        if kind=='course_ratio':
            plan['steps'] += [{'id':'all','op':'count'},{'id':'result','op':'ratio','inputs':['part','all']}]
            expected=Decimal(n+m)/len(rows)*100;unit='%';required_rows=row_ids
            phrases=[f'慕课“{a}”和慕课“{b}”对应记录合计，占全表记录的百分比是多少？',f'慕课“{b}”与慕课“{a}”对应行数之和，占全部数据行多少百分比？',f'合并慕课“{a}”和慕课“{b}”的对应记录，以全部明细行为分母，求百分比。',f'慕课“{b}”及慕课“{a}”的对应行占全表数据行的比例是多少？',f'先数出慕课“{a}”或慕课“{b}”的全部对应行，再与整表行数比较，百分比为多少？',f'全表校内课程记录中，来源为慕课“{b}”或慕课“{a}”的记录占多少百分比？']
        else:
            phrases=[f'慕课“{a}”对应的校内课程记录共有几条？每个课程代码所在行计一条。',f'按慕课名称“{a}”筛选，展开后的课程代码记录共多少条？',f'数出慕课“{a}”对应的全部数据行，有多少条校内课程记录？',f'慕课“{a}”可对应几条校内课程记录？按行计数，不按课程名称去重。',f'来源慕课为“{a}”的校内课程对应记录共有几个？一行代码算一条。',f'逐行核对课程对照表：慕课“{a}”对应多少条校内课程数据？相同名称也分别计数。']
        broken=json.loads(json.dumps(plan));broken['steps'][0]['filters'][0]['field_id']=cols[code_name]
    elif kind in {'credits_remaining','hours_remaining'}:
        unit='学分' if kind=='credits_remaining' else '学时'
        header=['课程信息 / 学分','课程明细 / 学分','学分'][rng.randrange(3)] if unit=='学分' else ['课程信息 / 学时','课程明细 / 学时','学时'][rng.randrange(3)]
        other='课程信息 / 学时' if unit=='学分' else '课程信息 / 学分'
        names=[prefix+f'实践课程{k}' for k in range(rng.randint(3,8))]
        values=[rng.randint(2,7) if unit=='学分' else rng.choice([16,24,32,48]) for _ in names]
        total=sum(values)
        done=rng.choice([0,total,rng.randint(1,total-1)])
        planned=rng.randint(0,total+6)
        while planned==done:planned=rng.randint(0,total+6)
        marker=rng.choice(['共计','合计','总计'])
        rows=[[name,str(value),str(32 if unit=='学分' else 2),str(k+1),'考核'] for k,(name,value) in enumerate(zip(names,values))]
        rows.append(['',str(total),str(32*len(names) if unit=='学分' else 2*len(names)),marker,''])
        did,tid,cols,headers,rows,row_ids=table(['课程名称',header,other,'序号','考核形式'],rows)
        use_total=index%2==0
        start={'id':'total','op':'lookup' if use_total else 'sum','field_id':cols[header]}
        if use_total:start['scope']='total'
        plan={'table_id':tid,'steps':[start,{'id':'done','op':'lookup','literal':str(done),'unit':unit},{'id':'result','op':'subtract','inputs':['total','done']}]}
        required_fields=[cols[header]]
        required_rows=[rid for row,rid in zip(rows,row_ids) if (row[headers.index('序号')]==marker)==use_total]
        expected=total-done
        target='表内合计' if use_total else '全部明细的合计'
        phrases=[f'总计{total}{unit}，已完成{done}{unit}，计划修读{planned}{unit}，还差多少{unit}？计划量不算已完成。',
            f'计划修读{planned}{unit}，总计{total}{unit}，已完成{done}{unit}，还剩多少{unit}？仅扣除实际完成量。',
            f'已完成{done}{unit}，总计{total}{unit}，计划修读{planned}{unit}，距离总量还差多少{unit}？',
            f'总计{total}{unit}，计划修读{planned}{unit}，已完成{done}{unit}，尚缺多少{unit}？',
            f'课程总计{total}{unit}，已完成{done}{unit}，计划完成{planned}{unit}，只减实际已完成量后还需多少{unit}？',
            f'已完成{done}{unit}，计划修读{planned}{unit}，课程总计{total}{unit}；现在还剩多少{unit}没有完成？']
        broken=json.loads(json.dumps(plan));broken['steps'][0].pop('scope',None);broken['steps'][0]['op']='lookup'
        broken['steps'][1]['literal']=str(total if total!=done else planned)
        if index%3==0:broken['steps'][0]=json.loads(json.dumps(plan['steps'][0]))
        source(prefix+'课程说明摘录',f'课程要求共{total}{unit}。')
    else:
        if kind=='grade_difference':
            unit='绩点';x=Decimal(rng.randint(18,29))/10;y=x-Decimal(rng.randint(2,8))/10
            a,b=prefix+'一般学生',prefix+'专项运动员'
            text=(f'按《{prefix}学籍管理办法》在校学习期间平均学分绩点为{x}及以上。{b}平均学分绩点要求为{y}及以上。' if index%2 else
                  f'{a}平均学分绩点不低于{x}。{b}平均学分绩点不低于{y}。')
            expected=x-y
            phrases=[f'{a}与{b}的平均学分绩点门槛相差多少？',f'{a}的平均学分绩点下限减去{b}的下限，差额是多少？',f'{a}与{b}的平均学分绩点下限之差是多少？按前者减后者。',f'{a}平均学分绩点门槛比{b}高多少？',f'比较{a}和{b}的平均学分绩点要求，前者比后者高多少？',f'两类学生的平均学分绩点下限差多少？按{a}减去{b}。']
        elif kind=='role_competition':
            x,y=rng.randint(20,99),rng.randint(100,199);n,m=rng.randint(2,18),rng.randint(2,18)
            text=f'{b}：{m}名，{y}元/人。{a}：{n}名，{x}元/人。'
            rate=index%2==0;unit='元/人' if rate else '人';expected=y-x if rate else n+m
            phrases=([f'{b}每人标准减去{a}每人标准，差额是多少元？',f'{b}与{a}人均标准相差多少元？',f'比较{b}和{a}人均金额，前者减后者是多少元？',f'{b}的人均标准比{a}高多少元？',f'每人标准按{b}减去{a}计算，得到多少元差额？',f'从{b}每人金额中扣掉{a}每人金额，差额多少元？'] if rate else
                [f'{a}和{b}人数合计多少人？',f'{b}与{a}的名额之和是多少人？',f'{a}及{b}一共有多少人？',f'将{a}和{b}人数相加，总计多少人？',f'只统计人数，{b}和{a}合并为多少人？',f'两组{a}、{b}的参加人数总和是多少人？'])
        else:
            x,n=rng.randint(3,18),rng.randint(2,7);cap=x*n+rng.randint(10,70)
            cycle=kind=='cycle_remaining';unit='小时' if cycle else '元'
            text=f'{a}每周时长不超过{x}小时。' if cycle else f'{a}小时报酬标准为{x}元/小时。'
            cap_in_source=cycle and index%2==0
            if cap_in_source:
                clauses=[f'{a}每周时长不超过{x}小时。',f'{a}每月时长不超过{cap}小时。']
                rng.shuffle(clauses);text=''.join(clauses)
            expected=cap-x*n
            phrases=([f'{a}连续{n}周每周按上限工作，总额度{cap}小时，还剩多少小时？',f'总额度{cap}小时，扣除{a}按每周时长执行{n}周的用量，还剩多少小时？',f'按每周标准连续执行{n}周，{cap}小时总预算减去{a}用量后，差额是多少小时？',f'{a}工作{n}周，按每周规定时长核算，距离{cap}小时总额还差多少小时？',f'用{cap}小时减去{a}连续{n}周按每周标准产生的用量，余量是多少小时？',f'{a}在{n}周内按每周上限工作，从{cap}小时额度中扣除后，还剩多少小时？'] if cycle else
                [f'{a}工作{n}小时，预算{cap}元，按小时标准支付后还剩多少元？',f'预算{cap}元扣除{a}的{n}小时报酬，还余多少元？',f'先按标准核算{a}工作{n}小时的报酬，再从预算{cap}元中减去，还剩多少元？',f'从{cap}元中减去{a}工作{n}小时应付金额，差额是多少元？',f'{a}已工作{n}小时，按小时标准发放报酬后，{cap}元预算还能剩多少元？',f'经费{cap}元，支付{a}的{n}小时工作报酬后，尚余多少元？'])
            if cap_in_source:
                phrases=[f'{a}连续{n}周每周按上限工作，与办法的每月上限相比，还剩多少小时？',f'{a}按每周规定时长工作{n}周，从每月工时上限中扣除后，还剩多少小时？',f'以每月上限为总额，减去{a}连续{n}周按每周上限产生的用量，余量多少小时？',f'{a}工作{n}周，每周均按上限核算，距离月度总上限还差多少小时？',f'将{a}每周时长按{n}周累计，与原文每月工时上限相比，差额多少小时？',f'{a}连续执行{n}周的每周上限后，每月上限还能余下多少小时？']
        did=source(policy,text)
        if kind=='grade_difference':source(prefix+'学生手册摘录',text)
        source(prefix+'另一规定',f'{prefix}外部事项预算为{rng.randint(211,899)}元。' if kind=='grade_difference' else f'{a}预算为{rng.randint(211,899)}元。{b}预算为{rng.randint(211,899)}元。')
    if rng.random()<.2:
        for j in range(3):
            source(prefix+f'辅助办法{j}',''.join(f'{prefix}辅助{j}{k}预算为{rng.randint(211,999)}元，申请人须先核实适用事项，再按原文流程提交材料并办理登记，完成审核后统一核算。' for k in range(4)))
    rng.shuffle(es)
    facts=NumericEngine.text_facts(es,academic_units=True)
    def fid(name,u):
        matches=[k for k,v in facts.items() if v['doc_id']==did and v['unit']==u and
            ((Decimal(v['value'])==(x if name==a else y)) if kind=='grade_difference' else
             v.get('scope',{}).get('period')=='每周' if kind=='cycle_remaining' else name in v['subject'])]
        assert len(matches)==1,(kind,name,u,facts)
        return matches[0]
    if kind=='grade_difference':
        plan['steps']=[{'id':'left','op':'lookup','fact_id':fid(a,unit)},{'id':'right','op':'lookup','fact_id':fid(b,unit)},{'id':'result','op':'subtract','inputs':['left','right']}]
        broken=json.loads(json.dumps(plan));broken['steps'][-1]['op']='gte'
    elif kind=='role_competition':
        left,right=(b,a) if rate else (a,b)
        plan['steps']=[{'id':'left','op':'lookup','fact_id':fid(left,unit)},{'id':'right','op':'lookup','fact_id':fid(right,unit)},{'id':'result','op':'subtract' if rate else 'add','inputs':['left','right']}]
        broken=json.loads(json.dumps(plan));broken['steps'][1]['fact_id']=fid(right,'人' if rate else '元/人')
    elif kind in {'cycle_remaining','wage_remaining'}:
        plan['steps']=[{'id':'rate','op':'lookup','fact_id':fid(a,'小时' if cycle else '元/小时')},{'id':'periods','op':'lookup','literal':str(n),'unit':'周' if cycle else '小时'},
            {'id':'used','op':'multiply','inputs':['rate','periods']},{'id':'cap','op':'lookup','literal':str(cap),'unit':unit},{'id':'result','op':'subtract','inputs':['cap','used']}]
        if cap_in_source:
            cap_ids=[k for k,v in facts.items() if v['doc_id']==did and v.get('scope',{}).get('period')=='每月']
            assert len(cap_ids)==1,(kind,facts)
            plan['steps'][3]={'id':'cap','op':'lookup','fact_id':cap_ids[0]}
        broken=json.loads(json.dumps(plan));broken['steps']=[s for s in broken['steps'] if s['id']!='used'];broken['steps'][-1]['inputs']=['cap','rate']
    qualifier='仅按每周上限核算，' if kind=='cycle_remaining' else '核算人均差额，' if kind=='role_competition' and rate else ''
    question=f'按《{policy}》'+('的课程表' if kind in {'credits_remaining','hours_remaining'} else '')+'，计算：'+qualifier+phrases[variant]
    if kind=='cycle_remaining':question=question.rstrip('？')+f'；另计划使用{rng.randint(1,30)}小时，不计入本次实际用量，还剩多少小时？'
    repair=rng.random()<.5
    if repair and kind=='cycle_remaining' and cap_in_source:
        broken=json.loads(json.dumps(plan))
        broken['steps'][0]['fact_id'],broken['steps'][3]['fact_id']=plan['steps'][3]['fact_id'],plan['steps'][0]['fact_id']
    if repair and index%5==0 and kind not in {'credits_remaining','hours_remaining'}:
        broken=json.loads(json.dumps(plan));broken['steps'][-1]['inputs']=['undefined_step','undefined_other']
    class Index:
        def search(self,*args,**kwargs):return es
    class Capture:
        def __init__(self):self.prompts=[];self.outputs=[]
        def generate(self,prompt,**kwargs):
            self.prompts.append(prompt)
            raw=json.dumps(broken if repair and len(self.prompts)==1 else plan,ensure_ascii=False)
            self.outputs.append(raw);return raw
    capture=Capture()
    service=RAGService(Settings(ROOT,ROOT,ROOT,ROOT,ROOT,numeric_shortcuts=False),Index(),capture,repo.connection)
    answer=service.answer(question)
    assert capture.prompts and answer.calculation and answer.calculation.ok,(kind,question,answer.answer,service.last_diagnostics)
    assert answer.calculation.unit==unit and abs(answer.calculation.value-Decimal(expected))<Decimal('.000001'),(kind,expected,asdict(answer))
    assert {d['doc_id'] for d in answer.calculation.details}=={did}
    assert {d['row_id'] for d in answer.calculation.details if 'row_id' in d}==set(required_rows)
    if repair:assert len(capture.prompts)==2,(kind,question,service.last_diagnostics)
    diag=service.last_diagnostics
    required_facts=[s['fact_id'] for s in plan['steps'] if 'fact_id' in s]
    assert set(required_facts)<={f['fact_id'] for f in diag['facts']}
    return {'id':f'v9-{split}-{kind}-{index}','family':kind,'kind':kind,'split':split,'template_family':f'{kind}-grammar-{variant}',
        'source_family':f'v9-{split}-{kind}-{index}','question':question,'prompt':capture.prompts[-1],
        'context':{'tables':diag['schema'],'facts':diag['facts']},'plan':plan,'expected':str(expected),'unit':unit,
        'table_ids':tables,'source_doc_ids':[did],'evidence_map':{e.evidence_id:asdict(e) for e in es},
        'reference':{'value':str(expected),'unit':unit,'measure_field_ids':required_fields,'row_ids':required_rows,'required_fact_ids':required_facts,'required_doc_ids':[did],'kind':kind,'raw_sources':raw_sources},
        'repair_previous':capture.outputs[0] if repair else None,'repair_feedback':diag['attempts'][-1].get('feedback') if repair else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--output',type=Path,help='New output directory; existing data is never overwritten.')
    args=parser.parse_args()
    out=args.output or ROOT/'data/eval/planner_lora_v9'/('pilot' if args.pilot else 'data');out.mkdir(parents=True,exist_ok=False)
    all_rows={}
    with Repository(out/'fixtures.db') as repo:
        repo.initialize()
        for split in ['train','validation','test']:
            rows=[];offset=12000
            for family,total in LEGACY.items():
                count=4 if args.pilot else total if split=='train' else total//5
                if not args.pilot and split!='train' and family=='retention':count+=120-sum(n//5 for n in LEGACY.values())
                for j in range(count):
                    grammar=(j//12)%2 if split=='train' else 2 if split=='validation' else 3
                    for k in ORIGINAL_WORDS:
                        phrase=ORIGINAL_WORDS[k][grammar] if grammar<3 else '计算：'+TEST_WORDS[k]
                        if split=='train' and k=='ratio' and (j//8)%2:
                            phrase='分母取{b}的{field}，分子取{a}的{field}，百分比为多少？'
                        legacy.WORDS[k]=[phrase]*3
                    with repo.transaction(): row=legacy.build(repo,split,offset+j,family)
                    row['id']=row['id'].replace('v4-','v9-',1);row['template_family']=f'legacy-{row["kind"]}-grammar-{grammar}'
                    rows.append(row)
                offset+=count
            for kind in KINDS:
                count=8 if args.pilot else KIND_COUNTS[kind] if split=='train' else KIND_COUNTS[kind]//5
                for j in range(count):
                    with repo.transaction():rows.append(build(repo,split,offset+j,kind))
                offset+=count
            random.Random(9909).shuffle(rows);all_rows[split]=rows
            (out/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
            print(json.dumps({'split':split,'count':len(rows)}),flush=True)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        for key in ['prompt','template_family','source_family']:
            assert not {r[key] for r in all_rows[a]}&{r[key] for r in all_rows[b]},key
    manifest={'counts':{k:len(v) for k,v in all_rows.items()},'reference_execution_checked':True,
        'scope':'Independent synthetic policies/tables with realistic flattened headers, one-to-many mapping, GPA, total rows and semantic repairs. No real regression labels used for training. Not a blind real-document accuracy claim.',
        'files':{f.name:digest(f) for f in out.iterdir() if f.is_file()}}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
