"""Fresh synthetic sources -> actual extraction/runtime prompts -> reviewed SFT.

Split-specific wording families are assigned before numerical expansion. Never
reads campus questions, historical answers, or any held-out evaluation labels.
"""
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from campus_rag.config import Settings
from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.rag.service import RAGService
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Chunk, Evidence, ParsedTable, SourceDocument

COUNTS = {'retention':240, 'constants':180, 'fields':180, 'direction':120,
    'source':120, 'budget':120, 'repair':120, 'unanswerable':120}
# Different grammatical formulations, not just different policy names.
WORDS = {
    'lookup':['{a}的{field}是多少{unit}？','查询{field}：{a}对应多少{unit}？','请给出{a}所对应的{field}数值，共多少{unit}？'],
    'sum':['所有明细的{field}合计多少{unit}？','将各条明细的{field}求和，总计多少{unit}？','计算全部记录的{field}之和，结果用{unit}表示。'],
    'subset_sum':['{a}和{b}的{field}之和是多少{unit}？','只对{a}与{b}的{field}求和，合计多少{unit}？','计算{a}、{b}两项{field}的总计，单位为{unit}。'],
    'table_total_ratio':['{a}与{b}的{field}之和占表内合计的百分比是多少？','{a}和{b}的{field}合并后，占表内合计的百分比是多少？','计算{a}、{b}的{field}总和相对于表内合计的百分比。'],
    'average':['全部明细的{field}平均多少{unit}？','求所有记录的{field}平均值，单位{unit}。','这些明细的{field}，平均数是多少{unit}？'],
    'count':['名称为“{a}”的记录有多少条？按行计数。','统计名称等于“{a}”的全部行，共有多少条记录？','请计算“{a}”对应的记录行数，不按名称去重。'],
    'subset_count':['名称为“{a}”或“{b}”的记录合计多少条？','统计名称等于“{a}”或“{b}”的行，共有多少条记录？','计算“{a}”和“{b}”对应的记录总行数，不按名称去重。'],
    'ratio':['{a}的{field}占{b}的{field}的百分比是多少？','{a}的{field}，占{b}对应{field}的百分比是多少？以占字后对象为分母。','计算{a}相对于{b}的{field}百分比。'],
    'full_ratio':['名称为“{a}”的记录占全表记录的百分比是多少？','全表记录中，“{a}”对应的行占多少百分比？','“{a}”的记录数占整张表总行数的比例是多少？'],
    'difference':['{a}的{field}减去{b}的{field}是多少{unit}？','{a}与{b}的{field}之差是多少{unit}？按前者减后者。','计算{field}差额：从{a}扣除{b}，得到多少{unit}？'],
    'product':['{a}单价乘以数量，合计多少元？','求{a}的金额：将单价与数量相乘。','根据{a}的单价和数量计算总金额，多少元？'],
    'hours':['{a}按标准工作{n}小时，报酬合计多少元？','完成{n}小时的{a}工作，按小时标准共需多少元？','请计算{a}的报酬总额，工作量为{n}小时。'],
    'completed':['{a}要求的时长已完成{n}小时，还差多少小时？','针对{a}的时长要求，已完成{n}小时，剩余多少小时？','{a}已完成{n}小时，距离规定时长的差额是多少小时？'],
    'cycle':['{a}每周用量连续{n}周，总计多少小时？','连续{n}周按{a}的每周用量计算，总共多少小时？','计算{a}在{n}周内的用量之和，每周按规定用量。'],
    'bounds':['{a}课程数量的上限比下限多多少门？','{a}的课程数量，上限与下限相差多少门？','计算{a}课程数量的上限与下限差额，单位为门。'],
    'budget':['{a}{n}人、{b}{m}人，按各自标准计算，预算合计多少元？','按各自人均标准，{a}有{n}人，{b}有{m}人，共需多少元？','计算总预算：{a}为{n}人；{b}为{m}人；分别采用本组标准。'],
    'body_budget':['{a}和{b}按各自人数和每人标准，预算合计多少元？','根据{a}与{b}的名额及人均标准，计算两组预算总额。','两组名额均按原文，{a}和{b}按各自人均标准核算后，预算总计多少元？'],
    'missing':['{a}工作报酬合计多少元？未提供实际工时。','缺少工作小时数时，{a}的报酬总额是多少元？','计算{a}的报酬合计，实际工作时长尚未给出。'],
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(repo, split, index, family):
    rng = random.Random(126001 + index + {'train':0,'validation':100000,'test':200000}[split])
    wording = {'train':0,'validation':1,'test':2}[split]
    prefix = {'train':'青岑','validation':'白栎','test':'紫榆'}[split]+str(index)
    policy = prefix+'核算办法'
    a,b = prefix+'技术组',prefix+'研习组'
    n,m = rng.randint(2,19),rng.randint(2,19)
    x,y = rng.randint(21,199),rng.randint(21,199)
    es=[]; tables=[]; required_docs=set(); required_rows=[]; required_fields=[]
    raw_sources=[]
    def source(title, text, required=True):
        url=f'https://example.edu/synthetic-planner/{split}/{index}/{len(es)}'
        did=repo.add_document(SourceDocument(title,url,sha256=hashlib.sha256((url+text).encode()).hexdigest()))
        eid=f'planner-{split}-{index}-{len(es)}'
        repo.add_chunks([Chunk(eid,did,text,'',len(text))])
        es.append(Evidence(eid,text,1,did,title,url,chunk_id=eid))
        raw_sources.append({'doc_id':did,'title':title,'text':text})
        if required: required_docs.add(did)
        return did
    def table(title, headers, rows, required=True):
        did=source(title, '公开核算明细表，具体数据见表格。', required)
        permutation=list(range(len(headers)));rng.shuffle(permutation)
        headers=[headers[i] for i in permutation];rows=[[r[i] for i in permutation] for r in rows]
        parsed=ParsedTable(title,headers,rows)
        tid=repo.add_table(did,parsed,[[normalize_cell(v) for v in row] for row in rows])
        es[-1].table_id=tid;tables.append(tid)
        raw_sources[-1].update(headers=headers,rows=rows,table_id=tid)
        cols={r['original_name']:r['id'] for r in repo.connection.execute('SELECT * FROM table_columns WHERE table_id=?',(tid,))}
        return tid,cols
    def phrase(kind, **extra):
        return WORDS[kind][wording].format(a=a,b=b,n=n,m=m,field='金额',unit='元',**extra)
    subtype=index%6
    kind=family
    repair=family=='repair'
    effective=['constants','fields','budget'][index%3] if repair else family
    plan={'table_id':None,'steps':[]};expected=None;unit='元'
    if effective in {'retention','fields','direction'}:
        subtype=index%6
        kind= ['count','full_ratio','subset_count'][index%3] if effective=='fields' else ('ratio' if index%2==0 else 'difference') if effective=='direction' else ['lookup','sum','average','difference','ratio','product','subset_sum','table_total_ratio'][index%8]
        names=[a,a,b,b,prefix+'拓展组',prefix+'服务组']
        vals=[x,y,x+3,y+5,x+9,y+11]
        if kind in {'lookup','difference','ratio','product'}:
            names=[a,b,prefix+'拓展组'];vals=[x,y,x+9]
        else:
            # Hold out row-count/wording combinations as well as source IDs.
            extra={'train':[0,2,4],'validation':[1,3,5],'test':[6,7,8]}[split][index%3]
            names += [prefix+f'扩展{k}组' for k in range(extra)]
            vals += [rng.randint(11,211) for _ in range(extra)]
        rows=[[str(i+1),name,'K'+str(i+11),str(value)+'元',str(i+2)+'台','常规类'] for i,(name,value) in enumerate(zip(names,vals))]
        if index%2==0:
            for row in rows[:3]:row[2]=row[1]  # Same value competes in the code column.
        if kind=='table_total_ratio':
            rows.append(['合计','合计','汇总',str(sum(vals))+'元','0台','汇总类'])
        headers=['序号','名称','代码','金额','数量','类别']
        if effective=='retention':
            optional=[0,2,4,5]
            keep={1,3,*optional[:[0,1,2,4][(index//8)%4]]}
            if kind=='product':keep.add(4)
            headers=[h for i,h in enumerate(headers) if i in keep]
            rows=[[v for i,v in enumerate(row) if i in keep] for row in rows]
        elif effective=='fields':
            for k in range({'train':index%3,'validation':3,'test':4}[split]):
                headers.append(f'备注{k}')
                for row in rows:row.append('仅用于记录核对，不作为数值字段')
        tid,cols=table(policy,headers,rows)
        # A competing schema must be encountered, not silently removed.
        if index%3==0:
            table(prefix+'另一办法',['名称','金额','代码'],[[a,str(y)+'元','Z9'],[b,str(x)+'元','Z8']],False)
        plan['table_id']=tid
        field=cols['金额'];entity=cols['名称'];required_fields=[field]
        filt=[{'field_id':entity,'op':'eq','value':a}]
        if kind in {'lookup','sum','average','subset_sum'}:
            if kind=='subset_sum':filt=[{'field_id':entity,'op':'in','value':[a,b]}]
            steps=[{'id':'result','op':'sum' if kind=='subset_sum' else kind,'field_id':field,**({'filters':filt} if kind in {'lookup','subset_sum'} else {})}]
            expected=vals[0] if kind=='lookup' else sum(v for name,v in zip(names,vals) if name in [a,b]) if kind=='subset_sum' else sum(vals) if kind=='sum' else Decimal(sum(vals))/len(vals)
        elif kind in {'count','full_ratio','subset_count'}:
            required_fields=[]
            if kind=='subset_count':filt=[{'field_id':entity,'op':'in','value':[a,b]}]
            steps=[{'id':'part','op':'count','filters':filt}]
            expected=names.count(a)+names.count(b) if kind=='subset_count' else names.count(a);unit='项'
            if kind=='full_ratio':
                steps += [{'id':'whole','op':'count'},{'id':'result','op':'ratio','inputs':['part','whole']}]
                expected=Decimal(names.count(a))/len(names)*100;unit='%'
        elif kind=='table_total_ratio':
            steps=[{'id':'part','op':'sum','field_id':field,'filters':[{'field_id':entity,'op':'in','value':[a,b]}]},
                {'id':'total','op':'lookup','field_id':field,'scope':'total'},
                {'id':'result','op':'ratio','inputs':['part','total']}]
            expected=Decimal(sum(v for name,v in zip(names,vals) if name in [a,b]))/sum(vals)*100;unit='%'
        elif kind=='product':
            steps=[{'id':sid,'op':'lookup','field_id':fid,'filters':filt} for sid,fid in [('price',field),('quantity',cols['数量'])]]
            steps.append({'id':'result','op':'multiply','inputs':['price','quantity']});expected=x*2
            required_fields.append(cols['数量'])
        else:
            steps=[{'id':sid,'op':'lookup','field_id':field,'filters':[{'field_id':entity,'op':'eq','value':name}]} for sid,name in [('left',a),('right',b)]]
            steps.append({'id':'result','op':'ratio' if kind=='ratio' else 'subtract','inputs':['left','right']})
            expected=Decimal(x)/y*100 if kind=='ratio' else x-y
            if kind=='ratio':unit='%'
        question=phrase(kind)
        plan['steps']=steps
        all_rows=[r['id'] for r in repo.connection.execute('SELECT id FROM table_rows WHERE table_id=? ORDER BY row_index',(tid,))]
        required_rows=all_rows if kind in {'sum','average','full_ratio'} else all_rows[:2] if kind in {'ratio','difference'} else [r for r,name in zip(all_rows,names) if name in ([a,b] if kind in {'subset_sum','subset_count','table_total_ratio'} else [a])]
        if kind=='table_total_ratio':required_rows.append(all_rows[-1])
    else:
        if effective=='budget':
            kind='budget' if index%2==0 else 'body_budget'
            text=(f'{a}标准为{x}元/人。{b}标准为{y}元/人。' if kind=='budget' else
                f'{a}：{n}名，{x}元/人。{b}：{m}名，{y}元/人。')
            source(policy,text);question=phrase(kind);expected=n*x+m*y
        elif effective in {'constants','unanswerable'}:
            kind=['hours','completed','cycle','bounds'][index%4] if effective=='constants' else 'missing'
            text= f'{a}课程数量不少于{min(x,y)}门。{a}课程数量不超过{max(x,y)}门。' if kind=='bounds' else f'{a}每周用量为{x}小时。' if kind=='cycle' else f'{a}时长要求为{x}小时。' if kind=='completed' else f'{a}标准为{x}元/小时。'
            missing_source=effective=='unanswerable' and index%2==0
            source(prefix+'另一核算办法' if missing_source else policy,text)
            question=phrase('hours' if missing_source else kind)
            expected=None if effective=='unanswerable' else x-n if kind=='completed' else x*n
            if kind=='bounds':expected=abs(x-y)
            unit='门' if kind=='bounds' else '小时' if kind in {'cycle','completed'} else '元'
        else:
            kind='difference'
            source(policy,f'{a}预算为{x}元。{b}预算为{y}元。')
            source(prefix+'另一核算办法',f'{a}预算为{y}元。{b}预算为{x}元。',False)
            question=phrase(kind);expected=x-y
        # Long contexts contain actual extracted distractors with full scopes.
        if index%4==0:
            descriptions=['经登记与审核后在对应活动结束时统一核算，申请人应按公开流程提交材料并核实所属事项',
                '适用于完成培训并取得负责人确认的参与者，具体办理时限与材料要求按本办法附则执行']
            for j in range(3):
                text='\n'.join(f'{prefix}辅助{j}{k}项目预算为{rng.randint(201,999)}元，'+descriptions[(j+k)%2]+'。' for k in range(6))
                source(prefix+f'辅助{j}办法',text,False)
            table(prefix+'辅助明细表',['名称','金额'],[[prefix+'附件',str(x)+'元']],False)
        rng.shuffle(es)
        facts=NumericEngine.text_facts(es)
        def fid(name,unit,comparison=None):
            matches=[key for key,f in facts.items() if f['doc_id'] in required_docs and name in f['subject'] and f['unit']==unit and (comparison is None or f['comparison']==comparison)]
            assert len(matches)==1,(split,index,kind,name,unit,matches)
            return matches[0]
        if expected is not None:
            if kind in {'budget','body_budget'}:
                steps=[]
                for i,(name,count) in enumerate([(a,n),(b,m)]):
                    steps += [{'id':f'n{i}','op':'lookup',**({'literal':str(count),'unit':'人'} if kind=='budget' else {'fact_id':fid(name,'人')})},
                        {'id':f'r{i}','op':'lookup','fact_id':fid(name,'元/人')},
                        {'id':f't{i}','op':'multiply','inputs':[f'n{i}',f'r{i}']}]
                steps.append({'id':'result','op':'add','inputs':['t0','t1']})
            elif kind=='difference':
                steps=[{'id':sid,'op':'lookup','fact_id':fid(name,'元')} for sid,name in [('left',a),('right',b)]]
                steps.append({'id':'result','op':'subtract','inputs':['left','right']})
            elif kind=='bounds':
                steps=[{'id':'upper','op':'lookup','fact_id':fid(a,'门','lte')},
                    {'id':'lower','op':'lookup','fact_id':fid(a,'门','gte')},
                    {'id':'result','op':'subtract','inputs':['upper','lower']}]
            else:
                steps=[{'id':'rate','op':'lookup','fact_id':fid(a,'元/小时' if kind=='hours' else '小时')},
                    {'id':'quantity','op':'lookup','literal':str(n),'unit':'周' if kind=='cycle' else '小时'},
                    {'id':'result','op':'subtract' if kind=='completed' else 'multiply','inputs':['rate','quantity']}]
            plan['steps']=steps
    question=f'按《{policy}》，'+question
    if not plan['steps']:plan['table_id']=None
    # Generate the prompt through the actual service, including fact caps,
    # source tags, examples, repair errors, and double-encoded raw previous.
    class Index:
        def search(self,*args,**kwargs):return es
    class Capture:
        def __init__(self):self.prompts=[];self.outputs=[]
        def generate(self,prompt,**kwargs):
            self.prompts.append(prompt)
            if repair and len(self.prompts)==1:
                if index%4==0:raw='{broken-json'
                else:
                    broken=json.loads(json.dumps(plan))
                    if index%4==1:
                        broken['steps'][-1]={'id':'result','op':'lookup','literal':'999999','unit':'元'}
                    elif index%4==2 and 'inputs' in broken['steps'][-1]:
                        broken['steps'][-1]['inputs']=['missing_operand',broken['steps'][-1]['inputs'][0]]
                    else:
                        broken['table_id']=999999
                    raw=json.dumps(broken,ensure_ascii=False)
            else:raw=json.dumps(plan,ensure_ascii=False)
            self.outputs.append(raw);return raw
    capture=Capture()
    service=RAGService(Settings(ROOT,ROOT,ROOT,ROOT,ROOT,numeric_shortcuts=False),Index(),capture,repo.connection)
    result=service.answer(question)
    assert capture.prompts,(family,kind,question,service.last_diagnostics)
    if expected is not None:
        assert result.calculation and result.calculation.ok,(family,kind,question,result.calculation.error if result.calculation else result.answer)
        assert result.calculation.unit==unit and abs(result.calculation.value-Decimal(expected))<Decimal('0.000001'),(kind,expected,asdict(result))
        assert {d['doc_id'] for d in result.calculation.details}==required_docs
        if required_rows:assert {d['row_id'] for d in result.calculation.details if 'row_id' in d}==set(required_rows)
    else:
        assert not result.references and not (result.calculation and result.calculation.ok)
    if repair:assert len(capture.prompts)==2
    diag=service.last_diagnostics
    payload={'tables':diag['schema'],'facts':diag['facts']}
    reference={'value':str(expected) if expected is not None else None,'unit':unit,
        'measure_field_ids':required_fields,'row_ids':required_rows,
        'required_fact_ids':[s['fact_id'] for s in plan['steps'] if 'fact_id' in s],
        'required_doc_ids':sorted(required_docs),'kind':kind,'raw_sources':raw_sources}
    assert set(reference['required_fact_ids'])<={f['fact_id'] for f in diag['facts']}
    return {'id':f'planner-{split}-{family}-{index}','family':family,'kind':kind,'split':split,
        'template_family':f'{kind}-wording-{wording}','source_family':f'{split}-{family}-{index}',
        'question':question,'prompt':capture.prompts[-1],'context':payload,'plan':plan,
        'expected':reference['value'],'unit':unit,'table_ids':tables,
        'source_doc_ids':sorted(required_docs) if expected is not None else [],
        'evidence_map':{e.evidence_id:asdict(e) for e in es},'reference':reference,
        'repair_previous':capture.outputs[0] if repair else None,
        'repair_feedback':diag['attempts'][-1].get('feedback') if repair else None}
