from __future__ import annotations

import sqlite3
import json
import re
from decimal import Decimal

from campus_rag.tools.numbers import convert_unit, question_quantities, asks_total
from campus_rag.types import CalculationResult, Evidence, FilterSpec, OperandSpec, QueryPlan
from campus_rag.tools.temporal import matches_period, question_periods
from campus_rag.ingest.navigation import navigation_table_ids
from campus_rag.database.normalization import compact, quantity, VERSION, header_unit, field_label, year_header


class PlanError(ValueError):
    """Only schema/plan mistakes are repairable; missing evidence is not."""


class QuestionConstraintError(PlanError):
    """An executable structure does not match explicit question constraints."""


def _field_question_score(column,question,values=()):
    name = compact(field_label(column.get('original_name',column.get('name',''))))
    name = re.sub(r'\([^)]*\)','',name)
    stem = re.sub(r'(?:名称|名字|类别|类型)$','',name)
    query = compact(question)
    adjacent = len(stem)>1 and any(re.search(re.escape(stem)+r'(?:名称)?(?:为|等于|是)?[“"‘\']?'+re.escape(compact(v)),query) for v in values)
    return (100 if adjacent else 0)+(len(name)+10 if len(name)>1 and name in query else 0)+(len(stem) if len(stem)>1 and stem in query else 0)


def _legal_fields(columns):
    return '；合法字段：'+','.join(str(fid)+':'+c['original_name']+'('+_field_role(c)+','+(c['unit'] or '未标单位')+')' for fid,c in columns.items())


def _excluded_object(value,question):
    clauses = re.findall(r'除(?:了)?(.*?)(?:之外|以外)|(?:不含|不包括|不包含|排除)([^，。；？?]+)',compact(question))
    return any(_object_mentioned(value,text) for parts in clauses for text in parts)


def _object_key(value):
    return re.sub(r'(?:工程)?实验教学中心|实验室|中心|学院|专业|场馆|组别','',compact(value))


def _object_query(question):
    query = compact(question)
    return re.sub(r'([一二三四五六七八九十]+(?:[、和与及][一二三四五六七八九十]+)+)等奖',
        lambda m:'、'.join(n+'等奖' for n in re.split('[、和与及]',m[1])),query)


def _award_alias(value):
    text = compact(value)
    return text[:-2] if re.search(r'[一二三四五六七八九十]+等奖学金$',text) else text


def _mentioned_objects(values,question,strict=False):
    labels = list(dict.fromkeys(v for v in values if _object_mentioned(v,question)))
    aliases = {}
    for value in labels:
        aliases.setdefault(_award_alias(value),set()).add(compact(value))
    if any(len(names)>1 and not all(n in _object_query(question) for n in names) for names in aliases.values()):
        if strict:
            raise QuestionConstraintError('问题约束：对象简称对应多个真实名称，需明确统计对象')
        return []
    return labels


def _object_mentioned(value, question):
    text, query = compact(value), _object_query(question)
    if not text or text in {'合计','总计','小计'}:
        return False
    if text in query:
        return True
    alias = _award_alias(text)
    if alias!=text and alias in query:
        return True
    suffix = re.search(r'((?:工程)?实验教学中心|实验室|中心|场馆)$',text)
    if not suffix:
        return False
    core = text[:suffix.start()]
    kind = re.sub(r'^工程','',suffix[1])
    return len(core)>=2 and core in query and kind in query


def _field_role(column, values=()):
    name = compact(field_label(column.get('original_name',column.get('name',''))))
    dtype = column.get('data_type',column.get('type',''))
    if re.search(r'序号|编号|代码|学号|工号|^id$',name):
        return 'identifier'
    if dtype=='number' and column.get('unit') and column['unit']!='年':
        return 'measure'
    if dtype=='year' or re.search(r'年度|年份|日期',name) or (
            re.search(r'学年|时间',name) and not (dtype=='number' and column.get('unit'))):
        return 'time'
    if dtype=='number':
        return 'measure'
    nonempty = [compact(v) for v in values if compact(v) and compact(v) not in {'合计','总计','小计'}]
    if re.search(r'类别|类型|学院|组别|分类',name) or len(set(nonempty))<len(nonempty):
        return 'category'
    return 'entity'


def _business_scope(text):
    return tuple(word for word in ['对照','替代','安排','汇总','明细','预算','名单','统计','本科','研究生'] if word in text)


class StructuredQueryEngine:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.connection.row_factory = sqlite3.Row
        self.navigation_tables = navigation_table_ids(connection)

    def schema_context(self, table_ids, periods, question=''):
        output = []
        seen_contents = {}
        for tid in dict.fromkeys(table_ids):
            if tid in self.navigation_tables:
                continue
            table = self.connection.execute('SELECT t.*,d.title,d.published_at FROM tables t JOIN documents d ON d.id=t.doc_id WHERE t.id=?',(tid,)).fetchone()
            if table is None or not matches_period(table['title'],table['published_at'],periods):
                continue
            columns = self.connection.execute('SELECT id,original_name,data_type,unit FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,)).fetchall()
            rows = self.connection.execute('SELECT id FROM table_rows WHERE table_id=? ORDER BY row_index',(tid,)).fetchall()
            examples = []
            # Prefer rows mentioned in the question, but never use samples as the execution domain.
            ranked = []
            for row in rows:
                cells = self.connection.execute('SELECT column_id,raw_value FROM table_cells WHERE row_id=? ORDER BY column_id',(row['id'],)).fetchall()
                mentions = [compact(c['raw_value']) for c in cells
                    if any(col['id']==c['column_id'] and col['data_type']=='text' and
                        _field_role(dict(col))!='identifier' for col in columns)
                    and _object_mentioned(c['raw_value'],question)]
                score = sum(map(len,mentions))
                position = min((compact(question).find(v) for v in mentions),default=len(question))
                ranked.append((score,position,row['id'],cells))
            for _,_,rid,cells in sorted(ranked,key=lambda x:(-x[0],x[1]))[:3]:
                examples.append({'row_id':rid,'cells':{str(c['column_id']):c['raw_value'][:120] for c in cells}})
            described, matched = [], []
            for col in columns:
                values = [next(c['raw_value'] for c in cells if c['column_id']==col['id']) for _,_,_,cells in ranked]
                role = _field_role(dict(col),values)
                described.append({'field_id':col['id'],'name':col['original_name'],
                    'type':col['data_type'],'unit':col['unit'],'role':role})
                labels = _mentioned_objects(values,question) if role in {'entity','category'} else []
                labels.sort(key=lambda v:_object_query(question).find(_award_alias(v)) if _award_alias(v) in _object_query(question)
                    else compact(question).find(_object_key(v)))
                if labels:
                    matched.append({'field_id':col['id'],'values':labels})
            business = _business_scope(table['title']+' '+table['name'])
            output.append({'table_id':tid,'name':table['name'],'document':table['title'],
                'periods':question_periods(table['title']+' '+table['name']),
                'columns':described,'question_matches':matched,'business_scope':business,
                'total_row_count':sum(any(compact(c['raw_value']) in {'合计','总计','小计'} for c in cells) for _,_,_,cells in ranked),
                'row_count':len(rows),'examples_only':True,'examples':examples})
            signature = (business,tuple((compact(c['original_name']),c['unit'],c['data_type']) for c in columns),
                tuple(tuple(compact(c['raw_value']) for c in cells) for _,_,_,cells in ranked))
            if signature in seen_contents:
                old_index = seen_contents[signature]
                old_periods = output[old_index]['periods']
                new_periods = output[-1]['periods']
                if not old_periods or not new_periods or old_periods==new_periods:
                    duplicate = output.pop()
                    if new_periods and not old_periods:
                        output[old_index] = duplicate
                    continue
            seen_contents[signature] = len(output)-1
        exact = [t for t in output if compact(t['name']) in compact(question) or compact(t['document']) in compact(question)]
        requested = set(_business_scope(question))-{'统计','本科','研究生'}
        scoped = [t for t in output if requested & set(t['business_scope'])] if requested else []
        return (exact or scoped or output)[:4]

    def execute_steps(self, plan, table_ids, question, facts=None):
        trace, used = [], []
        values = {}
        facts = facts or {}
        try:
            if not plan.steps:
                raise ValueError('证据不足，未生成有效计算步骤')
            if plan.table_id is not None:
                if plan.table_id not in table_ids or plan.table_id in self.navigation_tables:
                    raise PlanError('table_id不在实际候选Schema内；合法table_id：'+','.join(map(str,table_ids)))
                table, columns, rows = self._step_table(plan.table_id,plan.time_range)
                self._check_competing_tables(table,columns,rows,table_ids,plan.time_range,question)
            else:
                table, columns, rows = None, {}, []
            for step in plan.steps:
                sid, op = step['id'],step['op']
                inputs = step.get('inputs')
                if inputs is not None:
                    if set(step)-{'id','op','inputs'} or op not in {'add','subtract','multiply','divide','ratio','average','sum'}:
                        raise PlanError(sid+': 算术步骤字段不合法')
                    operands = [self._step_input(v,values,question) for v in inputs]
                    if op=='multiply':
                        self._check_person_rate_pair(operands,question,facts)
                    value,unit,expression = self._step_arithmetic(op,operands)
                    value_record = {'value':value,'unit':unit,'expression':expression,
                                    'details':[d for v in operands for d in v['details']],
                                    'question_inputs':[q for v in operands for q in v.get('question_inputs',[])]}
                    kinds = [v.get('kind','value') for v in operands]
                    value_record['kind'] = 'count' if op in {'add','sum'} and set(kinds)=={'count'} else op
                    if op=='divide' and kinds==['sum','count'] and {
                            d.get('row_id') for d in operands[0]['details']}=={d.get('row_id') for d in operands[1]['details']}:
                        value_record.update(kind='average',unit=operands[0]['unit'])
                    if op in {'add','sum','subtract','average'} and all(v.get('per_unit')==operands[0].get('per_unit') for v in operands) and operands[0].get('per_unit'):
                        value_record['per_unit'] = operands[0]['per_unit']
                    if op=='multiply' and any(v.get('per_unit') for v in operands):
                        value_record['expanded_period'] = True
                elif 'literal' in step:
                    if op!='lookup' or set(step)!={'id','op','literal','unit'}:
                        raise PlanError(sid+': 问题常数只能使用lookup和literal、unit')
                    value_record = self._step_input({'literal':step['literal'],'unit':step['unit']},values,question)
                elif 'fact_id' in step:
                    if op!='lookup' or set(step)-{'id','op','fact_id'} or step['fact_id'] not in facts:
                        raise PlanError(sid+': fact_id不在已核验正文事实中')
                    fact = facts[step['fact_id']]
                    self._check_fact_scope(fact,question)
                    value_record = {'value':Decimal(fact['value']),'unit':fact['unit'],
                                    'expression':fact['value'],'details':[fact]}
                    period = re.fullmatch(r'每(小时|分钟|天|周|月|年|学期|学年)',fact.get('scope',{}).get('period',''))
                    if period and '/' not in fact['unit']:
                        value_record['per_unit'] = period[1]
                else:
                    if table is None:
                        raise PlanError(sid+': 缺少真实table_id')
                    if op not in {'filter','lookup','sum','count','average'}:
                        raise PlanError(sid+': 缺少inputs')
                    selected = rows
                    if 'rows' in step:
                        previous = values.get(step['rows'],{})
                        if 'rows' not in previous:
                            raise PlanError(sid+': rows必须引用filter结果')
                        selected = previous['rows']
                    if 'rows' not in step or 'scope' in step:
                        scope = step.get('scope','detail')
                        selected = [r for r in selected if r['is_total']==(scope=='total')]
                    selected = self._step_filter(selected,columns,step.get('filters',[]),question)
                    if any(r.get('year_unknown') for r in selected):
                        raise ValueError(sid+': 匹配记录年份缺失，无法确认统计范围')
                    if not selected:
                        raise ValueError(sid+': 筛选后没有完整匹配记录，证据不足')
                    if 'expected_count' in step and len(selected)!=step['expected_count']:
                        raise ValueError(sid+': 匹配数量与明确范围不符')
                    if any(r['is_total'] for r in selected) and len(selected)>1 and op!='filter':
                        raise ValueError(sid+': 匹配多个合计或小计行，统计范围不明确')
                    if op=='filter':
                        values[sid] = {'rows':selected}
                        trace.append({'id':sid,'op':op,'row_ids':[r['id'] for r in selected],'count':len(selected)})
                        continue
                    field = step.get('field_id')
                    if (op!='count' or field is not None) and field not in columns:
                        raise PlanError(sid+': field_id不在所选表格Schema内'+_legal_fields(columns))
                    if op in {'sum','average'} and _field_role(columns[field]) in {'identifier','time'}:
                        raise PlanError(sid+': 标识或时间字段不可求和或平均，请选择度量字段或使用count统计记录')
                    details = []
                    for row in selected:
                        cell = row['cells'].get(field) if field else None
                        if op!='count' and (cell is None or cell['numeric_value'] is None):
                            raise ValueError(sid+': 参与计算的记录存在缺失或非确定数值')
                        if op!='count' and not cell['unit']:
                            raise ValueError(sid+': 数值单位未知，需要离线核对表头或原文')
                        details.append(self._step_detail(table,row,cell,columns.get(field)))
                    if op=='count':
                        value,unit,expression = Decimal(len(selected)),'项',f'count({len(selected)}条明细)'
                    else:
                        units = {d['unit'] for d in details}
                        if len(units)!=1:
                            raise ValueError(sid+': 单位无法统一')
                        unit = units.pop()
                        numbers = [Decimal(d['numeric_value']) for d in details]
                        if op=='lookup' and len(numbers)!=1:
                            raise ValueError(sid+': lookup匹配多个数值，统计范围不明确')
                        value = sum(numbers,Decimal(0))
                        expression = ' + '.join(format(v,'f') for v in numbers)
                        if op=='average':
                            value /= len(numbers)
                            expression = f'({expression}) / {len(numbers)}'
                    value_record = {'value':value,'unit':unit,'expression':expression,'details':details,
                                    'kind':'sum' if op=='lookup' and step.get('scope')=='total' else op}
                if not value_record['value'].is_finite():
                    raise ValueError(sid+': 计算结果非有限数值')
                values[sid] = value_record
                trace.append({'id':sid,'op':op,'operands':[
                    {'value':str(v['value']),'unit':v['unit'],**({'source':v['source']} if 'source' in v else {})} for v in operands] if inputs is not None else [],
                    'value':str(value_record['value']),'unit':value_record['unit'],
                    'expression':value_record['expression'],'details':value_record['details'],
                    **({'source':value_record['source']} if 'source' in value_record else {})})
            result = values[plan.output]
            if 'value' not in result:
                raise PlanError('最终步骤必须产生数值')
            needed = {plan.output}
            for step in reversed(plan.steps):
                if step['id'] in needed:
                    needed.update(v for v in step.get('inputs',[]) if isinstance(v,str))
                    if 'rows' in step:
                        needed.add(step['rows'])
            if needed!={step['id'] for step in plan.steps}:
                raise PlanError('存在未参与最终结果的步骤')
            try:
                self._check_question_operation(plan,question,result,facts)
                final = next(s for s in plan.steps if s['id']==plan.output)
                if final['op']=='ratio' and '占' in question:
                    sides = question.split('占',1)
                    for side,(part,item) in enumerate(zip(sides,final['inputs'])):
                        operand = self._step_input(item,values,question)
                        if table is not None:
                            try:
                                self._check_requested_objects(rows,columns,part,operand['details'])
                            except PlanError as exc:
                                raise PlanError(('分子' if side==0 else '分母')+'对象或范围不符：'+str(exc)) from exc
                        else:
                            expected = {f.get('subject','') for f in facts.values()
                                if _object_mentioned(f.get('subject',''),part)}
                            actual = {f.get('subject','') for f in operand['details']}
                            if expected and expected!=actual:
                                raise PlanError('比例操作数对象与“占”两侧不符，请分别核对分子和分母的fact_id')
            except PlanError as exc:
                raise QuestionConstraintError('问题约束：'+str(exc)) from exc
            # Include every executed source, including intermediate computations.
            used = [d for entry in trace for d in entry.get('details',[])]
            if table is not None:
                scope_details = result['details']
                final = plan.steps[-1]
                denominator_question = question.split('占',1)[-1] if '占' in question else ''
                if final['op']=='ratio' and re.search(r'全部|所有|整张|全表|表内合计|总面积|总额|总人数',denominator_question):
                    denominator = self._step_input(final['inputs'][1],values,question)
                    actual_rows = {d['row_id'] for d in denominator['details'] if 'row_id' in d}
                    detail_rows = {r['id'] for r in rows if not r['is_total']}
                    total_rows = {r['id'] for r in rows if r['is_total']}
                    if actual_rows!=detail_rows and not (len(total_rows)==1 and actual_rows==total_rows):
                        raise QuestionConstraintError('问题约束：分母未覆盖问题要求的全表明细或唯一合计行，不能使用局部计数')
                if final['op']=='ratio' and isinstance(final['inputs'][0],str) and any(
                    s['id']==final['inputs'][0] and s['op'] in {'sum','add','average','count'} for s in plan.steps):
                    scope_details = values[final['inputs'][0]]['details']
                self._check_requested_objects(rows,columns,question,scope_details)
            unique = {(d['table_id'],d['row_id'],d.get('cell_id')):d for d in used if 'table_id' in d}
            evidence = self._evidence_from_details(list(unique.values()))
            for d in used:
                if 'evidence' in d and d['evidence'] not in evidence:
                    evidence.append(d['evidence'])
            return CalculationResult(True,result['value'],result['unit'],result['expression'],used,
                                     evidence=evidence,steps=trace)
        except (ValueError,ArithmeticError,KeyError,TypeError) as exc:
            return CalculationResult(False,error=str(exc),steps=trace,repairable=isinstance(exc,PlanError))

    def _step_table(self,tid,periods):
        table = self.connection.execute('SELECT t.*,d.title,d.source_url,d.department,d.published_at FROM tables t JOIN documents d ON d.id=t.doc_id WHERE t.id=?',(tid,)).fetchone()
        try:
            metadata = self.connection.execute('SELECT * FROM table_metadata WHERE table_id=?',(tid,)).fetchone()
        except sqlite3.OperationalError:
            metadata = None
        if metadata is None or metadata['version']!=VERSION:
            raise ValueError('表格尚未离线规范化，请运行scripts/normalize_tables.py')
        if not matches_period(table['title'],table['published_at'],periods):
            raise ValueError('表格年份与问题不符')
        columns = {c['id']:dict(c) for c in self.connection.execute('SELECT * FROM table_columns WHERE table_id=?',(tid,))}
        row_meta = json.loads(metadata['row_metadata'])
        rows = []
        for row in self.connection.execute('SELECT * FROM table_rows WHERE table_id=? ORDER BY row_index',(tid,)):
            meta = row_meta.get(str(row['id']))
            if meta is None:
                raise ValueError('表格规范化元数据不完整')
            cells = {c['column_id']:dict(c) for c in self.connection.execute('SELECT * FROM table_cells WHERE row_id=?',(row['id'],))}
            if len(cells)!=len(columns):
                raise ValueError('表格存在缺失单元格，需要离线复核')
            for fid,cell in cells.items():
                if compact(cell['raw_value'])!=cell['normalized_value']:
                    raise ValueError('单元格原始值与规范化值不一致，需要离线复核')
                if columns[fid]['data_type']=='number':
                    expected,unit = quantity(cell['raw_value'],header_unit(columns[fid]['original_name']) or table['unit'])
                    stored = Decimal(cell['numeric_value']) if cell['numeric_value'] is not None else None
                    if expected!=stored or (expected is not None and unit!=cell['unit']):
                        raise ValueError('数值缺失或单位规范化与原始数据不一致，需要离线复核')
            actual_periods = {p for fid,cell in cells.items() if year_header(columns[fid]['original_name'])
                for p in question_periods(cell['raw_value'])}
            if actual_periods!=set(meta['periods']):
                raise ValueError('年份规范化与原始数据不一致，需要离线复核')
            # A year column imposes an additional row-level restriction.
            if periods and meta['periods'] and not any(p in meta['periods'] for p in periods):
                continue
            rows.append({**dict(row),**meta,'cells':cells,
                'year_unknown':bool(periods and any(year_header(c['original_name']) for c in columns.values()) and not actual_periods)})
        return dict(table),columns,rows

    def _check_competing_tables(self,table,columns,rows,table_ids,periods,question):
        if compact(table['name']) in compact(question) or compact(table['title']) in compact(question):
            return
        signature = [c['normalized_name'] for c in columns.values()]
        def contents(items):
            return [[c['raw_value'] for c in r['cells'].values()] for r in items]
        for tid in table_ids:
            if tid==table['id']:
                continue
            other,cols,candidates = self._step_table(tid,periods)
            if signature==[c['normalized_name'] for c in cols.values()] and contents(rows)!=contents(candidates):
                raise ValueError('多张同结构候选表数据不同，需明确资料或统计范围')

    def _check_requested_objects(self,rows,columns,question,details):
        # Rebuild the threshold domain from the question, independently of the
        # proposed plan, so omitted filters cannot legitimize extra records.
        numeric_filters = []
        relations = {'大于等于':'gte','不少于':'gte','至少':'gte','小于等于':'lte','不超过':'lte','至多':'lte','大于':'gt','超过':'gt','小于':'lt','低于':'lt'}
        for match in re.finditer('('+'|'.join(relations)+r')\s*(\d+(?:\.\d+)?)(万元|亿元|元|万人|千人|人|平方米|台|个|项|小时|%)?',question):
            candidates = [c for c in columns.values() if c['data_type']=='number' and c['unit'] and
                (not match[3] or quantity(match[2]+match[3])[1]==c['unit'])]
            best = max((_field_question_score(c,question) for c in candidates),default=-1)
            candidates = [c for c in candidates if _field_question_score(c,question)==best]
            if len(candidates)!=1:
                raise QuestionConstraintError('问题约束：数值阈值无法唯一绑定字段'+_legal_fields(columns))
            numeric_filters.append({'field_id':candidates[0]['id'],'op':relations[match[1]],'value':match[2]})
        domain = self._step_filter([r for r in rows if not r['is_total']],columns,numeric_filters,question)
        candidates = []
        for fid,col in columns.items():
            if col['data_type']!='text' or _field_role(col) in {'identifier','time'}:
                continue
            labels = set(_mentioned_objects([r['cells'][fid]['normalized_value'] for r in rows
                if not r['is_total']],question,strict=True))
            if labels:
                excluded = {v for v in labels if _excluded_object(v,question)}
                included = labels-excluded
                required = {r['id'] for r in domain if r['cells'][fid]['normalized_value'] not in excluded
                    and (not included or r['cells'][fid]['normalized_value'] in included)}
                candidates.append((len(labels),_field_question_score(col,question,labels),fid,required,labels))
        if not candidates:
            if numeric_filters or re.search(r'全部|所有|整张',question):
                required = {r['id'] for r in domain}
                actual = {d['row_id'] for d in details if 'row_id' in d}
                if actual!=required:
                    raise QuestionConstraintError('问题约束：结果未覆盖问题指定的完整筛选范围')
            return
        # More complete object coverage and explicit field names disambiguate
        # repeated categories from another column containing a similar label.
        best = max((c[0],c[1]) for c in candidates)
        choices = [c for c in candidates if c[:2]==best]
        if len({tuple(sorted(c[3])) for c in choices})>1:
            raise QuestionConstraintError('问题约束：对象可绑定多个字段，统计范围不明确'+_legal_fields(columns))
        _,_,fid,required,labels = choices[0]
        actual = {d['row_id'] for d in details if 'row_id' in d}
        if required-actual:
            raise QuestionConstraintError('问题约束：计划未覆盖列举对象的全部匹配记录：'+'、'.join(sorted(labels)))
        detail_ids = {r['id'] for r in rows if not r['is_total']}
        if (actual & detail_ids)-required:
            raise QuestionConstraintError('问题约束：包含列举对象范围之外的记录，应核对筛选字段'+_legal_fields(columns))

    @staticmethod
    def _check_fact_scope(fact,question):
        scope = fact.get('scope',{})
        pattern = r'第[一二三四五六七八九十百\d]+条'
        excluded = set(re.findall(pattern,' '.join(re.findall(r'(?:不考虑|不含|排除)[^，。；]*',question))))
        required = set(re.findall(pattern,question))-excluded
        if scope.get('clause') in excluded or (required and scope.get('clause') not in required):
            raise QuestionConstraintError('问题约束：fact_id条款与问题指定范围不符，应选择'+ '、'.join(sorted(required)))
        comparison = fact.get('comparison','eq')
        upper = bool(re.search(r'上限|最高|最多|至多|不超过',question))
        lower = bool(re.search(r'下限|最低|至少|不少于|不低于|门槛',question))
        if upper and not lower and comparison in {'gte','gt'}:
            raise QuestionConstraintError('问题约束：问题要求上限，不能选择下限fact_id')
        if lower and not upper and comparison in {'lte','lt'}:
            raise QuestionConstraintError('问题约束：问题要求下限，不能选择上限fact_id')
        if comparison!='eq' and not re.search(r'上限|下限|最高|最低|至少|至多|不少于|不超过|不低于|门槛|标准|要求|界限',question):
            raise ValueError('证据只提供界限或标准，缺少实际发生值，不能据此确定实际结果')
        if re.search(r'示例|例如|假设|举例',scope.get('context','')) and re.search(r'我|本人|个人',question) and not re.search(r'示例|例如|假设|举例|仅按|只按|只作',question):
            raise ValueError('公共资料中的示例不能代替个人实际计算条件')

    @staticmethod
    def _fact_source_labels(fact):
        title = fact.get('source_title') or fact.get('title') or getattr(fact.get('evidence'),'title','')
        return [title,fact.get('source_section',''),fact.get('table_name','')]

    @staticmethod
    def requested_fact_sources(question,facts):
        requested = {m[1] for m in re.finditer(r'《([^》]+)》',question)
            if re.search(r'办法|规定|指南|章程|通知|标准|方案$',m[1])
            or re.search(r'按|根据|依据|参照|依照',question[max(0,m.start()-8):m.start()])}
        requested.update(re.findall(r'[“"]([^”"]+)[”"]\s*(?:评选)?(?:办法|规定|指南|章程|通知)',question))
        for fact in facts.values():
            for label in StructuredQueryEngine._fact_source_labels(fact):
                for name in re.findall(r'[“"]([^”"]+)[”"]',label):
                    if name in question and re.search(r'办法|规定|指南|章程|通知',label):
                        requested.add(name)
        return {re.sub('[《》“”"\']','',compact(v)) for v in requested}

    @staticmethod
    def _check_fact_sources(question,details,facts):
        used = [d for d in details if 'fact_id' in d or 'table_id' in d]
        if not used:
            return
        requested = StructuredQueryEngine.requested_fact_sources(question,
            {**facts,**{'used_'+str(i):d for i,d in enumerate(used)}})
        def normalized(value):
            return re.sub('[《》“”"\']','',compact(value))
        if not requested:
            return
        covered = set()
        for fact in used:
            labels = StructuredQueryEngine._fact_source_labels(fact)
            matched = {name for name in requested if any(name in normalized(label) for label in labels)}
            if not matched:
                raise PlanError('操作数来源与问题指定资料不符；要求：'+'、'.join(sorted(requested))+
                    '；当前来源：'+' / '.join(filter(None,labels)))
            covered.update(matched)
        if requested-covered:
            raise PlanError('计算未覆盖问题明确列举的资料：'+'、'.join(sorted(requested-covered)))

    @staticmethod
    def _check_person_rate_pair(operands,question,facts=None):
        if len(operands)!=2 or {v['unit'] for v in operands}!={'人','元/人'}:
            return
        people = next(v for v in operands if v['unit']=='人')
        rate = next(v for v in operands if v['unit']=='元/人')
        counts = [d for d in people['details'] if 'fact_id' in d]
        rates = [d for d in rate['details'] if 'fact_id' in d or 'table_id' in d]
        def subject(fact):
            return re.sub(r'(?:人数|名额|(?:人均)?标准)$','',compact(fact.get('subject') or fact.get('row_label','')))
        # Keep every matching occurrence: equal quantities can legitimately
        # belong to different groups without changing the public plan format.
        labels = {label for fact in [*(facts or {}).values(),*rates]
            for label in [subject(fact),compact(fact.get('scope',{}).get('group',''))] if label}
        for fee in rates:
            right = compact(fee.get('scope',{}).get('group','')) or subject(fee)
            if not fee.get('scope',{}).get('group') and re.match(r'^(?:人均|生均|每人|统一|共同|通用)',right):
                continue
            explicit = right and re.search(r'(?:按|采用|使用|参照)'+re.escape(right)+r'(?:的)?(?:人均)?标准',compact(question))
            shared = re.search(r'统一|共同|通用',fee.get('field','')) and re.search(r'统一|共同|通用',question)
            if (explicit or shared) and '各自' not in question:
                continue
            for item in people.get('question_inputs',[]):
                bindings = []
                for source in item.get('occurrences',[]):
                    prefix = question[:source['span'][0]]
                    clause = re.split(r'[，,。；;、：:]',prefix)[-1]
                    matches = [(m.end(),len(label),label) for label in labels
                        for m in re.finditer(re.escape(label),compact(clause))]
                    # Object names absent from the supplied facts still must
                    # not silently inherit a different group's rate.
                    local = re.search(r'([\u4e00-\u9fffA-Za-z]{2,20}(?:组|项目|人员))\s*(?:有|共|为)?$',clause)
                    if local and not matches:
                        matches.append((len(compact(clause)),len(local[1]),local[1]))
                    if matches:
                        binding = max(matches)[2]
                        source['subject'] = binding
                        bindings.append(binding)
                if right and bindings and right not in bindings:
                    raise QuestionConstraintError('问题约束：问题人数与费率对象配对不符')
                if not bindings and len(question_quantities(question,'人'))>1:
                    raise QuestionConstraintError('问题约束：问题人数所属对象不明确，无法确认费率配对')
                matched = next((source for source in item.get('occurrences',[]) if source.get('subject')==right),None)
                if matched and 'source' in people:
                    people['source'].update(matched)
        for count in counts:
            for fee in rates:
                if (count.get('doc_id')!=fee.get('doc_id') or
                    (count.get('source_section') and fee.get('source_section') and
                     count['source_section']!=fee['source_section'])):
                    raise QuestionConstraintError('问题约束：人数与费率来源配对不符，不能跨办法套用标准')
                cg,rg = count.get('scope',{}).get('group'),fee.get('scope',{}).get('group')
                if cg and rg and cg!=rg:
                    raise QuestionConstraintError('问题约束：人数与费率分组配对不符')
                left,right = subject(count),subject(fee)
                explicit_rate = right and re.search(r'(?:按|采用|使用)'+re.escape(right)+r'(?:的)?(?:人均)?标准',compact(question))
                shared = re.search(r'统一|共同|通用',fee.get('field','')) and re.search(r'统一|共同|通用',question)
                if left and right and left!=right and not ((explicit_rate or shared) and '各自' not in question):
                    raise QuestionConstraintError('问题约束：人数与费率主体配对不符，请按各自主体匹配fact_id')

    @staticmethod
    def _check_question_operation(plan,question,result,facts=None):
        StructuredQueryEngine._check_fact_sources(question,result['details'],facts or {})
        groups = {f.get('scope',{}).get('group','') for f in (facts or {}).values()}
        requested_groups = {g for g in groups if g and compact(g) in compact(question)}
        requested_groups = {g for g in requested_groups if not any(g!=other and g in other for other in requested_groups)}
        used_groups = {d.get('scope',{}).get('group','') for d in result['details'] if 'fact_id' in d}
        if requested_groups and used_groups and (used_groups-requested_groups or requested_groups-used_groups):
            raise PlanError('正文分组口径与问题不符；要求：'+'、'.join(sorted(requested_groups)))
        if not requested_groups:
            def scope_key(fact):
                scope = fact.get('scope',{})
                return (fact.get('doc_id'),fact.get('source_section',''),fact.get('subject'),
                    fact.get('field'),fact.get('unit'),fact.get('comparison'),scope.get('clause'),scope.get('period'))
            alternatives = {}
            for fact in (facts or {}).values():
                group = fact.get('scope',{}).get('group')
                if group:
                    alternatives.setdefault(scope_key(fact),set()).add(group)
            for detail in result['details']:
                possible = alternatives.get(scope_key(detail),set()) if 'fact_id' in detail else set()
                if len(possible)>1:
                    raise ValueError('正文口径不明确：同一主体和指标涉及多个分组，请明确：'+'、'.join(sorted(possible)))
        compare = bool(re.search(r'相差|差额|比.*(?:多|少|高|低).*多少|减去|还差|还剩',question))
        final = next(s for s in plan.steps if s['id']==plan.output)
        record_count = bool(re.search(r'(?:多少|几)(?:条|行)|共有几(?:个|项)|有多少个',question))
        if record_count and not re.search(r'占|百分比|平均',question) and not compare and result.get('kind')!='count':
            raise PlanError('问题要求记录计数，结果必须来自count或合法计数相加，不能对金额等度量求和')
        if record_count and result.get('kind')=='count' and result['value']!=len({
                (d['table_id'],d['row_id']) for d in result['details'] if 'row_id' in d}):
            raise PlanError('记录计数重复使用同一记录，不能把重叠集合重复相加')
        if not compare and not re.search(r'占|百分比',question):
            average = '平均' in question.replace('平均学分绩点','绩点')
            if average and result.get('kind')!='average':
                raise PlanError('问题要求平均值，应使用average或同范围sum除以count')
            if not average and re.search(r'合计|总计|之和|总额',question) and result.get('kind')=='average':
                raise PlanError('问题要求合计，不能返回平均值')
        if compare and final['op']!='subtract':
            raise PlanError('问题要求比较差额，最终步骤必须subtract，不能只返回一个lookup')
        total = asks_total(question)
        person_rates = [d for d in result['details'] if d.get('unit')=='元/人']
        if total and person_rates:
            people = question_quantities(question,'人')
            if any(re.search(r'(?:约|大约|近|至少|至多|不少于|不超过|[-—–－～~至到])\s*$',question[:q['span'][0]])
                    or re.match(r'\s*(?:左右|上下|以上|以下)',question[q['span'][1]:]) for q in people):
                raise PlanError('人数仅有范围或近似值，无法确定实际总额')
            from collections import Counter
            requested = Counter(Decimal(q['literal']) for q in people)
            used = Counter(Decimal(q['literal']) for q in result.get('question_inputs',[]) if q['unit']=='人')
            monetary_total = not compare and not re.search(r'占|百分比|平均',question)
            if requested-used or (monetary_total and (result['unit']!='元' or result.get('kind') not in {'multiply','add','sum'})):
                raise PlanError('问题要求总额：必须使用问题明确的人数和人均费率，不能只返回单人金额')
        def subject(fact):
            return re.sub(r'\([^)]*\)','',compact(fact.get('subject','')))
        text_only = bool(result['details']) and all('fact_id' in d for d in result['details'])
        named = {subject(f) for f in (facts or {}).values() if text_only and len(subject(f))>=2 and subject(f) in compact(question)}
        actual = {subject(f) for f in result['details']}
        if len(named)>1 and named-actual:
            raise PlanError('未覆盖问题列举的正文对象：'+'、'.join(sorted(named-actual)))
        if '上限' in question and '下限' in question:
            available = {f.get('comparison') for f in (facts or {}).values()}
            used_bounds = {f.get('comparison') for f in result['details']}
            if {'gte','lte'}<=available and not {'gte','lte'}<=used_bounds:
                raise PlanError('上限与下限必须分别使用对应的两条界限事实，不能重复同一边界')
        required = 'multiply' if re.search(r'相乘|乘积|乘以|单价.*数量|标准乘',question) else 'divide' if '除以' in question else None
        if required and final['op']!=required:
            raise PlanError('问题明确要求'+required+'，不能只返回一个操作数，必须保留两个操作数和最终运算')
        if not compare and re.search(r'百分比|占.*(?:比例|多少)',question) and final['op']!='ratio':
            raise PlanError('问题要求百分比，最终步骤必须ratio，不能只返回计数或求和')
        if result['value']<0 and re.search(r'相差|高多少|多多少|还剩|还差',question):
            raise PlanError('差额方向与问题不符：请核对被比较对象顺序和周期，不能直接相减不同周期的上限')
        target = re.search(r'多少(元|小时|平方米|学期|学分|学时|门)',compact(question))
        person_basis = (question.count('每人')>=2 or re.search(r'人均|生均',question)
            or ('每人' in question and '标准' in question))
        rate_difference = (compare and person_basis and result['unit']=='元/人'
            and not re.search(r'总额|预算|合计|总计|总金额|共需',question))
        target_unit = '元/人' if target and target[1]=='元' and rate_difference else target[1] if target else ''
        if target and result['unit']!=target_unit:
            raise PlanError('结果单位与问题要求不符：要求'+target[1]+'；请补齐费率、时长等必要运算')
        periods = {d.get('scope',{}).get('period') for d in result['details'] if isinstance(d.get('scope'),dict)}
        constants = [s['literal'] for s in plan.steps if 'literal' in s]
        constants += [v['literal'] for s in plan.steps for v in s.get('inputs',[]) if isinstance(v,dict)]
        from campus_rag.tools.numbers import chinese_integer
        for match in re.finditer(r'(?:连续|累计|为期|共计)(\d+|[一二两三四五六七八九十]+)(周|月|天|学期)',compact(question)):
            value = Decimal(match[1]) if match[1].isdigit() else Decimal(chinese_integer(match[1]))
            if '每'+match[2] in periods and value not in {Decimal(v) for v in constants}:
                raise PlanError('问题给定的周期数未参与计算，必须先算周期内已用量再比较总范围')

    @staticmethod
    def _step_filter(rows,columns,filters,question):
        selected = list(rows)
        for spec in filters:
            fid,op = spec['field_id'],spec['op']
            if fid not in columns:
                raise PlanError('筛选field_id不在所选表格中'+_legal_fields(columns))
            targets = spec['value'] if op in {'in','not_in'} else [spec['value']]
            matches = set()
            for target in targets:
                text = compact(target)
                if op in {'eq','in','not_in','contains'}:
                    if not _object_mentioned(text,question) and text not in {'合计','总计','小计'}:
                        raise PlanError('筛选对象未在问题中出现：'+str(target))
                    if _excluded_object(text,question)!=(op=='not_in'):
                        raise QuestionConstraintError('问题约束：筛选包含/排除方向与原问题不符；明确排除对象使用not_in')
                    if op=='contains' and not re.search(r'包含|含有',question):
                        raise QuestionConstraintError('问题约束：不能把明确对象擅自扩大为contains模糊匹配')
                    alternatives = [cid for cid,c in columns.items() if c['data_type']=='text' and
                        any(text==r['cells'][cid]['normalized_value'] for r in rows)]
                    if alternatives and _field_question_score(columns[fid],question,[text])<max(_field_question_score(columns[cid],question,[text]) for cid in alternatives):
                        raise QuestionConstraintError('问题约束：筛选字段与问题的对象类型不符'+_legal_fields(columns))
                    found = [r for r in selected if (text in r['cells'][fid]['normalized_value'] if op=='contains' else text==r['cells'][fid]['normalized_value'])]
                    if not found:
                        raise PlanError('筛选对象未匹配真实记录：'+str(target))
                    # A category may match many rows. Only lookup must be unique.
                else:
                    try:
                        matched = re.search(r'(?<![\d.])'+re.escape(str(target))+r'(?![\d.])\s*(万元|亿元|元|万人|千人|人|平方米|㎡|台|本|项|个|年|天|小时|%)?',question)
                        raw_threshold = matched.group() if matched else str(target)
                        threshold,threshold_unit = quantity(raw_threshold,columns[fid]['unit'])
                        if threshold is None:
                            raise PlanError('数值筛选条件不是确定数值')
                        if not columns[fid]['unit'] or threshold_unit!=columns[fid]['unit']:
                            raise ValueError('筛选阈值与字段单位无法统一')
                    except ArithmeticError as exc:
                        raise PlanError('数值筛选条件不是有效数字') from exc
                    if not threshold.is_finite():
                        raise PlanError('数值筛选条件非有限数值')
                    if not re.search(r'(?<![\d.])'+re.escape(str(target))+r'(?![\d.])',question):
                        raise PlanError('筛选阈值未在问题中出现')
                    relations = {'大于等于':'gte','不少于':'gte','至少':'gte','小于等于':'lte','不超过':'lte','至多':'lte','大于':'gt','超过':'gt','小于':'lt','低于':'lt'}
                    relation = re.search('('+'|'.join(relations)+r')\s*'+re.escape(str(target))+r'(?![\d.])',question)
                    if relation is None or relations[relation[1]]!=op:
                        raise QuestionConstraintError('问题约束：筛选比较关系与原问题不符，不得改变上下界')
                    found = []
                    for row in selected:
                        cell = row['cells'][fid]
                        if cell['numeric_value'] is None:
                            raise ValueError('筛选字段存在缺失数值，无法确定完整范围')
                        value = Decimal(cell['numeric_value'])
                        if {'gt':value>threshold,'gte':value>=threshold,'lt':value<threshold,'lte':value<=threshold}[op]:
                            found.append(row)
                matches.update(r['id'] for r in found)
            selected = [r for r in selected if (r['id'] not in matches if op=='not_in' else r['id'] in matches)]
        return selected

    @staticmethod
    def _step_detail(table,row,cell,column):
        return {'table_id':table['id'],'table_name':table['name'],'row_id':row['id'],'row_index':row['row_index'],
            'row_label':row['row_label'],'field':column['original_name'] if column else '行数',
            'field_id':column['id'] if column else None,'cell_id':cell['id'] if cell else None,
            'raw_value':cell['raw_value'] if cell else '1','numeric_value':cell['numeric_value'] if cell else '1',
            'unit':cell['unit'] if cell else '项','locator':cell['locator'] if cell else table['locator'],
            'doc_id':table['doc_id'],'title':table['title'],'source_url':table['source_url'],
            'department':table['department'],'published_at':table['published_at']}

    @staticmethod
    def _step_input(item,values,question):
        if isinstance(item,str):
            if item not in values or 'value' not in values[item]:
                raise PlanError('算术输入必须引用已有数值步骤')
            value = values[item]
            if value['unit']=='%':
                return {**value,'value':value['value']/100,'unit':'ratio','expression':'('+value['expression']+') / 100'}
            return value
        value = Decimal(item['literal'])
        # Question constants only. Do not accept model-invented operand values.
        from campus_rag.tools.numbers import chinese_integer
        tokens = re.findall(r'\d+(?:\.\d+)?|[一二两三四五六七八九十]+',question)
        grounded = any((Decimal(t) if re.fullmatch(r'\d+(?:\.\d+)?',t) else Decimal(chinese_integer(t)))==value for t in tokens)
        canonical_unit = '人' if item['unit'] in {'名','位'} else item['unit']
        if not grounded or canonical_unit not in {'','年','学年','学期','月','周','天','小时','分钟','次','个','台','人','元','平方米','门'}:
            raise PlanError('常数或单位没有问题原文支持')
        if canonical_unit:
            paired = [q for q in question_quantities(question,canonical_unit) if Decimal(q['literal'])==value]
            if not paired:
                raise PlanError('常数的单位未紧邻对应数值出现在问题中')
            source = paired[0]
            periods = set(re.findall(r'(?:每|本)(学年|学期|年|月|周|天)',question))
            completed = bool(re.search(r'已(?:经)?(?:完成|修读|修|使用|用)\s*$',question[:source['span'][0]]))
            return {'value':value,'unit':canonical_unit,'expression':str(value),'details':[],
                'question_inputs':[{'literal':str(value),'unit':canonical_unit,'occurrences':paired}],
                **({'within_period':next(iter(periods))} if completed and len(periods)==1 else {}),
                'source':{'kind':'question','quote':source['quote'],'span':source['span']}}
        token = next(t for t in tokens if (Decimal(t) if re.fullmatch(r'\d+(?:\.\d+)?',t) else Decimal(chinese_integer(t)))==value)
        located = re.search(r'(?<!\d)'+re.escape(token)+r'(?!\d)'+(r'\s*(?:个)?(?:学)?'+re.escape(item['unit']) if item['unit'] else ''),question)
        return {'value':value,'unit':item['unit'],'expression':str(value),'details':[],
            'question_inputs':[{'literal':str(value),'unit':item['unit']}],
            'source':{'kind':'question','quote':located.group(),'span':list(located.span())}}

    @staticmethod
    def _step_arithmetic(op,operands):
        numbers, units = [x['value'] for x in operands],[x['unit'] for x in operands]
        if op in {'subtract','divide','ratio'} and len(numbers)!=2:
            raise PlanError(op+'需要两个操作数')
        if len(numbers)<2 and op not in {'average','sum'}:
            raise PlanError(op+'至少需要两个操作数')
        if op in {'add','sum','subtract','average','ratio'} and len(set(units))!=1:
            raise ValueError('操作数单位不一致')
        if op in {'add','sum','subtract','average'}:
            if len({x.get('per_unit') for x in operands})>1 and not (
                    op=='subtract' and len(operands)==2 and
                    sum(bool(x.get('per_unit')) for x in operands)==1 and
                    (any(x.get('expanded_period') for x in operands) or any(
                        x.get('within_period')==next(v['per_unit'] for v in operands if v.get('per_unit'))
                        for x in operands))):
                raise QuestionConstraintError('问题约束：不同周期数或周期范围不能直接加减，需先用问题给定周期数计算实际用量')
            value = numbers[0]-numbers[1] if op=='subtract' else sum(numbers,Decimal(0))
            unit = units[0]
            if op=='average':
                value /= len(numbers)
        elif op in {'divide','ratio'}:
            if not numbers[1]:
                raise ValueError('分母为零')
            value = numbers[0]/numbers[1]
            unit = 'ratio' if units[0]==units[1] else units[0]+'/'+units[1]
            if op=='ratio':
                value,unit = value*100,'%'
        else:
            value,unit = Decimal(1),''
            periodic = [x for x in operands if x.get('per_unit')]
            if periodic:
                if len(operands)!=2 or len(periodic)!=1:
                    raise ValueError('周期用量仅支持乘以明确的周期数')
                rate = periodic[0]
                duration = operands[1] if operands[0] is rate else operands[0]
                if duration['unit']!=rate['per_unit']:
                    raise ValueError('周期单位不一致，不能猜测换算')
                return rate['value']*duration['value'],rate['unit'],' × '.join('('+x['expression']+')' for x in operands)
            for n,u in zip(numbers,units):
                value *= n
                if u in {'','ratio'}:
                    continue
                if not unit:
                    unit = u
                elif '/' in unit and unit.rsplit('/',1)[1]==u:
                    unit = unit.rsplit('/',1)[0]
                elif '/' in u and u.rsplit('/',1)[1]==unit:
                    unit = u.rsplit('/',1)[0]
                elif unit in {'台','个','本','件','项','次'} and u=='元':
                    unit = '元'
                elif u in {'台','个','本','件','项','次'} and unit=='元':
                    unit = '元'
                else:
                    raise ValueError('乘法单位组合尚不支持')
        symbols = {'add':'+','sum':'+','subtract':'-','multiply':'×','divide':'/','ratio':'/','average':'+'}
        expression = (' '+symbols[op]+' ').join('('+o['expression']+')' for o in operands)
        if op=='ratio':
            expression += ' × 100%'
        if op=='average':
            expression = '('+expression+') / '+str(len(numbers))
        return value,unit,expression

    def execute(self, plan: QueryPlan, table_ids: list[int] | None = None) -> CalculationResult:
        if not plan.needs_table or plan.calculation.operation == "none":
            return CalculationResult(ok=False, error="该问题不需要结构化计算")
        if not plan.calculation.operands:
            return CalculationResult(ok=False, error='没有有效的计算操作数，拒绝返回空集合合计')
        tables = self._candidate_tables(plan, table_ids)
        if not tables:
            return CalculationResult(ok=False, error="未找到与问题匹配的结构化表格")
        all_details: list[dict] = []
        operand_values: list[tuple[Decimal, str, list[dict]]] = []
        for operand in plan.calculation.operands:
            try:
                matches = self._collect_operand(tables, operand, plan.filters)
            except ValueError as exc:
                return CalculationResult(ok=False,error=str(exc))
            if not matches:
                return CalculationResult(ok=False, error=f"字段“{operand.field}”没有满足条件的数据")
            try:
                if len({item['table_id'] for item in matches}) > 1:
                    raise ValueError('同一操作数匹配多张表，需明确范围，拒绝跨表重复累计')
                value, unit = self._aggregate_operand(operand, matches)
            except ValueError as exc:
                return CalculationResult(ok=False, error=str(exc))
            operand_values.append((value, unit, matches))
            all_details.extend(matches)
        try:
            result, expression, unit = self._calculate(plan, operand_values)
        except (ValueError, ArithmeticError) as exc:
            return CalculationResult(ok=False, error=str(exc), details=all_details)
        evidence = self._evidence_from_details(all_details)
        return CalculationResult(True, result, unit, expression, all_details, evidence=evidence)

    def _candidate_tables(self, plan: QueryPlan, table_ids: list[int] | None) -> list[sqlite3.Row]:
        if table_ids is not None:
            if not table_ids:
                return []
            placeholders = ",".join("?" for _ in table_ids)
            rows = self.connection.execute(
                f"SELECT t.*, d.title, d.source_url, d.department, d.published_at "
                f"FROM tables t JOIN documents d ON d.id=t.doc_id WHERE t.id IN ({placeholders})",
                tuple(table_ids),
            ).fetchall()
            return [r for r in rows if r['id'] not in self.navigation_tables and matches_period(r['title'], r['published_at'], plan.time_range)]
        terms = [*plan.keywords, *plan.entities, *plan.fields]
        rows = self.connection.execute(
            "SELECT t.*, d.title, d.source_url, d.department, d.published_at "
            "FROM tables t JOIN documents d ON d.id=t.doc_id"
        ).fetchall()
        scored = []
        for row in rows:
            if row['id'] in self.navigation_tables or not matches_period(row['title'], row['published_at'], plan.time_range):
                continue
            haystack = f"{row['name']} {row['search_text']}".lower()
            score = sum(1 for term in terms if term and term.lower() in haystack)
            if score:
                scored.append((score, row))
        return [row for _, row in sorted(scored, key=lambda item: item[0], reverse=True)[:10]]

    def _collect_operand(self, tables: list[sqlite3.Row], operand: OperandSpec, common_filters: list[FilterSpec]) -> list[dict]:
        matches: list[dict] = []
        for table in tables:
            columns = self.connection.execute(
                "SELECT * FROM table_columns WHERE table_id=? ORDER BY column_index", (table["id"],)
            ).fetchall()
            target = _match_column(operand.field, columns)
            if target is None:
                continue
            rows = self.connection.execute(
                "SELECT * FROM table_rows WHERE table_id=? ORDER BY row_index", (table["id"],)
            ).fetchall()
            for row in rows:
                cells = self.connection.execute(
                    """SELECT tc.*, col.original_name, col.normalized_name
                       FROM table_cells tc JOIN table_columns col ON col.id=tc.column_id
                       WHERE tc.row_id=?""", (row["id"],)
                ).fetchall()
                if not self._row_matches(cells, [*common_filters, *operand.filters]):
                    continue
                cell = next((cell for cell in cells if cell["column_id"] == target["id"]), None)
                if cell is None or cell["numeric_value"] is None:
                    raise ValueError(f'符合条件的字段“{operand.field}”存在缺失或非数值单元格')
                matches.append({
                    "table_id": int(table["id"]), "table_name": table["name"],
                    "row_id": int(row["id"]), "row_index": int(row["row_index"]),
                    "row_label": row["row_label"], "field": target["original_name"],
                    "raw_value": cell["raw_value"], "numeric_value": cell["numeric_value"],
                    "unit": cell["unit"] or target["unit"] or table["unit"],
                    "doc_id": int(table["doc_id"]), "title": table["title"],
                    "source_url": table["source_url"], "department": table["department"],
                    "published_at": table["published_at"],
                })
        return matches

    @staticmethod
    def _row_matches(cells: list[sqlite3.Row], filters: list[FilterSpec]) -> bool:
        for item in filters:
            column = _match_column(item.field, cells)
            if column is None:
                return False
            cell = next((candidate for candidate in cells if candidate["column_id"] == column["column_id"]), None)
            if cell is None or not _compare(cell, item):
                return False
        return True

    @staticmethod
    def _aggregate_operand(operand: OperandSpec, matches: list[dict]) -> tuple[Decimal, str]:
        units = {item["unit"] for item in matches if item["unit"]}
        if len(units) > 1:
            raise ValueError(f"字段“{operand.field}”存在不一致单位：{sorted(units)}")
        values = [Decimal(item["numeric_value"]) for item in matches]
        if operand.aggregate == "count":
            return Decimal(len(matches)), "项"
        if operand.aggregate == "sum":
            return sum(values, Decimal(0)), next(iter(units), "")
        if operand.aggregate == "avg":
            return sum(values, Decimal(0)) / Decimal(len(values)), next(iter(units), "")
        if len(values) != 1:
            raise ValueError(f"字段“{operand.field}”匹配到 {len(values)} 个值，请增加筛选条件")
        return values[0], next(iter(units), "")

    @staticmethod
    def _calculate(plan: QueryPlan, operands: list[tuple[Decimal, str, list[dict]]]) -> tuple[Decimal, str, str]:
        operation = plan.calculation.operation
        values = [item[0] for item in operands]
        units = [item[1] for item in operands]
        if operation in {"sum", "count", "average"}:
            if operation == "count":
                value, unit = Decimal(sum(int(value) for value in values)), "项"
            elif operation == "average":
                value, unit = sum(values, Decimal(0)) / Decimal(len(values)), units[0]
            else:
                if len({unit for unit in units if unit}) > 1:
                    raise ValueError("求和字段单位不一致")
                value, unit = sum(values, Decimal(0)), units[0] if units else ""
            expression = f" {operation} ".join(format(value, "f") for value in values)
        elif operation == "ratio":
            if len(values) != 2 or values[1] == 0:
                raise ValueError("比例计算需要两个操作数，且分母不能为零")
            value, unit = values[0] / values[1], plan.calculation.output_unit or "%"
            if unit == "%":
                value *= Decimal(100)
            expression = f"{values[0]} / {values[1]}" + (" * 100" if unit == "%" else "")
        elif operation == "difference":
            if len(values) != 2:
                raise ValueError("差值计算需要两个操作数")
            if units[0] != units[1]:
                raise ValueError('差值计算的单位不一致')
            value, unit = values[0] - values[1], units[0]
            expression = f"{values[0]} - {values[1]}"
        elif operation == "multiply":
            if not values:
                raise ValueError("乘法计算缺少操作数")
            value = Decimal(1)
            for number in values:
                value *= number
            if plan.calculation.multiplier is not None:
                value *= plan.calculation.multiplier
                values.append(plan.calculation.multiplier)
            unit = plan.calculation.output_unit or next((unit for unit in units if unit), "")
            expression = " * ".join(format(number, "f") for number in values)
        else:
            raise ValueError(f"不支持的计算操作：{operation}")
        if plan.calculation.output_unit and unit and unit != plan.calculation.output_unit and operation != "ratio":
            value = convert_unit(value, unit, plan.calculation.output_unit)
            unit = plan.calculation.output_unit
        return value, expression, unit

    @staticmethod
    def _evidence_from_details(details: list[dict]) -> list[Evidence]:
        grouped: dict[tuple[int, int], dict] = {}
        for detail in details:
            key = (detail["doc_id"], detail["table_id"])
            grouped.setdefault(key, {**detail, "rows": []})["rows"].append(detail["row_id"])
        return [Evidence(
            evidence_id=f"table:{table_id}",
            text=f"{item['table_name']}，使用行：{sorted(set(item['rows']))}", score=1.0,
            doc_id=doc_id, title=item["title"], source_url=item["source_url"],
            department=item["department"], published_at=item["published_at"],
            table_id=table_id, row_ids=sorted(set(item["rows"])),
        ) for (doc_id, table_id), item in grouped.items()]


def _match_column(field: str, columns: list[sqlite3.Row]) -> sqlite3.Row | None:
    needle = "".join(field.lower().split())
    exact = [column for column in columns if needle in {column["normalized_name"], "".join(column["original_name"].lower().split())}]
    if len(exact) == 1:
        return exact[0]
    aliases = [column for column in columns if needle == column['normalized_name'].split('/')[-1]]
    if len(aliases) == 1:
        return aliases[0]
    partial = [column for column in columns if needle in column["normalized_name"] or column["normalized_name"] in needle]
    if len(partial) == 1:
        return partial[0]
    aliases = [column for column in columns if column['normalized_name'].split('/')[-1] in needle]
    return aliases[0] if len(aliases) == 1 else None


def _compare(cell: sqlite3.Row, item: FilterSpec) -> bool:
    raw = str(cell["normalized_value"]).lower()
    expected = str(item.value).strip().lower()
    if item.operator == "contains":
        return expected in raw
    if item.operator == "eq":
        return expected == raw
    if cell["numeric_value"] is None:
        return False
    try:
        left, right = Decimal(cell["numeric_value"]), Decimal(expected)
    except Exception:
        return False
    return {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[item.operator]

