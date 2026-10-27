from __future__ import annotations

import sqlite3
import json
import re
from decimal import Decimal

from campus_rag.tools.numbers import convert_unit
from campus_rag.types import CalculationResult, Evidence, FilterSpec, OperandSpec, QueryPlan
from campus_rag.tools.temporal import matches_period, question_periods
from campus_rag.ingest.navigation import navigation_table_ids
from campus_rag.database.normalization import compact, quantity, VERSION


class PlanError(ValueError):
    """Only schema/plan mistakes are repairable; missing evidence is not."""


def _object_key(value):
    return re.sub(r'(?:工程)?实验教学中心|实验室|中心|学院|专业|场馆|组别','',compact(value))


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
                mentions = [_object_key(c['raw_value']) for c in cells if len(_object_key(c['raw_value']))>1 and _object_key(c['raw_value']) in compact(question)]
                score = sum(map(len,mentions))
                position = min((compact(question).find(v) for v in mentions),default=len(question))
                ranked.append((score,position,row['id'],cells))
            for _,_,rid,cells in sorted(ranked,key=lambda x:(-x[0],x[1]))[:3]:
                examples.append({'row_id':rid,'cells':{str(c['column_id']):c['raw_value'][:120] for c in cells}})
            output.append({'table_id':tid,'name':table['name'],'document':table['title'],
                'periods':question_periods(table['title']+' '+table['name']),
                'columns':[{'field_id':c['id'],'name':c['original_name'],'type':c['data_type'],'unit':c['unit']} for c in columns],
                'row_count':len(rows),'examples_only':True,'examples':examples})
            signature = (tuple((compact(c['original_name']),c['unit'],c['data_type']) for c in columns),
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
        return (exact or output)[:4]

    def execute_steps(self, plan, table_ids, question, facts=None):
        trace, used = [], []
        values = {}
        facts = facts or {}
        try:
            if not plan.steps:
                raise ValueError('证据不足，未生成有效计算步骤')
            if plan.table_id is not None:
                if plan.table_id not in table_ids or plan.table_id in self.navigation_tables:
                    raise PlanError('table_id不在实际候选Schema内')
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
                    value,unit,expression = self._step_arithmetic(op,operands)
                    value_record = {'value':value,'unit':unit,'expression':expression,
                                    'details':[d for v in operands for d in v['details']]}
                elif 'fact_id' in step:
                    if op!='lookup' or set(step)-{'id','op','fact_id'} or step['fact_id'] not in facts:
                        raise PlanError(sid+': fact_id不在已核验正文事实中')
                    fact = facts[step['fact_id']]
                    value_record = {'value':Decimal(fact['value']),'unit':fact['unit'],
                                    'expression':fact['value'],'details':[fact]}
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
                        raise PlanError(sid+': field_id不在所选表格Schema内')
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
                    value_record = {'value':value,'unit':unit,'expression':expression,'details':details}
                if not value_record['value'].is_finite():
                    raise ValueError(sid+': 计算结果非有限数值')
                values[sid] = value_record
                trace.append({'id':sid,'op':op,'operands':[
                    {'value':str(v['value']),'unit':v['unit']} for v in operands] if inputs is not None else [],
                    'value':str(value_record['value']),'unit':value_record['unit'],
                    'expression':value_record['expression'],'details':value_record['details']})
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
            # Include every executed source, including intermediate computations.
            used = [d for entry in trace for d in entry.get('details',[])]
            if table is not None:
                scope_details = result['details']
                final = plan.steps[-1]
                if final['op']=='ratio' and isinstance(final['inputs'][0],str) and any(
                    s['id']==final['inputs'][0] and s['op'] in {'sum','add','average'} for s in plan.steps):
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
            # A year column imposes an additional row-level restriction.
            if periods and meta['periods'] and not any(p in meta['periods'] for p in periods):
                continue
            rows.append({**dict(row),**meta,'cells':cells})
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

    @staticmethod
    def _check_requested_objects(rows,columns,question,details):
        if not re.search(r'和|与|、|及',question):
            return
        for fid,col in columns.items():
            if col['data_type']!='text':
                continue
            labels = [r['cells'][fid]['normalized_value'] for r in rows if not r['is_total']]
            if len(labels)!=len(set(labels)):
                continue
            required = set()
            for row in rows:
                text = row['cells'][fid]['normalized_value']
                core = _object_key(text)
                if not row['is_total'] and len(core)>=2 and (text in compact(question) or core in compact(question)):
                    required.add(row['id'])
            actual = {d['row_id'] for d in details if 'row_id' in d}
            missing = required-actual
            if len(required)>1 and missing:
                labels = [r['cells'][fid]['raw_value'] for r in rows if r['id'] in missing]
                raise PlanError('计划遗漏问题明确列举的筛选对象：'+'、'.join(labels))
            detail_ids = {r['id'] for r in rows if not r['is_total']}
            if len(required)>1 and (actual & detail_ids)-required:
                raise PlanError('计划包含问题列举范围之外的明细对象')

    @staticmethod
    def _step_filter(rows,columns,filters,question):
        selected = list(rows)
        for spec in filters:
            fid,op = spec['field_id'],spec['op']
            if fid not in columns:
                raise PlanError('筛选field_id不在所选表格中')
            targets = spec['value'] if op=='in' else [spec['value']]
            matches = set()
            for target in targets:
                text = compact(target)
                if op in {'eq','in','contains'}:
                    core = _object_key(text)
                    if text not in compact(question) and (not core or core not in compact(question)) and text not in {'合计','总计','小计'}:
                        raise PlanError('筛选对象未在问题中出现：'+str(target))
                    found = [r for r in selected if (text in r['cells'][fid]['normalized_value'] if op=='contains' else text==r['cells'][fid]['normalized_value'])]
                    if not found:
                        raise PlanError('筛选对象未匹配真实记录：'+str(target))
                    # "in" enumerates specific objects, each must map to one record.
                    if op=='in' and len(found)!=1:
                        raise ValueError('筛选对象重复，无法唯一确定匹配数量：'+str(target))
                else:
                    try:
                        matched = re.search(r'(?<!\d)'+re.escape(str(target))+r'\s*(万元|亿元|元|万人|千人|人|平方米|㎡|台|本|项|个|年|天|小时|%)?',question)
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
                    if not re.search(r'(?<!\d)'+re.escape(str(target))+r'(?!\d)',question):
                        raise PlanError('筛选阈值未在问题中出现')
                    found = []
                    for row in selected:
                        cell = row['cells'][fid]
                        if cell['numeric_value'] is None:
                            raise ValueError('筛选字段存在缺失数值，无法确定完整范围')
                        value = Decimal(cell['numeric_value'])
                        if {'gt':value>threshold,'gte':value>=threshold,'lt':value<threshold,'lte':value<=threshold}[op]:
                            found.append(row)
                matches.update(r['id'] for r in found)
            selected = [r for r in selected if r['id'] in matches]
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
        if not grounded or item['unit'] not in {'','年','学年','学期','月','周','天','小时','分钟','次','个','台','人','元','平方米'}:
            raise PlanError('常数或单位没有问题原文支持')
        if item['unit']:
            paired = [t for t in tokens if (Decimal(t) if re.fullmatch(r'\d+(?:\.\d+)?',t) else Decimal(chinese_integer(t)))==value
                and re.search(re.escape(t)+r'(?:个)?(?:学)?'+re.escape(item['unit']),question)]
            if not paired:
                raise PlanError('常数的单位未紧邻对应数值出现在问题中')
        return {'value':value,'unit':item['unit'],'expression':str(value),'details':[]}

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

