"""Evidence-bound numeric tools. Unsupported or ambiguous inputs fail closed.

No answer constants, model-authored SQL, or executable expressions are accepted.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

from campus_rag.database.query import _match_column
from campus_rag.types import CalculationResult, Evidence


def compact(text):
    return re.sub(r'\s+', '', text).replace('，', ',').replace('％', '%')


def number(value):
    value = Decimal(str(value))
    return format(value, 'f').rstrip('0').rstrip('.') if '.' in format(value, 'f') else format(value, 'f')


def _fact_heading(text):
    pattern = (r'(?:^|(?<=[。\n]))[ \t]*(?:\d+[.．、][ \t]*'
        r'(?P<numbered>[^。；;\n，,：:（）()]{2,32})(?=[（(\n。:：])|'
        r'(?P<caption>[^。；;\n，,：:（）()]{2,32})[（(][^。\n]{1,32}[）)](?=[ \t]*(?:\n|$)))')
    headings = list(re.finditer(pattern,text))
    if not headings:
        return '',0
    match = headings[-1]
    field = 'numbered' if match['numbered'] is not None else 'caption'
    return match[field].strip(),match.start(field)


def _fact_group(text):
    """Only explicit, standalone subsection captions define an object group."""
    groups = list(re.finditer(r'(?m)^[ \t]*[（(][一二三四五六七八九十]+[）)][ \t]*'
        r'([^\n。；;：:]{2,48})[：:][ \t]*$',text))
    if not groups:
        return ''
    resets = list(re.finditer(r'(?m)^[ \t]*(?:[一二三四五六七八九十]+、|第[一二三四五六七八九十百\d]+条)',text))
    return compact(groups[-1][1]) if not resets or groups[-1].start()>resets[-1].start() else ''


@dataclass
class NumericOutcome:
    handled: bool = False
    result: CalculationResult = field(default_factory=lambda: CalculationResult(False))
    canonical: str = ''
    facts: list[dict] = field(default_factory=list)


class NumericEngine:
    def __init__(self, connection, excluded_tables=None):
        self.connection = connection
        self.excluded_tables = excluded_tables or set()

    def resolve(self, question, evidence):
        try:
            if '学费' in question and any(word in question for word in ['文科','理科','工科','艺术']):
                return self.fees(question, self.support(evidence, ['章程'], ['学费','住宿费']))
            if any(word in question for word in ['专任教师','外聘教师','折合教师','折算系数','生师比']):
                return self.staff(question, self.support(evidence, ['教学质量报告'], ['教师','折合在校生']))
            if '首选' in question and any(word in question for word in ['计划','可报']):
                return self.admission(question, evidence)
        except (ValueError, ArithmeticError) as exc:
            return NumericOutcome(True, CalculationResult(False,error=str(exc)))
        return NumericOutcome()

    def support(self, evidence, title_terms, text_terms):
        if not evidence:
            return []
        preferred = [e for e in evidence if any(term in e.title for term in title_terms)]
        output = []
        # Do not silently resolve competing documents by selecting the first hit.
        templates = {e.doc_id:e for e in (preferred or evidence)}
        for doc_id,template in templates.items():
            for row in self.connection.execute("SELECT id,text FROM chunks WHERE doc_id=? AND kind='text' ORDER BY id", (doc_id,)):
                if any(term in row['text'] for term in text_terms):
                    output.append(Evidence(row['id'],row['text'],template.score,doc_id,template.title,
                        template.source_url,template.department,template.published_at,chunk_id=row['id']))
        return output

    @staticmethod
    def text_facts(evidence):
        """Extract explicit quantities, preserving bounds, scope and original offsets.

        Whitespace (including PDF line breaks) is ignored for matching only. Quotes
        always point back to the unmodified evidence; missing values are not inferred.
        """
        from campus_rag.database.normalization import quantity
        facts, seen = {}, set()
        units = r'万元|亿元|元|万人|千人|人|名|平方米|㎡|台|本|项|个|门|学年|学期|年|月|周|天|小时|分钟|%|％'
        amount = re.compile(r'(?<![\d.])(?P<number>\d[\d,]*(?:\.\d+)?)(?:[-—–～~至到](?P<upper>\d[\d,]*(?:\.\d+)?))?(?P<unit>'+units+r')?(?P<denominator>/(?:小时|分钟|天|周|月|学年|年|人|次|台|个))?')
        relations = {'不超过':'lte','不高于':'lte','至多':'lte','最多':'lte','上限为':'lte',
            '不低于':'gte','不少于':'gte','至少':'gte','最低为':'gte','下限为':'gte',
            '超过':'gt','高于':'gt','低于':'lt','小于':'lt','大于':'gt',
            '能够完成':'eq','完成':'eq','共计':'eq','共':'eq','设置':'eq',
            '为':'eq','有':'eq','是':'eq',':':'eq','：':'eq'}
        connector = re.compile('('+'|'.join(sorted(relations,key=len,reverse=True))+r')(?:每(?:小时|分钟|天|周|月|年|人))?$')
        for item in evidence:
            if item.table_id is not None:
                continue
            offsets = [i for i,char in enumerate(item.text) if not char.isspace()]
            # One-character normalization keeps the original offset map intact.
            text = ''.join(item.text[i] for i in offsets).replace('／','/')
            previous_count = None
            for match in amount.finditer(text):
                start = max(text.rfind(p,0,match.start()) for p in '。；;！？')+1
                prefix = text[start:match.start()]
                # A nearby number alone is not a fact (page numbers, dates, codes).
                local = re.split('[，,]',prefix)[-1]
                if match['unit'] in {'人','名'}:
                    row_label = re.search(r'(?:^|(?<=[\n。；;，,]))[ \t]*'
                        r'(?P<label>[^\n。；;，,：:\d]{1,40})[：:][ \t]*$',item.text[:offsets[match.start()]])
                    if row_label:
                        start = bisect_left(offsets,row_label.start('label'))
                        prefix = text[start:match.start()]
                        local = prefix
                paired = (previous_count if match['unit'] in {'元','万元','亿元'}
                    and match['denominator']=='/人' and previous_count is not None
                    and not re.match(r'左右|上下|不等|至|到|[-—–～~]',text[match.end():])
                    and re.fullmatch(r'[，,](?:每(?:学年|学期|年|月))?'
                        r'(?:(?:奖学金|补助|补贴|奖励|收费|资助)?(?:标准|金额)(?:为|是|：|:)?)?',
                        text[previous_count['end']:match.start()]) else None)
                if match['unit'] in {'年','月'} and (re.match(r'\d+(?:月|日)',text[match.end():])
                        or re.search(r'日期|公示|截止|评选时间',local)):
                    continue
                relation = connector.search(local)
                suffix = re.match(r'(及以上|及以下|以上|以下)',text[match.end():])
                gpa = bool(re.search(r'(?:平均学分)?绩点(?:要求)?(?:为|不低于|不超过|高于|低于)?$',local))
                gpa = gpa or bool('绩点' in local and re.search(r'(?:但|且)(?:高于|低于|不低于|不高于)$',local))
                if not paired and not relation and not suffix and not re.search(r'(?:额度|金额|标准|预算|期限|时长|人数)$',local):
                    continue
                if not match['unit'] and not gpa:
                    continue
                if re.search(r'约|大约|左右|大致',local[-12:]):
                    continue
                comparison = relations[relation[1]] if relation else 'eq'
                if suffix:
                    comparison = 'gte' if '以上' in suffix[1] else 'lte'
                label = previous_count['field']+'人均标准' if paired else local[:relation.start()] if relation else local
                if not label:
                    # A second bound may share the explicitly stated subject.
                    first = re.search('|'.join(sorted(relations,key=len,reverse=True)),prefix)
                    label = prefix[:first.start()] if first else ''
                if re.fullmatch(r'每(?:小时|分钟|天|周|月|年|人)',label):
                    label = prefix.split('每',1)[0]+label
                label = re.sub(r'每(?:小时|分钟|天|周|月|年|人)$','',label).strip('：:，,')
                label = re.sub(r'^第[一二三四五六七八九十百\d]+条','',label)
                label = re.sub(r'^（[一二三四五六七八九十]+）','',label)
                if not label or not re.search(r'[\u4e00-\u9fffA-Za-z]',label):
                    continue
                periods = re.findall(r'每(?:人每年|学年|学期|小时|分钟|天|周|月|年|人)',prefix)
                period = periods[-1] if periods else ''
                clauses = re.findall(r'第[一二三四五六七八九十百\d]+条',text[:match.start()])
                clause = clauses[-1] if clauses else ''
                subject = re.split(r'每|(?:平均学分)?绩点|额度|金额|预算|时长',label,maxsplit=1)[0] or label
                if paired:
                    subject = paired['subject']
                end = match.end()+(len(suffix[1]) if suffix else 0)
                original_start,original_end = offsets[start],offsets[end-1]+1
                if paired:
                    original_start = paired['original_start']
                heading,heading_start = _fact_heading(item.text[:offsets[match.start()]])
                sections = re.findall(r'(?m)^[ \t]*[《“"]?([^\n《》“"]{2,70}(?:办法|规定|指南|章程|通知))[》”"]?[ \t]*$',
                    item.text[:offsets[match.start()]])
                sections = [s for s in sections if not re.match(r'(?:第[一二三四五六七八九十百\d]+条|[（(]?[一二三四五六七八九十]+[、.）)])',s)]
                if sections:
                    for caption in ('《'+sections[-1]+'》',sections[-1]):
                        if subject.startswith(caption):
                            subject = subject[len(caption):].strip('：:，,')
                        if label.startswith(caption):
                            label = label[len(caption):].strip('：:，,')
                if heading and (subject in {'奖励','金额','补贴','补助','时长','课程','标准','预算'}
                        or re.fullmatch(r'每(?:学年|学期|年|月|周|天)',label)):
                    subject = heading
                    original_start = min(original_start,heading_start)
                boundary = re.search('[。；;！？]',text[end:])
                scope_end = end+boundary.start() if boundary else len(text)
                scope = {'clause':clause,'period':period,'context':compact(item.text[original_start:offsets[scope_end-1]+1])}
                group = _fact_group(item.text[:offsets[match.start()]])
                if group:
                    scope['group'] = group
                values = [(match['number'],comparison)]
                if match['upper']:
                    if Decimal(match['upper'].replace(',',''))<Decimal(match['number'].replace(',','')):
                        continue
                    values = [(match['number'],'gte'),(match['upper'],'lte')]
                for token,comparison in values:
                    raw = token+(match['unit'] or '')
                    value,unit = quantity(token+'人' if match['unit']=='名' else raw) if match['unit'] else (Decimal(token),'绩点')
                    if value is None and match['unit'] in {'学年','学期','月','周','分钟','门'}:
                        value,unit = Decimal(token.replace(',','')),match['unit']
                    if value is None:
                        continue
                    per = re.search(r'每(小时|分钟|天|周|月|年|人)$',prefix)
                    if match['denominator']:
                        unit += match['denominator']
                    elif per and unit=='元':
                        unit += '/'+per[1]
                    elif unit=='元' and re.search(r'生均|人均',prefix):
                        unit += '/人'
                    if unit=='人' and comparison=='eq' and not match['upper']:
                        # Only an accepted, adjacent, exact headcount can supply
                        # the subject of an otherwise unlabeled per-person rate.
                        previous_count = {'subject':subject,'field':label,
                            'end':match.end(),'original_start':original_start}
                    key = (item.doc_id,sections[-1] if sections else '',group,subject,label,clause,period,comparison,str(value),unit)
                    if key in seen:
                        continue
                    seen.add(key)
                    fid = 'f'+str(len(facts)+1)
                    facts[fid] = {'fact_id':fid,'field':label,'subject':subject,
                        'raw_value':item.text[offsets[match.start()]:offsets[match.end()-1]+1]
                            if match['denominator'] else match.group() if match['upper'] else raw,
                        'value':str(value),'unit':unit,'comparison':comparison,'scope':scope,
                        'quote':item.text[original_start:original_end],'span':[original_start,original_end],
                        'doc_id':item.doc_id,'source_title':item.title,'source_section':sections[-1] if sections else '',
                        'evidence_id':item.evidence_id,'source_url':item.source_url,'evidence':item}
        # Conflicts concern the same subject, clause, period and comparison, not
        # unrelated objects that happen to have the same measure name.
        groups = {}
        for fid,fact in facts.items():
            key = (fact['doc_id'],fact['source_section'],fact['scope'].get('group',''),fact['subject'],fact['field'],fact['scope']['clause'],
                fact['scope']['period'],fact['comparison'])
            groups.setdefault(key,[]).append(fid)
        conflicts = {fid for ids in groups.values()
            if len({(facts[f]['value'],facts[f]['unit']) for f in ids})>1 for fid in ids}
        return {fid:f for fid,f in facts.items() if fid not in conflicts}
    @staticmethod
    def extract(label, pattern, unit, evidence):
        matches = []
        for item in evidence:
            text = compact(item.text)
            for match in re.finditer(pattern, text):
                value = Decimal(match.group('value').replace(',', ''))
                if not value.is_finite() or value < 0:
                    raise ValueError(f'{label}包含无效数值')
                matches.append((value, item, {'field':label, 'raw_value':match.group('value'),
                    'value':number(value),'unit':unit, 'quote':match.group(),
                    'normalized_span':[match.start(),match.end()], 'evidence_id':item.evidence_id,
                    'doc_id':item.doc_id}))
        values = {m[0] for m in matches}
        if not values:
            raise ValueError(f'缺少可定位的{label}证据')
        if len(values) != 1:
            raise ValueError(f'{label}存在多个统计口径或冲突值，需指定范围')
        return matches[0]

    @staticmethod
    def finish(answer, expression, value, unit, extracts, extra_evidence=None):
        facts = [x[2] for x in extracts]
        used = {x[1].evidence_id:x[1] for x in extracts}
        used.update({e.evidence_id:e for e in extra_evidence or []})
        result = CalculationResult(True,value,unit,expression,facts,evidence=list(used.values()))
        return NumericOutcome(True,result,answer,facts)

    def fees(self, question, evidence):
        categories = [s for s in ['文科','理科','工科','艺术'] if s in question]
        if len(categories) != 1:
            raise ValueError('需要明确一个专业类别')
        category = categories[0]
        base = self.extract(category+'基准学费', category+r'类专业每生每年(?P<value>\d+(?:\.\d+)?)元', '元/生·年',evidence)
        extracts = [base]
        tuition = base[0]
        expression = number(tuition)
        if any(word in question for word in ['优势','上浮']):
            rate = self.extract('学费上浮比例',r'上述标准基础上上浮(?P<value>\d+(?:\.\d+)?)%', '%',evidence)
            if rate[0] > 100:
                raise ValueError('学费上浮比例超出当前工具验证范围')
            tuition *= Decimal(1) + rate[0]/Decimal(100)
            extracts.append(rate)
            expression += f' × (1 + {number(rate[0])}%)'
        answer = f'{category}类' + ('优势学科专业' if len(extracts)>1 else '专业') + f'学费为{number(tuition)}元/生·年；算式：{expression} = {number(tuition)}元。'
        lodging = None
        if '住宿' in question:
            lodging = self.extract('住宿费预收标准',r'住宿费(?:暂按|按|预收标准为|标准为|为)(?P<value>\d+(?:\.\d+)?)元','元/生·学年',evidence)
            extracts.append(lodging)
            answer += f'住宿费按{number(lodging[0])}元/生·学年预收，实际按住宿条件结算。'
        total = tuition
        if re.search(r'合计|总计|预收额|一共',question):
            if lodging is None:
                raise ValueError('费用合计需要明确是否包含住宿费')
            durations = re.findall(r'(?<![\d一二两三四五六七八九十])([一二两三四五六七八九十]{1,3}|\d{1,2})(?:个)?(?:学)?年',question)
            years = Decimal(1)
            if durations:
                from campus_rag.tools.numbers import chinese_integer
                if len(set(durations)) != 1:
                    raise ValueError('计费年限存在歧义')
                token = durations[0]
                years = Decimal(token) if token.isdigit() else Decimal(chinese_integer(token))
                if years < 1 or years > 10:
                    raise ValueError('计费年限超出当前工具验证范围')
            total = (tuition + lodging[0])*years
            expression = f'({expression} + {number(lodging[0])}) × {number(years)}'
            answer += f'假设收费标准不变，{number(years)}年预收合计为{number(total)}元；算式：{expression} = {number(total)}元。'
        return self.finish(answer,expression,total,'元',extracts)

    def staff(self, question, evidence):
        patterns = {
            '专任教师':r'专任教师(?:人数|数量|数)?(?:为|有)?(?P<value>\d+(?:\.\d+)?)人',
            '外聘教师':r'(?:外聘教师|聘请校外教师)(?:人数|数量|数)?(?:为|有)?(?P<value>\d+(?:\.\d+)?)人',
            '折合教师':r'折合教师(?:总数|人数|数)?(?:为|有)?(?P<value>\d+(?:\.\d+)?)(?![\d.万亿])',
            '折合在校生':r'折合在校生(?:人数|总数|数)?(?:为|有)?(?P<value>\d+(?:\.\d+)?)(?![\d.万亿])',
        }
        derive = any(term in question for term in ['折算','反推','生师比'])
        labels = list(patterns) if derive else [label for label in patterns if label in question]
        extracts = [self.extract(label,patterns[label],'人',evidence) for label in labels]
        facts = dict(zip(labels,extracts))
        if any(item[0] != item[0].to_integral_value() for label,item in facts.items() if label in {'专任教师','外聘教师'}):
            raise ValueError('实际教师人数不是整数，需要核对统计口径')
        answer = '；'.join(f'{label}为{number(item[0])}人' for label,item in facts.items())+'。'
        expression = '逐项原文抽取，无加总'
        value = extracts[0][0]
        unit = '人'
        if derive:
            permanent, external, effective, students = [facts[label][0] for label in patterns]
            if external <= 0 or effective <= 0 or effective < permanent:
                raise ValueError('师资折算数据不满足计算条件')
            coefficient = (effective-permanent)/external
            if not Decimal(0) <= coefficient <= Decimal(1):
                raise ValueError('反推折算系数不在0至1范围内')
            ratio = (students/effective).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
            ratio_pattern = r'(?:全校)?生师比(?:为)?(?P<value>\d+(?:\.\d+)?)(?![\d.])'
            if any(re.search(ratio_pattern,compact(e.text)) for e in evidence):
                reported = self.extract('报告生师比',ratio_pattern,':1',evidence)
                if abs(reported[0]-ratio) > Decimal('0.01'):
                    raise ValueError('复算生师比与报告数值不一致，需核对口径')
                extracts.append(reported)
            expression = f'({number(effective)} - {number(permanent)}) / {number(external)} = {number(coefficient)}；{number(students)} / {number(effective)} ≈ {number(ratio)}'
            answer += f'外聘折算系数为{number(coefficient)}，复算生师比约为{number(ratio)}:1。算式：{expression}。'
            value, unit = ratio, ':1'
        return self.finish(answer,expression,value,unit,extracts)

    def admission(self, question, evidence):
        primary = re.search(r'首选(?:科目)?(?:为|是)?[：:]?(物理|历史)',question)
        secondary = re.search(r'再选(?:科目)?(?:为|是)?[：:]?([^，,。；;]+)',question)
        if not primary or not secondary:
            raise ValueError('缺少明确的首选和再选科目信息')
        selected = set(re.findall(r'物理|历史|化学|生物|政治|地理',secondary.group(1)))
        if len(selected) != 2:
            raise ValueError('需要明确两个再选科目')
        tids = {e.table_id for e in evidence if e.table_id is not None} - self.excluded_tables
        candidates = []
        for tid in sorted(tids):
            columns = self.connection.execute('SELECT * FROM table_columns WHERE table_id=? ORDER BY column_index',(tid,)).fetchall()
            matched = [_match_column(name,columns) for name in ['首选科目','再选科目','专业名称','计划']]
            if all(c is not None for c in matched):
                candidates.append((tid,matched))
        if len(candidates) != 1:
            raise ValueError('选科计划表缺失或存在多张候选表，无法唯一确定统计范围')
        tid, columns = candidates[0]
        rows = self.connection.execute('SELECT id,row_index FROM table_rows WHERE table_id=? ORDER BY row_index',(tid,)).fetchall()
        details, total = [], Decimal(0)
        seen = set()
        for row in rows:
            cells = {c['column_id']:c for c in self.connection.execute('SELECT * FROM table_cells WHERE row_id=?',(row['id'],))}
            first, second, major, count = [cells[c['id']] for c in columns]
            if first['raw_value'] not in {'物理','历史'}:
                if '合计' in major['raw_value'] or '总计' in major['raw_value']:
                    continue
                raise ValueError('首选科目包含无法识别的行，拒绝静默漏计')
            if first['raw_value'] != primary.group(1) or not eligible(second['raw_value'],selected):
                continue
            if major['raw_value'] in seen:
                raise ValueError('符合条件的专业出现重复行，需要核对专业组')
            seen.add(major['raw_value'])
            if count['numeric_value'] is None:
                raise ValueError('符合条件的计划数缺失')
            if not re.fullmatch(r'\d+(?:人|名)?',compact(count['raw_value'])):
                raise ValueError('计划数原始单元格不是确定整数')
            value = Decimal(count['numeric_value'])
            if value < 0 or value != value.to_integral_value():
                raise ValueError('招生计划数必须是非负整数')
            total += value
            details.append({'field':major['raw_value'],'raw_value':count['raw_value'],'value':number(value),
                'unit':'人','row_id':row['id'],'table_id':tid,'cell_id':count['id'],
                'quote':f"{first['raw_value']} / {second['raw_value']} / {major['raw_value']} / {count['raw_value']}"})
        template = next(e for e in evidence if e.table_id == tid)
        used = Evidence(f'verified-table:{tid}', '\n'.join(d['quote'] for d in details),1,template.doc_id,
            template.title,template.source_url,template.department,template.published_at,table_id=tid,
            row_ids=[d['row_id'] for d in details])
        expression = ' + '.join(d['value'] for d in details) or '0'
        answer = f"首选{primary.group(1)}、再选{'和'.join(sorted(selected))}，仅按选科要求筛选，符合条件的专业计划合计{number(total)}人。"
        answer += '筛选明细：'+'；'.join(f"{d['field']} {d['value']}人" for d in details)+f'。算式：{expression} = {number(total)}人。'
        result = CalculationResult(True,total,'人',expression,details,evidence=[used])
        return NumericOutcome(True,result,answer,details)


def render_calculation(result):
    value = result.value
    if value.as_tuple().exponent < -6:
        value = value.quantize(Decimal('0.000001'),rounding=ROUND_HALF_UP)
    bounds = {d.get('comparison') for d in result.details if d.get('comparison') in {'gte','gt','lte','lt'}}
    qualifier = ('按下限标准计算，' if bounds<={'gte','gt'} else '按上限标准计算，' if bounds<={'lte','lt'} else '按所引界限计算，') if bounds else ''
    return f'{qualifier}结果为{number(value)}{result.unit}。算式：{result.expression}。'


def eligible(requirement, selected):
    text = compact(requirement)
    if text in {'不限','不提科目要求','无科目要求'}:
        return True
    subjects = set(re.findall(r'化学|生物|政治|地理|物理|历史',text))
    if not subjects or re.search(r'除|不选|非|不得|不能|不含|优先',text):
        raise ValueError('再选科目要求无法可靠解释：'+text)
    if len(subjects) == 1:
        return subjects <= selected
    if re.search(r'或|任选|至少.*一|选考.*一',text):
        return bool(subjects & selected)
    if re.search(r'和|及|均|同时',text):
        return subjects <= selected
    raise ValueError('多科目要求未明确且/或关系：'+text)


def check_draft(draft, canonical):
    """Only render an exact grounded paraphrase; otherwise use the checked template.

    A small model can swap labels while keeping all numeric tokens. Exact normalized
    comparison is intentionally conservative until semantic claim verification exists.
    """
    valid = compact(draft) == compact(canonical)
    return (draft if valid else canonical), {'verified':True, 'used_template':not valid,
        'reason':'exact_verified_text' if valid else 'draft_not_identical_to_verified_facts',
        'model_draft':draft}
