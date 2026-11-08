"""Evidence-bound numeric tools. Unsupported or ambiguous inputs fail closed.

No answer constants, model-authored SQL, or executable expressions are accepted.
"""
from __future__ import annotations

import re
from bisect import bisect_left
from decimal import Decimal, ROUND_HALF_UP, localcontext

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


class NumericEngine:
    @staticmethod
    def text_facts(evidence, academic_units=False):
        """Extract explicit quantities, preserving bounds, scope and original offsets.

        Whitespace (including PDF line breaks) is ignored for matching only. Quotes
        always point back to the unmodified evidence; missing values are not inferred.
        """
        from campus_rag.database.normalization import quantity
        facts, seen = {}, set()
        units = r'万元|亿元|元|万人|千人|人|名|平方米|㎡|台|本|项|个|门|学年|学期|年|月|周|天|小时|分钟|%|％'
        if academic_units:
            units = '学分|学时|'+units
        amount = re.compile(r'(?<![\d.])(?P<number>\d[\d,]*(?:\.\d+)?)(?:[-—–～~至到](?P<upper>\d[\d,]*(?:\.\d+)?))?(?P<unit>'+units+r')?(?P<denominator>/(?:小时|分钟|天|周|月|学年|年|人|次|台|个))?')
        relations = {'不超过':'lte','不高于':'lte','至多':'lte','最多':'lte','上限为':'lte',
            '不低于':'gte','不少于':'gte','至少':'gte','最低为':'gte','下限为':'gte',
            '超过':'gt','高于':'gt','低于':'lt','小于':'lt','大于':'gt',
            '能够完成':'eq','完成':'eq','共计':'eq','共':'eq','设置':'eq',
            '暂按':'eq','按':'eq','上浮':'eq','提高':'eq','下调':'eq',
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
                if match['unit']=='学年' and re.search(r'截止|截至|年度',local):
                    continue
                relation = connector.search(local)
                suffix = re.match(r'(及以上|及以下|以上|以下)',text[match.end():])
                gpa = bool(re.search(r'绩点(?:要求)?(?:为|不低于|不超过|高于|低于)?$',local))
                gpa = gpa or bool('绩点' in local and re.search(r'(?:但|且)(?:高于|低于|不低于|不高于)$',local))
                annual = re.search(r'每(?:生|人)每(?:学)?年$',local)
                # A bare labelled quantity is explicit evidence too. Dates,
                # page numbers and unitless numbers still need a typed label.
                page_prefix = re.match(r'[1-9]\d{0,2}[ \t]*\r?\n',item.text[offsets[start]:offsets[match.start()]])
                bare_label = re.sub(r'^\d+','',local) if page_prefix else local
                labelled = bool(match['unit'] and re.fullmatch(r'[\u4e00-\u9fffA-Za-z]{2,}',bare_label)
                    and not re.search(r'(?:由|前|后|至|于|第|比|较|在|占)$',bare_label))
                count_label = re.search(r'(?:人数|数量|总数)(?:为|有)?$',local)
                ratio_label = re.search(r'比(?:为)?$',local) and re.match(r'[:：]1(?!\d)',text[match.end():])
                if not paired and not relation and not suffix and not annual and not labelled and not count_label and not re.search(r'(?:额度|金额|标准|预算|期限|时长|人数)$',local):
                    continue
                inherited = None
                if count_label and not match['unit']:
                    prior = [m for m in amount.finditer(text[start:match.start()]) if m['unit'] in {'人','名','个','项','台','本'}]
                    inherited = '人' if '人数' in count_label.group() else prior[-1]['unit'] if prior else None
                if not match['unit'] and not gpa and not inherited and not ratio_label:
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
                label = re.sub(r'每(?:生每年|人每年|小时|分钟|天|周|月|年|人)$','',label).strip('：:，,')
                label = re.sub(r'^第[一二三四五六七八九十百\d]+条','',label)
                label = re.sub(r'^（[一二三四五六七八九十]+）','',label)
                if not relation:
                    if page_prefix:
                        label = re.sub(r'^\d+','',label)
                    label = re.sub(r'^.*?(?:现有|现设|设有|聘请|共有|有)(?=[\u4e00-\u9fff]{2,})','',label)
                label = re.sub(r'每(?:生|人)每(?:学)?年$','',label)
                if not label or not re.search(r'[\u4e00-\u9fffA-Za-z]',label):
                    continue
                periods = re.findall(r'每(?:生每年|人每年|学年|学期|小时|分钟|天|周|月|年|人)',prefix)
                period = periods[-1] if periods else ''
                if annual:
                    period = '每年'
                annual_denominator = bool(match['unit']=='元' and (
                    re.match(r'/(?:生|人)[·•・]?(?:学年|年)',text[match.end():]) or
                    (match['denominator']=='/人' and re.match(r'[·•・](?:学年|年)',text[match.end():]))))
                if annual_denominator:
                    period = '每年'
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
                if annual:
                    scope['basis'] = '生' if '生' in annual.group() else '人'
                    scope['period_label'] = '学年' if '学年' in annual.group() else '年'
                elif annual_denominator:
                    scope['basis'] = '人' if match['denominator']=='/人' else '生' if text[match.end():].startswith('/生') else '人'
                    scope['period_label'] = '学年' if re.match(r'(?:/(?:生|人))?[·•・]?(?:学年)',text[match.end():]) else '年'
                note = re.search(r'预收[,，][^。]+',text[match.end():scope_end]) if match['unit']=='元' else None
                if note:
                    scope['note'] = note.group()
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
                    value,unit = quantity(token+'人' if match['unit']=='名' else raw) if match['unit'] else (Decimal(token),'ratio' if ratio_label else inherited or '绩点')
                    if value is None and match['unit'] in {'学年','学期','月','周','分钟','门'}:
                        value,unit = Decimal(token.replace(',','')),match['unit']
                    if value is None:
                        continue
                    per = re.search(r'每(小时|分钟|天|周|月|年|人)$',prefix)
                    if match['denominator'] and not annual_denominator:
                        unit += match['denominator']
                    elif per and unit=='元' and not annual:
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

def render_calculation(result):
    if result.outputs:
        answers = []
        for i,item in enumerate(result.outputs,1):
            value = item['value']
            if value.as_tuple().exponent < -6:
                value = value.quantize(Decimal('0.000001'),rounding=ROUND_HALF_UP)
            unit = '' if item['unit']=='ratio' else item['unit']
            if item.get('basis'):
                unit += '/'+item['basis']
            if item.get('per_unit'):
                unit += ('·' if item.get('basis') else '/')+item.get('period_label',item['per_unit'])
            note = '（'+item['note']+'）' if item.get('note') else ''
            if item['unit']=='ratio' and item['value'].as_tuple().exponent < -2:
                with localcontext() as context:
                    context.prec = max(context.prec,len(item['value'].as_tuple().digits)+2)
                    approximate = item['value'].quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
                if approximate!=item['value']:
                    note += '（约'+number(approximate)+':1）'
            answers.append(f"{item.get('label') or '结果'+str(i)}为{number(value)}{unit}{note}。算式：{item['expression']}。")
        return '\n'.join(answers)
    value = result.value
    if value.as_tuple().exponent < -6:
        value = value.quantize(Decimal('0.000001'),rounding=ROUND_HALF_UP)
    bounds = {d.get('comparison') for d in result.details if d.get('comparison') in {'gte','gt','lte','lt'}}
    qualifier = ('按下限标准计算，' if bounds<={'gte','gt'} else '按上限标准计算，' if bounds<={'lte','lt'} else '按所引界限计算，') if bounds else ''
    return f'{qualifier}结果为{number(value)}{result.unit}。算式：{result.expression}。'


def check_draft(draft, canonical):
    """Only render an exact grounded paraphrase; otherwise use the checked template.

    A small model can swap labels while keeping all numeric tokens. Exact normalized
    comparison is intentionally conservative until semantic claim verification exists.
    """
    valid = compact(draft) == compact(canonical)
    return (draft if valid else canonical), {'verified':True, 'used_template':not valid,
        'reason':'exact_verified_text' if valid else 'draft_not_identical_to_verified_facts',
        'model_draft':draft}
