from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_BIG_UNITS = {"万": 10000, "亿": 100000000}
_UNIT_PATTERNS = (
    ("%", "%"), ("％", "%"), ("万元", "万元"), ("元", "元"),
    ("平方米", "平方米"), ("㎡", "平方米"), ("人", "人"), ("年", "年"),
)


def question_quantities(question: str, unit: str) -> list[dict]:
    """Explicit unit-adjacent question inputs, not years or arbitrary numbers."""
    aliases = r'(?:人|名|位)' if unit=='人' else re.escape(unit)
    pattern = r'(?<![\d.+-])(?P<value>\d+(?:\.\d+)?|[一二两三四五六七八九十百千]+)\s*(?:个)?'+aliases
    output = []
    for match in re.finditer(pattern,question):
        token = match['value']
        value = Decimal(token) if token[0].isdigit() else chinese_integer(token)
        if value is not None:
            output.append({'literal':str(value),'unit':unit,'quote':match.group(),'span':list(match.span())})
    return output


def asks_total(question: str) -> bool:
    return bool(re.search(r'共需|共计|合计|总计|总额|总共|一共|总金额|共.*多少元',question))


def question_operand_roles(question: str, unit: str) -> list[dict]:
    """Locate explicit quantity roles; leave implicit or ambiguous roles unknown."""
    output=[]
    patterns={
        'completed':r'已(?:经)?(?:完成|修读|修满|修|使用|用|支付|支出|消耗|缴纳)(?:了)?',
        'total':r'总计|共计|合计|总量|总额|总数|总学分|总学时|总预算|总额度|总共|共需|预算',
        'remaining':r'还需|还差|还剩|尚需|尚缺|尚余|剩余(?:量)?|余量',
        'planned':r'计划(?:完成|修读|使用|支付|支出)?|拟(?:完成|修读|使用)?|预计(?:完成|修读|使用)?',
    }
    previous_end=0
    for quantity in question_quantities(question,unit):
        start,end=quantity['span']
        left=question[previous_end:start];previous_end=end
        left=re.split(r'[，,。；;！？?]',left)[-1]
        right=question[end:end+10]
        matches=[]
        for role,pattern in patterns.items():
            for match in re.finditer(pattern,left):
                if re.fullmatch(r'(?:为|是|有|共|的|了|约|大约|至少|至多|不少于|不超过|\s)*',left[match.end():]):
                    matches.append((match.start(),role,match.group()))
        role=max(matches)[1] if matches else 'unknown'
        if re.search(r'(?:计划|拟|预计)(?:已)?(?:经)?(?:完成|修读|使用)(?:了)?\s*$',left):role='planned'
        if role=='unknown' and re.match(r'\s*(?:已完成|已修读|已使用)',right):role='completed'
        exact=not bool(re.search(r'约|大约|至少|至多|不少于|不超过',left))
        output.append({**quantity,'role':role,'exact':exact})
    return output


def chinese_integer(text: str) -> int | None:
    if not text or any(ch not in _DIGITS | _SMALL_UNITS | _BIG_UNITS for ch in text):
        return None
    if all(ch in _DIGITS for ch in text):
        return int("".join(str(_DIGITS[ch]) for ch in text))
    total = section = number = 0
    for ch in text:
        if ch in _DIGITS:
            number = _DIGITS[ch]
        elif ch in _SMALL_UNITS:
            section += (number or 1) * _SMALL_UNITS[ch]
            number = 0
        else:
            section += number
            total += section * _BIG_UNITS[ch]
            section = number = 0
    return total + section + number


def parse_decimal(value: str) -> Decimal | None:
    text = str(value).strip().replace(",", "").replace("，", "")
    text = re.sub(r"^[约近超]", "", text)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match:
        try:
            return Decimal(match.group())
        except InvalidOperation:
            return None
    integer = chinese_integer(re.sub(r"[余多左右约]", "", text))
    return Decimal(integer) if integer is not None else None


def detect_unit(value: str) -> str:
    text = str(value)
    for marker, unit in _UNIT_PATTERNS:
        if marker in text:
            return unit
    return ""


def normalize_cell(value: object) -> dict[str, str | None]:
    raw = "" if value is None else str(value).strip()
    normalized = re.sub(r"\s+", " ", raw).replace("％", "%")
    unit = detect_unit(normalized)
    number = parse_decimal(normalized)
    if number is not None and unit == "%":
        number = number / Decimal(100)
    return {
        "raw_value": raw,
        "normalized_value": normalized.lower(),
        "numeric_value": format(number, "f") if number is not None else None,
        "unit": unit,
    }


def convert_unit(value: Decimal, source: str, target: str) -> Decimal:
    if not target or source == target:
        return value
    conversions = {
        ("万元", "元"): Decimal(10000),
        ("元", "万元"): Decimal("0.0001"),
        ("%", "ratio"): Decimal(1),
        ("ratio", "%"): Decimal(100),
    }
    try:
        return value * conversions[(source, target)]
    except KeyError as exc:
        raise ValueError(f"不支持的单位转换：{source} -> {target}") from exc

