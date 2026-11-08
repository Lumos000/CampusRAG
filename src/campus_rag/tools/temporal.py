"""Explicit question dates are authoritative; document titles precede publication dates."""
import re

_QUANTITY_TAIL = r'(?:-20\d{2})?(?:\.\d+)?(?:万元|亿元|元|万人|千人|人|名|台|个|项|门|本|件|次|小时|分钟|学分|学时|平方米|㎡|%|％)'


def _year_values(text):
    return [m[1] for m in re.finditer(r'(?<!\d)(20\d{2})(?!\d)',text)
            if not re.match(_QUANTITY_TAIL,text[m.end():])]


def normalize_period(text):
    return re.sub(r'[—–－~～至/]', '-', str(text)).replace(' ', '')


def question_periods(question):
    text = normalize_period(question)
    # Four digits followed by a quantity unit are operands, not dates. This
    # also covers the first endpoint of a monetary/count range.
    spans = [(m[1],m[2]) for m in re.finditer(r'(?<!\d)(20\d{2})年?-(20\d{2})(?:年|学年)?', text)
             if not re.match(_QUANTITY_TAIL,text[m.end():])]
    if spans:
        return [f'{first}-{last}' for first, last in spans]
    return list(dict.fromkeys(_year_values(text)))


def matches_period(title, published_at, periods):
    if not periods:
        return True
    normalized = normalize_period(title)
    title_years = set(_year_values(normalized))
    for period in periods:
        period = normalize_period(period)
        if re.fullmatch(r'20\d{2}-20\d{2}', period):
            first, last = period.split('-')
            if re.search(first + r'年?-' + last, normalized):
                return True
        elif re.fullmatch(r'20\d{2}', period):
            if title_years:
                if period in title_years:
                    return True
            elif str(published_at or '').startswith(period + '-'):
                return True
    return False
