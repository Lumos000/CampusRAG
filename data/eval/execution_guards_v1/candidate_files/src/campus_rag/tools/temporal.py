"""Explicit question dates are authoritative; document titles precede publication dates."""
import re


def normalize_period(text):
    return re.sub(r'[—–－~～至/]', '-', str(text)).replace(' ', '')


def question_periods(question):
    text = normalize_period(question)
    spans = re.findall(r'(?<!\d)(20\d{2})年?-(20\d{2})(?:年|学年)?', text)
    if spans:
        return [f'{first}-{last}' for first, last in spans]
    return list(dict.fromkeys(re.findall(r'(?<!\d)(20\d{2})(?!\d)', text)))


def matches_period(title, published_at, periods):
    if not periods:
        return True
    normalized = normalize_period(title)
    title_years = set(re.findall(r'(?<!\d)20\d{2}(?!\d)', normalized))
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
