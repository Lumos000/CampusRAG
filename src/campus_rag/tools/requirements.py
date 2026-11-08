"""Conservative evaluation of textual all/any requirements."""
import re


def satisfies(requirement, selected):
    text = re.sub(r'\s+', '', requirement)
    if text in {'不限', '不提科目要求', '无科目要求', '无要求'}:
        return True
    if re.search(r'除|不选|非|不得|不能|不含|优先|[()（）]', text):
        raise ValueError('条件包含尚不支持或有歧义的要求：'+text)
    mode = None
    suffix = re.search(r'(任选(?:其)?一|至少(?:满足|选择|选考)?一(?:项|门)?|均(?:须|需)?(?:选考|满足)?|同时(?:选考|满足)?|必须(?:选考|满足))$',text)
    if suffix:
        mode = 'any' if re.match(r'任选|至少',suffix[1]) else 'all'
        text = text[:suffix.start()]
    if text.startswith('同时'):
        if mode=='any':
            raise ValueError('混合且/或条件需要明确范围：'+requirement)
        mode,text = 'all',text[2:]
    if re.search(r'均|同时|任选|至少|选考|必须|须|满足',text):
        raise ValueError('条件表达未完整解析，不能静默漏计：'+requirement)
    any_of = '或' in text or mode=='any'
    has_all = bool(re.search(r'和|及|与',text))
    if mode=='any' and '或' in text and has_all:
        raise ValueError('混合且/或条件需要明确括号范围：'+requirement)
    all_of = (has_all and mode!='any') or mode=='all'
    if any_of and all_of:
        raise ValueError('混合且/或条件需要明确括号范围：'+text)
    if not any_of and not all_of and re.search(r'[、,，/]', text):
        raise ValueError('多个条件未明确且/或关系：'+text)
    parts = re.split(r'或|和|及|与|[、,，/]', text)
    if any(not re.fullmatch(r'[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_-]{0,30}', p) for p in parts):
        raise ValueError('条件无法可靠解释：'+text)
    matches = [p in selected for p in parts]
    return any(matches) if any_of else all(matches)
