from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation

from campus_rag.types import CalculationSpec, FilterSpec, OperandSpec, QueryPlan


PLAN_PROMPT = """你是校园资料查询规划器。只输出一个 JSON 对象，不要输出解释、Markdown 或 SQL。
允许的问题类型：fact,list,aggregate,compare,ratio,fee,unanswerable。
允许的筛选运算：contains,eq,gt,gte,lt,lte。
允许的聚合：value,sum,count,avg。允许的计算：none,sum,count,ratio,difference,multiply,average。
JSON 字段必须为：question_type,time_range,keywords,entities,fields,filters,group_by,unit,
needs_text,needs_table,calculation。calculation 包含 operation,operands,multiplier,output_unit；
每个 operand 包含 field,aggregate,filters。不要生成 SQL 或 Python。
完整格式示例（只参考结构，不复制内容）：{example}
可用表格信息：{context}
选科筛选时，考生必须满足所有要求；“不限”表示不限制再选科目。不能把考生选了政治和地理误解释为专业要求同时包含这两科。
问题：{question}"""

EXAMPLE = {'question_type':'aggregate','time_range':[],'keywords':['学生人数'],'entities':[],
           'fields':['人数'],'filters':[{'field':'类别','operator':'eq','value':'本科'}],
           'group_by':[],'unit':'人','needs_text':False,'needs_table':True,
           'calculation':{'operation':'sum','operands':[{'field':'人数','aggregate':'sum','filters':[]}],
                          'multiplier':None,'output_unit':'人'}}


class QueryPlanner:
    def __init__(self, generator):
        self.generator = generator

    def plan(self, question: str, context: str = '') -> QueryPlan:
        prompt = PLAN_PROMPT.format(question=question,context=context,example=json.dumps(EXAMPLE,ensure_ascii=False))
        raw = self.generator.generate(prompt, thinking=False, temperature=0.0, max_new_tokens=512)
        try:
            return parse_plan(raw)
        except (ValueError, json.JSONDecodeError):
            repair = self.generator.generate(
                prompt + '\n上次格式错误，请按照示例修复，只输出JSON：\n' + raw,
                thinking=False, temperature=0.0, max_new_tokens=512,
            )
            try:
                return parse_plan(repair)
            except (ValueError, json.JSONDecodeError):
                return QueryPlan(keywords=[question], needs_text=True, needs_table=False)


def parse_plan(raw: str) -> QueryPlan:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("规划结果不包含 JSON")
    data = json.loads(match.group())
    if not isinstance(data, dict):
        raise ValueError('查询计划必须是对象')
    for key in ['needs_text','needs_table']:
        if key in data and not isinstance(data[key], bool):
            raise ValueError('查询计划布尔类型不合法')
    for key in ['filters','keywords','entities','fields','group_by','time_range']:
        if key in data and not isinstance(data[key], list):
            raise ValueError('查询计划列表类型不合法')
    filters = [_filter(item) for item in data.get("filters", [])]
    calc_data = data.get("calculation") or {}
    if not isinstance(calc_data, dict) or not isinstance(calc_data.get('operands',[]),list):
        raise ValueError('计算计划类型不合法')
    for item in calc_data.get('operands',[]):
        if not isinstance(item,dict) or not isinstance(item.get('filters',[]),list):
            raise ValueError('操作数必须是对象')
    operands = [OperandSpec(
        field=str(item.get("field", "")),
        aggregate=item.get("aggregate", "value"),
        filters=[_filter(value) for value in item.get("filters", [])],
    ) for item in calc_data.get("operands", []) if item.get("field")]
    multiplier = calc_data.get("multiplier")
    try:
        multiplier = Decimal(str(multiplier)) if multiplier is not None else None
        if multiplier is not None and not multiplier.is_finite():
            raise ValueError('乘数不是有限数值')
    except InvalidOperation as exc:
        raise ValueError('乘数不合法') from exc
    calculation = CalculationSpec(
        operation=calc_data.get("operation", "none"), operands=operands,
        multiplier=multiplier,
        output_unit=str(calc_data.get("output_unit", "")),
    )
    if not isinstance(calculation.operation,str) or calculation.operation not in {'none','sum','count','ratio','difference','multiply','average'}:
        raise ValueError('不支持的计算操作')
    if any(not isinstance(o.aggregate,str) or o.aggregate not in {'value','sum','count','avg'} for o in operands):
        raise ValueError('不支持的聚合操作')
    allowed_types = {"fact", "list", "aggregate", "compare", "ratio", "fee", "unanswerable"}
    question_type = data.get("question_type", "fact")
    if not isinstance(question_type,str) or question_type not in allowed_types:
        raise ValueError("不支持的问题类型")
    return QueryPlan(
        question_type=question_type,
        time_range=_strings(data.get("time_range", [])), keywords=_strings(data.get("keywords", [])),
        entities=_strings(data.get("entities", [])), fields=_strings(data.get("fields", [])),
        filters=filters, group_by=_strings(data.get("group_by", [])), unit=str(data.get("unit", "")),
        needs_text=bool(data.get("needs_text", True)), needs_table=bool(data.get("needs_table", False)),
        calculation=calculation,
    )


def _filter(data: dict) -> FilterSpec:
    if not isinstance(data,dict) or not isinstance(data.get('field'),str) or not data['field'].strip():
        raise ValueError('筛选条件必须是含有字段名的对象')
    operator = data.get("operator", "contains")
    if not isinstance(operator,str) or operator not in {"contains", "eq", "gt", "gte", "lt", "lte"}:
        raise ValueError("不支持的筛选运算")
    return FilterSpec(str(data.get("field", "")), operator, str(data.get("value", "")))


def _strings(value) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []

