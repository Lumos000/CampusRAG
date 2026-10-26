from __future__ import annotations

import sqlite3
import json
import re
from decimal import Decimal

from campus_rag.config import Settings
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.citations import build_references
from campus_rag.rag.planner import QueryPlanner
from campus_rag.types import AnswerResult, CalculationResult, Evidence
from campus_rag.tools.temporal import question_periods, matches_period
from campus_rag.rag.numeric import NumericEngine, check_draft, number


ANSWER_PROMPT = """你是南京邮电大学校园资料问答助手。只能根据下面经过系统验证的证据回答。
不得补充证据中没有的事实，不得生成链接、资料名称或参考资料列表。
分别询问多个数量时，逐项对应回答，不得相加或混用其他统计口径。证据出现百分比调整时必须说明基数和调整比例，不能把基数冒充调整后的最终金额。
若证据不足或计算失败，明确说明现有资料不足。回答应简洁，并在数值题中写出关键筛选条件和算式。

问题：{question}

文本证据：
{evidence}

结构化计算：
{calculation}
"""


class RAGService:
    def __init__(self, settings: Settings, index, generator, connection: sqlite3.Connection):
        self.settings = settings
        self.index = index
        self.generator = generator
        self.planner = QueryPlanner(generator)
        self.structured = StructuredQueryEngine(connection)
        self.numeric = NumericEngine(connection,self.structured.navigation_tables)

    def answer(self, question: str) -> AnswerResult:
        if not question.strip():
            raise ValueError("问题不能为空")
        plan = self.planner.plan(question)
        plan.time_range = question_periods(question)
        self.last_plan = plan
        search_query = question
        evidence = self.index.search(
            search_query, self.settings.bm25_top_k, self.settings.dense_top_k,
            self.settings.fusion_top_k, self.settings.rerank_top_k,
            time_range=plan.time_range,
        )
        evidence = [e for e in evidence if matches_period(e.title, e.published_at, plan.time_range)]
        if not evidence:
            return AnswerResult(question,'现有离线资料不足，未找到符合问题年份和资料类型的证据。',[],[],None,
                {'verified':False,'reason':'no_matching_evidence'})
        checked = self.numeric.resolve(question,evidence)
        if checked.handled:
            if not checked.result.ok:
                return AnswerResult(question,'现有证据不足，无法可靠给出数值：'+checked.result.error,[],[],checked.result,
                    {'verified':False,'reason':checked.result.error})
            draft = self.generator.generate(
                '请组织下面已经由Python核验的答案，不添加或更改任何字段、数值、单位或来源。\n问题：'+question+'\n已核验答案：'+checked.canonical,
                thinking=False,temperature=0.0,max_new_tokens=self.settings.max_new_tokens)
            answer, verification = check_draft(draft,checked.canonical)
            verification['method'] = 'evidence_bound_numeric_tool'
            verification['facts'] = checked.facts
            return AnswerResult(question,answer,build_references(checked.result.evidence),checked.result.evidence,
                checked.result,verification)
        table_ids = sorted({item.table_id for item in evidence if item.table_id is not None})
        calculation_required = bool(re.search(r'合计|总计|之和|占.*(?:比例|多少)|复算|反推|计划.*多少|生师比|折算系数',question))
        if not calculation_required:
            plan.needs_text, plan.needs_table = True, False
        fields_valid = True
        from campus_rag.database.query import _match_column
        for item in [*plan.filters, *plan.calculation.operands]:
            if not any(_match_column(item.field, self.structured.connection.execute('SELECT * FROM table_columns WHERE table_id=?',(tid,)).fetchall()) is not None for tid in table_ids):
                fields_valid = False
        if calculation_required and table_ids and not fields_valid:
            schema = []
            for table_id in table_ids[:4]:
                columns = self.structured.connection.execute('SELECT original_name FROM table_columns WHERE table_id=? ORDER BY column_index',(table_id,)).fetchall()
                table = self.structured.connection.execute('SELECT name FROM tables WHERE id=?',(table_id,)).fetchone()
                schema.append({'table':table['name'],'columns':[c['original_name'].split('/')[-1].strip() for c in columns]})
            plan = self.planner.plan(question, json.dumps(schema,ensure_ascii=False))
            plan.time_range = question_periods(question)
            self.last_plan = plan
        calculation = self.structured.execute(plan, table_ids or None) if plan.needs_table else None
        if calculation_required and (calculation is None or not calculation.ok):
            reason = calculation.error if calculation is not None else '未生成有效的结构化计算计划'
            return AnswerResult(question, '现有资料或计算条件不足，无法可靠计算：' + reason, [], [], calculation)
        if re.search(r'多少|几(?:人|个|名|本|元)|百分比|预收额',question) and not (calculation and calculation.ok):
            return AnswerResult(question,'现有数值抽取工具尚不能验证该问题，无法可靠给出数值。',[],[],calculation,
                {'verified':False,'reason':'unsupported_numeric_question'})
        used = self._select_used_evidence(plan.needs_text, evidence, calculation)
        if not used and (calculation is None or not calculation.ok):
            answer = "现有离线资料不足，无法可靠回答该问题。"
        else:
            prompt = ANSWER_PROMPT.format(
                question=question,
                evidence=self._format_evidence(used if plan.needs_text else []),
                calculation=self._format_calculation(calculation),
            )
            answer = self.generator.generate(
                prompt, thinking=False,
                temperature=0.0, max_new_tokens=self.settings.max_new_tokens,
            )
        references = build_references(used)
        verification = None
        if calculation and calculation.ok:
            canonical = f'经结构化筛选和Python计算，结果为{number(calculation.value)}{calculation.unit}。算式：{calculation.expression}。'
            answer, verification = check_draft(answer,canonical)
        return AnswerResult(question, answer, references, used, calculation,verification)

    @staticmethod
    def _select_used_evidence(needs_text: bool, evidence: list[Evidence], calculation: CalculationResult | None) -> list[Evidence]:
        selected = list(evidence) if needs_text else []
        if calculation and calculation.ok:
            selected.extend(calculation.evidence)
        unique: dict[str, Evidence] = {}
        for item in selected:
            unique[item.evidence_id] = item
        return list(unique.values())

    @staticmethod
    def _format_evidence(evidence: list[Evidence]) -> str:
        return "\n\n".join(f"[证据{i}] {item.text}" for i, item in enumerate(evidence, 1)) or "无"

    @staticmethod
    def _format_calculation(result: CalculationResult | None) -> str:
        if result is None:
            return "不需要结构化计算"
        if not result.ok:
            return f"计算失败：{result.error}"
        value = format(result.value, "f") if isinstance(result.value, Decimal) else str(result.value)
        rows = [f"{item['table_name']} / {item['row_label']} / {item['field']} = {item['raw_value']}" for item in result.details]
        return f"筛选结果：{'；'.join(rows)}\n算式：{result.expression}\n结果：{value}{result.unit}"

