from __future__ import annotations

import sqlite3
import json
import re

from campus_rag.config import Settings
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.citations import build_references
from campus_rag.rag.planner import QueryPlanner, bind_legacy_plan
from campus_rag.types import AnswerResult, CalculationResult, Evidence, QueryPlan
from campus_rag.tools.temporal import question_periods, matches_period
from campus_rag.rag.numeric import NumericEngine, render_calculation


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
        periods = question_periods(question)
        self.last_plan = QueryPlan(time_range=periods)
        evidence = self.index.search(
            question, self.settings.bm25_top_k, self.settings.dense_top_k,
            self.settings.fusion_top_k, self.settings.rerank_top_k, time_range=periods,
        )
        evidence = [e for e in evidence if matches_period(e.title,e.published_at,periods)]
        if not evidence:
            return self._refuse(question,'未找到符合问题年份和资料类型的证据。')
        numeric_question = bool(re.search(
            r'计算|求和|合计|总计|之和|占.*(?:比例|多少)|百分比|平均|相差|差额|多多少|少多少|乘|除以|复算|反推|生师比|折算|多少|几(?:人|个|名|本|元)|预收额',question))
        if not numeric_question:
            prompt = ANSWER_PROMPT.format(question=question,evidence=self._format_evidence(evidence),calculation='不需要计算')
            answer = self.generator.generate(prompt,thinking=False,temperature=0.0,max_new_tokens=self.settings.max_new_tokens)
            return AnswerResult(question,answer,build_references(evidence),evidence)
        if self.settings.numeric_shortcuts:
            checked = self.numeric.resolve(question,evidence)
            if checked.handled:
                if not checked.result.ok:
                    return self._refuse(question,checked.result.error,checked.result)
                return AnswerResult(question,checked.canonical,build_references(checked.result.evidence),
                    checked.result.evidence,checked.result,{'verified':True,'used_template':True,
                    'method':'evidence_bound_numeric_tool','facts':checked.facts})
        table_ids = list(dict.fromkeys(e.table_id for e in evidence if e.table_id is not None))
        schema = self.structured.schema_context(table_ids,periods,question)
        allowed_ids = [table['table_id'] for table in schema]
        facts = self.numeric.text_facts(evidence)
        context = json.dumps({'tables':schema,'facts':[
            {k:v for k,v in fact.items() if k!='evidence'} for fact in facts.values()][:24]},ensure_ascii=False)
        # Only supplied facts are available to the executor.
        facts = dict(list(facts.items())[:24])
        if not schema and not facts:
            return self._refuse(question,'没有可验证的表格或正文数值。')
        error, previous = '', None
        for attempt in range(2):
            try:
                plan = self.planner.step_plan(question,context,error,previous)
                if plan.steps and plan.table_id is None and any('field_id' in s or s.get('op')=='filter' for s in plan.steps):
                    if len(schema)==1:
                        plan.table_id = schema[0]['table_id']
                if not plan.steps:
                    plan = bind_legacy_plan(plan,schema) if plan.calculation.operands else plan
                plan.time_range = periods
                self.last_plan = plan
            except (ValueError,TypeError) as exc:
                error = str(exc)
                previous = getattr(self.planner,'last_raw',None)
                if attempt == 0:
                    continue
                return self._refuse(question,'计划校验失败：'+error)
            calculation = self.structured.execute_steps(plan,allowed_ids,question,facts)
            if calculation.ok:
                return AnswerResult(question,render_calculation(calculation),build_references(calculation.evidence),
                    calculation.evidence,calculation,{'verified':True,'used_template':True,
                    'method':'validated_step_plan','steps':calculation.steps})
            if not calculation.repairable or attempt == 1:
                return self._refuse(question,calculation.error,calculation)
            error = calculation.error
            previous = {'table_id':plan.table_id,'steps':plan.steps,'output':plan.output}
        return self._refuse(question,'未生成有效计算计划。')

    @staticmethod
    def _refuse(question,reason,calculation=None):
        return AnswerResult(question,'现有证据或计算条件不足，无法可靠给出数值：'+reason,[],[],calculation,
                            {'verified':False,'reason':reason})

    @staticmethod
    def _format_evidence(evidence: list[Evidence]) -> str:
        return "\n\n".join(f"[证据{i}] {item.text}" for i, item in enumerate(evidence, 1)) or "无"

