from __future__ import annotations

import sqlite3
import json
import re
from dataclasses import asdict

from campus_rag.config import Settings
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.citations import build_references
from campus_rag.rag.planner import QueryPlanner, bind_legacy_plan, repair_json_delimiter, parse_plan
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
        # Per-request diagnostics are local data, never inserted into model prompts.
        self.last_diagnostics = {'route':'retrieval','retrieved':[], 'facts':[],
            'schema':[], 'attempts':[], 'citations':[], 'semantic_correct':None}
        self.planner.last_raw = ''
        self.planner.last_prompt = ''
        result = self._answer(question)
        self.last_diagnostics['citations'] = list(result.references)
        self.last_diagnostics['cited_evidence_ids'] = [e.evidence_id for e in result.evidence]
        self.last_diagnostics['execution_ok'] = result.calculation.ok if result.calculation else None
        self.last_diagnostics['refusal_reason'] = (result.numeric_verification or {}).get('reason')
        return result

    def _answer(self, question: str) -> AnswerResult:
        if not question.strip():
            raise ValueError("问题不能为空")
        periods = question_periods(question)
        self.last_plan = QueryPlan(time_range=periods)
        evidence = self.index.search(
            question, self.settings.bm25_top_k, self.settings.dense_top_k,
            self.settings.fusion_top_k, self.settings.rerank_top_k, time_range=periods,
        )
        evidence = [e for e in evidence if matches_period(e.title,e.published_at,periods)]
        self.last_diagnostics['retrieved'] = [e.evidence_id for e in evidence]
        if not evidence:
            return self._refuse(question,'未找到符合问题年份和资料类型的证据。')
        numeric_question = bool(re.search(
            r'计算|求和|合计|总计|之和|占.*(?:比例|多少)|百分比|平均|相差|差额|多多少|少多少|乘|除以|复算|反推|生师比|折算|多少|几(?:人|个|名|本|元)|预收额',question))
        if not numeric_question:
            self.last_diagnostics['route'] = 'text'
            prompt = ANSWER_PROMPT.format(question=question,evidence=self._format_evidence(evidence),calculation='不需要计算')
            answer = self.generator.generate(prompt,thinking=False,temperature=0.0,max_new_tokens=self.settings.max_new_tokens)
            return AnswerResult(question,answer,build_references(evidence),evidence)
        missing = re.search(r'(?:我|本人)(?:还)?(?:没有|尚未|未)(?:提供|给出|明确)([^。；;？?]+)',question)
        hypothetical = bool(re.search(r'(?:仅|只)(?:按|作|做)|假设|纯计算',question))
        if missing and not hypothetical:
            self.last_diagnostics['route'] = 'missing_question_inputs'
            return self._refuse(question,'未提供'+missing[1].strip('，, ')+
                '，不能用公共资料中的示例或上下限代替个人计算条件。')
        if self.settings.numeric_shortcuts:
            checked = self.numeric.resolve(question,evidence)
            if checked.handled:
                self.last_diagnostics['route'] = 'numeric_shortcut'
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
        self.last_diagnostics.update(route='numeric_plan',schema=schema,
            facts=[{k:v for k,v in fact.items() if k!='evidence'} for fact in facts.values()])
        if not schema and not facts:
            return self._refuse(question,'没有可验证的表格或正文数值。')
        error, previous, syntax_repair = '', None, None
        for attempt in range(2):
            diagnostic = {'attempt':attempt+1,'kind':'initial' if attempt==0 else 'repair',
                'feedback':error, 'syntax_ok':False, 'execution_ok':None,'semantic_correct':None}
            self.last_diagnostics['attempts'].append(diagnostic)
            try:
                if syntax_repair is not None:
                    plan = parse_plan(syntax_repair,require_final=True)
                    diagnostic.update(kind='syntax_repair',raw_plan=syntax_repair,syntax_ok=True,
                        normalization='single_closing_delimiter; repair budget consumed')
                else:
                    plan = self.planner.step_plan(question,context,error,previous)
                    diagnostic.update(raw_plan=self.planner.last_raw,syntax_ok=True)
                diagnostic['prompt'] = getattr(self.planner,'last_prompt','')
                if plan.steps and plan.table_id is None and any(
                        s.get('op') in {'filter','lookup','sum','average','count'} and
                        not any(k in s for k in ['fact_id','literal','inputs']) for s in plan.steps):
                    fields = {s['field_id'] for s in plan.steps if 'field_id' in s}
                    fields.update(f['field_id'] for s in plan.steps for f in s.get('filters',[]))
                    candidates = [t for t in schema if fields and fields<={c['field_id'] for c in t['columns']}]
                    if len(schema)==1:
                        plan.table_id = schema[0]['table_id']
                    elif len(candidates)==1:
                        plan.table_id = candidates[0]['table_id']
                if not plan.steps:
                    plan = bind_legacy_plan(plan,schema) if plan.calculation.operands else plan
                plan.time_range = periods
                self.last_plan = plan
                diagnostic['bound_plan'] = asdict(plan)
                diagnostic['field_bindings'] = [{'table_id':table['table_id'],**column}
                    for table in schema for column in table['columns']
                    if column['field_id'] in {field_id for step in plan.steps
                        for field_id in [step.get('field_id'),*[f.get('field_id') for f in step.get('filters',[])]]}]
            except (ValueError,TypeError) as exc:
                error = str(exc)
                diagnostic.update(raw_plan=getattr(self.planner,'last_raw',''),error=error,
                    failure_stage='plan_validation',prompt=getattr(self.planner,'last_prompt',''))
                previous = getattr(self.planner,'last_raw',None)
                if attempt == 0:
                    syntax_repair = repair_json_delimiter(previous) if isinstance(previous,str) else None
                    continue
                return self._refuse(question,'计划校验失败：'+error)
            calculation = self.structured.execute_steps(plan,allowed_ids,question,facts)
            diagnostic.update(execution_ok=calculation.ok,repairable=calculation.repairable,
                execution=asdict(calculation),failure_stage=None if calculation.ok else 'execution_validation')
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

