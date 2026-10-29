import json
import re

import pytest

from campus_rag.config import Settings, load_settings
from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.rag.numeric import NumericEngine
from campus_rag.tools.numbers import normalize_cell
from campus_rag.index.hybrid import validate_records
from campus_rag.rag.service import RAGService
from campus_rag.rag.generator import ContextBudgetError, TransformersGenerator
from campus_rag.types import Chunk, Evidence, ParsedTable, SourceDocument


class SelectAll:
    def generate(self, prompt, **kwargs):
        return json.dumps({'sentence_ids': re.findall(r'\[(S\d+)\]', prompt)})


def service(repo, tmp_path, evidence):
    class Index:
        def search(self, *args, **kwargs): return evidence
    return RAGService(Settings(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path),
                      Index(), SelectAll(), repo.connection)


def test_default_is_named_public_corpus():
    settings = load_settings()
    assert settings.database_path.name == 'campus_public.db'
    assert settings.index_dir.parent == settings.database_path.parent


@pytest.mark.parametrize('question', [
    '计算机学院选课失败应该怎么办？', '计算机专业申请转专业有哪些条件？',
    '课程体系共多少学分？', '实验课程需要多少学时？',
])
def test_factual_requests_do_not_enter_arithmetic(tmp_path, question):
    with Repository(tmp_path/'rules.db') as repo:
        repo.initialize()
        item = Evidence('a', '完成登记后可向开课部门提交申请。课程共12学分，实验共24学时。', 1, 1, '课程指南', 'https://example.edu/a')
        rag = service(repo, tmp_path, [item])
        result = rag.answer(question)
        assert rag.last_diagnostics['route'] == 'text'
        assert result.references


def test_credit_lookup_excludes_unrelated_table_and_notice(tmp_path):
    with Repository(tmp_path/'credits.db') as repo:
        repo.initialize()
        items = [Evidence('a','课程=报名名单 | 编号=12',1,1,'2026报名名单','https://example.edu/a',table_id=1),
                 Evidence('b','星河学院开设微专业，报名方式另行通知。',.9,2,'星河学院微专业通知','https://example.edu/b'),
                 Evidence('c','星河学院微专业课程共12学分。',.8,3,'培养方案','https://example.edu/c')]
        result = service(repo,tmp_path,items).answer('星河学院微专业课程共多少学分？')
        assert '12学分' in result.answer and result.references == ['培养方案｜https://example.edu/c']


def test_credit_lookup_refuses_without_a_stated_quantity(tmp_path):
    with Repository(tmp_path/'no_credits.db') as repo:
        repo.initialize()
        item = Evidence('a','星河学院开设微专业，报名方式另行通知。',1,1,'通知','https://example.edu/a')
        result = service(repo,tmp_path,[item]).answer('星河学院微专业共多少学分？')
        assert '无法确认' in result.answer and not result.references


def test_credit_lookup_does_not_replace_body_with_heading_only(tmp_path):
    with Repository(tmp_path/'credit_sections.db') as repo:
        repo.initialize()
        doc = repo.add_document(SourceDocument('课程通知','https://example.edu/notice',sha256='notice'))
        repo.add_chunks([Chunk('heading',doc,'7. 海洋技术\n\n开办学院：星河学院','',20)])
        item = Evidence('body','星河学院海洋技术微专业课程共12学分。\n\n8. 航海技术\n\n开办学院：其他学院',
                        1,doc,'课程通知','https://example.edu/notice')
        result = service(repo,tmp_path,[item]).answer('星河学院海洋技术微专业课程共多少学分？')
        assert '12学分' in result.answer and '航海技术' not in result.answer


def test_credit_sum_cannot_be_replaced_by_record_count(tmp_path):
    with Repository(tmp_path/'sum_credits.db') as repo:
        repo.initialize()
        doc = repo.add_document(SourceDocument('课程方案','https://example.edu/courses',sha256='credits'))
        table = ParsedTable('课程学分',['课程','学分（学分）'],[['课程甲','3'],['课程乙','5']])
        tid = repo.add_table(doc,table,[[normalize_cell(v) for v in row] for row in table.rows])
        fid = repo.connection.execute('SELECT id FROM table_columns WHERE table_id=? ORDER BY column_index DESC',(tid,)).fetchone()[0]
        engine = StructuredQueryEngine(repo.connection)
        good = parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'total','op':'sum','field_id':fid}]}))
        result = engine.execute_steps(good,[tid],'所有课程合计多少学分？')
        assert result.ok and result.value==8 and result.unit=='学分'
        bad = parse_plan(json.dumps({'table_id':tid,'steps':[{'id':'total','op':'count'}]}))
        result = engine.execute_steps(bad,[tid],'所有课程合计多少学分？')
        assert not result.ok and result.repairable


def test_body_credits_keep_the_stated_measure():
    item = Evidence('a','课程共12学分。实验时长为24学时。',1,1,'课程方案','https://example.edu/courses')
    facts = list(NumericEngine.text_facts([item],academic_units=True).values())
    assert {(f['value'],f['unit']) for f in facts} == {('12','学分'),('24','学时')}
    assert not NumericEngine.text_facts([item])


def test_body_relevance_prevents_similar_title_from_hiding_answer(tmp_path):
    with Repository(tmp_path/'rules.db') as repo:
        repo.initialize()
        evidence = [Evidence('a', '星河学院设有测绘工程、海洋技术两个本科专业。', 1, 1,
                             '学院简介', 'https://example.edu/a'),
                    Evidence('b', '星河学院开设海洋技术微专业，共12学分。', .8, 2,
                             '星河学院海洋技术微专业招生简章', 'https://example.edu/b')]
        result = service(repo, tmp_path, evidence).answer('星河学院设有哪些本科专业？')
        assert '测绘工程' in result.answer and '微专业' not in result.answer


def test_named_rule_expands_to_matching_section_with_conditions(tmp_path):
    with Repository(tmp_path/'rules.db') as repo:
        repo.initialize()
        doc = repo.add_document(SourceDocument('设备借用规则', 'https://example.edu/rules', sha256='rules'))
        repo.add_chunks([
            Chunk('a', doc, '1. 借用\n\n凭工作证登记后借出设备。', '', 20),
            Chunk('b', doc, '2. 续借\n\n无人预约时可网上续借一次，过期设备不得续借。', '', 30)])
        item = Evidence('a', '1. 借用\n\n凭工作证登记后借出设备。', 1, doc,
                        '设备借用规则', 'https://example.edu/rules', chunk_id='a')
        result = service(repo, tmp_path, [item]).answer('设备借用规则中，如何续借设备？')
        assert '网上续借' in result.answer and '过期设备不得续借' in result.answer
        assert '工作证' not in result.answer


def test_faq_answer_keeps_all_steps_without_other_questions(tmp_path):
    text = ('问：实验室进不去怎么办？1\n提交《场地使用申请表》。2\n交实验中心审核。'
            '问：考试迟到怎么办？1\n联系监考老师。')
    with Repository(tmp_path/'faq.db') as repo:
        repo.initialize()
        item = Evidence('a', text, 1, 1, '场地常见问题', 'https://example.edu/faq')
        result = service(repo, tmp_path, [item]).answer('实验室进不去怎么办？')
        assert '场地使用申请表' in result.answer and '实验中心' in result.answer
        assert '监考' not in result.answer


def test_index_validation_checks_content_and_source_not_only_size(tmp_path):
    with Repository(tmp_path/'index.db') as repo:
        repo.initialize()
        doc = repo.add_document(SourceDocument('规则', 'https://example.edu/rules', sha256='index'))
        repo.add_chunks([Chunk('a', doc, '公开规则正文。', '', 8)])
        records = [dict(r) for r in repo.index_records()]
        validate_records(records, repo.connection)
        for field, value in [('text','被替换的正文'), ('doc_id',999), ('source_url','https://example.edu/other')]:
            with pytest.raises(ValueError, match='索引与数据库'):
                validate_records([dict(records[0], **{field:value})], repo.connection)


def test_oversized_prompt_is_rejected_before_generation():
    from types import SimpleNamespace
    generator = TransformersGenerator.__new__(TransformersGenerator)
    generator.max_context_tokens = 20
    generator.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs:
        {'input_ids': SimpleNamespace(shape=(1, 25))})
    with pytest.raises(ContextBudgetError):
        generator.generate('引用资料', max_new_tokens=10)


def test_output_reservation_cannot_exceed_model_positions():
    from types import SimpleNamespace
    generator = TransformersGenerator.__new__(TransformersGenerator)
    generator.max_context_tokens = 20
    generator.max_position_tokens = 20
    generator.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs:
        {'input_ids': SimpleNamespace(shape=(1,15))})
    with pytest.raises(ContextBudgetError):
        generator.generate('引用资料',max_new_tokens=10)


def test_context_budget_refusal_has_no_unverified_references(tmp_path):
    class Generator:
        def generate(self, *args, **kwargs): raise ContextBudgetError('预算不足')
    with Repository(tmp_path/'budget.db') as repo:
        repo.initialize()
        item = Evidence('a','提交登记材料后可申请使用场地。',1,1,'指南','https://example.edu/a')
        rag = service(repo,tmp_path,[item])
        rag.generator = Generator()
        result = rag.answer('使用场地的条件是什么？')
        assert '上下文预算' in result.answer and not result.references
