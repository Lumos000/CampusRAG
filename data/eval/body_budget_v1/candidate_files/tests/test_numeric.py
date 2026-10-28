from decimal import Decimal

import pytest

from campus_rag.database import Repository
from campus_rag.rag.numeric import NumericEngine, check_draft, eligible
from campus_rag.tools.numbers import normalize_cell
from campus_rag.types import Evidence, ParsedTable, SourceDocument


def evidence(text):
    return Evidence('chunk-a',text,1,1,'2026章程','https://example.edu/charter',chunk_id='chunk-a')


def test_fee_values_are_extracted_and_composed_not_constants():
    engine = NumericEngine(None)
    source = evidence('40\n工科类专业每生每年5800元。江苏高校优势学科专业在上述标准基础上上浮10%。住宿费暂按1500元/生·学年预收。')
    result = engine.fees('工科类优势学科专业学费加住宿费，假设标准四年不变，预收额合计多少？',[source])
    assert result.result.value == Decimal('31520')
    assert '6380' in result.canonical
    assert result.result.evidence == [source]
    assert all(f['quote'] and f['normalized_span'] for f in result.facts)
    changed = evidence(source.text.replace('5800','6000').replace('10%','5%').replace('1500','1200'))
    assert engine.fees('工科类优势学科专业学费加住宿费，三年合计多少？',[changed]).result.value == Decimal('22500')


def test_staff_ignores_page_number_and_verifies_derived_values():
    engine = NumericEngine(None)
    source = evidence('40\n学校现有专任教师2318人，外聘教师864人，折合教师总数为2750，折合在校生人数40863.2。')
    count = engine.staff('专任教师和外聘教师分别多少人？',[source])
    assert '2318' in count.canonical and '864' in count.canonical
    assert all(f['value'] != '40' for f in count.facts)
    derived = engine.staff('反推外聘折算系数，再复算生师比',[source])
    assert derived.result.value == Decimal('14.86')
    assert '0.5' in derived.canonical
    altered = evidence('专任教师2000人，外聘教师1000人，折合教师总数为2500，折合在校生人数40000。')
    assert engine.staff('反推折算系数和生师比',[altered]).result.value == Decimal('16.00')


def test_conflicts_missing_facts_and_zero_denominator_refuse():
    engine = NumericEngine(None)
    with pytest.raises(ValueError,match='冲突'):
        engine.staff('专任教师多少人？',[evidence('专任教师10人。专任教师20人。')])
    with pytest.raises(ValueError,match='缺少'):
        engine.staff('专任教师多少人？',[evidence('40\n专任教师队伍结构合理。')])
    with pytest.raises(ValueError,match='计算条件'):
        engine.staff('生师比',[evidence('专任教师20人，外聘教师0人，折合教师总数为20，折合在校生人数100。')])
    with pytest.raises(ValueError,match='缺少'):
        engine.staff('生师比',[evidence('专任教师20人，外聘教师10人，折合教师总数为25，折合在校生人数4.08万人。')])


def test_duration_not_silently_truncated():
    source = evidence('工科类专业每生每年6000元。住宿费暂按1000元/生·学年预收。')
    with pytest.raises(ValueError,match='年限'):
        NumericEngine(None).fees('工科类学费加住宿费三十年合计多少？',[source])


def test_checked_answer_rejects_number_changes_label_swaps_and_new_claims():
    canonical = '专任教师为2318人；外聘教师为864人。'
    for draft in ['专任教师为40人；外聘教师为864人。','专任教师为864人；外聘教师为2318人。',canonical+'共计9999人。']:
        answer, audit = check_draft(draft,canonical)
        assert answer == canonical and audit['used_template']
    assert not check_draft(canonical,canonical)[1]['used_template']


def test_subject_logic_and_ambiguity():
    selected = {'政治','地理'}
    assert eligible('不限',selected)
    assert eligible('政治或化学',selected)
    assert eligible('政治和地理',selected)
    assert not eligible('化学',selected)
    assert not eligible('政治和化学',selected)
    with pytest.raises(ValueError):
        eligible('政治、化学',selected)


def test_admission_filters_complete_table_and_excludes_total(tmp_path):
    with Repository(tmp_path/'test.db') as repo:
        repo.initialize()
        doc_id = repo.add_document(SourceDocument('2026综合评价','https://example.edu/a',sha256='abc'))
        table = ParsedTable('计划',['首选科目','再选科目','专业名称','计划'],[
            ['物理','不限','专业甲','11'],['物理','不限','专业乙','17'],
            ['物理','化学','专业丙','20'],['历史','不限','专业丁','30'],['合计','合计','合计','78']])
        tid = repo.add_table(doc_id,table,[[normalize_cell(v) for v in row] for row in table.rows])
        source = Evidence('t','表',1,doc_id,'2026综合评价','https://example.edu/a',table_id=tid)
        result = NumericEngine(repo.connection).admission('首选物理，再选政治和地理，计划合计多少？',[source])
        assert result.result.value == Decimal(28)
        assert len(result.facts) == 2
        assert len(result.result.evidence[0].row_ids) == 2
        assert all(f['cell_id'] for f in result.facts)


def test_ambiguous_multiple_tables_do_not_double_count(tmp_path):
    with Repository(tmp_path/'test.db') as repo:
        repo.initialize()
        doc_id = repo.add_document(SourceDocument('2026综合评价','u',sha256='a'))
        table = ParsedTable('计划',['首选科目','再选科目','专业名称','计划'],[['物理','不限','专业甲','10']])
        tids = [repo.add_table(doc_id,table,[[normalize_cell(v) for v in row] for row in table.rows]) for _ in range(2)]
        sources = [Evidence(str(t),'表',1,doc_id,'2026综合评价','u',table_id=t) for t in tids]
        with pytest.raises(ValueError,match='多张'):
            NumericEngine(repo.connection).admission('首选物理，再选政治和地理，计划多少？',sources)
