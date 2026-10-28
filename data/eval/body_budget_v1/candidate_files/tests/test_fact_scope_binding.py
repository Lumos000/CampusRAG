import json

import pytest

from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.types import Evidence


def evidence(text):
    return Evidence('policy',text,1,1,'公开资助办法','https://example.edu/policy')


@pytest.mark.parametrize('counter',['人','名'])
@pytest.mark.parametrize('separator',['；','\n'])
def test_person_rows_keep_subject_metric_and_period(counter,separator):
    text=f'三、资助标准\n技术组：4{counter}，每学年补助标准为137元/人{separator}研习组：6{counter}，每学年补助标准为89元／人。'
    facts=list(NumericEngine.text_facts([evidence(text)]).values())
    assert {(f['subject'],f['value'],f['unit']) for f in facts}=={
        ('技术组','4','人'),('技术组','137','元/人'),('研习组','6','人'),('研习组','89','元/人')}
    for f in facts:
        assert text[slice(*f['span'])]==f['quote'] and f['subject'] in f['quote']
        assert f['scope']['period']==('每学年' if f['unit']=='元/人' else '')


def test_group_titles_separate_identical_subjects_without_hiding_conflicts():
    text='（一）远山补助：\n技术组：4名，137元/人。\n（二）清溪补助：\n技术组：6名，89元/人。'
    facts=list(NumericEngine.text_facts([evidence(text)]).values())
    assert len(facts)==4
    assert {(f['scope']['group'],f['value']) for f in facts}=={
        ('远山补助','4'),('远山补助','137'),('清溪补助','6'),('清溪补助','89')}
    conflict=NumericEngine.text_facts([evidence('（一）远山补助：\n技术组：4名，137元/人；技术组：4名，139元/人。')])
    assert {(f['value'],f['unit']) for f in conflict.values()}=={('4','人')}


def test_group_scope_is_validated_against_explicit_question(tmp_path):
    facts=NumericEngine.text_facts([evidence('（一）远山补助：\n技术组：4名，137元/人。\n（二）清溪补助：\n技术组：6名，89元/人。')])
    with Repository(tmp_path/'groups.db') as repo:
        repo.initialize();engine=StructuredQueryEngine(repo.connection)
        for group,expected in [('远山补助',True),('清溪补助',False)]:
            chosen={f['unit']:fid for fid,f in facts.items() if f['scope'].get('group')==group}
            assert set(chosen)=={'人','元/人'}
            steps=[{'id':sid,'op':'lookup','fact_id':chosen[u]} for sid,u in [('people','人'),('rate','元/人')]]
            steps.append({'id':'result','op':'multiply','inputs':['people','rate']})
            result=engine.execute_steps(parse_plan(json.dumps({'steps':steps})),[],
                '按《公开资助办法》，远山补助中技术组按人数和标准计算，预算合计多少元？',facts)
            assert result.ok==expected
            if expected:assert result.value==548 and result.unit=='元'


def test_higher_heading_ends_group_scope():
    facts=NumericEngine.text_facts([evidence('（一）远山补助：\n技术组：4名，137元/人。\n四、申请流程\n处理时长为9天。')])
    duration=next(f for f in facts.values() if f['unit']=='天')
    assert not duration['scope'].get('group')


@pytest.mark.parametrize('second_rate',[137,89])
def test_unspecified_group_cannot_pick_a_scope_even_when_values_equal(tmp_path,second_rate):
    facts=NumericEngine.text_facts([evidence('（一）远山补助：\n技术组：4名，137元/人。\n'
        f'（二）清溪补助：\n技术组：6名，{second_rate}元/人。')])
    chosen={f['unit']:fid for fid,f in facts.items() if f['scope'].get('group')=='远山补助'}
    steps=[{'id':sid,'op':'lookup','fact_id':chosen[u]} for sid,u in [('people','人'),('rate','元/人')]]
    steps.append({'id':'result','op':'multiply','inputs':['people','rate']})
    with Repository(tmp_path/'ambiguous.db') as repo:
        repo.initialize()
        result=StructuredQueryEngine(repo.connection).execute_steps(parse_plan(json.dumps({'steps':steps})),[],
            '按《公开资助办法》，技术组按名额和人均标准计算，预算合计多少元？',facts)
    assert not result.ok and not result.repairable
    assert '口径不明确' in result.error


@pytest.mark.parametrize('text',[
    '技术组：4名；每学年标准为137元/人。',
    '技术组：4名，另一组标准为137元/人。',
    '技术组：不超过4名，每学年标准为137元/人。'])
def test_expanded_pairing_does_not_guess_a_subject(text):
    facts=list(NumericEngine.text_facts([evidence(text)]).values())
    assert not any(f['subject']=='技术组' and f['unit']=='元/人' for f in facts)
