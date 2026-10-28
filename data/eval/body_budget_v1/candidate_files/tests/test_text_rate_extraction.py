import json
from decimal import Decimal

import pytest

from campus_rag.database import Repository
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.rag.numeric import NumericEngine
from campus_rag.rag.planner import parse_plan
from campus_rag.types import Evidence


def source(text, key='rates'):
    return Evidence(key,text,1,1,'合成补助办法','https://example.edu/rates')


@pytest.mark.parametrize('slash',['/','／'])
@pytest.mark.parametrize('denominator',['人','小时','台'])
def test_rate_slash_preserves_unit_and_original_span(slash,denominator):
    item=source(f'技术组标准为 37 元 {slash} {denominator}。')
    fact,=NumericEngine.text_facts([item]).values()
    assert fact['unit']=='元/'+denominator
    assert fact['value']=='37' and fact['comparison']=='eq'
    assert fact['quote']==item.text[slice(*fact['span'])]
    assert slash in fact['quote'] and denominator in fact['raw_value']
    assert fact['evidence'] is item and fact['source_url']==item.source_url


@pytest.mark.parametrize('slash',['/','／'])
@pytest.mark.parametrize('comma',[',','，\n'])
def test_adjacent_headcount_and_rate_bind_to_the_same_subject(slash,comma):
    item=source(f'技术组：3人{comma}37元{slash}人；研习组：7人{comma}53元{slash}人。')
    facts=list(NumericEngine.text_facts([item]).values())
    assert {(f['subject'],f['value'],f['unit']) for f in facts}=={
        ('技术组','3','人'),('技术组','37','元/人'),('研习组','7','人'),('研习组','53','元/人')}
    for f in facts:
        assert f['quote']==item.text[slice(*f['span'])]
        assert f['subject'] in f['quote']


def test_paired_rates_execute_from_verified_facts(tmp_path):
    item=source('技术组：3人，37元／人；研习组：7人，53元/人。')
    facts=NumericEngine.text_facts([item]);steps=[]
    for name,sid in [('技术组','first'),('研习组','second')]:
        for kind,unit in [('count','人'),('rate','元/人')]:
            fid=next(fid for fid,f in facts.items() if f['subject']==name and f['unit']==unit)
            steps.append({'id':sid+'_'+kind,'op':'lookup','fact_id':fid})
        steps.append({'id':sid,'op':'multiply','inputs':[sid+'_count',sid+'_rate']})
    steps.append({'id':'result','op':'add','inputs':['first','second']})
    with Repository(tmp_path/'rates.db') as repo:
        repo.initialize()
        result=StructuredQueryEngine(repo.connection).execute_steps(parse_plan(json.dumps({'steps':steps})),[],
            '按《合成补助办法》，技术组和研习组按名额及人均标准计算，预算合计多少元？',facts)
    assert result.ok and result.value==Decimal('482') and result.unit=='元'
    assert result.evidence==[item]
    assert {d['fact_id'] for d in result.details}==set(facts)


@pytest.mark.parametrize('text',[
    '技术组：3人；37元/人。', '技术组：3人。37元/人。',
    '3人，37元/人。', '技术组：不超过3人，37元/人。',
    '技术组：3至7人，37元/人。', '技术组：约3人，37元/人。',
    '技术组：3人，约37元/人。', '技术组：3人，37元。'])
def test_ambiguous_or_missing_pair_does_not_infer_a_rate(text):
    assert not any(f['unit']=='元/人' for f in NumericEngine.text_facts([source(text)]).values())


@pytest.mark.parametrize('tail',['左右','至53元/人'])
def test_pair_does_not_turn_post_value_uncertainty_into_a_fixed_rate(tail):
    facts=NumericEngine.text_facts([source('技术组：3人，37元/人'+tail+'。')])
    assert not any(f['unit']=='元/人' for f in facts.values())


def test_subject_does_not_leak_across_evidence_or_override_explicit_subject():
    facts=NumericEngine.text_facts([source('技术组：3人，','first'),source('37元/人。','second')])
    assert not any(f['unit']=='元/人' for f in facts.values())
    facts=NumericEngine.text_facts([source('技术组：3人，另一组标准为37元/人。')])
    rate,=[f for f in facts.values() if f['unit']=='元/人']
    assert rate['subject']=='另一组标准'


def test_conflicting_paired_rates_are_not_silently_selected():
    facts=NumericEngine.text_facts([source('技术组：3人，37元/人；技术组：3人，41元/人。')])
    assert {(f['value'],f['unit']) for f in facts.values()}=={('3','人')}
