import importlib
import json
from dataclasses import asdict, replace

import pytest

from campus_rag.config import ROOT, load_settings
from campus_rag.ingest.chunker import CharacterTokenizer
from campus_rag.ingest.cleaning import content_key
from campus_rag.ingest.pipeline import ingest_document
from campus_rag.types import ParsedDocument, SourceDocument


def test_rebuild_retains_baseline_and_protects_existing_files(tmp_path, monkeypatch):
    import campus_rag.ingest.pipeline as pipeline
    monkeypatch.setattr(pipeline, '_tokenizer', lambda _: (CharacterTokenizer.encode, CharacterTokenizer.decode))
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('rebuild_college_database')
    baseline_db = tmp_path / 'original.db'
    base = ParsedDocument(SourceDocument('学校简介', 'https://www.njupt.edu.cn/a', sha256='base'),
                          [('正文', '学校公开介绍。' * 30)])
    added = ParsedDocument(SourceDocument('学院简介', 'https://ai.njupt.edu.cn/a', sha256='added'),
                           [('正文', '人工智能学院公开介绍。' * 30)])
    ingest_document(base, replace(load_settings(), database_path=baseline_db))

    def entry(doc):
        return {'content_key': content_key(doc), 'document': asdict(doc),
                'sources': [{'url': doc.source.source_url, 'title': doc.source.title}]}

    baseline = tmp_path / 'base.jsonl'
    baseline.write_text(json.dumps(entry(base)), encoding='utf-8')
    additional = tmp_path / 'additional.jsonl'
    reparsed = ParsedDocument(base.source, [('正文', '同一原始文件在新版解析器中的不同文本表示。' * 10)])
    additional.write_text('\n'.join(json.dumps(entry(d)) for d in [base, reparsed, added]), encoding='utf-8')
    before = module.digest(baseline_db)
    report = module.rebuild(baseline_db, baseline, additional, tmp_path / 'new', database_name='campus_public.db')
    assert (tmp_path / 'new/campus_public.db').is_file()
    assert report['counts']['documents'] == 2
    assert report['added_unique_documents'] == 1
    assert len(report['same_raw_parser_variants']) == 1
    assert report['baseline_unchanged'] and all(report['checks'].values())
    assert module.digest(baseline_db) == before
    assert report['source_hosts']['ai.njupt.edu.cn'] == 1
    with pytest.raises(ValueError, match='预期文档数'):
        module.rebuild(baseline_db, baseline, additional, tmp_path / 'wrong_count', expected_documents=518)
    assert not (tmp_path / 'wrong_count').exists()
    with pytest.raises(ValueError, match='必需学院'):
        module.rebuild(baseline_db, baseline, additional, tmp_path / 'missing_college',
                       required_hosts=['cs.njupt.edu.cn', 'ai.njupt.edu.cn'])
    assert not (tmp_path / 'missing_college').exists()
    excluded_report = module.rebuild(baseline_db, baseline, additional, tmp_path / 'excluded',
                                     excluded_urls=[added.source.source_url])
    assert excluded_report['counts']['documents'] == 1
    assert len(excluded_report['excluded_documents']) == 1
    assert excluded_report['checks']['baseline_retained']
    with pytest.raises(ValueError, match='不能排除原库'):
        module.rebuild(baseline_db, baseline, additional, tmp_path / 'protected',
                       excluded_urls=[base.source.source_url])
    assert not (tmp_path / 'protected').exists()
    with pytest.raises(ValueError, match='未匹配'):
        module.rebuild(baseline_db, baseline, additional, tmp_path / 'unmatched',
                       excluded_urls=['https://ai.njupt.edu.cn/missing'])
    assert not (tmp_path / 'unmatched').exists()
    with pytest.raises(FileExistsError):
        module.rebuild(baseline_db, baseline, additional, tmp_path / 'new')
    baseline.write_text(json.dumps(entry(added)), encoding='utf-8')
    with pytest.raises(ValueError, match='原库不一致'):
        module.rebuild(baseline_db, baseline, additional, tmp_path / 'mismatch')
    assert not (tmp_path / 'mismatch').exists()


def test_comparison_keeps_original_and_college_metrics_separate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('compare_college_retrieval')
    rows = [{'group': group, 'variant': variant, 'mode': 'bm25', 'rank': rank}
            for group, variant, rank in [('original_topics', 'baseline', 1),
                                         ('original_topics', 'candidate', 6),
                                         ('college_new', 'baseline', None),
                                         ('college_new', 'candidate', 2)]]
    metrics = module.summarize(rows)
    assert metrics['college_new']['baseline']['bm25']['source_hit_at_5'] == 0
    assert metrics['college_new']['candidate']['bm25']['source_hit_at_5'] == 1
    assert metrics['original_topics']['candidate']['bm25']['source_hit_at_5'] == 0
    assert metrics['original_topics']['candidate']['bm25']['source_hit_at_10'] == 1
    assert metrics['all']['candidate']['bm25']['cases'] == 2


def test_comparison_rejects_duplicate_ids_and_wrong_source_metadata(tmp_path, monkeypatch):
    import numpy as np
    import campus_rag.ingest.pipeline as pipeline
    from campus_rag.database import Repository

    monkeypatch.setattr(pipeline, '_tokenizer', lambda _: (CharacterTokenizer.encode, CharacterTokenizer.decode))
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('compare_college_retrieval')
    settings = replace(load_settings(), database_path=tmp_path / 'test.db', index_dir=tmp_path / 'index')
    for i in range(2):
        parsed = ParsedDocument(SourceDocument(f'学院介绍{i}', f'https://ai.njupt.edu.cn/{i}', sha256=str(i)),
                                [('正文', f'学院专业{i}。' * 30)])
        ingest_document(parsed, settings)
    with Repository(settings.database_path) as repository:
        records = [dict(row) for row in repository.index_records()]
    settings.index_dir.mkdir()
    np.save(settings.index_dir / 'vectors.npy', np.zeros((len(records), 512)))

    def write_records(rows):
        (settings.index_dir / 'records.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in rows), encoding='utf-8')

    write_records(records)
    assert len(module.load_profile(settings, None, None)[1]) == 2
    write_records([records[0]] * len(records))
    with pytest.raises(ValueError, match='索引与数据库'):
        module.load_profile(settings, None, None)
    write_records([dict(row, doc_id=999) for row in records])
    with pytest.raises(ValueError, match='索引与数据库'):
        module.load_profile(settings, None, None)




def test_public_corpus_recipe_keeps_the_original_baseline(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('rebuild_college_database')
    captured = []
    monkeypatch.setattr(module, 'rebuild', lambda *args: captured.append(args) or {})
    monkeypatch.setattr(sys, 'argv', ['rebuild_college_database.py', '--recipe',
                                    str(ROOT / 'config/dataset_campus_public.json'), '--build-index'])
    module.main()
    args, = captured
    assert args[0] == ROOT / 'data/processed/campus.db'
    assert args[3] == load_settings().database_path.parent
    assert args[6] == 518 and args[8] == 'campus_public.db'


def test_comparison_rejects_comparing_the_same_database(monkeypatch, tmp_path):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('compare_college_retrieval')
    profile = replace(load_settings(),database_path=tmp_path/'same.db',index_dir=tmp_path/'index')
    monkeypatch.setattr(module,'load_settings',lambda _:profile)
    monkeypatch.setattr(sys,'argv',['compare_college_retrieval.py','--baseline-config','a.json',
        '--candidate-config','b.json','--cases','cases.json','--output',str(tmp_path/'results')])
    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code==2 and not (tmp_path/'results').exists()
