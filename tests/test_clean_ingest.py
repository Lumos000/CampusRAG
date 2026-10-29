import json
from dataclasses import replace
from pathlib import Path

import pytest

from campus_rag.config import load_settings
from campus_rag.database import Repository
from campus_rag.ingest.chunker import CharacterTokenizer
from campus_rag.ingest.cleaning import content_key, prepare_manifest
from campus_rag.ingest.pipeline import ingest_document
from campus_rag.types import ParsedDocument, ParsedTable, SourceDocument


def test_clean_html_excludes_navigation_and_merges_aliases(tmp_path):
    import hashlib
    content = '<nav>菜单噪声</nav><div class="wp_articlecontent"><p>' + '学校办事规定。' * 20 + '</p></div>'
    path = tmp_path / 'a.html'
    path.write_text(content, encoding='utf-8')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records = [{'status': 'ok', 'url': f'https://jwc.njupt.edu.cn/{name}.htm',
        'local_path': str(path), 'sha256': digest, 'title': '2026规定'} for name in ['page', 'pagem']]
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    report = prepare_manifest(manifest, tmp_path / 'clean')
    assert report['accepted_unique_documents'] == 1
    assert report['decisions']['duplicate'] == 1
    result = json.loads((tmp_path / 'clean/clean_documents.jsonl').read_text(encoding='utf-8'))
    assert len(result['sources']) == 2
    assert '菜单噪声' not in str(result['document']['sections'])


def test_idempotency_table_limit_and_transaction_rollback(tmp_path, monkeypatch):
    import campus_rag.ingest.pipeline as pipeline
    monkeypatch.setattr(pipeline, '_tokenizer', lambda _: (CharacterTokenizer.encode, CharacterTokenizer.decode))
    settings = replace(load_settings(), database_path=tmp_path / 'campus.db')
    parsed = ParsedDocument(SourceDocument('2026测试', 'https://jwc.njupt.edu.cn/a', sha256='abc'),
        [('正文', '办事规定' * 40)], [ParsedTable('测试表', ['项目', '人数'], [[f'项目{i}', str(i)] for i in range(100)])])
    first = ingest_document(parsed, settings)
    with Repository(settings.database_path) as repo:
        before = [repo.connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in ['documents','tables','table_cells','chunks']]
        assert repo.connection.execute('SELECT max(token_count) FROM chunks').fetchone()[0] <= 480
        assert repo.connection.execute('SELECT count(*) FROM table_rows').fetchone()[0] == 100
    assert ingest_document(parsed, settings) == first
    with Repository(settings.database_path) as repo:
        after = [repo.connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in ['documents','tables','table_cells','chunks']]
    assert before == after
    broken = ParsedDocument(SourceDocument('2025另一个', 'https://jwc.njupt.edu.cn/b', sha256='def'),
        [('正文', '另外的资料')], parsed.tables)
    def fail(*_):
        raise RuntimeError('simulated failure')
    monkeypatch.setattr(Repository, 'add_table', fail)
    with pytest.raises(RuntimeError):
        ingest_document(broken, settings)
    with Repository(settings.database_path) as repo:
        assert repo.connection.execute('SELECT count(*) FROM documents').fetchone()[0] == before[0]
        assert repo.connection.execute('PRAGMA foreign_key_check').fetchall() == []


def test_same_body_different_year_not_merged():
    a = ParsedDocument(SourceDocument('2025规定', 'a'), [('正文','一样的条款')])
    b = ParsedDocument(SourceDocument('2026规定', 'b'), [('正文','一样的条款')])
    assert content_key(a) != content_key(b)


def test_scoped_college_cleaning_and_coverage(tmp_path):
    import hashlib
    from campus_rag.config import ROOT
    from campus_rag.crawl.coverage import coverage_report
    from campus_rag.ingest.cleaning import prepare_item

    config = json.loads((ROOT / 'config/sites_topics.json').read_text(encoding='utf-8'))
    path = tmp_path / 'college.html'
    path.write_text('<div class="wp_articlecontent">' + '学院本科培养计划。' * 20 + '</div>', encoding='utf-8')
    record = {'status': 'ok', 'url': 'https://scie.njupt.edu.cn/a.htm',
              'title': '本科培养计划', 'local_path': str(path),
              'published_at': '2026-08-01',
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert prepare_item(record)[1] == 'extended_site_scope_check'
    assert prepare_item(record, config)[0] == 'accept'
    assert prepare_item(dict(record, title='科研项目申报通知'), config)[1] == 'topic_scope_check'
    assert prepare_item(dict(record, title='研究生专业培养方案'), config)[1] == 'excluded_title_scope_check'
    assert prepare_item(dict(record, title='学生资助政策宣讲会'), config)[1] == 'news_or_activity_scope_check'
    assert prepare_item(dict(record, final_url='https://pg.njupt.edu.cn/a.htm'), config)[1] == 'extended_site_scope_check'
    manifest = tmp_path / 'manifest.jsonl'
    alias = dict(record, url='https://jwc.njupt.edu.cn/b.htm', published_at='2026-09-01')
    manifest.write_text('\n'.join(json.dumps(r) for r in [alias, record]), encoding='utf-8')
    prepare_manifest(manifest, tmp_path / 'clean', scope=config)
    report = coverage_report(tmp_path / 'clean', config)
    assert report['accepted_unique_documents'] == 1
    assert report['colleges']['scie.njupt.edu.cn']['coverage']['本科培养']['accepted_documents'] == 1
    assert report['colleges']['scie.njupt.edu.cn']['coverage']['本科培养']['examples'][0]['published_at'] == '2026-08-01'
    assert report['topics']['招生']['招生计划']['status'] == 'missing'


def test_image_only_article_is_reviewed_for_ocr(tmp_path):
    import hashlib
    from campus_rag.ingest.cleaning import prepare_item

    path = tmp_path / 'image.html'
    path.write_text('<div class="wp_articlecontent"><img src="intro.jpg"><iframe src="blank.htm"></iframe></div>', encoding='utf-8')
    record = {'url': 'https://jwc.njupt.edu.cn/a.htm', 'title': '本科专业介绍',
              'local_path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert prepare_item(record)[:2] == ('review', 'image_content_requires_ocr')


def test_collection_refuses_existing_output(tmp_path):
    import subprocess
    import sys
    from campus_rag.config import ROOT

    marker = tmp_path / 'keep.txt'
    marker.write_text('original', encoding='utf-8')
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/collect_topics.py'),
                             '--skip-crawl', '--output', str(tmp_path)], capture_output=True)
    assert result.returncode != 0
    assert marker.read_text(encoding='utf-8') == 'original'
    assert not (tmp_path / 'sites.json').exists()
