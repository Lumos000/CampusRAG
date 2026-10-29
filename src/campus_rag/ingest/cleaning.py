"""Conservative cleaning: retain raw snapshots and explain every decision."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from campus_rag.crawl.crawler import is_error_page
from campus_rag.ingest.parsers import parse_file
from campus_rag.types import ParsedDocument, ParsedTable, SourceDocument


SUPPORTED = {'.html', '.htm', '.pdf', '.docx', '.xlsx', '.xlsm', '.txt', '.md'}
CORE_HOSTS = {'zs.njupt.edu.cn', 'jwc.njupt.edu.cn', 'xsc.njupt.edu.cn',
              'lib.njupt.edu.cn', 'cwc.njupt.edu.cn', 'www.njupt.edu.cn'}


def normalize_text(text):
    # Preserve digits, dates, units, punctuation and logical paragraph boundaries.
    text = text.replace('\u200b', '').replace('\ufeff', '').replace('\xa0', ' ')
    return '\n'.join(re.sub(r'[ \t\r]+', ' ', line).strip() for line in text.split('\n')).strip()


def content_key(parsed):
    data = {
        'sections': [normalize_text(text) for _, text in parsed.sections],
        'tables': [{'headers': t.headers, 'rows': t.rows, 'unit': t.unit, 'notes': t.notes} for t in parsed.tables],
        # Do not merge identical short notices belonging to different years.
        'title_years': re.findall(r'20\d{2}', parsed.source.title),
    }
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def prepare_item(item, scope=None):
    path = Path(item['local_path'])
    if path.suffix.lower() in {'.doc', '.xls'}:
        return 'review', 'legacy_office_requires_conversion', None
    if path.suffix.lower() not in SUPPORTED:
        return 'exclude', 'unsupported_format', None
    trusted_hosts = set(scope['departments']) if scope else CORE_HOSTS
    if any(urlparse(url).hostname not in trusted_hosts for url in (item['url'], item.get('final_url') or item['url'])):
        return 'review', 'extended_site_scope_check', None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item.get('sha256'):
        return 'review', 'hash_mismatch', None
    if is_error_page(item.get('title', '')):
        return 'exclude', 'error_page', None
    title = item.get('title', '')
    if scope and '本科' not in title and any(word in title for word in scope.get('excluded_title_keywords', [])):
        return 'review', 'excluded_title_scope_check', None
    if scope and not any(word in title for groups in scope['coverage'].values() for words in groups.values() for word in words):
        return 'review', 'topic_scope_check', None
    if scope and any(word in title for word in scope.get('news_title_keywords', [])):
        return 'review', 'news_or_activity_scope_check', None
    if re.search(r'党日|党委|支部|来我校|来校|调研交流|召开|举办|喜报|获奖|纪念馆|读书班', title) and not re.search(r'办法|规定|指南|流程|章程|实施细则|收费标准', title):
        return 'review', 'news_or_activity_scope_check', None
    if path.suffix.lower() in {'.html', '.htm'}:
        soup = BeautifulSoup(raw, 'lxml')
        article = soup.select_one('.wp_articlecontent, .Article_Content, .article-content, article')
        if article is None:
            return 'review', 'no_reliable_article_selector', None
        if not article.get_text(' ', strip=True) and article.select('img'):
            return 'review', 'image_content_requires_ocr', None
        if not article.get_text(' ', strip=True) and article.select('[pdfsrc], iframe, embed, object'):
            return 'exclude', 'attachment_container', None
    parsed = parse_file(path, item.get('final_url') or item['url'], item.get('department', ''), item.get('published_at'))
    parsed.source.title = item.get('title') or parsed.source.title
    parsed.source.crawled_at = item.get('crawled_at')
    parsed.sections = [(normalize_text(h), normalize_text(t)) for h, t in parsed.sections if normalize_text(t)]
    for table in parsed.tables:
        table.headers = [normalize_text(h) for h in table.headers]
        table.rows = [[normalize_text(v) for v in row] for row in table.rows]
    text_chars = sum(len(t) for _, t in parsed.sections)
    if not text_chars and not any(t.rows for t in parsed.tables):
        return 'review', 'empty_or_scanned_document', parsed
    if parsed.warnings:
        return 'review', 'parser_warnings_check_required', parsed
    if not parsed.tables and text_chars < 60:
        return 'review', 'short_content', parsed
    return 'accept', 'parsed_with_provenance', parsed


def prepare_manifest(manifest, output, limit=None, scope=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    latest = {}
    for line in Path(manifest).read_text(encoding='utf-8').splitlines():
        if line.strip():
            item = json.loads(line)
            latest[item['url']] = item
    # Prefer dated desktop pages as canonical sources; keep every alias.
    ordered = sorted(latest.values(), key=lambda r: (not bool(r.get('published_at')),
        '/pagem.' in r['url'], not r['url'].startswith('https:'), r['url']))
    groups, decisions = {}, []
    examined = 0
    for item in ordered:
        if item.get('status') != 'ok':
            decisions.append(dict(url=item['url'], decision='exclude', reason='crawl_' + item.get('status', 'unknown')))
            continue
        if limit is not None and examined >= limit:
            break
        examined += 1
        try:
            decision, reason, parsed = prepare_item(item, scope)
        except Exception as exc:
            decision, reason, parsed = 'review', 'parse_failed_' + type(exc).__name__, None
        record = {'url': item['url'], 'title': item.get('title'), 'local_path': item.get('local_path'),
                  'decision': decision, 'reason': reason}
        if parsed:
            record.update(text_chars=sum(len(t) for _, t in parsed.sections), tables=len(parsed.tables), warnings=parsed.warnings)
        if decision == 'accept':
            key = content_key(parsed)
            source = {name: item.get(name) for name in ('url', 'final_url', 'title', 'department', 'published_at', 'crawled_at', 'parent_url', 'local_path', 'sha256')}
            if key in groups:
                groups[key]['sources'].append(source)
                record.update(decision='duplicate', reason='identical_clean_content')
            else:
                groups[key] = {'content_key': key, 'document': asdict(parsed), 'sources': [source]}
            record['content_key'] = key
        decisions.append(record)
        if examined % 25 == 0:
            print(f'[CLEAN] examined={examined} accepted_unique={len(groups)}', flush=True)
    for filename, records in [('clean_documents.jsonl', groups.values()), ('clean_decisions.jsonl', decisions)]:
        temporary = output / (filename + '.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        temporary.replace(output / filename)
    from collections import Counter
    report = {'examined_successful_urls': examined, 'partial_run': limit is not None,
        'accepted_unique_documents': len(groups),
        'decisions': dict(Counter(r['decision'] for r in decisions)),
        'reasons': dict(Counter(r['reason'] for r in decisions)),
        'missing_date_accepted': sum(not g['document']['source']['published_at'] for g in groups.values())}
    (output / 'clean_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def restore_document(data):
    return ParsedDocument(SourceDocument(**data['source']), [tuple(v) for v in data['sections']],
                          [ParsedTable(**t) for t in data['tables']], data['warnings'])
