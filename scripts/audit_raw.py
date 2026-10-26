"""Inspect raw snapshots without loading any language model or modifying sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401
from bs4 import BeautifulSoup

from campus_rag.config import ROOT
from campus_rag.crawl.crawler import is_error_page


def audit(directory: Path) -> dict:
    latest = {}
    for line in (directory / 'manifest.jsonl').read_text(encoding='utf-8').splitlines():
        if line.strip():
            item = json.loads(line)
            latest[item['url']] = item
    entries = []
    for item in latest.values():
        row = {key: item.get(key) for key in ['url', 'title', 'status', 'department', 'published_at', 'parent_url', 'local_path']}
        issues = []
        row['usable'] = item.get('status') == 'ok'
        if row['usable']:
            path = Path(item['local_path'])
            try:
                content = path.read_bytes()
                row['bytes'] = len(content)
                if hashlib.sha256(content).hexdigest() != item['sha256']:
                    issues.append('hash_mismatch')
                if path.suffix in {'.html', '.htm'}:
                    soup = BeautifulSoup(content, 'lxml')
                    if is_error_page(item.get('title', '')):
                        issues.append('error_page')
                    body = soup.select_one('.wp_articlecontent, .Article_Content, .article-content')
                    row['page_kind'] = 'article' if body else 'navigation_or_other'
                    row['text_chars'] = len((body or soup).get_text(' ', strip=True))
                elif path.suffix == '.pdf':
                    import fitz
                    with fitz.open(path) as pdf:
                        row['pages'] = len(pdf)
                        row['text_chars'] = sum(len(page.get_text()) for page in pdf)
                        if not row['text_chars']:
                            issues.append('needs_ocr')
                elif path.suffix in {'.docx', '.xlsx', '.xlsm'}:
                    if not zipfile.is_zipfile(path):
                        issues.append('invalid_office_file')
                elif path.suffix in {'.doc', '.xls'}:
                    issues.append('unsupported_legacy_office')
            except Exception as exc:
                issues.append('unreadable_' + type(exc).__name__)
            if not item.get('published_at'):
                issues.append('missing_date')
            row['usable'] = not any(issue not in {'missing_date', 'needs_ocr'} for issue in issues)
        row['issues'] = issues
        entries.append(row)
    return {'unique_urls': len(entries), 'statuses': dict(Counter(row['status'] for row in entries)),
        'usable_snapshots': sum(row['usable'] for row in entries),
        'departments': dict(Counter(row['department'] for row in entries if row['usable'])),
        'issues': dict(Counter(issue for row in entries for issue in row['issues'])),
        'entries': entries}


def main():
    parser = argparse.ArgumentParser(description='检查采集文件、错误页、哈希、日期及附件可读性')
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'data/raw')
    args = parser.parse_args()
    report = audit(args.raw_dir)
    destination = args.raw_dir / 'audit_report.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'entries'}, ensure_ascii=False, indent=2))
    print(f'Report: {destination}')


if __name__ == '__main__':
    main()
