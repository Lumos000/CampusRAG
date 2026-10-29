"""Collect and clean into a new directory without importing or opening a database."""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401
from campus_rag.config import ROOT
from campus_rag.crawl import CampusCrawler
from campus_rag.crawl.coverage import coverage_report, read_jsonl
from campus_rag.ingest.cleaning import prepare_manifest


def main():
    parser = argparse.ArgumentParser(description='独立采集五类主题及学院资料，不写入数据库')
    parser.add_argument('--config', type=Path, default=ROOT / 'config/sites_topics.json')
    parser.add_argument('--output', type=Path, help='必须为不存在的新目录，默认 data/collections/topics_时间戳')
    parser.add_argument('--max-pages', type=int, help='本轮请求数上限')
    parser.add_argument('--max-depth', type=int, help='链接跟踪深度')
    parser.add_argument('--reuse-manifest', type=Path, default=ROOT / 'data/raw/manifest.jsonl',
                        help='只读复用已有原始资料；原文件保持原位')
    parser.add_argument('--skip-crawl', action='store_true', help='只读复核已有资料，不联网')
    args = parser.parse_args()
    if args.max_pages is not None and args.max_pages < 1:
        parser.error('--max-pages 必须大于 0')
    if args.max_depth is not None and args.max_depth < 0:
        parser.error('--max-depth 不能小于 0')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = (args.output or ROOT / 'data/collections' / ('topics_' + stamp)).resolve()
    # Refuse all existing destinations, including the original data directories.
    if output.exists():
        parser.error('输出目录已存在，请指定新目录；已有数据不会覆盖。')
    missing = [name for name in ('fitz', 'docx', 'openpyxl') if importlib.util.find_spec(name) is None]
    if missing:
        parser.error('缺少项目解析依赖：' + ', '.join(missing) + '；请先激活已配置的 rag 环境。')
    output.mkdir(parents=True, exist_ok=False)
    print(f'[OUTPUT] {output}', flush=True)
    config_path = output / 'sites.json'
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.skip_crawl:
        CampusCrawler(config_path, output / 'raw', args.max_pages, args.max_depth).run()
    latest = {}
    for manifest in (args.reuse_manifest, output / 'raw/manifest.jsonl'):
        if manifest.exists():
            for item in read_jsonl(manifest):
                latest[item['url']] = item
    manifest = output / 'combined_manifest.jsonl'
    with manifest.open('w', encoding='utf-8') as handle:
        for item in latest.values():
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    prepare_manifest(manifest, output / 'clean', scope=config)
    report = coverage_report(output / 'clean', config)
    (output / 'coverage_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(output), 'accepted_unique_documents': report['accepted_unique_documents'],
                      'database_modified': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
