"""Rebuild a separate database from the verified baseline and supplemental corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from urllib.parse import urlparse

import _bootstrap  # noqa: F401
from campus_rag.config import ROOT, load_settings
from campus_rag.crawl.coverage import read_jsonl
from campus_rag.database import Repository
from campus_rag.database.normalization import normalize_existing_tables
from campus_rag.ingest.cleaning import content_key, restore_document
from campus_rag.ingest.pipeline import ingest_document


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def merge_corpus(baseline, additional):
    groups, raw_sources, variants = {}, {}, []
    for path in (baseline, additional):
        for item in read_jsonl(path):
            key = content_key(restore_document(item['document']))
            if key != item['content_key']:
                raise ValueError('清洗内容与 content_key 不一致')
            source = item['document']['source']
            identity = (source['source_url'], source['sha256'])
            if identity in raw_sources and raw_sources[identity] != key:
                # Same raw source parsed by different parser versions: retain the
                # verified baseline representation instead of duplicating a document.
                variants.append({'source_url': source['source_url'], 'title': source['title'],
                                 'retained_content_key': raw_sources[identity], 'skipped_content_key': key})
                key = raw_sources[identity]
            raw_sources[identity] = key
            if key not in groups:
                groups[key] = item
            else:
                sources = {s['url']: s for s in groups[key]['sources']}
                for source in item['sources']:
                    sources.setdefault(source['url'], source)
                groups[key]['sources'] = list(sources.values())
    return groups, variants


def rebuild(baseline_db, baseline_clean, additional_clean, output, build_vectors=False, excluded_urls=(),
            expected_documents=None, required_hosts=(), database_name='campus_colleges.db'):
    if Path(database_name).name != database_name or not database_name.endswith('.db'):
        raise ValueError('数据库名称必须是以 .db 结尾的文件名')
    baseline_db, output = Path(baseline_db).resolve(), Path(output).resolve()
    before = digest(baseline_db)
    groups, variants = merge_corpus(baseline_clean, additional_clean)
    with closing(sqlite3.connect(baseline_db.as_uri() + '?mode=ro', uri=True)) as connection:
        baseline_keys = {r[0] for r in connection.execute('SELECT content_key FROM ingestion_state')}
        baseline_count = connection.execute('SELECT count(*) FROM documents').fetchone()[0]
    supplied_keys = {item['content_key'] for item in read_jsonl(baseline_clean)}
    if baseline_keys != supplied_keys or baseline_count != len(baseline_keys):
        raise ValueError('基线清洗清单与原库不一致，停止重建以避免遗漏原资料')
    excluded, matched = [], set()
    requested = set(excluded_urls)
    for key, item in list(groups.items()):
        source = item['document']['source']
        urls = {source['source_url']} | {s['url'] for s in item['sources']}
        hits = requested & urls
        if not hits:
            continue
        if key in baseline_keys:
            raise ValueError('不能排除原库文档；此流程必须完整保留基线资料')
        matched.update(hits)
        excluded.append({'content_key': key, 'title': source['title'], 'source_url': source['source_url']})
        del groups[key]
    if requested - matched:
        raise ValueError('存在未匹配的排除 URL，停止以避免生成与预期不符的数据集')
    if expected_documents is not None and len(groups) != expected_documents:
        raise ValueError(f'预期文档数为 {expected_documents}，实际合并为 {len(groups)}，停止重建')
    source_hosts = {urlparse(item['document']['source']['source_url']).hostname for item in groups.values()}
    if set(required_hosts) - source_hosts:
        raise ValueError('缺少必需学院官网资料，停止重建')
    # This also protects every preexisting database/index below the destination.
    output.mkdir(parents=True, exist_ok=False)
    settings = replace(load_settings(), database_path=output / database_name, index_dir=output / 'index')
    with (output / 'clean_documents.jsonl').open('w', encoding='utf-8') as handle:
        for item in groups.values():
            handle.write(json.dumps(item, ensure_ascii=False) + '\n')
    for index, item in enumerate(groups.values(), 1):
        ingest_document(restore_document(item['document']), settings, item['sources'])
        if index % 25 == 0:
            print(f'[REBUILD] {index}/{len(groups)}', flush=True)
    with Repository(settings.database_path) as repository:
        with repository.transaction():
            normalize_existing_tables(repository.connection)
        keys = {r[0] for r in repository.connection.execute('SELECT content_key FROM ingestion_state')}
        counts = {table: repository.connection.execute(f'SELECT count(*) FROM {table}').fetchone()[0]
                  for table in ['documents', 'chunks', 'tables', 'table_cells', 'document_sources']}
        hosts = {}
        for row in repository.connection.execute('SELECT source_url FROM documents'):
            host = urlparse(row[0]).hostname
            hosts[host] = hosts.get(host, 0) + 1
        checks = {'integrity': repository.connection.execute('PRAGMA integrity_check').fetchone()[0] == 'ok',
                  'foreign_keys': not repository.connection.execute('PRAGMA foreign_key_check').fetchall(),
                  'baseline_retained': baseline_keys <= keys, 'all_documents_imported': keys == set(groups),
                  'token_limit': repository.connection.execute('SELECT count(*) FROM chunks WHERE token_count > ?',
                                                               (settings.chunk_hard_max,)).fetchone()[0] == 0}
        records = [dict(row) for row in repository.index_records()]
    profile = json.loads((ROOT / 'config/settings.json').read_text(encoding='utf-8'))
    profile.update(database_path=str(settings.database_path), index_dir=str(settings.index_dir))
    (output / 'settings.json').write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
    if not all(checks.values()):
        raise ValueError('重建数据库检查未通过')
    if build_vectors:
        from campus_rag.index.hybrid import build_index
        build_index(records, settings.index_dir, settings.embedding_model)
    after = digest(baseline_db)
    report = {'baseline_database': str(baseline_db), 'baseline_sha256_before': before,
              'baseline_sha256_after': after, 'baseline_unchanged': before == after,
              'baseline_documents': baseline_count, 'added_unique_documents': len(keys - baseline_keys),
              'same_raw_parser_variants': variants,
              'excluded_documents': excluded,
              'input_sha256': {'baseline_clean': digest(baseline_clean), 'additional_clean': digest(additional_clean)},
              'counts': counts, 'source_hosts': hosts, 'checks': checks,
              'index_built': build_vectors, 'database': str(settings.database_path)}
    (output / 'rebuild_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if before != after:
        raise ValueError('原数据库哈希发生变化，请检查是否有其他进程同时写入')
    return report


def main():
    parser = argparse.ArgumentParser(description='保留原资料，在全新目录重建独立学院扩充数据库')
    parser.add_argument('--recipe', type=Path, help='读取构建配置，例如 config/dataset_campus_public.json；命令行路径参数可覆盖配置')
    parser.add_argument('--baseline-db', type=Path)
    parser.add_argument('--baseline-clean', type=Path)
    parser.add_argument('--additional-clean', type=Path)
    parser.add_argument('--output', type=Path, help='必须是不存在的新目录')
    parser.add_argument('--build-index', action='store_true', help='在新目录构建独立检索索引')
    parser.add_argument('--exclude-url', action='append', default=[], help='排除指定补采文档（可重复），不允许排除原库资料')
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding='utf-8')) if args.recipe else {}

    def path_value(name, default=None):
        value = getattr(args, name) or recipe.get(name) or default
        if value is None:
            parser.error(f'缺少 --{name.replace("_", "-")} 或对应构建配置')
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    print(json.dumps(rebuild(
        path_value('baseline_db', load_settings().database_path),
        path_value('baseline_clean', 'data/processed/clean/clean_documents.jsonl'),
        path_value('additional_clean'), path_value('output'), args.build_index,
        recipe.get('excluded_urls', []) + args.exclude_url,
        recipe.get('expected_documents'), recipe.get('required_hosts', []),
        recipe.get('database_name', 'campus_colleges.db')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
