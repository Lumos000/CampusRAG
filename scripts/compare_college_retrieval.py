"""Compare frozen source-retrieval cases with unchanged models and parameters."""
from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
from campus_rag.config import ROOT, load_settings
from campus_rag.index.hybrid import HybridIndex, LocalEmbedder, LocalReranker
from rebuild_college_database import digest


def load_profile(settings, embedder, reranker):
    with (settings.index_dir / 'records.jsonl').open(encoding='utf-8') as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    vectors = np.load(settings.index_dir / 'vectors.npy')
    if vectors.shape != (len(records), 512) or not np.isfinite(vectors).all():
        raise ValueError('索引记录与向量不匹配')
    with closing(sqlite3.connect(settings.database_path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        keys = dict(connection.execute('SELECT doc_id,content_key FROM ingestion_state'))
        chunks = {r['id']: dict(r) for r in connection.execute(
            'SELECT c.id,c.text,c.kind,c.doc_id,c.table_id,c.heading_path,'
            'd.title,d.source_url,d.department,d.published_at '
            'FROM chunks c JOIN documents d ON d.id=c.doc_id')}
    if (len(chunks) != len(records) or {r['id'] for r in records} != set(chunks)
            or any(any(r.get(field) != value for field, value in chunks[r['id']].items()) for r in records)):
        raise ValueError('索引与数据库分块不一致')
    return HybridIndex(records, vectors, embedder, reranker), keys


def summarize(results):
    report = {}
    for group in sorted({r['group'] for r in results} | {'all'}):
        report[group] = {}
        for variant in ('baseline', 'candidate'):
            report[group][variant] = {}
            for mode in sorted({r['mode'] for r in results}):
                rows = [r for r in results if r['variant'] == variant and r['mode'] == mode
                        and (group == 'all' or r['group'] == group)]
                if rows:
                    report[group][variant][mode] = {'cases': len(rows),
                        'source_hit_at_5': sum(r['rank'] is not None and r['rank'] <= 5 for r in rows) / len(rows),
                        'source_hit_at_10': sum(r['rank'] is not None and r['rank'] <= 10 for r in rows) / len(rows),
                        'mrr_at_20': sum(1 / r['rank'] if r['rank'] else 0 for r in rows) / len(rows)}
    return report


def main():
    parser = argparse.ArgumentParser(description='使用冻结问题集比较原库和学院扩充库的来源检索表现')
    parser.add_argument('--baseline-config', type=Path)
    parser.add_argument('--candidate-config', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True, help='JSON: cases，含 group/question/relevant_content_keys')
    parser.add_argument('--output', type=Path, required=True, help='必须为不存在的新目录')
    parser.add_argument('--modes', nargs='+', choices=['bm25', 'hybrid', 'rerank'], default=['bm25', 'hybrid', 'rerank'])
    args = parser.parse_args()
    baseline = load_settings(args.baseline_config) if args.baseline_config else replace(
        load_settings(), database_path=ROOT/'data/processed/campus.db', index_dir=ROOT/'data/index')
    candidate = load_settings(args.candidate_config)
    if baseline.database_path.resolve()==candidate.database_path.resolve() or baseline.index_dir.resolve()==candidate.index_dir.resolve():
        parser.error('基线与候选必须使用各自独立的数据库和配套索引')
    if baseline.embedding_model != candidate.embedding_model or baseline.reranker_model != candidate.reranker_model:
        parser.error('比较要求使用相同模型')
    cases = json.loads(args.cases.read_text(encoding='utf-8'))['cases']
    if not cases or any(not c.get('relevant_content_keys') for c in cases):
        parser.error('问题集必须包含正例来源内容键')
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = {'cases_sha256': digest(args.cases),
                  'case_count': len(cases), 'embedding_model': str(baseline.embedding_model),
                  'reranker_model': str(baseline.reranker_model),
                  'parameters': {'bm25_top_k': 40, 'dense_top_k': 40, 'fusion_top_k': 25, 'final_top_k': 20},
                  'baseline_database_sha256': digest(baseline.database_path),
                  'candidate_database_sha256': digest(candidate.database_path),
                  'baseline_index_records_sha256': digest(baseline.index_dir / 'records.jsonl'),
                  'candidate_index_records_sha256': digest(candidate.index_dir / 'records.jsonl'),
                  'method': '固定模型和检索参数；按相关文档内容键计算来源命中，不等于答案事实正确率。',
                  'limitations': '同一助手编写学院定向问题，样本量有限；未评估生成答案或拒答。'}
    embedder = LocalEmbedder(baseline.embedding_model) if any(m != 'bm25' for m in args.modes) else None
    reranker = LocalReranker(baseline.reranker_model) if 'rerank' in args.modes else None
    profiles = {'baseline': load_profile(baseline, embedder, reranker),
                'candidate': load_profile(candidate, embedder, reranker)}
    results = []
    for number, case in enumerate(cases, 1):
        relevant = set(case['relevant_content_keys'])
        for variant, (index, keys) in profiles.items():
            for mode in args.modes:
                evidence = index.search(case['question'], bm25_top_k=40, dense_top_k=40,
                                        fusion_top_k=25, final_top_k=20, mode=mode)
                rank = next((i for i, e in enumerate(evidence, 1) if keys.get(e.doc_id) in relevant), None)
                results.append({'case_id': case['id'], 'group': case['group'], 'question': case['question'],
                                'variant': variant, 'mode': mode, 'rank': rank,
                                'top_sources': [{'title': e.title, 'url': e.source_url, 'doc_id': e.doc_id}
                                                for e in evidence[:5]]})
        print(f'[COMPARE] {number}/{len(cases)} {case["id"]}', flush=True)
        (args.output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'provenance': provenance, 'metrics': summarize(results)}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
