"""Small reproducible local-model regression comparison, not a general benchmark."""
import argparse
import hashlib
import json
import sqlite3
import tempfile
import time
from contextlib import closing
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

import _bootstrap  # noqa: F401
from campus_rag.config import ROOT, load_settings
from campus_rag.database import Repository
from campus_rag.index.hybrid import HybridIndex, LocalEmbedder, LocalReranker
from campus_rag.ingest.pipeline import ingest_document
from campus_rag.rag.generator import TransformersGenerator
from campus_rag.rag.service import RAGService
from campus_rag.types import ParsedDocument, ParsedTable, SourceDocument


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=ROOT/'data/eval/numeric_regression.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=['all','fixture','campus'], default='all')
    args = parser.parse_args()
    data = json.loads(args.dataset.read_text(encoding='utf-8'))
    settings = load_settings()
    generator = TransformersGenerator(settings.generator_model)
    embedder = LocalEmbedder(settings.embedding_model)
    reranker = LocalReranker(settings.reranker_model)
    report = {'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              'description':data['description'], 'results':[]}
    (ROOT/'.tmp').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT/'.tmp') as directory:
        fixture = replace(settings, database_path=Path(directory)/'fixture.db')
        for n, table in enumerate(data['tables']):
            # Pad absent cells explicitly; missing numeric values must remain missing.
            rows = [row + ['']*(len(table['headers'])-len(row)) for row in table['rows']]
            parsed = ParsedDocument(SourceDocument(table['name'],f'https://example.edu/fixture/{n}',
                published_at='2026-01-01',sha256=f'fixture-{n}'), tables=[ParsedTable(table['name'],table['headers'],rows)])
            ingest_document(parsed,fixture)
        with Repository(fixture.database_path) as repo:
            records = [dict(row) for row in repo.index_records()]
        indexes = {'fixture':HybridIndex(records,embedder.encode_documents([r['text'] for r in records]),embedder,reranker)}
        if args.suite in {'all','campus'}:
            # Reuse model instances for the production index.
            import numpy as np
            with (settings.index_dir/'records.jsonl').open(encoding='utf-8') as handle:
                production = [json.loads(line) for line in handle if line.strip()]
            indexes['campus'] = HybridIndex(production,np.load(settings.index_dir/'vectors.npy'),embedder,reranker)
        for case in data['cases']:
            suite = case['suite']
            if args.suite != 'all' and suite != args.suite:
                continue
            active = fixture if suite == 'fixture' else settings
            started = time.perf_counter()
            with closing(sqlite3.connect(active.database_path)) as connection:
                service = RAGService(active,indexes[suite],generator,connection)
                try:
                    result = service.answer(case['question'])
                    calc = result.calculation
                    if case.get('refuse'):
                        verification = result.numeric_verification or {}
                        correct = not (calc and calc.ok) and not result.references and bool(
                            (calc and not calc.ok) or verification.get('verified') is False)
                    else:
                        correct = bool(calc and calc.ok and result.references and
                            abs(calc.value-Decimal(case['expected'])) <= Decimal(case.get('tolerance','0.000001')))
                    row = {'id':case['id'],'suite':suite,'correct':correct,'result':asdict(result),
                           'plan':asdict(service.last_plan),'raw_plan':getattr(service.planner,'last_raw','')}
                except Exception as exc:
                    row = {'id':case['id'],'suite':suite,'correct':False,'error':type(exc).__name__+': '+str(exc)}
            row['seconds'] = round(time.perf_counter()-started,2)
            report['results'].append(row)
            for group in ['fixture','campus','all']:
                selected = [r for r in report['results'] if group=='all' or r['suite']==group]
                if selected:
                    report[group] = {'correct':sum(r['correct'] for r in selected),'total':len(selected),
                                     'accuracy':sum(r['correct'] for r in selected)/len(selected)}
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            print(json.dumps({k:row[k] for k in ['id','correct','seconds']},ensure_ascii=False),flush=True)
    print(json.dumps({k:report[k] for k in ['fixture','campus','all'] if k in report},ensure_ascii=False))


if __name__ == '__main__':
    main()
