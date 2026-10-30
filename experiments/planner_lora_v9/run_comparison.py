"""Paired current-code/current-database comparison; no production changes."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sqlite3
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from campus_rag.config import load_settings
from campus_rag.database.query import StructuredQueryEngine
from campus_rag.index.hybrid import HybridIndex
from campus_rag.rag.generator import TransformersGenerator
from campus_rag.rag.service import RAGService
from campus_rag.rag.planner import parse_plan
from evaluate_formal import save, digest

OUTPUT = ROOT / 'data/eval/planner_lora_v9/current_corpus'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--adapter', type=Path, required=True, help='Explicit experiment adapter; leaves the default service unchanged.')
    args = parser.parse_args()
    output = args.output
    import torch
    from peft import PeftModel
    torch.set_num_threads(4)
    settings = load_settings()
    release = json.loads((ROOT / 'data/eval/campus_public/verified_release.json').read_text(encoding='utf-8'))
    selected = {'path': args.adapter.resolve().relative_to(ROOT).as_posix()}
    cases = [dict(case) for case in release['cases']]
    prefix = '按《人工智能学院人工智能微专业招生简章》的课程表，'
    for key, text, value, unit in [
        ('NEW01', '五门核心课程的学分合计多少学分？', '10', '学分'),
        ('NEW02', '五门核心课程的学时合计多少学时？', '160', '学时'),
        ('NEW03', '图像处理和模式识别两门课的学分合计多少学分？', '4', '学分'),
        ('NEW04', '图像处理和模式识别两门课的学时合计多少学时？', '64', '学时'),
        ('NEW05', '总计10学分，已完成4学分，还差多少学分？', '6', '学分'),
        ('NEW06', '总计160学时，已完成32学时，还差多少学时？', '128', '学时'),
    ]:
        cases.append({'id': key, 'kind': 'new_numeric', 'question': prefix + text, 'expected': value, 'unit': unit})
    tracked = [*sorted((ROOT / 'src').rglob('*.py')), ROOT / 'config/settings.json', settings.database_path,
        settings.index_dir / 'records.jsonl', settings.index_dir / 'vectors.npy', Path(__file__),
        ROOT / selected['path'] / 'adapter_model.safetensors', ROOT / selected['path'] / 'adapter_config.json']
    protocol = {'purpose': '54 paired current-runtime diagnostics; exposed 48-case regression plus six new credit/hour arithmetic cases, not a blind benchmark.',
        'adapter': selected['path'], 'files': {path.relative_to(ROOT).as_posix(): digest(path) for path in tracked},
        'cases': cases, 'adapter_scope': 'Only step planning and repairs; all text calls disable adapter; no revised prompts or fallback overlay.'}
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        assert json.loads((output / 'protocol.json').read_text(encoding='utf-8')) == protocol
    else:
        save(output / 'protocol.json', protocol)
    network = []
    def denied(*args, **kwargs):
        network.append('blocked')
        raise RuntimeError('Network disabled during model comparison')
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    index = HybridIndex.load(settings.index_dir, settings.embedding_model, settings.reranker_model, settings.database_path)
    class CachedIndex:
        def __init__(self):
            self.cache = {}
        def search(self, *args, **kwargs):
            key = repr((args, kwargs))
            if key not in self.cache:
                self.cache[key] = index.search(*args, **kwargs)
            return list(self.cache[key])
    cached_index = CachedIndex()
    original = TransformersGenerator(settings.generator_model, settings.max_context_tokens)
    original.model = PeftModel.from_pretrained(original.model, ROOT / selected['path'], is_trainable=False)
    original.model.eval()
    class Generator:
        planning_calls = 0
        base_calls = 0
        def generate(self, *args, **kwargs):
            self.base_calls += 1
            with original.model.disable_adapter():
                return original.generate(*args, **kwargs)
        def planning_generate(self, *args, **kwargs):
            self.planning_calls += 1
            return original.generate(*args, **kwargs)
    generator = Generator()
    connection = sqlite3.connect(settings.database_path.resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('BEGIN')
    services = {mode: RAGService(settings, cached_index, generator, connection) for mode in ['base', 'adapter']}
    services['adapter'].planner.generator = SimpleNamespace(generate=generator.planning_generate)
    path = output / 'results.json'
    report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'results': [], 'complete': False}
    done = {(row['id'], row['mode']) for row in report['results']}
    for case in cases:
        for mode in ['base', 'adapter']:
            if (case['id'], mode) in done:
                continue
            service = services[mode]
            before = (generator.base_calls, generator.planning_calls)
            started = time.perf_counter()
            answer = service.answer(case['question'])
            row = {'id': case['id'], 'kind': case['kind'], 'mode': mode, 'question': case['question'],
                'result': asdict(answer), 'plan': asdict(service.last_plan), 'raw_plan': service.planner.last_raw,
                'diagnostics': service.last_diagnostics, 'seconds': time.perf_counter() - started,
                'base_calls': generator.base_calls - before[0], 'adapter_calls': generator.planning_calls - before[1]}
            report['results'].append(row)
            report['network_attempts'] = len(network)
            save(path, report)
            print(json.dumps({key: row[key] for key in ['id', 'mode', 'seconds', 'adapter_calls']}) , flush=True)
    # Independently test whether reference plans for the new table operations
    # are executable before attributing any failure to the language model.
    engine = StructuredQueryEngine(connection)
    audit = []
    for case in cases:
        if case['kind'] != 'new_numeric':
            continue
        field = 18222 if case['unit'] == '学分' else 18223
        filters = [{'field_id': 18221, 'op': 'in', 'value': ['图像处理', '模式识别']}] if case['id'] in {'NEW03', 'NEW04'} else []
        steps = [{'id': 'total', 'op': 'sum', 'field_id': field, 'filters': filters}]
        if case['id'] in {'NEW05', 'NEW06'}:
            steps += [{'id': 'done', 'op': 'lookup', 'literal': '4' if case['id'] == 'NEW05' else '32', 'unit': case['unit']},
                      {'id': 'result', 'op': 'subtract', 'inputs': ['total', 'done']}]
        raw = json.dumps({'table_id': 784, 'steps': steps}, ensure_ascii=False)
        result = engine.execute_steps(parse_plan(raw, require_final=True), [784], case['question'])
        audit.append({'id': case['id'], 'ok': result.ok, 'error': result.error,
                      'column': dict(connection.execute('SELECT id,original_name,unit,data_type FROM table_columns WHERE id=?', (field,)).fetchone())})
    save(output / 'reference_execution_audit.json', audit)
    for name, checksum in protocol['files'].items():
        assert digest(ROOT / name) == checksum, ('Input changed during comparison', name)
    report['complete'] = True
    save(path, report)
    print('Complete paired comparison; explicit changed-answer review remains.', flush=True)


if __name__ == '__main__':
    main()
