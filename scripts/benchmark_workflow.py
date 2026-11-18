from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
import types

import _bootstrap  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def serializable(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def metrics(rows):
    times = sorted(row['seconds'] for row in rows)
    return {
        'requests': len(rows),
        'mean_seconds': statistics.mean(times),
        'median_seconds': statistics.median(times),
        'p95_seconds': times[max(0, int(len(times)*0.95+0.999999)-1)],
        'total_seconds': sum(times),
        'model_calls': sum(len(row['model_calls']) for row in rows),
        'mean_model_seconds': statistics.mean(sum(call['seconds'] for call in row['model_calls']) for row in rows),
        'peak_allocated_mib': max(row['peak_allocated_mib'] for row in rows),
        'peak_reserved_mib': max(row['peak_reserved_mib'] for row in rows),
        'errors': sum(row['error'] is not None for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description='真实本地模型的原流程 / LangGraph 配对性能对比')
    parser.add_argument('--input', type=Path, default=ROOT/'data/eval/campus_public/questions.csv')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--baseline-ref', default='HEAD')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--limit', type=int, default=0, help='0 表示完整题集')
    args = parser.parse_args()
    if args.repeats < 1 or args.limit < 0:
        parser.error('repeats 必须为正数，limit 不能为负数')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', LANGSMITH_TRACING='false',
                      LANGCHAIN_TRACING_V2='false', TOKENIZERS_PARALLELISM='false')

    network_attempts = []

    def block_network(*unused_args, **unused_kwargs):
        network_attempts.append('blocked')
        raise OSError('Benchmark forbids socket connections')

    socket.socket.connect = block_network
    socket.socket.connect_ex = block_network
    socket.create_connection = block_network

    import torch
    from campus_rag.config import load_settings
    from campus_rag.index import HybridIndex
    from campus_rag.rag.generator import TransformersGenerator
    from campus_rag.rag.service import RAGService
    import sqlite3

    def git(*arguments):
        return subprocess.check_output(['git', '-c', 'safe.directory='+ROOT.as_posix(), *arguments], cwd=ROOT)

    baseline_ref = git('rev-parse', '--verify', args.baseline_ref+'^{commit}').decode().strip()
    baseline_source = git('show', baseline_ref+':src/campus_rag/rag/service.py')
    if b'build_workflow' in baseline_source or b'langgraph' in baseline_source:
        raise ValueError('基线版本已经使用 LangGraph，请指定改造前的提交')
    (args.output_dir/'baseline_service.py').write_bytes(baseline_source)
    module = types.ModuleType('campus_rag.rag._benchmark_baseline')
    sys.modules[module.__name__] = module
    exec(compile(baseline_source, str(args.output_dir/'baseline_service.py'), 'exec'), module.__dict__)
    with args.input.open(encoding='utf-8-sig', newline='') as stream:
        questions = [row['问题'].strip() for row in csv.DictReader(stream) if row['问题'].strip()]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise ValueError('题集不能为空')
    settings = load_settings()
    if not torch.cuda.is_available():
        raise RuntimeError('本次 GPU 对比要求 CUDA 可用')
    print('Loading local retrieval models and index', flush=True)
    start = time.perf_counter()
    index = HybridIndex.load(settings.index_dir, settings.embedding_model, settings.reranker_model, settings.database_path)
    torch.cuda.synchronize()
    index_load_seconds = time.perf_counter()-start
    print('Loading local generator and planner adapter', flush=True)
    start = time.perf_counter()
    generator = TransformersGenerator(settings.generator_model, settings.max_context_tokens,
                                      planner_adapter_path=settings.planner_adapter_path)
    torch.cuda.synchronize()
    generator_load_seconds = time.perf_counter()-start
    connection = sqlite3.connect(settings.database_path.resolve().as_uri()+'?mode=ro', uri=True)
    services, init_seconds = {}, {}
    for name, cls in [('baseline', module.RAGService), ('langgraph', RAGService)]:
        start = time.perf_counter()
        services[name] = cls(settings, index, generator, connection)
        init_seconds[name] = time.perf_counter()-start

    manifest = {
        'baseline_ref': baseline_ref, 'baseline_service_sha256': digest(baseline_source),
        'current_source_sha256': {str(path.relative_to(ROOT)):digest(path.read_bytes())
            for path in [ROOT/'src/campus_rag/rag/service.py', ROOT/'src/campus_rag/rag/workflow.py',
                         ROOT/'src/campus_rag/rag/generator.py', ROOT/'src/campus_rag/rag/planner.py',
                         ROOT/'src/campus_rag/database/query.py', ROOT/'config/settings.json']},
        'input_sha256':digest(args.input.read_bytes()), 'questions':len(questions), 'repeats':args.repeats,
        'python':platform.python_version(), 'gpu':torch.cuda.get_device_name(),
        'versions':{name:version(name) for name in ['torch','transformers','sentence-transformers','peft','langgraph','pydantic']},
        'model':str(settings.generator_model), 'adapter':str(settings.planner_adapter_path),
        'index_load_seconds':index_load_seconds, 'generator_load_seconds':generator_load_seconds,
        'service_init_seconds':init_seconds,
        'method':'Shared real models; greedy decoding; paired alternating order; two warmup questions per workflow; synchronized CUDA timing. PyTorch allocator peaks include shared models. No response replay.',
    }
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    calls = []
    original_generate = generator._generate

    def measured_generate(prompt, *positional, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        raw = original_generate(prompt, *positional, **kwargs)
        torch.cuda.synchronize()
        calls.append({'seconds':time.perf_counter()-start,
                      'prompt_sha256':digest(prompt.encode()), 'response_sha256':digest(raw.encode())})
        return raw

    generator._generate = measured_generate
    rows = []
    try:
        warmup = ['南京邮电大学的前身是什么？', questions[0]]
        for name, service in services.items():
            for question in warmup:
                service.answer(question)
            print(json.dumps({'warmup':name},ensure_ascii=False),flush=True)
        with (args.output_dir/'requests.jsonl').open('w',encoding='utf-8') as stream:
            for repeat in range(args.repeats):
                for number, question in enumerate(questions,1):
                    order = ['baseline','langgraph'] if (repeat+number)%2 else ['langgraph','baseline']
                    for name in order:
                        calls.clear()
                        torch.manual_seed(0)
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        start = time.perf_counter()
                        error, result = None, None
                        try:
                            result = services[name].answer(question)
                        except Exception as exc:
                            error = {'type':type(exc).__name__, 'message':str(exc)}
                        torch.cuda.synchronize()
                        row = {'workflow':name,'repeat':repeat+1,'number':number,'question':question,
                               'seconds':time.perf_counter()-start,
                               'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                               'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                               'model_calls':list(calls),'error':error,
                               'result':serializable(asdict(result)) if result else None,
                               'diagnostics':serializable(services[name].last_diagnostics)}
                        rows.append(row)
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        stream.flush()
                        print(json.dumps({key:row[key] for key in ['workflow','repeat','number','seconds','error']},ensure_ascii=False),flush=True)
    finally:
        connection.close()
    paired = {(row['repeat'],row['number'],row['workflow']):row for row in rows}
    mismatches, differences = [], []
    for repeat in range(1,args.repeats+1):
        for number in range(1,len(questions)+1):
            left, right = [paired[repeat,number,name] for name in ['baseline','langgraph']]
            matching_calls = [{k:v for k,v in call.items() if k!='seconds'} for call in left['model_calls']] == [
                {k:v for k,v in call.items() if k!='seconds'} for call in right['model_calls']]
            if left['result'] != right['result'] or left['error'] != right['error'] or not matching_calls:
                mismatches.append({'repeat':repeat,'number':number,'result_equal':left['result']==right['result'],
                                   'model_calls_equal':matching_calls,'error_equal':left['error']==right['error']})
            differences.append(right['seconds']-left['seconds'])
    summary = {name:metrics([row for row in rows if row['workflow']==name]) for name in services}
    summary['by_route'] = {name:{route:metrics([row for row in rows if row['workflow']==name
        and row['diagnostics']['route']==route]) for route in sorted({row['diagnostics']['route']
        for row in rows if row['workflow']==name})} for name in services}
    summary.update(pairs=len(differences), mismatches=mismatches, network_attempts=len(network_attempts),
                   mean_paired_delta_ms=statistics.mean(differences)*1000,
                   median_paired_delta_ms=statistics.median(differences)*1000)
    (args.output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
