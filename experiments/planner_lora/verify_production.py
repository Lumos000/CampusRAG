"""Compare the actual batch CLI with the planner adapter disabled/enabled."""
import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from campus_rag.config import load_settings
from common import digest, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'data/eval/planner_lora/production')
    parser.add_argument('--worker', choices=['base', 'adapter'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker:
        network = []

        def denied(*args, **kwargs):
            network.append('blocked')
            raise RuntimeError('Network disabled during official inference verification')

        socket.socket.connect = denied
        socket.socket.connect_ex = denied
        sys.argv = ['batch_answer.py', '--input', str(output / 'questions.csv'),
                    '--output', str(output / f'{args.worker}.csv'),
                    '--audit-output', str(output / f'{args.worker}_audit.json')]
        if args.worker == 'base':
            sys.argv.append('--no-adapter')
        started = time.perf_counter()
        try:
            runpy.run_path(str(ROOT / 'scripts/batch_answer.py'), run_name='__main__')
        finally:
            save(output / f'{args.worker}_runtime.json',
                 {'seconds': time.perf_counter() - started, 'network_attempts': len(network)})
        return

    output.mkdir(parents=True, exist_ok=False)
    settings = load_settings()
    assert settings.planner_adapter_path, 'Configure the selected planner adapter first.'
    release = json.loads((ROOT / 'data/eval/campus_public/verified_release.json').read_text(encoding='utf-8'))
    cases = list(release['cases'])
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
    with (output / 'questions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['问题'])
        writer.writeheader()
        writer.writerows({'问题': case['question']} for case in cases)
    tracked = [*sorted((ROOT / 'src').rglob('*.py')), ROOT / 'config/settings.json',
               ROOT / 'scripts/batch_answer.py', ROOT / 'app.py', Path(__file__),
               settings.database_path, settings.index_dir / 'records.jsonl', settings.index_dir / 'vectors.npy',
               settings.planner_adapter_path / 'adapter_config.json',
               settings.planner_adapter_path / 'adapter_model.safetensors']
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in tracked}
    save(output / 'protocol.json', {'purpose': 'Official batch CLI comparison; exposed development regression, not blind testing.',
                                  'cases': cases, 'settings': asdict(settings), 'files': hashes})
    rows, runtimes = [], {}
    environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                       PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4')
    for mode in ['base', 'adapter']:
        save(output / 'status.json', {'phase': 'inference', 'mode': mode, 'cases_per_mode': len(cases)})
        with (output / f'{mode}.log').open('w', encoding='utf-8') as log:
            try:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--output', str(output), '--worker', mode],
                               cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError:
                save(output / 'status.json', {'phase': 'failed', 'mode': mode,
                                             'log': str(output / f'{mode}.log'), 'results_complete': False})
                raise
        audits = json.loads((output / f'{mode}_audit.json').read_text(encoding='utf-8'))
        assert len(audits) == len(cases)
        for case, audit in zip(cases, audits):
            diagnostics = audit.pop('diagnostics')
            assert audit['question'] == case['question']
            assert diagnostics['planner_adapter_enabled'] == (mode == 'adapter')
            calls = sum(a['kind'] != 'syntax_repair' for a in diagnostics['attempts'])
            rows.append({'id': case['id'], 'kind': case['kind'], 'mode': mode, 'result': audit,
                         'diagnostics': diagnostics, 'adapter_calls': calls if mode == 'adapter' else 0})
        runtimes[mode] = json.loads((output / f'{mode}_runtime.json').read_text(encoding='utf-8'))
        print(json.dumps({'mode': mode, 'completed': len(audits), **runtimes[mode]}), flush=True)
    for name, checksum in hashes.items():
        assert digest(ROOT / name) == checksum, ('Input changed during comparison', name)
    save(output / 'results.json', {'results': rows, 'complete': True,
                                  'network_attempts': sum(r['network_attempts'] for r in runtimes.values())})
    subprocess.run([sys.executable, str(Path(__file__).with_name('review_comparison.py')), '--directory', str(output)],
                   cwd=ROOT, check=True)
    save(output / 'status.json', {'phase': 'complete', 'cases_per_mode': len(cases), 'runtimes': runtimes})


if __name__ == '__main__':
    main()
