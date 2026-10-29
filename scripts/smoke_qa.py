"""Real local-model QA smoke test, not a benchmark or a claim of correctness."""
import argparse
import json
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401
from campus_rag.config import load_settings
from campus_rag.index.hybrid import HybridIndex
from campus_rag.rag.generator import TransformersGenerator
from campus_rag.rag.service import RAGService


QUESTIONS = [
    '根据2026年本科招生章程，工科类优势学科专业的学费和住宿费预收标准分别是多少？',
    '2026年综合评价中，首选物理、再选政治和地理，仅按选科条件筛选，可报专业的计划合计多少人？',
    '图书馆借阅规则中，学生如何办理借还手续？',
    '根据2023—2024学年本科教学质量报告，专任教师和外聘教师分别有多少人？',
    '根据2038年本科招生章程，学费是多少？',
    '按2026年本科招生章程，工科类优势学科专业的学费上浮后，加住宿费，假设标准四年不变，预收额合计多少？',
    '根据2023—2024学年本科教学质量报告，反推外聘教师折算系数，再复算生师比。',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/eval/smoke_qa.json'))
    parser.add_argument('--limit', type=int, help='最多检查的问题数，默认检查问题列表中的全部剩余问题')
    parser.add_argument('--start', type=int, default=0, help='从第几个问题开始（0起算）')
    parser.add_argument('--config', type=Path, help='指定独立数据库和索引配置')
    parser.add_argument('--questions-file', type=Path, help='JSON 字符串数组，用于定向问答检查')
    args = parser.parse_args()
    if args.start < 0 or (args.limit is not None and args.limit < 1):
        parser.error('--start 不能小于0，--limit 必须大于0')
    questions = json.loads(args.questions_file.read_text(encoding='utf-8')) if args.questions_file else QUESTIONS
    if not isinstance(questions, list) or not questions or any(not isinstance(q, str) or not q.strip() for q in questions):
        parser.error('问题文件必须为非空字符串数组')
    settings = load_settings(args.config)
    index = HybridIndex.load(settings.index_dir, settings.embedding_model, settings.reranker_model)
    generator = TransformersGenerator(settings.generator_model)
    report = []
    with closing(sqlite3.connect(settings.database_path.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
        service = RAGService(settings, index, generator, connection)
        end = args.start + args.limit if args.limit is not None else None
        for question in questions[args.start:end]:
            started = time.perf_counter()
            try:
                result = service.answer(question)
                record = asdict(result)
                record['plan'] = asdict(service.last_plan)
            except Exception as exc:
                record = {'question':question, 'error':type(exc).__name__, 'detail':str(exc)}
            record['seconds'] = round(time.perf_counter()-started,2)
            report.append(record)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            print(json.dumps({key:record.get(key) for key in ['question','answer','error','seconds']},ensure_ascii=False,default=str),flush=True)
    if any('error' in row for row in report):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
