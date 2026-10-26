from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
import sqlite3
from pathlib import Path

import _bootstrap  # noqa: F401

from campus_rag.config import load_settings
from campus_rag.index import HybridIndex
from campus_rag.rag import RAGService
from campus_rag.rag.generator import TransformersGenerator


def _read_questions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return []
    key = "问题" if "问题" in rows[0] else next(iter(rows[0]))
    return [row[key].strip() for row in rows if row.get(key, "").strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="离线批量回答测试问题")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument('--audit-output', type=Path, help='另存原文数值、计算过程和答案核对日志')
    args = parser.parse_args()
    settings = load_settings()
    index = HybridIndex.load(settings.index_dir, settings.embedding_model, settings.reranker_model)
    generator = TransformersGenerator(settings.generator_model)
    connection = sqlite3.connect(settings.database_path)
    try:
        service = RAGService(settings, index, generator, connection)
        rows = []
        audits = []
        for question in _read_questions(args.input):
            result = service.answer(question)
            if args.audit_output:
                audits.append(asdict(result))
            rows.append({
                "问题": question,
                "答案": result.answer,
                "参考资料": "\n".join(result.references),
            })
    finally:
        connection.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["问题", "答案", "参考资料"])
        writer.writeheader()
        writer.writerows(rows)
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True,exist_ok=True)
        args.audit_output.write_text(json.dumps(audits,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    print(f"已写入 {len(rows)} 条结果：{args.output}")


if __name__ == "__main__":
    main()
