from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from campus_rag.config import ROOT, load_settings
from campus_rag.ingest import ingest_path
from campus_rag.crawl.crawler import is_error_page
from campus_rag.ingest.cleaning import restore_document
from campus_rag.ingest.pipeline import ingest_document


SUPPORTED = {".html", ".htm", ".pdf", ".docx", ".xlsx", ".xlsm", ".txt", ".md"}


def main() -> None:
    parser = argparse.ArgumentParser(description="将公开资料解析到 SQLite")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "raw" / "manifest.jsonl")
    parser.add_argument('--clean-manifest', type=Path, help='导入 clean_data.py 生成的 clean_documents.jsonl')
    parser.add_argument('--database', type=Path, help='指定数据库，便于隔离试运行')
    parser.add_argument('--limit', type=int, help='本次最多导入 N 份文档')
    args = parser.parse_args()
    settings = load_settings()
    if args.database:
        settings = replace(settings, database_path=args.database.resolve())
    if args.limit is not None and args.limit < 1:
        parser.error('--limit 必须大于 0')
    if args.clean_manifest:
        count = errors = 0
        with args.clean_manifest.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                if args.limit is not None and count + errors >= args.limit:
                    break
                item = json.loads(line)
                try:
                    parsed = restore_document(item['document'])
                    doc_id = ingest_document(parsed, settings, item['sources'])
                    count += 1
                    print(f'[OK] doc_id={doc_id} {parsed.source.title}', flush=True)
                except Exception as exc:
                    errors += 1
                    print(f'[ERROR] {type(exc).__name__}: {exc}', flush=True)
        print(f'processed={count} errors={errors} database={settings.database_path}')
        sys.exit(1 if errors else 0)
    jobs: list[tuple[Path, dict]] = []
    if args.manifest.exists():
        latest = {}
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get('status') == 'ok':
                latest[item['url']] = item
        for item in latest.values():
            if is_error_page(item.get('title', '')):
                print(f"[SKIP] 错误提示页：{item['url']}")
                continue
            path = Path(item.get("local_path", ""))
            if item.get("status") == "ok" and path.suffix.lower() in SUPPORTED:
                jobs.append((path, item))
    jobs.extend((path, {}) for path in args.paths if path.suffix.lower() in SUPPORTED)
    if not jobs:
        raise SystemExit("没有可导入的文件。可传入路径或先运行 crawl_sites.py。")
    for path, metadata in jobs:
        try:
            doc_id = ingest_path(
                path, settings, source_url=metadata.get("final_url") or metadata.get("url", ""),
                department=metadata.get("department", ""), published_at=metadata.get("published_at"),
                source_title=metadata.get('title', ''),
            )
            print(f"[OK] doc_id={doc_id} {path}")
        except Exception as exc:
            print(f"[ERROR] {path}: {exc}")


if __name__ == "__main__":
    main()
