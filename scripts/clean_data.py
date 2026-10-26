from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from campus_rag.config import ROOT
from campus_rag.ingest.cleaning import prepare_manifest


def main():
    parser = argparse.ArgumentParser(description='生成可追溯的清洗结果、复核清单和去重正文')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/raw/manifest.jsonl')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/clean')
    parser.add_argument('--limit', type=int, help='只检查前 N 个成功 URL，用于试运行')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit 必须大于 0')
    print(json.dumps(prepare_manifest(args.manifest, args.output, args.limit), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
