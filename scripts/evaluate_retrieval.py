from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from campus_rag.config import load_settings
from campus_rag.index import HybridIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="计算检索 Recall@K 和 MRR")
    parser.add_argument("--dataset", required=True, type=Path, help="JSONL: question, relevant_ids")
    parser.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid", "rerank"])
    args = parser.parse_args()
    settings = load_settings()
    index = HybridIndex.load(settings.index_dir, settings.embedding_model, settings.reranker_model)
    cases = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    count = max(1, len(cases))
    report = {}
    for mode in args.modes:
        hits = {5: 0, 10: 0, 20: 0}
        reciprocal_ranks = []
        for case in cases:
            results = index.search(case["question"], fusion_top_k=40, final_top_k=20, mode=mode)
            ids = [item.evidence_id for item in results]
            relevant = {str(value) for value in case["relevant_ids"]}
            for k in hits:
                hits[k] += int(bool(set(ids[:k]) & relevant))
            rank = next((i for i, value in enumerate(ids, 1) if value in relevant), None)
            reciprocal_ranks.append(1 / rank if rank else 0)
        metrics = {f"Recall@{k}": value / count for k, value in hits.items()}
        metrics["MRR"] = sum(reciprocal_ranks) / count
        report[mode] = metrics
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
