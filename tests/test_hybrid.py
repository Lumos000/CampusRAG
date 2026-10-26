import numpy as np

from campus_rag.index.hybrid import HybridIndex, reciprocal_rank_fusion


class FakeEmbedder:
    def encode_queries(self, queries):
        return np.array([[1.0, 0.0]], dtype=np.float32)


def test_rrf_and_hybrid_search_without_reranker():
    assert reciprocal_rank_fusion([[(0, 2.0), (1, 1.0)], [(1, 3.0)]])[0][0] == 1
    records = [
        {"id": "a", "text": "招生 计划 人数", "doc_id": 1, "title": "招生", "source_url": "u1", "table_id": None},
        {"id": "b", "text": "图书馆 开放 时间", "doc_id": 2, "title": "图书馆", "source_url": "u2", "table_id": None},
    ]
    index = HybridIndex(records, np.array([[1.0, 0.0], [0.0, 1.0]]), FakeEmbedder())
    results = index.search("招生人数", bm25_top_k=2, dense_top_k=2, final_top_k=1)
    assert results[0].evidence_id == "a"
    assert index.search("图书馆", final_top_k=1, mode="bm25")[0].evidence_id == "b"
