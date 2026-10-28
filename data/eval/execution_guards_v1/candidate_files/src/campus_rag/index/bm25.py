from __future__ import annotations

import math
import re
from collections import Counter


def lexical_tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*|[\u4e00-\u9fff]", lowered)
    chinese = [token for token in words if len(token) == 1 and "\u4e00" <= token <= "\u9fff"]
    bigrams = [chinese[i] + chinese[i + 1] for i in range(len(chinese) - 1)]
    return words + bigrams


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.tokenized = [lexical_tokens(document) for document in documents]
        self.lengths = [len(tokens) for tokens in self.tokenized]
        self.avg_length = sum(self.lengths) / max(1, len(self.lengths))
        self.k1, self.b = k1, b
        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized]
        document_frequency = Counter()
        for tokens in self.tokenized:
            document_frequency.update(set(tokens))
        count = len(documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        query_terms = lexical_tokens(query)
        scored: list[tuple[int, float]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            length_norm = 1 - self.b + self.b * self.lengths[index] / max(1, self.avg_length)
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
            if score > 0:
                scored.append((index, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

