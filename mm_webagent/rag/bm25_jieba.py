"""BM25 with jieba tokenization for Chinese/English web-task text."""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    try:
        import jieba
    except ImportError:
        jieba = None

    if jieba:
        tokens = [token.strip().lower() for token in jieba.cut(text) if token.strip()]
    else:
        tokens = re.findall(r"[\w]+", text.lower())
    return tokens


class BM25JiebaIndex:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = [tokenize(text) for text in texts]
        self.doc_freq = Counter()
        for doc in self.docs:
            self.doc_freq.update(set(doc))
        self.avg_len = sum(len(doc) for doc in self.docs) / len(self.docs) if self.docs else 0.0

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores = []
        total_docs = len(self.docs)
        for doc in self.docs:
            term_counts = Counter(doc)
            doc_len = len(doc)
            score = 0.0
            for term in query_terms:
                freq = term_counts[term]
                if not freq:
                    continue
                df = self.doc_freq[term]
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avg_len, 1e-6))
                score += idf * freq * (self.k1 + 1) / denom
            scores.append(score)
        return scores
