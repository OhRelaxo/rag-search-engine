import os

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .utils import get_movies


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        bm25 = self._bm25_search(query, limit * 500)
        semantic = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = [score for doc_id, score in bm25]
        normalized_bm25_scores = normalize_scores(bm25_scores)

        for i, (doc_id, original_score) in enumerate(bm25):
            normalized_score = normalized_bm25_scores[i]

        print(type(semantic[0]))


    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores: list[float]) -> list[float]:
    min_score = min(scores)
    range_score = max(scores) - min_score

    if range_score == 0:
        return [1.0] * len(scores)

    normalized_scores = []
    for score in scores:
        normalized_score = (score - min_score) / range_score
        normalized_scores.append(normalized_score)

    return normalized_scores

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search(query: str, alpha: float = 0.5, limit: int = 5):
    documents = get_movies()
    search = HybridSearch(documents["movies"])
    search.weighted_search(query, alpha, limit)