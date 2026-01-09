import os
from typing import Any

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .utils import get_movies
from .gemini import spell_correct, rewrite_query


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

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[tuple[Any, Any]]:
        bm25 = self._bm25_search(query, limit * 500)
        semantic = self.semantic_search.search_chunks(query, limit * 500)

        bm25_scores = [score for doc_id, score in bm25]
        normalized_bm25_scores = normalize_scores(bm25_scores)

        semantic_scores = []
        doc_id_to_idx_document = {}
        for i, data in enumerate(semantic):
            semantic_score = data["score"]
            semantic_scores.append(semantic_score)
            doc_id = data["id"]
            document = data["document"]
            doc_id_to_idx_document[doc_id] = {"idx": i, "document": document, "title": data["title"]}
        normalized_semantic_scores = normalize_scores(semantic_scores)

        search_result = {}
        for i, (doc_id, original_score) in enumerate(bm25):
            normalized_bm25_score = normalized_bm25_scores[i]
            semantic_data = doc_id_to_idx_document[doc_id]
            semantic_idx = semantic_data["idx"]
            document = semantic_data["document"]
            normalized_semantic_score = normalized_semantic_scores[semantic_idx]
            hy_score = hybrid_score(normalized_bm25_score, normalized_semantic_score, alpha)
            search_result[doc_id] = {"title": semantic_data["title"], "document": document, "keyword_score": normalized_bm25_score, "semantic_score": normalized_semantic_score, "hybrid_score": hy_score}

        sorted_result = sorted(search_result.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)
        return sorted_result

    def rrf_search(self, query: str, k: int = 60, limit: int = 5):
        bm25 = self._bm25_search(query, limit * 500)
        semantic = self.semantic_search.search_chunks(query, limit * 500)

        search_result = {}
        for i, data in enumerate(semantic, 1):
            doc_id = data["id"]
            title = data["title"]
            document = data["document"]
            search_result[doc_id] = {"title": title, "document": document, "semantic_rank": i}

        for i, (doc_id, score) in enumerate(bm25, 1):
            item = search_result.get(doc_id)
            if item:
                item["bm25_rank"] = i
                search_result[doc_id] = item
                continue
            document = self.documents[doc_id]
            title = document["title"]
            description = document["description"]
            search_result[doc_id] = {"title": title, "document": description[:100], "bm25_rank": i}

        for key, v in search_result.items():
            bm25_rank = v.get("bm25_rank")
            semantic_rank = v.get("semantic_rank")

            if bm25_rank and semantic_rank:
                score = rrf_score(bm25_rank, k) + rrf_score(semantic_rank, k)
                v["rrf_score"] = score
                search_result[key] = v
                continue

            if bm25_rank:
                score = rrf_score(bm25_rank, k)
                v["rrf_score"] = score
                search_result[key] = v
                continue

            score = rrf_score(semantic_rank, k)
            v["rrf_score"] = score
            search_result[key] = v

        sorted_search_result = sorted(search_result.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        return sorted_search_result


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

def hybrid_score(bm25_score: float, semantic_score: float, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search(query: str, alpha: float = 0.5, limit: int = 5) -> list[tuple[Any, Any]]:
    documents = get_movies()
    search = HybridSearch(documents["movies"])
    result = search.weighted_search(query, alpha, limit)
    return result[:limit]

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def rrf_search(query: str, k: int, limit: int, enhancement: str) -> list[tuple[Any, Any]]:
    documents = get_movies()
    search = HybridSearch(documents["movies"])

    enhanced_query = ""

    match enhancement:
        case "spell":
            enhanced_query = spell_correct(query)
        case "rewrite":
            enhanced_query = rewrite_query(query)

    if enhanced_query != query:
        print(f"Enhanced query ({enhancement}): '{query}' -> '{enhanced_query}'\n")

    result = search.rrf_search(query, k)
    return result[:limit]