import pickle
import os
from collections import defaultdict, Counter
import math

from .text_processing import text_processing
from .utils import get_movies, CACHE_PATH, BM25_K1, BM25_B

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[str, str]] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = text_processing(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        self.doc_lengths[doc_id] = len(tokens)
        return

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_movie(self, doc_id: int) -> dict[str, str]:
        movie = self.docmap[doc_id]
        return movie

    def get_tf(self, doc_id: int, term: str) -> int:
        token = text_processing(term)
        if len(token) != 1:
            raise Exception("error in class InvertedIndex at method get_tf: too many tokens, can only process one token!")
        return self.term_frequencies.get(doc_id, Counter()).get(token[0], 0)

    def get_idf(self, term: str) -> float:
        token = text_processing(term)
        if len(token) != 1:
            raise Exception("error in class InvertedIndex at method get_bm25_id: too many tokens, can only process one token!")
        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(token[0], set()))
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = text_processing(term)
        if len(token) != 1:
            raise Exception("error in class InvertedIndex at method get_bm25_id: too many tokens, can only process one token!")
        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(token[0], set()))
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1 = BM25_K1, b = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        doc_count = len(self.doc_lengths)
        doc_length_count = 0.00
        for length in self.doc_lengths.values():
            doc_length_count += length
        return doc_length_count / doc_count

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        tokens = text_processing(query)
        scores: dict[int, float] = {}
        for doc_id in self.doc_lengths:
            bm25_score = 0.00
            for token in tokens:
                bm25_score += self.bm25(doc_id, token)
            scores[doc_id] = bm25_score
        sorted_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_score[:limit]

    def build(self) -> None:
        movies = get_movies()
        for movie in movies["movies"]:
            doc_id = movie["id"]
            doc_description = f"{movie['title']} {movie['description']}"
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, doc_description)
        return

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)
        return

    def load(self) -> None:
        try:
            with open(self.index_path, "rb") as f:
                self.index= pickle.load(f)
        except FileNotFoundError:
            print("No file with the name index.pkl was found in the cache directory")
        except Exception as e:
            print(f"error while opening the index.pkl file: {e}")

        try:
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        except FileNotFoundError:
            print("No file with the name docmap.pkl was found in the cache directory")
        except Exception as e:
            print(f"error while opening the docmap.pkl file: {e}")

        try:
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
        except FileNotFoundError:
            print("No file with the name term_frequencies.pkl was found in the cache directory")
        except Exception as e:
            print(f"error while opening the term_frequencies.pkl file: {e}")

        try:
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            print("No file with the name doc_lengths.pkl was found in the cache directory")
        except Exception as e:
            print(f"error while opening the doc_lengths.pkl file: {e}")
        return