import pickle
import os
from collections import defaultdict, Counter
import math

from .text_processing import text_processing
from .utils import get_movies, CACHE_PATH, BM25_K1

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict[int, str]] = {}
        self.term_frequencies: dict[int, Counter] = defaultdict(Counter)
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_PATH, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = text_processing(text)
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1
        return

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_movie(self, doc_id: int) -> dict[int, str]:
        movie = self.docmap[doc_id]
        return movie

    def get_tf(self, doc_id: int, term: str) -> int:
        token = text_processing(term)
        if len(token) > 1:
            raise Exception("error in class InvertedIndex at method get_tf: too many tokens, can only process one token!")
        return self.term_frequencies[doc_id][token[0]]

    def get_idf(self, term: str) -> float:
        token = text_processing(term)
        if len(token) > 1:
            raise Exception("error in class InvertedIndex at method get_bm25_id: too many tokens, can only process one token!")
        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(token[0], set()))
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        # log((N - df + 0.5) / (df + 0.5) + 1)
        token = text_processing(term)
        if len(token) > 1:
            raise Exception("error in class InvertedIndex at method get_bm25_id: too many tokens, can only process one token!")
        doc_count = len(self.docmap)
        term_doc_count = len(self.index.get(token[0], set()))
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1 = BM25_K1) -> float:
        tf = self.get_tf(doc_id, term)
        bm25_tf = (tf * (k1 + 1) / (tf + k1))
        return bm25_tf

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
        return