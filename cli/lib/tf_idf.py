import pickle
from .utils import get_movies, CACHE_PATH
from .search import Movie
from collections import defaultdict
import os

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        lower = text.lower()
        tokens = lower.split(" ")
        for token in tokens:
            self.index[token].add(doc_id)
        return

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index[term.lower()]
        return sorted(doc_ids)

    def build(self) -> None:
        movies = get_movies()
        for v in movies["movies"]:
            new_movie = Movie(v["id"], v["title"], v["description"])
            self.docmap[new_movie.id] = new_movie
            self.__add_document(new_movie.id, f"{new_movie.title} {new_movie.description}")
        return

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        return