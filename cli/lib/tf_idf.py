import pickle
from .text_processing import text_processing
from .utils import get_movies, CACHE_PATH
from collections import defaultdict
import os

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = text_processing(text)
        for token in tokens:
            self.index[token].add(doc_id)
        return

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_movie(self, doc_id: int):
        movie = self.docmap[doc_id]
        return movie

    def build(self) -> None:
        movies = get_movies()
        for v in movies["movies"]:
            doc_id = v["id"]
            doc_description = f"{v['title']} {v['description']}"
            self.docmap[doc_id] = v
            self.__add_document(doc_id, doc_description)
        return

    def save(self) -> None:
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
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
        return