from pathlib import Path
from utils import get_movies, CACHE_PATH, INDEX_PATH, DOCMAP_PATH, save_obj
from search import Movie

class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = {}  # tokens (lsit[str]): IDs sets of int -> {int, int, int}
        self.docmap: dict[int, Movie] = {}  # IDs (int): Documetn (obj)

    def __add_document(self, doc_id: int, text: str) -> None:
        lower = text.lower()
        tokens = lower.split(" ")
        for string in tokens:
            value = self.index[string]
            value.add(doc_id)
            self.index[string] = value
        return

    def get_documents(self, term: str) -> list[int]:
        value = self.index[term.lower()]
        return sorted(value)

    def build(self) -> None:
        movies = get_movies()
        for v in movies:
            new_movie = Movie(v["id"], v["title"], v["description"])
            self.docmap[new_movie.id] = new_movie
            self.__add_document(new_movie.id, f"{new_movie.title} {new_movie.description}")
        return

    def save(self) -> None:
        Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)
        save_obj(self.index, INDEX_PATH)
        save_obj(self.docmap, DOCMAP_PATH)
        return