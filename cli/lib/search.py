from utils import get_movies

from text_processing import compare, text_processing

class Movie:
    def __init__(self, id: int, title: str, description: str) -> None:
        self.id = id
        self.title = title
        self.description = description

    def __repr__(self) -> str:
        return f"{self.id}: {self.title}"

class Movies:
    def __init__(self) -> None:
        self._movies_list: list[Movie] = []

    def append(self, movie: Movie) -> None:
        self._movies_list.append(movie)

    def __getitem__(self, index: int) -> Movie:
        return self._movies_list[index]

    def __setitem__(self, index: int, value: Movie) -> None:
        self._movies_list[index] = value

    def __len__(self) -> int:
        return len(self._movies_list)

    def __repr__(self) -> str:
        return f"{self._movies_list}"

    def print_search_result(self) -> None:
        for i, v in enumerate(self._movies_list, 1):
            print(f"{i}. {v.title}")
        return

    def sort_by_id(self) -> None:
        self._movies_list = sorted(self._movies_list, key=lambda x: x.id)
        return

    def truncate_by_five(self) -> None:
        self._movies_list = self._movies_list[:5]
        return

def get_search_result(query: str) -> Movies:
    movies = get_movies()

    filtered_query = text_processing(query)
    result = Movies()
    for v in movies["movies"]:
        title = text_processing(v["title"])
        if compare(filtered_query, title):
            result.append(Movie(v["id"], v["title"], v["description"]))

    return result