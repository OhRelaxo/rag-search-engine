from .tf_idf import InvertedIndex
from .text_processing import text_processing


def print_search_result(movie_list) -> None:
    for i, v in enumerate(movie_list, 1):
        print(f"{i}. {v["title"]}, id: {v["id"]}")
    return

def get_search_result(query: str, index: InvertedIndex):
    index.load()
    tokens = text_processing(query)

    result = []
    seen_ids = []
    for token in tokens:
        doc_ids: list[int] = index.get_documents(token)
        for doc_id in doc_ids:
            if len(result) >= 5:
                break
            if doc_id in seen_ids:
                continue
            movie = index.get_movie(doc_id)
            result.append(movie)
            seen_ids.append(doc_id)
        if len(result) >= 5:
            break
    return result