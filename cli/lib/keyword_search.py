from .inverted_index import InvertedIndex
from .text_processing import text_processing

def print_search_result(movie_list) -> None:
    for i, v in enumerate(movie_list, 1):
        print(f"{i}. {v["title"]}, id: {v["id"]}")
    return

def get_search_result(query: str, index: InvertedIndex):
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

def command_search(query: str, index: InvertedIndex) -> None:
    index.load()
    print(f"Searching for: {query}")
    result = get_search_result(query, index)
    print_search_result(result)
    return

def command_tf(doc_id: int, term: str, index: InvertedIndex) -> None:
    index.load()
    tf = index.get_tf(doc_id, term)
    print(f"Term frequency of '{term}' in document '{doc_id}': {tf}")
    return

def command_idf(term: str, index : InvertedIndex) -> None:
    index.load()
    idf = index.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")
    return

def command_tfidf(doc_id: int, term: str, index: InvertedIndex) -> None:
    index.load()
    tf = index.get_tf(doc_id, term)
    idf = index.get_idf(term)
    tf_idf = tf * idf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
    return

def command_bm25idf(term: str, index: InvertedIndex) -> None:
    index.load()
    bm25idf = index.get_bm25_idf(term)
    print(f"BM25-IDF score of '{term}': {bm25idf:.2f}")
    return

def command_bm25tf(doc_id: int, term: str, k1: float, b: float, index: InvertedIndex) -> None:
    index.load()
    bm25tf = index.get_bm25_tf(doc_id, term, k1, b)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")

def command_bm25search(query: str, index: InvertedIndex) -> None:
    index.load()
    search_result = index.bm25_search(query, 5)
    for i, id_score in enumerate(search_result, 1):
        doc_id = id_score[0]
        score = id_score[1]
        movie = index.docmap[doc_id]
        title = movie["title"]
        print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")
    return