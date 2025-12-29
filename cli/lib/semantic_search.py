import json
import os.path
import re

from sentence_transformers import SentenceTransformer
import numpy as np

from . import utils
from .utils import CACHE_PATH, get_movies

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.movie_embeddings_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or text.isspace():
            raise ValueError("error in class SemanticSearch in method generate_embedding: text is either empty or just whitespace!")
        return self.model.encode([text])[0]

    def search(self, query: str, limit: int) -> list[dict[str, str | float]]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        embedding = self.generate_embedding(query)
        similarities: list[tuple[float, dict[str, float | str]]] = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))
        similarities.sort(key=lambda x: x[0], reverse=True)

        result: list[dict[str, str | float]] = []
        for score, doc in similarities:
            result.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return result[:limit]

    def build_embeddings(self, documents: list[dict]):
        self.documents = documents
        self.document_map = {}
        movie_title_desc = []
        for movie in self.documents:
            self.document_map[movie["id"]] = movie
            movie_title_desc.append(f"{movie["title"]} {movie["description"]}")
        self.embeddings = self.model.encode(movie_title_desc, show_progress_bar=True)

        os.makedirs(os.path.dirname(self.movie_embeddings_path), exist_ok=True)
        np.save(self.movie_embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for movie in self.documents:
            self.document_map[movie["id"]] = movie

        if os.path.exists(self.movie_embeddings_path):
            self.embeddings = np.load(self.movie_embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_embeddings_path = os.path.join(CACHE_PATH, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_PATH, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for movie_idx, movie in enumerate(self.documents):
            description = movie["description"]
            if not description:
                continue
            current_chunks = semantic_chunk(description, 4, 1)
            for chunk_idx, chunks in enumerate(current_chunks):
                all_chunks.append(chunks)
                chunk_metadata.append({"movie_idx": movie_idx, "chunk_idx": chunk_idx, "total_chunks": len(current_chunks)})

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(self.chunk_embeddings_path), exist_ok=True)
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        os.makedirs(os.path.dirname(self.chunk_metadata_path), exist_ok=True)

        with open(self.chunk_metadata_path, "w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                chunk_metadata = json.load(f)
                self.chunk_metadata = chunk_metadata["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        embeddings = self.generate_embedding(query)
        chunk_score: list[dict] = []

        # cosine_similarity = dot_product(A, B) / (magnitude(A) × magnitude(B))
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            similarity = cosine_similarity(embeddings, chunk_embedding)
            metadata = self.chunk_metadata[i]
            chunk_score.append({"chunk_idx": i, "movie_idx": metadata["movie_idx"], "score": similarity})

        movie_score = {} # movie_idx: cosine_similarity
        for c_score in chunk_score:
            if movie_score.get(c_score["movie_idx"]) is None or c_score["score"] > movie_score.get(c_score["movie_idx"]):
                movie_score[c_score["movie_idx"]] = c_score["score"]

        sorted_movie_score = sorted(movie_score.items(), key=lambda items: items[1], reverse=True)
        sorted_movie_score = sorted_movie_score[:limit]
        print(sorted_movie_score)
        result: list[dict] = []

        for s in sorted_movie_score:
            doc = self.documents[s[0]]
            doc_id = doc["id"]
            title = doc["title"]
            document = doc["description"]
            score = s[1]
            metadata = self.chunk_metadata[s[0]]
            result.append({"id": doc_id, "title": title, "document": document[:100], "score": score, "metadata": metadata})
        return result

def verify_model() -> None:
    model = SemanticSearch()
    print(f"Model loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")
    return

def embed_text(text: str) -> None:
    model = SemanticSearch()
    embedding = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    return

def verify_embeddings() -> None:
    model = SemanticSearch()
    documents = get_movies()
    embeddings = model.load_or_create_embeddings(documents["movies"])
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    return

def embed_query_text(query: str) -> None:
    model = SemanticSearch()
    embedding = model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")
    return

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def semantic_search(query: str, limit: int) -> None:
    model = SemanticSearch()
    movies = get_movies()
    model.load_or_create_embeddings(movies["movies"])
    result = model.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(result)} results:")
    print()

    for i, movie in enumerate(result, 1):
        print(
            f"{i}. {movie["title"]} (score: {movie["score"]:.4f})\n"
            f"{movie["description"][:100]}...\n"
        )
    return

def fixed_size_chunking(text: str, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
    return

def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if text is None:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        return sentences

    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break

        filtered_chunks = []
        for sentence in chunk_sentences:
            sentence = sentence.strip()
            if sentence:
               filtered_chunks.append(sentence)

        chunks.append(" ".join(filtered_chunks))
        i += max_chunk_size - overlap
    return chunks


def semantic_chunk_text(text: str, max_chunk_size: int, overlap: int) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def embed_chunks() -> None:
    movies = utils.get_movies()
    search = ChunkedSemanticSearch()
    embeddings = search.load_or_create_chunk_embeddings(movies["movies"])
    print(f"Generated {len(embeddings)} chunked embeddings")
    return

def search_chunked(query: str, limit: int):
    movies = utils.get_movies()
    search = ChunkedSemanticSearch()
    search.load_or_create_chunk_embeddings(movies["movies"])
    result = search.search_chunks(query, limit)
    for i, r in enumerate(result, 1):
        print(f"\n{i}. {r["title"]} (score: {r["score"]:.4f})")
        print(f"   {r["document"]}...")
    return
