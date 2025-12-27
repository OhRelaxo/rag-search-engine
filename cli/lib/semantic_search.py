import os.path
import re

from sentence_transformers import SentenceTransformer
import numpy as np

from .utils import CACHE_PATH, get_movies

class SemanticSearch:
    def __init__(self):
        self.model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.movie_embeddings_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")
    
    def generate_embedding(self, text: str):
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

def chunk(text: str, chunk_size: int, overlap: int) -> None:
    parts = text.split()
    print(f"Chunking {len(text)} characters")
    chunking(parts, chunk_size, overlap)
    return

def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> None:
    parts = re.split(r"(?<=[.!?])\s+", text)
    print(f"Semantically chunking {len(text)} characters")
    chunking(parts, max_chunk_size, overlap)
    return

def chunking(parts: list[str], chunk_size: int, overlap: int) -> None:
    i = 1
    while len(parts) >= 1:
        part = parts[:chunk_size]
        text = " ".join(part)
        print(f"{i}. {text}")
        parts = parts[chunk_size - overlap:]
        i += 1
    return
