import os.path

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

def verify_embeddings():
    model = SemanticSearch()
    documents = get_movies()
    embeddings = model.load_or_create_embeddings(documents["movies"])
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")