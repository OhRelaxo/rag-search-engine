import json
import os

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")

def get_movies():
    with open(DATA_PATH, "r") as f:
        movies = json.load(f)
    return movies

def get_stop_words() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        text = f.read()
        return text.splitlines()