from string import punctuation
from .utils import get_stop_words
from nltk.stem import PorterStemmer

def filter_on_stop_words(tokens: list[str]) -> list[str]:
    stop_words = get_stop_words()
    new_tokens: list[str] = []
    for t in tokens:
        if t in stop_words:
            continue
        new_tokens.append(t)
    return new_tokens

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    stemmed_tokens = []
    for word in tokens:
        stemmed_tokens.append(stemmer.stem(word))
    return stemmed_tokens

def text_processing(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))
    tokenized = text.split(" ")
    filtered_tokens = filter_on_stop_words(tokenized)
    stemmed_tokens = stem_tokens(filtered_tokens)
    return stemmed_tokens
