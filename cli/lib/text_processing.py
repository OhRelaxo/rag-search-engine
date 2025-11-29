from string import punctuation
from utils import get_stop_words
from nltk.stem import PorterStemmer

def filter_on_stop_words(tokens: list[str]) -> list[str]:
    stop_words = get_stop_words()
    new_tokens: list[str] = []
    for t in tokens:
        if t in stop_words:
            continue
        new_tokens.append(t)
    return new_tokens

def compare(query: list[str], title: list[str]) -> bool:
    stemmer = PorterStemmer()
    for q in query:
        sq = stemmer.stem(q, False)
        for t in title:
            st = stemmer.stem(t, False)
            if sq in st:
                return True
    return False

def remove_punctuation(text: str) -> str:
    new_text = ""
    for v in text:
        if v in punctuation:
            continue
        new_text += v
    return new_text


def tokenized_text(text: str) -> list[str]:
    return text.split(" ")

def text_processing(text: str) -> list[str]:
    lower_text = text.lower()
    no_punctuation = remove_punctuation(lower_text)
    tokenized = tokenized_text(no_punctuation)
    return filter_on_stop_words(tokenized)
