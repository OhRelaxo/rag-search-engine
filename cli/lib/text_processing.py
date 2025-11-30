from string import punctuation
from nltk.stem import PorterStemmer

from .utils import get_stop_words

def text_processing(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", punctuation))

    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)

    stop_words = get_stop_words()
    filtered_tokens = []
    for token in valid_tokens:
        if token not in stop_words:
            filtered_tokens.append(token)

    stemmer = PorterStemmer()
    stemmed_tokens = []
    for word in tokens:
        stemmed_tokens.append(stemmer.stem(word))
    return stemmed_tokens
