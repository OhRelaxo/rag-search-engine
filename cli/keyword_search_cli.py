import argparse

from lib.keyword_search import command_search, command_tf, command_idf, command_tfidf, command_bm25idf, command_bm25search, command_bm25tf
from lib.inverted_index import InvertedIndex
from lib.utils import BM25_K1, BM25_B


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build an Inverse Index of the available movies")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given document ID and term")
    tf_parser.add_argument("doc_id", type=int, help="The document id")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF score for a given document ID and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Get the BM25-IDF score")
    bm25idf_parser.add_argument("term", type=str, help="Term to get BM25-IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    index = InvertedIndex()
    match args.command:
        case "search":
            command_search(args.query, index)
        case "build":
            index.build()
            index.save()
        case "tf":
            command_tf(args.doc_id, args.term, index)
        case "idf":
            command_idf(args.term, index)
        case "tfidf":
            command_tfidf(args.doc_id, args.term, index)
        case "bm25idf":
            command_bm25idf(args.term, index)
        case "bm25tf":
            command_bm25tf(args.doc_id, args.term, args.k1, args.b, index)
        case "bm25search":
            command_bm25search(args.query, index)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
