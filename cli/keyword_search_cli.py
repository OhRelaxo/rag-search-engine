import argparse
import math

from lib.text_processing import text_processing
from lib.search import print_search_result
from lib.inverted_index import InvertedIndex
from lib.search import get_search_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build an Inverse Index of the available movies")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given document ID and term")
    tf_parser.add_argument("doc_id", type=int, help="The document id")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="")
    idf_parser.add_argument("term", type=str, help="")

    args = parser.parse_args()

    index = InvertedIndex()
    match args.command:
        case "search":
            index.load()
            print(f"Searching for: {args.query}")
            result = get_search_result(args.query, index)
            print_search_result(result)
        case "build":
            index.build()
            index.save()
        case "tf":
            if not args.doc_id or not args.term:
                print("error for using the tf command you need a doc_id and a term!")
                exit(1)
            index.load()
            tf = index.get_tf(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")
        case "idf":
            index.load()
            processed_term = text_processing(args.term)
            doc_count = len(index.docmap)
            term_doc_count = len(index.index.get(processed_term[0], set()))
            idf = math.log((doc_count + 1) / (term_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
