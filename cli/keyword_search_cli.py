import argparse
from lib.search import print_search_result
from lib.tf_idf import InvertedIndex
from lib.search import get_search_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="build an Inverse Index of the available movies")

    args = parser.parse_args()

    index = InvertedIndex()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = get_search_result(args.query, index)
            print_search_result(result)
        case "build":
            index.build()
            index.save()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
