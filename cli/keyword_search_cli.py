#!/usr/bin/env python3

import argparse
from lib.tf_idf import InvertedIndex
from lib.search import get_search_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="build an Inverse Index of the available movies")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = get_search_result(args.query)
            result.sort_by_id()
            result.truncate_by_five()
            result.print_search_result()
        case "build":
            newIndex = InvertedIndex()
            newIndex.build()
            newIndex.save()
            docs = newIndex.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
