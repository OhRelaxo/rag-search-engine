#!/usr/bin/env python3

import argparse

from keyword_search.search import get_search_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = get_search_result(args.query)
            result.sort_by_id()
            result.truncate_by_five()
            result.print_search_result()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
