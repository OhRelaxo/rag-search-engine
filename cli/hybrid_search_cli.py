import argparse
from lib.hybrid_search import normalize_scores, weighted_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="get a normalized BM25 score based on the inputted data")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="")
    weighted_search_parser.add_argument("query", type=str, help="the query to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="an optional parameter to control the alpha of the weighted search")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="an optional parameter to set the limit of the output")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.scores:
                normalized = normalize_scores(args.scores)
                for i, score in enumerate(normalized, 1):
                    print(f"{i}. {score:.4f}")
        case "weighted-search":
                weighted_search(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()