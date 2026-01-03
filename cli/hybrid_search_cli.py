import argparse
from lib.hybrid_search import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="get a normalized BM25 score based on the inputted data")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.scores:
                normalized = normalize_scores(args.scores)
                for i, score in enumerate(normalized, 1):
                    print(f"{i}. {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()