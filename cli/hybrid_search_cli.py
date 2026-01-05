import argparse
from lib.hybrid_search import normalize_scores, weighted_search, rrf_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="get a normalized BM25 score based on the inputted data")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="the query to search for")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="an optional parameter to control the alpha of the weighted search")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="an optional parameter to set the limit of the output")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform a Reciprocal Rank Fusion (rrf) search")
    rrf_search_parser.add_argument("query", type=str, help="the query to search for")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="on optional parameter to set the k parameter, the default is 60")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="an optional parameter to set the limit of the output")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            if args.scores:
                normalized = normalize_scores(args.scores)
                for i, score in enumerate(normalized, 1):
                    print(f"{i}. {score:.4f}")
        case "weighted-search":
                result = weighted_search(args.query, args.alpha, args.limit)
                for i, (doc_id, data) in enumerate(result, 1):
                    print(f"{i}. {data["title"]}")
                    print(f"Hybrid Score: {data["hybrid_score"]:.3f}")
                    print(f"BM25: {data["keyword_score"]:.3f}, Semantic: {data["semantic_score"]:.3f}")
                    print(f"{data["document"]}...")
        case "rrf-search":
            result = rrf_search(args.query, args.k, args.limit)
            for i, (doc_id, data) in enumerate(result, 1):
                bm25_rank = data.get("bm25_rank")
                semantic_rank = data.get("semantic_rank")

                if not bm25_rank:
                    bm25_rank = "-"
                if not semantic_rank:
                    semantic_rank = "-"

                print(f"{i}. {data["title"]}")
                print(f"RRF Score: {data["rrf_score"]:.3f}")
                print(f"BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                print(f"{data["document"]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()