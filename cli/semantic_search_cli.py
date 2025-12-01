import argparse

from lib.semantic_search import SemanticSearch
from lib.utils import get_movies
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="verifies the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings for the movie dataset")

    embedquery_parser = subparsers.add_parser("embedquery", help="Generate an embedding for a query")
    embedquery_parser.add_argument("query", type=str, help="query to embed")

    search_parser = subparsers.add_parser("search", help="Search for movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, required=False, help="optional parameter to specify search limit, default is 5")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            model = SemanticSearch()
            movies = get_movies()
            model.load_or_create_embeddings(movies["movies"])
            result = model.search(args.query, args.limit)
            for i, movie in enumerate(result, 1):
                print(f"{i}. {movie["title"]} (score: {movie["score"]:.4f})\n{movie["description"]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()