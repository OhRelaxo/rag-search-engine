import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, semantic_search, chunk_text, semantic_chunk_text, embed_chunks

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

    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks with optional overlap")
    chunk_parser.add_argument("text", type=str, help="the text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help="the character size per chunk, the default value is 200")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of words to overlap between chunks, the default value is 0" )

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text on sentence boundaries to preserve meaning")
    semantic_chunk_parser.add_argument("text", type=str, help="the text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="the maximum size per chunk, the default value is 4")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Number of words to overlap between chunks, the default value is 0")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate embeddings for chunked documents")

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
            semantic_search(args.query, args.limit)
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()