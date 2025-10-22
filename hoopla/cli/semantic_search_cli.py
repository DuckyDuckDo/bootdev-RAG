import argparse
from lib.semantic_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help = "Verify installation and initiation of sentence transformer model")

    embed_text_parser = subparsers.add_parser("embed_text", help = "Embed the text into a vector space using preloaded model")
    embed_text_parser.add_argument("text", help = "Text of which we want to generate embedding for")

    subparsers.add_parser("verify_embeddings", help = "Verifies that the model has proper embeddings for our movies")

    embed_query_parser = subparsers.add_parser("embedquery", help = "Embeds the search query for similarity matching")
    embed_query_parser.add_argument("query", help = "query for which the model can get embedding for")

    search_parser = subparsers.add_parser("search", help = "command to perform a search on the embeddings")
    search_parser.add_argument("query", help = "text of which to search for matches")
    search_parser.add_argument("--limit", type = int, nargs = '?', default = 5, help = "limit for the number of movies returned")

    chunk_parser = subparsers.add_parser("chunk", help = "Chunks input text into chunks of input_size")
    chunk_parser.add_argument("text", help = "Text to chunk")
    chunk_parser.add_argument("--chunk-size", type = int, nargs = '?', default = 200)
    chunk_parser.add_argument("--overlap", type = int, nargs = '?', default = 0)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help = "Semantically chunks input text")
    semantic_chunk_parser.add_argument("text", help = "Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type = int, nargs = '?', default = 200)
    semantic_chunk_parser.add_argument("--overlap", type = int, nargs = '?', default = 0)

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        
        case "embed_text":
            embed_text(args.text)
        
        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embedquery(args.query)
        
        case "search":
            search(args.query, args.limit)
        
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()