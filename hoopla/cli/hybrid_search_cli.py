import argparse
from lib.hybrid_search import *


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help = "Normalize scores into range 0-1 using min/max normalization")
    normalize_parser.add_argument("scores", nargs = "+", help = "Scores to normalize", type = float)

    weighted_parser = subparsers.add_parser("weighted-search", help = "Perform a weighted search combining BM25 and Semantic")
    weighted_parser.add_argument("query", help = "query to search")
    weighted_parser.add_argument("--limit", nargs = "?", default = 5, type = int)
    weighted_parser.add_argument("--alpha", nargs = "?", default = 0.5, type = float)

    rrf_parser = subparsers.add_parser("rrf-search", help = "Perform a RRF search Reciprocal Rank Fusion, normalizing scores by rank and not values")
    rrf_parser.add_argument("query", help = "query to search")
    rrf_parser.add_argument("--limit", nargs = "?", default = 5, type = int)
    rrf_parser.add_argument("--k", nargs = "?", default = 60, type = float)
    rrf_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="LLM Rerank Method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)

        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)

        case "rrf-search":
            method = args.enhance
            rerank_method = args.rerank_method
            # Based on different ways to enhance search query, perform different AI calls/functions
            # Then performs the rrf_search on the new query
            match method:
                case "spell":
                    enhanced_query = spellcheck_query(args.query)
                    if enhanced_query != args.query:
                        print( f"Enhanced query ({method}): '{args.query}' -> '{enhanced_query}'\n")
                
                case "rewrite":
                    enhanced_query = rewrite_query(args.query)
                    if enhanced_query != args.query:
                        print( f"Enhanced query ({method}): '{args.query}' -> '{enhanced_query}'\n")
                
                case "expand":
                    enhanced_query = expand_query(args.query)
                    if enhanced_query != args.query:
                        print( f"Enhanced query ({method}): '{enhanced_query}'\n")              

                case _:
                    enhanced_query = args.query
            
            rrf_results = rrf_search_command(enhanced_query, args.k, args.limit, rerank_method)
            format_rrf_results(rrf_results)
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()