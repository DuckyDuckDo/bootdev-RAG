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

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()