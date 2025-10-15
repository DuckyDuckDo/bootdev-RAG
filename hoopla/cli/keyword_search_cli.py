from utils import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help = "Build inverted index of movie docs")
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            # matching_movies = keyword_search_by_title(args.query)
            matching_movies = keyword_search_by_inverted_index(args.query)
            for title in (matching_movies):
                print(title)
        
        case "build":
            print(f"Building Inverted Index for all movies")
            build_command()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()