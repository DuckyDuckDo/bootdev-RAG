from InvertedIndex import *
from utils import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help = "Build inverted index of movie docs")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            matching_movies = keyword_search_by_title(args.query)
            for i, title in enumerate(matching_movies):
                print(f"{i+1}. {title}")
        
        case "build":
            index_class = InvertedIndex()
            index_class.build()
            index_class.save()
            
            # Test Case
            docs_with_merida = index_class.index["merida"]
            print(f"First document for token 'merida' = {docs_with_merida[0]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()