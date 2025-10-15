from utils import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help = "Build inverted index of movie docs")

    tf_parser = subparsers.add_parser("tf", help = "Get term frequency of a term from document id and desired term")
    tf_parser.add_argument("doc_id", type = int, help = "document id")
    tf_parser.add_argument("term", type = str, help = "Desired term to get frequency of")

    idf_parser = subparsers.add_parser("idf", help = "Get inverse document frequency of a term")
    idf_parser.add_argument("term", type = str, help = "Term for which to calculate inverse document frequency")

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
        
        case "tf":
            term_frequency = get_tf_command(args.doc_id, args.term)
            print(f"{args.term} appears {term_frequency} times")
        
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverted document frequency of term: {idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()