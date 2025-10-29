from hoopla.cli.lib.keyword_search import *

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

    tf_idf_parser = subparsers.add_parser("tfidf", help = "Calculate TF-IDF")
    tf_idf_parser.add_argument("doc_id", type = int, help = "document id")
    tf_idf_parser.add_argument("term", type = str, help = "Desired term to get frequency of")

    bm25tf_parser = subparsers.add_parser("bm25tf", help = "Get term frequency of a term from document id and desired term")
    bm25tf_parser.add_argument("doc_id", type = int, help = "document id")
    bm25tf_parser.add_argument("term", type = str, help = "Desired term to get frequency of")
    bm25tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25idf_parser = subparsers.add_parser("bm25idf", help = "Get inverse document frequency of a term")
    bm25idf_parser.add_argument("term", type = str, help = "Term for which to calculate inverse document frequency")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("limit", type = int, nargs = '?', default = 5, help = "Limits the result length of the Bm25 search")

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
            term_frequency = tf_command(args.doc_id, args.term)
            print(f"{args.term} appears {term_frequency} times")
        
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverted document frequency of term: {idf:.2f}")
        
        case "tfidf":
            tf_idf = tf_idf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25tf":
            bm25tf = bm25tf_command(args.doc_id, args.term)
            print(f"BM25TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25idf":
            bm25idf = bm25idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25search":
            top_x_bm25_scores = bm25search_command(args.query)
            for item in top_x_bm25_scores:
                print(f'{item[0]}: {item[1]:.2f}')

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()