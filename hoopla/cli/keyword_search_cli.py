import json
import argparse
import string
from nltk.stem import PorterStemmer

DATA_PATH = "./data/movies.json"
SEARCH_LIMIT = 5
STOP_WORDS_PATH = "./data/stopwords.txt"

def get_stop_words():
    """
    Loads stop words from a path
    """
    with open(STOP_WORDS_PATH, "r") as f:
        stop_words = f.read()
        stop_words = stop_words.splitlines()
    return stop_words

def has_matching_tokens(query_tokens, title_tokens):
    """
    Checks query and title tokens for a match
    """
    # Looks for one matching token as the comparison operator
    for token in query_tokens:
        for title_token in title_tokens:
            if token in title_token:
                return True
    return False

def remove_stop_words(token_list):
    """
    Removes stop words given a token list
    """
    stop_words = get_stop_words()
    result = []
    for word in token_list:
        if word not in stop_words:
            result.append(word)
    return result

def stem_tokens(token_list):
    """
    Stems all tokens from a token list
    """
    result = []
    stemmer = PorterStemmer()
    for token in token_list:
        result.append(stemmer.stem(token))
    return result

def keyword_search_by_title(query):
    """
    Opens and loads all the movie data and returns title based on search query and only returns exact matches   
    """
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # To get rid of punctuation, map each punctuation key to an empty string, and perform a translation/replacement
    punctuation_map = {
        punctuation: "" for punctuation in string.punctuation
    }
    translate_table = str.maketrans(punctuation_map)

    # Apply transformations to query
    query = query.translate(translate_table)
    query = query.lower()
    query_tokens = query.split()
    query_tokens = remove_stop_words(query_tokens)
    query_tokens = stem_tokens(query_tokens)
    
    results = []

    # Loops through movie in movies, applying each transformation of text processing before comparing query to movie title. 
    for movie in data["movies"]:
        # Transform the titles
        cleaned_title = movie["title"].translate(translate_table)
        cleaned_title = cleaned_title.lower()
        cleaned_tokens = cleaned_title.split()
        cleaned_tokens = remove_stop_words(cleaned_tokens)
        cleaned_tokens = stem_tokens(cleaned_tokens)
        
        if has_matching_tokens(query_tokens, cleaned_tokens):
            results.append(movie["title"])

        if len(results) >= SEARCH_LIMIT:
            return results
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            matching_movies = keyword_search_by_title(args.query)
            for i, title in enumerate(matching_movies):
                print(f"{i+1}. {title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()