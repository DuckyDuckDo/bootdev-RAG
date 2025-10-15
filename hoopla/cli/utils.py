import json
import argparse
import string
from nltk.stem import PorterStemmer
from InvertedIndex import *

DATA_PATH = "./data/movies.json"
SEARCH_LIMIT = 5
STOP_WORDS_PATH = "./data/stopwords.txt"

def translate_and_lower(text):
    """
    Gets rid of punctuation in text and lower cases the text
    """
    # To get rid of punctuation, map each punctuation key to an empty string, and perform a translation/replacement
    punctuation_map = {
        punctuation: "" for punctuation in string.punctuation
    }
    translate_table = str.maketrans(punctuation_map)
    return text.translate(translate_table)

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

def tokenize(text):
    """
    Applies functions to a body of text and returns the final array of tokens
    """
    text = translate_and_lower(text)
    tokens = text.split()
    tokens = stem_tokens(remove_stop_words(tokens))
    return tokens

def keyword_search_by_title(query):
    """
    Opens and loads all the movie data and returns title based on search query and only returns exact matches   
    """
    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    # # Apply transformations to query
    query_tokens = tokenize(query)
    results = []

    # Loops through movie in movies, applying each transformation of text processing before comparing query to movie title. 
    for movie in data["movies"]:
        # Transform the titles
        cleaned_tokens = tokenize(movie["title"])
        if has_matching_tokens(query_tokens, cleaned_tokens):
            results.append(movie["title"])

        if len(results) >= SEARCH_LIMIT:
            return results
    return results

def keyword_search_by_inverted_index(query, index):
    """
    Searches for movies through each query token using inverted index
    """
    query_tokens = tokenize(query)
    results = set()

    # Loops through each token in query
    for token in query_tokens:
        # Gets all documents containing the token
        docs_with_tokens = index.index.get(token, [])
        for doc in docs_with_tokens:
            if len(results) == SEARCH_LIMIT:
                break
            results.add(index.docmap[doc]["title"])

    return results