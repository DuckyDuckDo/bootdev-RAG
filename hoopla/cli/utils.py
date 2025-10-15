import json
import argparse
import string
from nltk.stem import PorterStemmer
import os
import pickle
import json
import math
from collections import defaultdict


######## Inverted Index Class ##########
class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)  # Dictionary that maps tokens to docs they appear in
        self.docmap: dict[int, dict] = {}# Dictionary that maps document ids to their objects
        self.term_frequencies = {}# Dictionary of dictionaries mapping document id to a frequency dictionary of the tokens in that document
    
    def __add_document(self, doc_id, text):
        """
        Add all text of a document into the index
        """
        tokens = tokenize(text)
        self.term_frequencies[doc_id] = {}
        for token in set(tokens):
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] = self.term_frequencies[doc_id].get(token, 0) + 1

    def get_documents(self, term):
        """
        Retrieves the document list if the term exists in index
        """
        term = term.lower()
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id, term):
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term):
        tokens = tokenize(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def build(self):
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            movies = data["movies"]
        
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self):
        save_path = "./cache/"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        index_file = "./cache/index.pkl"
        docmap_file = "./cache/docmap.pkl"
        tf_file = "./cache/term_frequencies.pkl"

        with open(index_file, 'wb') as file:
            pickle.dump(self.index, file, protocol = pickle.HIGHEST_PROTOCOL)
        with open(docmap_file, 'wb') as file:
            pickle.dump(self.docmap, file, protocol = pickle.HIGHEST_PROTOCOL)
        with open(tf_file, "wb") as file:
            pickle.dump(self.term_frequencies, file, protocol = pickle.HIGHEST_PROTOCOL)
        
    
    def load(self):
        try:
            with open("./cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)
            with open("./cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)
            with open("./cache/term_frequencies.pkl", "rb") as file:
                self.term_frequencies = pickle.load(file)
        except:
            raise Exception("Cache files not found/does not exist, build up a new index first")
        
###### CONSTANTS #######
DATA_PATH = "./data/movies.json"
SEARCH_LIMIT = 5
STOP_WORDS_PATH = "./data/stopwords.txt"

###### Helper Functions for Tokenization ########
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

######## Commands called from CLI ############
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

def keyword_search_by_inverted_index(query):
    """
    Searches for movies through each query token using inverted index
    """
    index = InvertedIndex()
    try:
        index.load()
    except:
        raise Exception("Could not load index")
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

def build_command():
    """
    Builds out the inverted index with a test case to verify should return document 4651
    """
    index = InvertedIndex()
    index.build()
    index.save()

    # TEST CASE
    docs_with_merida = index.index["merida"]
    print(f"First document for token 'merida' = {list(docs_with_merida)[0]}")

def get_tf_command(doc_id, term):
    """
    Gets term frequency from a given doc_id and term
    """
    index = InvertedIndex()
    try:
        index.load()
    except:
        raise Exception("Could not load index")
    return index.term_frequencies[doc_id].get(term, 0)

def idf_command(term):
    index = InvertedIndex()
    try:
        index.load()
    except:
        raise Exception("Could not load index")
    return index.get_idf(term)
    
