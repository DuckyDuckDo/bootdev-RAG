from utils import *
import os
import pickle

class InvertedIndex:
    def __init__(self):
        self.index = {} # Dictionary that maps tokens to an array of documents
        self.docmap = {}# Dictionary that maps document ids to their objects
    
    def __add_document(self, doc_id, text):
        tokens = tokenize(text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = []
            self.index[token].append(doc_id)

    def get_documents(self, term):
        term = term.lower()
        return sorted(self.index[term])

    def build(self):
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            movies = data["movies"]
        
        for i, movie in enumerate(movies):
            input_text = f"{movie["title"]} {movie["description"]}"
            self.docmap[i+1] = movie
            self.__add_document(i+1, input_text)

    def save(self):
        save_path = "./cache/"
        if not os.path.isdir(save_path):
            os.mkdir(save_path)

        index_file = "./cache/index.pkl"
        docmap_file = "./cache/docmap.pkl"

        with open(index_file, 'wb') as file:
            pickle.dump(self.index, file, protocol = pickle.HIGHEST_PROTOCOL)
        with open(docmap_file, 'wb') as file:
            pickle.dump(self.docmap, file, protocol = pickle.HIGHEST_PROTOCOL)
        
