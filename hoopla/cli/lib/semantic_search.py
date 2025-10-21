from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json

DATA_PATH = "././data/movies.json"

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        """
        Uses the model to get embedding for a input text sequence
        """
        encodings = self.model.encode([text])
        return encodings[0]
    
    def build_embeddings(self, documents):
        """
        Parses through all documents in our movies database, populates the maps, and generates embeddings for all of them
        """
        self.documents = documents
        movies_info = []
        for document in self.documents:
            self.document_map[document['id']] = document
            document_info = f"{document['title']}: {document['description']}"
            movies_info.append(document_info)
        embeddings = self.model.encode(movies_info, show_progress_bar = True)
        self.embeddings = embeddings
        np.save('cache/movie_embeddings.npy', self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for document in self.documents:
            self.document_map[document['id']] = document
        
        if os.path.exists('cache/movie_embeddings.npy'):
            embeddings = np.load('cache/movie_embeddings.npy')
            if len(embeddings) == len(documents):
                self.embeddings = embeddings
                return self.embeddings
        else:
            return self.build_embeddings(documents)
    
def verify_model():
    semantic_search_model = SemanticSearch()
    print(f"Model loaded: {semantic_search_model}")
    print(f"Max sequence length: {semantic_search_model.model.max_seq_length}")

def embed_text(text):
    semantic_search_model = SemanticSearch()
    embedding = semantic_search_model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search_model = SemanticSearch()
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]

    embeddings = semantic_search_model.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

