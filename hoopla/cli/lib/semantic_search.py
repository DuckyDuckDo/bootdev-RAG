from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import re

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
        """
        Either loads in the embeddings of the movies dataset, or calls build_embeddings to create them for first time
        """
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
    
    def search(self, query, limit):
        """
        Performs a search of the query across the embeddings in the model
        """
        # Checks for embeddings to exist
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        # Get the query embedding
        query_embedding = self.generate_embedding(query)

        # Get the similarity scores, doc_id is decremented b/c movie_id starts at 1, but python indexing is at 0
        similarity_scores = []
        for doc in self.documents:
            similarity_scores.append((cosine_similarity(query_embedding, self.embeddings[doc['id'] - 1]), doc))
        
        # Grab the top_x_scores
        top_x_scores = sorted(similarity_scores, key = lambda x: x[0], reverse = True)
        final_result = []
        
        # Build out the final result
        for score, doc in top_x_scores:
            if len(final_result) >= limit:
                break
            final_result.append({'score': score, 
                                 'title': doc['title'], 
                                 'description': doc['description']
                                 })
        return final_result
        

#### UTIL FUNCTION FOR SIMILARITY MATCHING
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
    
# HIGHER LEVEL COMMANDS CALLED ON SEMANTIC SEARCH CLI
def verify_model():
    """
    Checks that the model works
    """
    semantic_search_model = SemanticSearch()
    print(f"Model loaded: {semantic_search_model}")
    print(f"Max sequence length: {semantic_search_model.model.max_seq_length}")

def embed_text(text):
    """
    Similar to embedquery
    """
    semantic_search_model = SemanticSearch()
    embedding = semantic_search_model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    """
    Called from CLI, checks for embeddings in the dataset
    """
    semantic_search_model = SemanticSearch()
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]

    embeddings = semantic_search_model.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embedquery(query):
    """
    Embeds the search query which can then be used for similarity matching to the rest of the embedded database
    """
    semantic_search_model = SemanticSearch()
    embedding = semantic_search_model.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def search(query, limit):
    semantic_search_model = SemanticSearch()
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]
    semantic_search_model.load_or_create_embeddings(movies)
    top_matches = semantic_search_model.search(query, limit)
    for movie in top_matches:
        print(f"{movie['title']} (score: {movie['score']})")



### Word sized Chunking
def fixed_size_chunking(text, chunk_size):
    """
    Chunks text with a fixed size
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
    return chunks

def overlap_chunking(text, chunk_size, overlap):
    """
    Chunking with potential overlap between start and end of each chunk
    """
    words = text.split()
    chunks = []

    # Generate the chunks by looping through indexes which are incremented by chunk_size
    i = 0
    while i < len(words) - overlap:
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap

    return chunks

def chunk_text(text, chunk_size, overlap):
    """
    Performs the chunking based on the mode we pass in 
    """
    print(f"Chunking {len(text)} characters")
    if not overlap:
        chunks = fixed_size_chunking(text, chunk_size)
    else:
        chunks = overlap_chunking(text, chunk_size, overlap)
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

### Semantic Chunking
def semantic_chunking(text, max_chunk_size, overlap):   
    """
    Semantic chunking based on sentences
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    i = 0
    chunks = []
    while i < len(sentences) - overlap:
        chunk = " ".join(sentences[i: i + max_chunk_size])
        chunks.append(chunk)
        i += max_chunk_size - overlap
    return chunks


def semantic_chunk_text(text, max_chunk_size, overlap):
    """
    Calls the semantic chunking function and prints the chunks to the terminal
    """
    print(f"Semantically chunking {len(text)} characters")
    chunks = semantic_chunking(text, max_chunk_size, overlap)
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")