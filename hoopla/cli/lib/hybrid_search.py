from lib.keyword_search import *
from lib.semantic_search import ChunkedSemanticSearch
import os

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        self.idx.load()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha = 0.5, limit=5):
        """
        Performs both BM25 and Semantic Search and store results as a weighted sum of the two scores
        """
        top_bm_scores = self.idx.bm25search(query, limit) # array of [(id, score)] pairs
        top_semantic_scores = self.semantic_search.search_chunks(query, limit) # array of dictionaries with id, score, description, title
        score_mapping = {i + 1: {"BM25": 0, "Semantic": 0} for i in range(len(self.documents))} # maps movie id to a dictionary of their scores

        bm_scores = normalize_command([bm_score[1] for bm_score in top_bm_scores])
        semantic_scores = normalize_command([movie["score"] for movie in top_semantic_scores])

        bm_scores = [(item[0], bm_scores[i]) for i, item in enumerate(top_bm_scores)]
        semantic_scores = [(movie["id"], semantic_scores[i]) for i, movie in enumerate(top_semantic_scores)]

        for id, score in bm_scores:
            score_mapping[id]["BM25"] = score
        for id, score in semantic_scores:
            score_mapping[id]["Semantic"] = score

        results = []
        for id in score_mapping:
            movie_data = self.semantic_search.document_map[id]
            title, description = movie_data["title"], movie_data["description"]
            hybrid_score = alpha * score_mapping[id]["BM25"] + (1-alpha) * score_mapping[id]["Semantic"]
            movie_result = {
                "id": id, 
                "title": title,
                "description": description,
                "BM25": score_mapping[id]["BM25"],
                "Semantic": score_mapping[id]["Semantic"],
                "hybrid": hybrid_score
            }
            results.append(movie_result)
        results = sorted(results, key=lambda d: d['hybrid'], reverse = True)[:limit]
        return results


    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def normalize_command(scores):
    """
    Given a list of scores, apply min-max normalization so that bm25 search scores are weighted equally against semantic search scores
    """
    if not scores:
        return 
    
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return [1.0] * len(scores)
    
    normalized = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized

def weighted_search_command(query, alpha = 0.5, limit = 5):
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]
    hybrid_search = HybridSearch(movies)
    top_search_results = hybrid_search.weighted_search(query, alpha, limit)
    for i, movie_data in enumerate(top_search_results):
        print(f"{i+1}. {movie_data["title"]}")
        print(f"Hybrid Score: {movie_data["hybrid"]:.4f}")
        print(f"BM25: {movie_data["BM25"]:.4f}, Semantic: {movie_data["Semantic"]:.4f}")

    