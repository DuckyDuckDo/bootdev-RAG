from lib.keyword_search import *
from lib.semantic_search import ChunkedSemanticSearch
import os
from google import genai
from dotenv import load_dotenv
import time
from sentence_transformers import CrossEncoder

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")

####### HYBRID SEARCH CLASS ##########

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
        # Calls BM25 and Semantic Search with the desired query
        top_bm_scores = self.idx.bm25search(query, limit * 500) # array of [(id, score)] pairs
        top_semantic_scores = self.semantic_search.search_chunks(query, limit * 500) # array of dictionaries with id, score, description, title
        
        # Initialize a mapping of movie_id to their respective scores
        score_mapping = {i + 1: {"BM25": 0, "Semantic": 0} for i in range(len(self.documents))} # maps movie id to a dictionary of their scores

        # Normalize the scores
        bm_scores = normalize_command([bm_score[1] for bm_score in top_bm_scores])
        semantic_scores = normalize_command([movie["score"] for movie in top_semantic_scores])

        # Standardize the scores arrays to just id, score pairs
        bm_scores = [(item[0], bm_scores[i]) for i, item in enumerate(top_bm_scores)]
        semantic_scores = [(movie["id"], semantic_scores[i]) for i, movie in enumerate(top_semantic_scores)]

        # Update the score mapping
        for id, score in bm_scores:
            score_mapping[id]["BM25"] = score
        for id, score in semantic_scores:
            score_mapping[id]["Semantic"] = score

        # Compile into results
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

    def get_rrf_score(self, rank, k):
        """
        Formula for getting RRF scores
        """
        return 1 / (rank + k)

    def rrf_search(self, query, k, limit=10, rerank = False):
        """
        Performs hybrid search but with the reciprocal rank fusion (RRF) metric to order scores instead of normalization. 
        Returns an [] of {} with top search results
        """
        # Call BM25 and Semantic
        top_bm_scores = self.idx.bm25search(query, limit * 5) # array of [(id, score)] pairs
        top_semantic_scores = self.semantic_search.search_chunks(query, limit * 5) # array of dictionaries with id, score, description, title
       
        # Initialize the Mapping
        rrf_score_mapping = {i + 1: 0 for i in range(len(self.documents))} # maps   movie id to a dictionary of their scores

        bm_rrfs = [(item[0], self.get_rrf_score(i + 1, k)) for i, item in enumerate(top_bm_scores)]
        semantic_rrfs = [(movie["id"], self.get_rrf_score(i + 1, k)) for i, movie in enumerate(top_semantic_scores)]

        for id, score in bm_rrfs:
            rrf_score_mapping[id] += score
        for id, score in semantic_rrfs:
            rrf_score_mapping[id] += score
        
        # Compile into results
        results = []
        for id in rrf_score_mapping:
            movie_data = self.semantic_search.document_map[id]
            title, description = movie_data["title"], movie_data["description"]
            movie_result = {
                "id": id, 
                "title": title,
                "description": description,
                "score": rrf_score_mapping[id]
            }
            results.append(movie_result)
        results = sorted(results, key=lambda d: d['score'], reverse = True)[:limit*5]

        # If reranking is selected, rerank calling rerank_results which leads to LLM generation or CrossEncoder model calls
        if rerank == "individual":
            results = sorted(individual_rerank_results(query, results), key = lambda d: d["score"], reverse = True)

        if rerank == "batch":
            results = batch_rerank_results(query, results)
        
        if rerank == "cross_encoder":
            results = cross_encoder_rerank_results(query, results)


        return results[:limit]

##### COMMANDS MAPPING TO COMMAND LINE ARGUMENTS #######
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
    
def rrf_search_command(query, k = 60, limit = 5, rerank = False):
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]
    hybrid_search = HybridSearch(movies)
    top_search_results = hybrid_search.rrf_search(query, k, limit, rerank)
    for i, movie_data in enumerate(top_search_results):
        print(f"{i+1}. {movie_data["title"]}")
        print(f"RRF Score: {movie_data["score"]:.4f}")

def spellcheck_query(query):
    """
    Calls Google Gemini model to enhance the query that we pass in through proper spell-checking
    """
    client = genai.Client(api_key = API_KEY)
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001", 
        contents = f"""Fix any spelling errors in this movie search query.
                        Only correct obvious typos. Don't change correctly spelled words.
                        Query: "{query}"
                        If no errors, return the original query.
                        Corrected:"""
    )
    return response.text.strip("\n")

def rewrite_query(query):
    """
    Calls Google Gemini model to rewrite vague user queries into more specific ones that the search engine can be more
    relevant results for
    """
    client = genai.Client(api_key = API_KEY)
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001", 
        contents = f"""Rewrite this movie search query to be more specific and searchable.
                        Original: "{query}"
                        Consider:
                        - Common movie knowledge (famous actors, popular films)
                        - Genre conventions (horror = scary, animation = cartoon)
                        - Keep it concise (under 10 words)
                        - It should be a google style search query that's very specific
                        - Don't use boolean logic
                        Examples:
                        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
                        Rewritten query:"""
    )
    return response.text.strip("\n")

def expand_query(query):
    """
    Calls Google Gemini Model to expand user queries to include more synonyms that maybe relevant
    """
    client = genai.Client(api_key = API_KEY)
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001", 
        contents = f"""Expand this movie search query with related terms.

                        Add synonyms and related concepts that might appear in movie descriptions.
                        Keep expansions relevant and focused.
                        This will be appended to the original query.

                        Examples:

                        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                        - "action movie with bear" -> "action thriller bear chase fight adventure"
                        - "comedy with bear" -> "comedy funny bear humor lighthearted"

                        Query: "{query}"""
    )
    return response.text.strip("\n")

def individual_rerank_results(query, rrf_results):
    """
    Calls Google Gemini Model to rerank search results if user chooses this option
    This will mutate rrf_results list of dictionaries with the new score
    """
    client = genai.Client(api_key = API_KEY)
    for movie in rrf_results:
        response = client.models.generate_content(
            model = "gemini-2.0-flash-001", 
            contents = f"""Rate how well this movie matches the search query.
                        Query: "{query}"
                        Movie: {movie.get("title", "")} - {movie.get("description", "")}
                        Consider:
                        - Direct relevance to query
                        - User intent (what they're looking for)
                        - Content appropriateness
                        Rate 0-10 (10 = perfect match).
                        Give me ONLY the float number in your response, no other text or explanation.
                        Score:"""
        )
        movie["score"] = float(response.text)
        print(f"Rescored {movie["title"]}: {movie["score"]}")
        time.sleep(3)

    return rrf_results

def batch_rerank_results(query, rrf_results):
    """
    Calls Google Gemini Model to rerank search results in a big batch.
    This will return a new list of dictionaries with added field of batch rerank score
    """
    client = genai.Client(api_key = API_KEY)
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001", 
        contents = f"""Rank these movies by relevance to the search query.

                    Query: "{query}"
                    Movies:
                    {rrf_results}

                    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON object with one field containing a Python list. 
                    Response: {{
                    data: [76, 1, 26, 2, 31]
                    }}
                    """
    )
    cleaned_response = response.text.strip().strip("```")[5:] # Clean the JSON response from Gemini API
    rankings = json.loads(cleaned_response)['data']
    ranking_map = {key: i for i, key in enumerate(rankings)}
    rrf_results = sorted(rrf_results, key = lambda item: ranking_map[item["id"]]) # Sort results based on the ranking_map from AI Reranking
    return rrf_results

def cross_encoder_rerank_results(query, rrf_results):
    """
    Performs a cross encoder similarity match which compares two sentences and gives a relevance score. In our case
    we measure relevance between user query and movie title + description
    rrf_results is an array of dictionaries each containing information for the top search results from RRF
    """
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    pairs = []
    # Generate the inputs for the cross encoder model
    for movie in rrf_results:
        pairs.append([query, f"{movie.get('title', '')} - {movie.get('description', '')}"])
    
    # Calculate the scores
    scores = cross_encoder_model.predict(pairs)

    # Populate each movie with its new score
    for i, movie in enumerate(rrf_results):
        movie["cross_encoder_score"] = scores[i]
    
    # Return the new movie results based on CE score
    results = sorted(rrf_results, key = lambda d: d["cross_encoder_score"], reverse = True)
    return results