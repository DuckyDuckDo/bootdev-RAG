import argparse
from lib.hybrid_search import *

######## RAG COMMANDS ##########
def rag(query):
    """
    Given a query, pass it through hybrid search to get relevant movies, inject the result into the prompt, and have the LLM
    answer the query with the additional retrieved context
    """

    # load the hybrid search engine that is used for RAG
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]
    hybrid_search = HybridSearch(movies)
    top_search_results = hybrid_search.rrf_search(query, limit = 10)
    
    client = genai.Client(api_key = API_KEY)
    prompt = prompt = f"""
        Answer the question or provide information based on the provided documents. 
        This should be tailored to Hoopla users. Hoopla is a movie streaming service
        Query: {query}

        Documents:
        {[movie['title'] for movie in top_search_results]}

        Provide a comprehensive answer that addresses the query:
        """
    
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001", 
        contents = prompt
    )
    print(response.text.strip())
    return response.text.strip()

def summarize(query):
    """
    Given a query, perform the RRF search and using those results prompt the LLM to summarize the results into a comprehensive answer
    """
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]
    hybrid_search = HybridSearch(movies)
    top_search_results = hybrid_search.rrf_search(query, limit = 10)
    client = genai.Client(api_key = API_KEY)
    prompt = prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {[(movie["title"]) for movie in top_search_results]}
    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""
    
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001", 
        contents = prompt
    )
    print(response.text.strip())
    return response.text.strip()

def citations(query):
    """
    Given a query, perform the RRF search and using those results prompt the LLM to summarize the results into a comprehensive answer with proper citations
    """
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]
    hybrid_search = HybridSearch(movies)
    top_search_results = hybrid_search.rrf_search(query, limit = 10)
    client = genai.Client(api_key = API_KEY)
    prompt = f"""Answer the question or provide information based on the provided documents.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

            Query: {query}

            Documents:
            {[movie["title"] for movie in top_search_results]}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources using [1], [2], etc. format when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
    
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001", 
        contents = prompt
    )
    print(response.text.strip())
    return response.text.strip()

def question(query):
    """
    Given a query, perform the RRF search and using those results prompt the LLM to summarize the results into a comprehensive answer.
    """
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]
    hybrid_search = HybridSearch(movies)
    top_search_results = hybrid_search.rrf_search(query, limit = 10)
    client = genai.Client(api_key = API_KEY)
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Question: {query}

    Documents:
    {[(movie["title"], movie["description"]) for movie in top_search_results[:3]]}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""
        
    response = client.models.generate_content(
        model = "gemini-2.0-flash-001", 
        contents = prompt
    )
    print(response.text.strip())
    return response.text.strip()