# bootdev-RAG
Retrieval Augmented Generation Project 
- Starts with Building out a Keyword Search Application for Movie Data 
    Begins with Inverted Index, followed by TF-IDF, and then BM25
- Build out Semantic Search Application for Movie Data
    Starts out with full document embeddings and then chunked embeddings for more accurate retrieval

- Builds out a hybrid search engine that uses reciprocal rank fusion along with different varieties of reranking techniques to sort the best results
- The hybrid search engine can then be used as a RAG source document for LLM to hit and call every time it needs a query 
    - For now this is hardcoded in as the query passed to the LLM will be the same one passed into the search engine
- Finally, explores multimodal search capabilities by providing relevant movie recommendations based on a user image input. 

Integrate with LLM through Gemini API and use the retrieval mechanisms to implement a RAG

