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

Integrate with LLM through Gemini API and use the retrieval mechanisms to implement a RAG that is capable of summarization of movies, question answering of movies, and more context informed suggestion of movies based on user query. 

Key Learning Outcomes
- How search works based on user query and the dataset. Two types of searching uses keyword and semantic search both of which must be considered to retrieve the most relevant search results. Keyword search focuses on the word's frequency throughout the different documents giving even more weight to words that appear less (TF-IDF). Semantic search focuses on the word's actual meaning by encoding vectors into an embedding space of vectors that encode meaning. From there cosine similarity can match up the user query with the embedding of the documents to retrieve the embeddings that are most similar and then decode back to text. 
- Learned about hybrid search which uses a mix of keyword and semantic search. This can be done through a weighted search (weighted average of normalized scores) or reciprocal rank fusion (RRF) where the individual rankings across both search methods factor into a final search score. 
- Learned about reranking methods with the help of LLMs. LLMs can be used to clean up and enhance user queries, can provide additional reranking after the search hits our database through prompt engineering. 
- Learned about cross encoding another reranking mechanism which is its own pretrained model that takes in two text queries and provides a score based on how similar they are. 
- Learned RAG, the retrieval step is performed with search, but the results of the retrieval augments the prompt that goes into LLM that plays into cleaner and higher quality of results being generated. 
- Finally, touched up on image search using CLIP pretrained model that embeds images and texts into one shared embedding space and then using cosine similarity once again to find highest scoring match. 