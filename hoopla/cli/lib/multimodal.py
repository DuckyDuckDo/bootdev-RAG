import argparse
from lib.hybrid_search import *
from lib.semantic_search import *
import mimetypes
from google.genai import types
from PIL import Image
from sentence_transformers import SentenceTransformer

# Class that will perform multimodal search equipped with model for Image to Embedding space
class MultimodalSearch():
    def __init__(self, documents, model_name = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [
            f"{doc["title"]}: {doc["description"]}"
            for doc in self.documents
        ]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar = True)
    
    def embed_image(self, image_path):
        """
        Given an input image_path, returns an output embedding based on the model declared in constructor
        """
        image_data = Image.open(image_path)
        image_embedding = self.model.encode([image_data])[0]
        return image_embedding

    def search_with_image(self, image_path, limit = 5):
        """
        Given an image_path, we get the embedding of the image and perform cosine similarity with all the text embeddings
        """
        search_results = [] # list of dicts containing document_ID, title, description, similarity score    
        image_embedding = self.embed_image(image_path)
        similarity_scores = []

        for doc in self.documents:
            similarity_scores.append((cosine_similarity(image_embedding, self.text_embeddings[doc['id'] - 1]), doc)) # - 1 b/c doc ids start indexed at 1 and not 0

            # Grab the top_x_scores
        top_x_scores = sorted(similarity_scores, key = lambda x: x[0], reverse = True)
        final_result = []
        
        # Build out the final result
        for score, doc in top_x_scores:
            if len(final_result) >= limit:
                break
            final_result.append({
                                'id': doc['id'], 
                                'score': score, 
                                'title': doc['title'], 
                                'description': doc['description']
                                 })
        return final_result

###### COMMANDS FROM describe_image CLI and multimodal_search 
def describe_image(image_path, query):
    """
    Given an image_path and a user query, rewrite a new prompt that the LLM can use to answer the quesiton as well as can be passed into the 
    database
    """
    client = genai.Client(api_key = API_KEY)
    system_prompt = f"""
            Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary
        """
    
    # Prepare the image contents
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    with open(image_path, "rb") as img:
        image_binary = img.read()
    
    # Set up the Parts that will be passed into the LLM call
    parts = [
        system_prompt, 
        types.Part.from_bytes(data = image_binary, mime_type = mime),
        query.strip()
    ]

    response = client.models.generate_content(
        model = "gemini-2.0-flash-001", 
        contents = parts
    )

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")
    
def verify_image_embedding(image_path):
    """
    Command called to check if our multimodal search model can verify image embedding
    """
    model = MultimodalSearch()
    image_embedding = model.embed_image(image_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")

def image_search_command(image_path, limit = 5):
    """
    Command to be called from CLI that loads movies and creates MultiModalSearch instance to do search with image based on
    images and text being embedded on the same vector space
    """
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
        movies = data["movies"]
    model = MultimodalSearch(movies)
    top_search_results = model.search_with_image(image_path, limit)

    for movie in top_search_results:
        print(f"{movie["title"]} (similarity: {movie["score"]:.3f})")