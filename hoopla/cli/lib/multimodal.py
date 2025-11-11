import argparse
from lib.hybrid_search import *
import mimetypes
from google.genai import types


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