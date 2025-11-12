import argparse
from lib.multimodal import *

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser("verify_image_embedding", help = "verifies a given image embedding")
    verify_image_embedding_parser.add_argument("image", help = "image path for verification")

    image_search_parser = subparsers.add_parser("image_search", help = "search the database with an image")
    image_search_parser.add_argument("image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image)
        
        case "image_search":
            image_search_command(args.image, limit = 5)
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()