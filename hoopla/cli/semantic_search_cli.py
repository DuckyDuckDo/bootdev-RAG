import argparse
from lib.semantic_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help = "Verify installation and initiation of sentence transformer model")

    embed_text_parser = subparsers.add_parser("embed_text", help = "Embed the text into a vector space using preloaded model")
    embed_text_parser.add_argument("text", help = "Text of which we want to generate embedding for")

    verify_embedding_parser = subparsers.add_parser("verify_embeddings", help = "Verifies that the model has proper embeddings for our movies")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        
        case "embed_text":
            embed_text(args.text)
        
        case "verify_embeddings":
            verify_embeddings()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()