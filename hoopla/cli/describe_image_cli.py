import argparse
from lib.multimodal import *

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_argument("--image", help = "path to an image file for multimodal search")
    parser.add_argument("--query", help = "text query to go along with image as input into search")

    args = parser.parse_args()

    describe_image(args.image, args.query)


if __name__ == "__main__":
    main()