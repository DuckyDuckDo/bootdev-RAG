import argparse
from lib.hybrid_search import *

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()

    # run evaluation logic here, get the results, and then call format to print the results
    eval_results = evaluate_model(args.limit)
    format_eval_results(eval_results, args.limit)


if __name__ == "__main__":
    main()