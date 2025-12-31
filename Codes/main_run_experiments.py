import argparse
from qmeta.experiment_runner import run_experiments_for_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run QNN experiments on a dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset.")
    parser.add_argument("--target_col", type=str, required=True, help="Name of target column.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/experiments.csv",
        help="Path to output CSV log file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiments_for_dataset(
        data_path=args.data_path,
        target_col=args.target_col,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
