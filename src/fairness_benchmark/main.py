import argparse
from loguru import logger

from fairness_benchmark.process.process_dataset import process
from fairness_benchmark.benchmark.benchmark_dataset import benchmark


log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
logger.add(
    "logger.log",
    colorize=False,
)


# Control Flow for Accessing, Transforming and Saving a Dataset.
## Datasets are accessed either under the ./data/dataset/ dir or imported from AIF360.datasets.
## Datasets are saved under ./data/processed_dataset/.
def process_dataset(args):
    logger.info(f"Processing Dataset: {args.dataset} using {args.preprocess}.")
    process(args)


# Control Flow for Benchmarking the Dataset.
def benchmark_dataset(args):
    logger.info(
        f"Running Benchmark for Dataset: {args.dataset} using {args.evaluation}."
    )
    # TODO: Add Logging
    benchmark(args)


# Add additional arguments here: These will be added to either process() or benchmark() depending on the --task arg.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="process",
        help="Choice between 'process' and 'benchmark'. Process transforms a dataset and uses the process arguments. Benchmark uses a transformed dataset and uses Benchmark arguments. ",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        choices=["adult", "bank", "compas", "german", "meps", "custom"],
        help="Select a dataset from: adult, bank, compas, german, meps, custom. If selecting custom fill --dataset_path.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Enter a custom path to a dataset within in ./data/dataset/. E.G. fairness_benchmark/data/dataset/my_custom_data/custom_data.data",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="income",
        help="Target attribute for prediction",
    )
    parser.add_argument(
        "--sensitive",
        type=str,
        default="sex",
        help="Sensitive attribute for fairness analysis",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="DPI",
        choices=["rw", "dir", "lfr", "op", "none"],
        help="Pre-Processing Algorithm: rw, dir, lfr, op, or none",
    )
    parser.add_argument(
        "--evaluation",
        type=str,
        default="acc,ap,dp,eopp,eodd",
        help="Evaluation metrics separated by commas, e.g., acc,ap,dp",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lr",
        choices=["lr", "mlp", "nb"],
        help="What Model to use in the benchmarking, e.g. lr, mlp, nb",
    )

    args = parser.parse_args()

    match args.task:
        case "process":
            process_dataset(args)
        case "benchmark":
            benchmark_dataset(args)
        case _:
            print("Invalid --task argument")
