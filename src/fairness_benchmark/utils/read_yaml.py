from time import time
from loguru import logger
import yaml
from argparse import Namespace

from fairness_benchmark.benchmark.benchmark_dataset import benchmark
from fairness_benchmark.process.process_dataset import process


def read_yaml():
    with open("tasks.yaml", "r") as f:
        runs_info = yaml.safe_load(f)

    if len(runs_info["run"]) == 0:
        logger.info("There are no values in 'run' inside of tasks.yaml.")
    for run in runs_info["run"]:
        args = Namespace(**runs_info["jobs"].get(run))

        match args.task:
            case "process":
                process(args)
            case "benchmark":
                benchmark(args)
            case _:
                logger.info("Invalid --task argument")


def read_yaml_alt():
    with open("tasks-alt.yaml", "r") as f:
        runs_info = yaml.safe_load(f)

    if len(runs_info["datasets"]) == 0:
        logger.info("There are no values in 'datasets' inside of tasks-alt.yaml.")
        quit()

    if len(runs_info["fairness"]) == 0:
        logger.info("There are no values in 'fairness' inside of tasks-alt.yaml.")
        quit()

    if len(runs_info["models"]) == 0:
        logger.info("There are no values in 'models' inside of tasks-alt.yaml.")
        quit()

    for dataset_element in runs_info["datasets"]:
        dataset, sensitive = dataset_element

        for fairness in runs_info["fairness"]:
            for model in runs_info["models"]:
                args = Namespace(
                    **{
                        "dataset": dataset,
                        "sensitive": sensitive,
                        "preprocess": fairness,
                        "model": model,
                    }
                )
                process(args)
                benchmark(args)
