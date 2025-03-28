from fairness_benchmark.benchmark.run_dataset import run_dataset
from fairness_benchmark.benchmark.plotting import create_plots
from fairness_benchmark.utils.loading import get_dataset
from loguru import logger


def benchmark(args):
    # Load saved Datasets.
    dataset_original = get_dataset(args, "Original")
    dataset_proc = get_dataset(args, "Processed")

    logger.info("Running Original Dataset...")
    best_class_threshold_orig, class_threshold_orig, bal_orig, disp_orig, avg_orig = (
        run_dataset(args, dataset_original, "Original")
    )

    logger.info("Running Processed Dataset...")
    best_class_threshold_proc, class_threshold_proc, bal_proc, disp_proc, avg_proc = (
        run_dataset(args, dataset_proc, "Processed")
    )

    logger.info("Creating Plots for Original Dataset...")
    create_plots(
        args,
        "Original",
        best_class_threshold_orig,
        class_threshold_orig,
        bal_orig,
        disp_orig,
        avg_orig,
    )

    logger.info("Creating Plots for Processed Dataset...")
    create_plots(
        args,
        "Processed",
        best_class_threshold_proc,
        class_threshold_proc,
        bal_proc,
        disp_proc,
        avg_proc,
    )

    return
