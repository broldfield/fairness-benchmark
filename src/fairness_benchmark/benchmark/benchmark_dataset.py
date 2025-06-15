from fairness_benchmark.benchmark.run_dataset import run_dataset
from fairness_benchmark.benchmark.plotting import create_plots
from fairness_benchmark.utils.loading import get_dataset, load_metric_df, save_metric
from loguru import logger


def benchmark(args):
    # Load saved Datasets.
    dataset_original = get_dataset(args, "Original")
    dataset_proc = get_dataset(args, "Processed")

    logger.info(f"Label = {dataset_proc.label_names}")
    # logger.info(f"Label = {dataset_proc.labels}")

    logger.info("Running Original Dataset...")
    (best_class_threshold_orig, class_threshold_orig, metric_df_orig) = run_dataset(
        args, dataset_original, "Original"
    )

    logger.info("Running Processed Dataset...")
    (best_class_threshold_proc, class_threshold_proc, metric_df_proc) = run_dataset(
        args, dataset_proc, "Processed"
    )

    logger.info("Creating Plots for Original Dataset...")
    create_plots(
        args,
        "Original",
        best_class_threshold_orig,
        class_threshold_orig,
        metric_df_orig,
    )

    logger.info("Creating Plots for Processed Dataset...")
    create_plots(
        args,
        "Processed",
        best_class_threshold_proc,
        class_threshold_proc,
        metric_df_proc,
    )
    # load_metric_df(
    #     args, "Original", bal_orig, disp_orig, avg_orig, stat_orig, eq_orig, theil_orig
    # )
    # load_metric_df(
    #     args, "Processed", bal_proc, disp_proc, avg_proc, stat_proc, eq_proc, theil_proc
    # )
    #
    save_metric(args, "Original", metric_df_orig)

    save_metric(args, "Processed", metric_df_proc)

    return
