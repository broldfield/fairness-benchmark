from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric

from fairness_benchmark.process.preprocessing import get_privileged_list
from loguru import logger

import pandas as pd
from fairness_benchmark.utils.loading import save_fairness_metric


def pre_processor_metrics(dataset, sensitive_attr):
    privileged_groups, unprivileged_groups = get_privileged_list(sensitive_attr)

    dataset_metrics = BinaryLabelDatasetMetric(
        dataset, unprivileged_groups, privileged_groups
    )

    return dataset_metrics


def metrics(args, dataset: StandardDataset, type: str):
    dataset_metrics = pre_processor_metrics(dataset, args.sensitive)
    logger.info("Generating Metrics...")
    base = round(dataset_metrics.base_rate(), 7)

    logger.info("Creating Consistency Metric (May take 30s-1m).")
    consistency = dataset_metrics.consistency()
    disp = round(dataset_metrics.disparate_impact(), 7)
    mean_diff = round(dataset_metrics.mean_difference(), 7)
    num_pos = dataset_metrics.num_positives()
    num_neg = dataset_metrics.num_negatives()
    smooth = round(dataset_metrics.smoothed_empirical_differential_fairness(), 7)

    metric_data = {
        "Base Rate": base,
        "Consistency": consistency,
        "Disparate": disp,
        "Mean Difference": mean_diff,
        "Num Positives": num_pos,
        "Num Negatives": num_neg,
        "Empirical Diff": smooth,
    }

    metric_df = pd.DataFrame(data=metric_data)
    save_fairness_metric(args, type, metric_df)

    logger.info(f"{type} Dataset - Base Rate: {base}")
    logger.info(f"{type} Dataset - Consistency: {consistency}")
    logger.info(f"{type} Dataset - Disparate Impact: {disp}")
    logger.info(f"{type} Dataset - Mean Difference: {mean_diff}")
    logger.info(f"{type} Dataset - Num Positives: {num_pos} | Num Negatives: {num_neg}")
    logger.info(f"{type} Dataset - Smoothed Empirical Differenctial Fairness: {smooth}")

    return
