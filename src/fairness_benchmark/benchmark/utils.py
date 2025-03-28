from sklearn.preprocessing import StandardScaler, MinMaxScaler

from collections import OrderedDict
from aif360.metrics import ClassificationMetric
import os
import numpy as np
from tqdm import tqdm

from loguru import logger

from fairness_benchmark.process.preprocessing import get_privileged_list


def save_metrics(args, metrics):
    metric_location = "src/fairness_benchmark/data/metric/"
    name = f"metrics_{args.dataset}_{args.preprocess}_{args.sensitive}_{args.target}"
    path = metric_location + name + ".csv"

    save_path = os.path.join(os.getcwd(), path)

    logger.info(f"Saving Metrics to: {save_path}")

    return


def scale_data(args, dataset, weight=None):
    scaler = MinMaxScaler()
    if weight is not None:
        scaled_dataset = scaler.fit_transform(dataset)
        return scaled_dataset

    scaled_dataset = scaler.fit_transform(dataset)
    return scaled_dataset


def splitter(args, dataset):
    X = scale_data(args, dataset.features)
    y = dataset.labels.ravel()
    w = dataset.instance_weights.ravel()
    s = dataset.protected_attributes.ravel()

    return X, y, w, s


# This was copied from an AIF360 example: https://github.com/Trusted-AI/AIF360/blob/main/examples/common_utils.py here.
def compute_metrics(
    dataset_true, dataset_pred, unprivileged_groups, privileged_groups, disp=True
):
    """Compute the key metrics"""
    classified_metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5 * (
        classified_metric_pred.true_positive_rate()
        + classified_metric_pred.true_negative_rate()
    )
    metrics["Statistical parity difference"] = (
        classified_metric_pred.statistical_parity_difference()
    )
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = (
        classified_metric_pred.average_odds_difference()
    )
    metrics["Equal opportunity difference"] = (
        classified_metric_pred.equal_opportunity_difference()
    )
    metrics["Theil index"] = classified_metric_pred.theil_index()

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics


def optim_threshold(args, dataset, dataset_pred, original=True):
    privileged_groups, unprivileged_groups = get_privileged_list(args.sensitive)

    num_thresh = 100
    ba_arr = np.zeros(num_thresh)
    class_thresh_arr = np.linspace(0.01, 0.99, num_thresh)
    for idx, class_thresh in enumerate(class_thresh_arr):
        fav_inds = dataset_pred.scores > class_thresh
        dataset_pred.labels[fav_inds] = dataset_pred.favorable_label
        dataset_pred.labels[~fav_inds] = dataset_pred.unfavorable_label

        classified_metric_orig_valid = ClassificationMetric(
            dataset,
            dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        ba_arr[idx] = 0.5 * (
            classified_metric_orig_valid.true_positive_rate()
            + classified_metric_orig_valid.true_negative_rate()
        )

    best_ind = np.where(ba_arr == np.max(ba_arr))[0][0]
    best_class_thresh = class_thresh_arr[best_ind]
    strPreprocess = ""

    if original:
        strPreprocess = f"Without {args.preprocess}"
    if not original:
        strPreprocess = f"With{args.preprocess}"

    logger.info(f"Best balanced accuracy ({strPreprocess}) = %.4f" % np.max(ba_arr))
    print("Optimal classification threshold (no reweighing) = %.4f" % best_class_thresh)

    return best_class_thresh, class_thresh_arr


def threshold(args, dataset, dataset_pred, best_class_thresh, class_thresh_arr):
    privileged_groups, unprivileged_groups = get_privileged_list(args.sensitive)
    bal_acc_arr_orig = []
    disp_imp_arr_orig = []
    avg_odds_diff_arr_orig = []

    logger.info("Classification threshold used = %.4f" % best_class_thresh)
    for thresh in tqdm(class_thresh_arr):
        if thresh == best_class_thresh:
            disp = True
        else:
            disp = False

        fav_inds = dataset_pred.scores > thresh
        dataset_pred.labels[fav_inds] = dataset_pred.favorable_label
        dataset_pred.labels[~fav_inds] = dataset_pred.unfavorable_label

        metric_test_bef = compute_metrics(
            dataset,
            dataset_pred,
            unprivileged_groups,
            privileged_groups,
            disp=disp,
        )
        save_metrics(args, metric_test_bef)

        bal_acc_arr_orig.append(metric_test_bef["Balanced accuracy"])
        avg_odds_diff_arr_orig.append(metric_test_bef["Average odds difference"])
        disp_imp_arr_orig.append(metric_test_bef["Disparate impact"])

    return bal_acc_arr_orig, avg_odds_diff_arr_orig, disp_imp_arr_orig


def split(args, dataset):
    dataset_train, dataset_leftover = dataset.split([0.7], shuffle=True)
    dataset_test, dataset_valid = dataset_leftover.split([0.5], shuffle=True)
    return dataset_train, dataset_test, dataset_valid
