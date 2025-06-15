from aif360.datasets import StandardDataset
import os
from loguru import logger
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def save_dataset(args, dataset: StandardDataset, type):
    dataset_location = "src/fairness_benchmark/data/processed_dataset/"
    name = f"{args.dataset}/{args.preprocess}/{args.sensitive}/{type}"
    path_with_file = dataset_location + name + "/dataset.pkl"
    path = dataset_location + name

    Path(path).mkdir(parents=True, exist_ok=True)

    save_path = os.path.join(os.getcwd(), path_with_file)

    logger.info(f"Saving data to: {save_path}")

    with open(save_path, "wb") as file:
        pickle.dump(dataset, file)

    return


def get_dataset(args, type) -> StandardDataset:
    dataset_location = "src/fairness_benchmark/data/processed_dataset/"
    name = f"{args.dataset}/{args.preprocess}/{args.sensitive}/{type}"

    path = dataset_location + name + "/dataset.pkl"

    load_path = os.path.join(os.getcwd(), path)

    logger.info(f"Loading Dataset at: {load_path}")

    with open(load_path, "rb") as file:
        dataset = pickle.load(file)

    return dataset

    save_path = os.path.join(os.getcwd(), path_with_file)


def to_csv(args, dataset: StandardDataset, type):
    dataset_location = "src/fairness_benchmark/data/processed_dataset/"
    name = f"{args.dataset}/{args.preprocess}/{args.sensitive}/{type}"
    path = dataset_location + name + "/dataset.csv"

    save_path = os.path.join(os.getcwd(), path)

    logger.info(f"Saving data to: {save_path}")

    # Use the StandardDataset method .convert_to_dataframe() to create a Tuple of Dataframe and Attributes
    dataset_pd, _ = dataset.convert_to_dataframe()

    # Save the dataset to csv.
    dataset_pd.to_csv(save_path)

    return


def load_metric_df(args, bal, disp, avg, stat, eq, theil):
    data = {
        "Balanced Average": bal,
        "Average Odds Difference": avg,
        "Disparate Impact": disp,
        "Statistical Parity Difference": stat,
        "Equal Opportunity Difference": eq,
        "Theil Index": theil,
    }
    logger.info(f"Average {avg}")

    metric_df = pd.DataFrame(data=data)

    save_metric(args, "Checking", metric_df)
    return metric_df


def save_metric(args, type, metric_df):
    dataset_location = "src/fairness_benchmark/data/metric/"
    name = f"{args.dataset}/{args.preprocess}/{args.sensitive}/{args.model}/{type}"
    path_with_file = dataset_location + name + "/metric.csv"
    path = dataset_location + name
    Path(path).mkdir(parents=True, exist_ok=True)

    save_path = os.path.join(os.getcwd(), path_with_file)

    logger.info(f"Saving metrics to: {save_path}")

    # Save the dataset to csv.
    metric_df.to_csv(save_path)
    return


def save_fairness_metric(args, type, metric_df):
    dataset_location = "src/fairness_benchmark/data/metric/"
    name = f"{args.dataset}/{args.preprocess}/{args.sensitive}/{type}"
    path_with_file = dataset_location + name + "/metric.csv"
    path = dataset_location + name
    Path(path).mkdir(parents=True, exist_ok=True)

    save_path = os.path.join(os.getcwd(), path_with_file)

    logger.info(f"Saving metrics to: {save_path}")

    # Save the dataset to csv.
    metric_df.to_csv(save_path, index=False)
    return


def save_plot(args, type, name, fig):
    plot_location = "src/fairness_benchmark/data/plots/"
    name_path = f"{args.dataset}/{args.preprocess}/{args.sensitive}/{args.model}/{type}"
    path_with_file = plot_location + name_path + "/" + name + ".png"
    path = plot_location + name_path
    Path(path).mkdir(parents=True, exist_ok=True)

    save_path = os.path.join(os.getcwd(), path_with_file)

    fig.savefig(save_path)

    return
