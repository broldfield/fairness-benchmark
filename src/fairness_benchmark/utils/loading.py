from aif360.datasets import StandardDataset
import os
from loguru import logger
import pickle


def save_dataset(args, dataset: StandardDataset, type):
    dataset_location = "src/fairness_benchmark/data/processed_dataset/"
    name = f"{type}_{args.dataset}_{args.preprocess}_{args.sensitive}_{args.target}"
    path = dataset_location + name + ".pkl"

    save_path = os.path.join(os.getcwd(), path)

    logger.info(f"Saving data to: {save_path}")

    with open(save_path, "wb") as file:
        pickle.dump(dataset, file)

    return


def get_dataset(args, type) -> StandardDataset:
    dataset_location = "src/fairness_benchmark/data/processed_dataset/"
    name = f"{type}_{args.dataset}_{args.preprocess}_{args.sensitive}_{args.target}"

    path = dataset_location + name + ".pkl"

    load_path = os.path.join(os.getcwd(), path)

    logger.info(f"Loading Dataset at: {load_path}")

    with open(load_path, "rb") as file:
        dataset = pickle.load(file)

    return dataset


def to_csv(args, dataset: StandardDataset, type):
    dataset_location = "src/fairness_benchmark/data/processed_dataset/"
    name = f"{type}_{args.dataset}_{args.preprocess}_{args.sensitive}_{args.target}"
    path = dataset_location + name + ".csv"

    save_path = os.path.join(os.getcwd(), path)

    logger.info(f"Saving data to: {save_path}")

    # Use the StandardDataset method .convert_to_dataframe() to create a Tuple of Dataframe and Attributes
    dataset_pd, _ = dataset.convert_to_dataframe()

    # Save the dataset to csv.
    dataset_pd.to_csv(save_path)

    return
