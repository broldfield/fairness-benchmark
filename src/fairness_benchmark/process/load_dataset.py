from aif360.datasets import (
    AdultDataset,
    BankDataset,
    BinaryLabelDataset,
    CompasDataset,
    GermanDataset,
    MEPSDataset21,
    StandardDataset,
)
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult,
    load_preproc_data_german,
    load_preproc_data_compas,
)

from loguru import logger

import numpy as np

from fairness_benchmark.process.custom_dataset import CustomDataset
from fairness_benchmark.process.preprocessing import get_privileged_list


def load_dataset(args) -> StandardDataset | BinaryLabelDataset:
    # TODO: Add in logic for loading custom datasets.
    if args.dataset.lower() == "none":
        dataset = load_custom_dataset(args)

        return dataset
    # TODO: Create Matches for different dataset types from AIF360.
    if args.dataset.lower() != "none":
        logger.info(f"Loading {args.dataset.lower()}")
        if args.preprocess == "op":
            match args.dataset.lower():
                case "adult":
                    dataset = load_preproc_data_adult([args.sensitive])
                    return dataset
                case "compas":
                    dataset = load_preproc_data_compas([args.sensitive])
                    return dataset
                case "german":
                    dataset = load_preproc_data_german([args.sensitive])
                    return dataset
                case _:
                    logger.info(
                        "OptimProc currently only supports adult, german, compas datasets. "
                    )
                    quit()

        match args.dataset.lower():
            case "adult":
                dataset = load_aif_adult_dataset(sensitive_attr=[args.sensitive])
                return dataset
            case "bank":
                dataset = load_aif_bank_dataset(sensitive_attr=[args.sensitive])
                return dataset
            case "compas":
                dataset = load_aif_compas_dataset(sensitive_attr=[args.sensitive])
                return dataset
            case "german":
                dataset = load_aif_german_dataset(sensitive_attr=[args.sensitive])
                return dataset
            case "meps":
                dataset = load_aif_meps_dataset(sensitive_attr=[str(args.sensitive)])
                return dataset
            case "custom":
                dataset = load_custom_dataset(args)
                return dataset
            case _:
                raise ValueError

    raise ValueError


def load_aif_adult_dataset(sensitive_attr=["sex"]) -> StandardDataset:
    logger.info("Loading Adult Dataset")
    # priv = ""
    # if sensitive_attr == ["sex"]:
    #     priv = [["Male"]]
    # if sensitive_attr == ["race"]:
    #     priv = [["White"]]

    adult_dataset = load_preproc_data_adult(sensitive_attr)

    # adult_dataset = AdultDataset(
    #     protected_attribute_names=sensitive_attr,
    #     privileged_classes=priv,
    #     categorical_features=[],
    #     features_to_keep=["age", "education-num"],
    # )
    return adult_dataset


def load_aif_bank_dataset(sensitive_attr=["age"]) -> StandardDataset:
    bank_dataset = BankDataset(protected_attribute_names=sensitive_attr)
    logger.info(bank_dataset.label_names)
    logger.info(bank_dataset.protected_attribute_names)
    logger.info(bank_dataset.favorable_label)

    return bank_dataset


def load_aif_compas_dataset(sensitive_attr=["sex"]) -> StandardDataset:
    # if sensitive_attr == ["sex"]:
    #     label_map = {1.0: "Did recid.", 0.0: "No recid."}
    #     protected_attribute_maps = [{1.0: "Male", 0.0: "Female"}]
    #     compas_dataset = CompasDataset(
    #         protected_attribute_names=["sex"],
    #         privileged_classes=[["Male"]],
    #         features_to_drop=["race"],
    #         metadata={
    #             "label_map": label_map,
    #             "protected_attribute_maps": protected_attribute_maps,
    #         },
    #     )
    #     return compas_dataset
    # else:
    #     label_map = {1.0: "Did recid.", 0.0: "No recid."}
    #     protected_attribute_maps = [{1.0: "Caucasian", 0.0: "Not Caucasian"}]
    #     compas_dataset = CompasDataset(
    #         protected_attribute_names=["race"],
    #         privileged_classes=[["Caucasian"]],
    #         features_to_drop=["sex"],
    #         metadata={
    #             "label_map": label_map,
    #             "protected_attribute_maps": protected_attribute_maps,
    #         },
    #     )
    #     return compas_dataset

    compas_dataset = load_preproc_data_compas(sensitive_attr)
    return compas_dataset


def load_aif_german_dataset(sensitive_attr=["age"]) -> StandardDataset:
    # priv = ""
    # if sensitive_attr == ["sex"]:
    #     priv = [["male"]]
    # if sensitive_attr == ["race"]:
    #     priv = lambda x: x > 25
    # german_dataset = GermanDataset(
    #     protected_attribute_names=sensitive_attr,
    #     privileged_classes=priv,
    # )
    # logger.info(f"Label names for german is: {german_dataset.labels}")
    german_dataset = load_preproc_data_german(sensitive_attr)
    return german_dataset


def load_aif_meps_dataset(sensitive_attr=["RACE"]) -> StandardDataset:
    meps_dataset = MEPSDataset21(protected_attribute_names=sensitive_attr)
    return meps_dataset


# TODO: Add in custom dataset handling.
def load_custom_dataset(args) -> StandardDataset:
    custom_dataset = CustomDataset(path=args.dataset_path)
    logger.info("Imported Custom Dataset.")
    return custom_dataset
