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

from fairness_benchmark.process.custom_dataset import CustomDataset


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


def load_aif_adult_dataset(
    sensitive_attr=["sex"], groups=[{1.0: "Male", 0.0: "Female"}], priv_group=[["Male"]]
) -> StandardDataset:
    label_map = {1.0: ">50K", 0.0: "<=50K"}
    adult_dataset = AdultDataset(
        protected_attribute_names=sensitive_attr,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "native-country",
            "race",
        ],
        privileged_classes=priv_group,
        metadata={"label_map": label_map, "protected_attribute_maps": groups},
    )

    return adult_dataset


def load_aif_bank_dataset(sensitive_attr=["age"]) -> StandardDataset:
    bank_dataset = BankDataset(protected_attribute_names=sensitive_attr)
    return bank_dataset


def load_aif_compas_dataset(sensitive_attr=["sex"]) -> StandardDataset:
    compas_dataset = CompasDataset(protected_attribute_names=sensitive_attr)

    return compas_dataset


def load_aif_german_dataset(sensitive_attr=["sex"]) -> StandardDataset:
    german_dataset = GermanDataset(protected_attribute_names=sensitive_attr)
    return german_dataset


def load_aif_meps_dataset(sensitive_attr=["race"]) -> StandardDataset:
    meps_dataset = MEPSDataset21()
    return meps_dataset


# TODO: Add in custom dataset handling.
def load_custom_dataset(args) -> StandardDataset:
    custom_dataset = CustomDataset(path=args.dataset_path)
    logger.info("Imported Custom Dataset.")
    return custom_dataset
