import os

from loguru import logger
import pandas as pd

from aif360.datasets import StandardDataset


default_mappings = {
    "label_maps": [{1.0: "Good Credit", 2.0: "Bad Credit"}],
    "protected_attribute_maps": [
        {1.0: "Male", 0.0: "Female"},
        {1.0: "Old", 0.0: "Young"},
    ],
}


def default_preprocessing(df):
    """Adds a derived sex attribute based on personal_status."""
    # TODO: ignores the value of privileged_classes for 'sex'
    status_map = {
        "A91": "male",
        "A93": "male",
        "A94": "male",
        "A92": "female",
        "A95": "female",
    }
    df["sex"] = df["personal_status"].replace(status_map)

    return df


class CustomDataset(StandardDataset):
    """Template for a Custom Dataset.

    This is adapted from the AIF360 GermanDataset code.
    `https://github.com/Trusted-AI/AIF360/blob/cd7e2138b7919e0796db7e7902bf49b20065f4f8/aif360/datasets/german_dataset.py`.
    """

    def __init__(
        self,
        path="data/dataset/example/example.data",
        label_name="credit",
        favorable_classes=[1],
        protected_attribute_names=["sex", "age"],
        privileged_classes=[["male"], lambda x: x > 25],
        instance_weights_name=None,
        categorical_features=[
            "status",
            "credit_history",
            "purpose",
            "savings",
            "employment",
            "other_debtors",
            "property",
            "installment_plans",
            "housing",
            "skill_level",
            "telephone",
            "foreign_worker",
        ],
        features_to_keep=[],
        features_to_drop=["personal_status"],
        na_values=[],
        custom_preprocessing=default_preprocessing,
        metadata=default_mappings,
    ):
        """See :obj:`StandardDataset` for a description of the arguments.

        By default, this code converts the 'age' attribute to a binary value
        where privileged is `age > 25` and unprivileged is `age <= 25` as
        proposed by Kamiran and Calders [1]_.

        References:
            .. [1] F. Kamiran and T. Calders, "Classifying without
               discriminating," 2nd International Conference on Computer,
               Control and Communication, 2009.

        Examples:
            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:

            >>> label_map = {1.0: 'Good Credit', 0.0: 'Bad Credit'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> gd = GermanDataset(protected_attribute_names=['sex'],
            ... privileged_classes=[['male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})

            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        filepath = os.path.join(os.getcwd(), path)
        logger.info(f"Loading Custom Dataset from: {filepath}")

        # INFO: Column Names in the dataset.
        column_names = [
            "status",
            "month",
            "credit_history",
            "purpose",
            "credit_amount",
            "savings",
            "employment",
            "investment_as_income_percentage",
            "personal_status",
            "other_debtors",
            "residence_since",
            "property",
            "age",
            "installment_plans",
            "housing",
            "number_of_credits",
            "skill_level",
            "people_liable_for",
            "telephone",
            "foreign_worker",
            "credit",
        ]
        try:
            df = pd.read_csv(
                filepath, sep=" ", header=None, names=column_names, na_values=na_values
            )
        except IOError as err:
            import sys

            logger.info(f"Error loading custom dataset: {err}")

            sys.exit(1)

        super(CustomDataset, self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
            metadata=metadata,
        )
