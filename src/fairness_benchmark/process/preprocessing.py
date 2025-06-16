from aif360.algorithms.preprocessing import (
    LFR,
    DisparateImpactRemover,
    OptimPreproc,
    Reweighing,
)
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import (
    get_distortion_adult,
    get_distortion_german,
    get_distortion_compas,
)
from loguru import logger


def get_optim_options(args):
    match args.dataset.lower():
        case "adult":
            optim_options = {
                "distortion_fun": get_distortion_adult,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [0.1, 0.05, 0],
            }
            return optim_options
        case "german":
            if args.sensitive == "sex":
                optim_options = {
                    "distortion_fun": get_distortion_german,
                    "epsilon": 0.05,
                    "clist": [0.99, 1.99, 2.99],
                    "dlist": [0.1, 0.05, 0],
                }
                return optim_options
            if args.sensitive == "age":
                optim_options = {
                    "distortion_fun": get_distortion_german,
                    "epsilon": 0.1,
                    "clist": [0.99, 1.99, 2.99],
                    "dlist": [0.1, 0.05, 0],
                }
            logger.info(
                "When using OptimProc, the German dataset requires a sensitive attribute of sex or age."
            )
            quit()
        case "compas":
            optim_options = {
                "distortion_fun": get_distortion_compas,
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [0.1, 0.05, 0],
            }
            return optim_options


def preprocessing(args, dataset: StandardDataset) -> BinaryLabelDataset:
    match args.preprocess.lower():
        case "rw":
            dataset = run_reweigh(args, dataset)
        case "lfr":
            dataset = run_lfr(args, dataset)
        case "dir":
            dataset = run_dir(args, dataset)
        case "op":
            dataset = run_op(args, dataset)
        case "none":
            dataset = dataset
        case _:
            raise ValueError

    return dataset


def run_reweigh(args, dataset):
    """
    Reweighs the dataset. Turns the dataframe into a BinaryLabelDataset.

    Returns: A 2 length tuple, with item 0 being the dataset and 1 being the new weights.
    """
    logger.info("Using Reweighing on the Dataset...")
    privileged_groups, unprivileged_groups = get_privileged_list(args.sensitive)

    rw = Reweighing(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )

    rw = rw.fit(dataset)
    result = rw.transform(dataset)
    # result_test = rw.transform(bld_test)
    return result


def run_dir(args, dataset):
    """
    Uses the DisparateImpactRemover preprocessing technique on the dataset. Turns the dataframe into a BinaryLabelDataset.

    Returns: A 2 length tuple, with item 0 being the dataset and 1 being the new weights.
    """

    logger.info("Using disparateImpactRemover on the Dataset...")

    dir_pp = DisparateImpactRemover(repair_level=1, sensitive_attribute=args.sensitive)

    result = dir_pp.fit_transform(dataset)

    return result


def run_op(args, dataset):
    logger.info("Using OptimPreproc on the Dataset...")

    privileged_groups, unprivileged_groups = get_privileged_list(args.sensitive)

    optim_options = get_optim_options(args)
    op_pp = OptimPreproc(
        OptTools,
        optim_options,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    result = op_pp.fit_transform(dataset)
    return result


def run_lfr(args, dataset):
    """
    Uses LFR preprocessing technique on the dataset. Turns the dataframe into a BinaryLabelDataset.

    Returns: A 2 length tuple, with item 0 being the dataset and 1 being the new weights.
    """
    print("Using LearnedFairRepresentations() on the Dataset...")

    privileged_groups, unprivileged_groups = get_privileged_list(args.sensitive)

    lfr_pp = LFR(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )

    lfr_pp = lfr_pp.fit(dataset)
    result = lfr_pp.transform(dataset)
    return result


def get_privileged_list(sensitive_attr):
    """
    Switch Case of the different sensitive attributes.
    To add new sensitive attributes, add a new case.

    Args:
        sensitive_attr: The Sensitive Attribute specified by the script.

    Returns: 2 Dict(list)s, first for privileged groups, 2nd for unprivileged groups.
    """
    match sensitive_attr:
        case "sex":
            return [{"sex": 1}], [{"sex": 0}]
        case "race":
            return [{"race": 1}], [{"race": 0}]
        case "age":
            return [{"age": 1}], [{"age": 0}]
        case "gender":
            return [{"gender": 1}], [{"gender": 0}]
        case "RACE":
            return [{"RACE": 1}], [{"RACE": 0}]
        case _:
            return [{"sex": 1}], [{"sex": 0}]
