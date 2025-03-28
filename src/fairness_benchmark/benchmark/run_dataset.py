from fairness_benchmark.benchmark.models import load_model
from fairness_benchmark.benchmark.plotting import plot_1, plot_2
from fairness_benchmark.benchmark.utils import (
    optim_threshold,
    split,
    splitter,
    threshold,
)
import numpy as np


def run_dataset(args, dataset, type):
    # Split  dataset into Training, Testing, and Validating sets.
    dataset_train, dataset_test, dataset_valid = split(args, dataset)

    ## Original
    X_train, y_train, w_train, s_train = splitter(args, dataset_train)

    # Load desired Model.
    model = load_model(args, dataset)

    # Train on Original
    match args.model.lower():
        case "mlp":
            model.fit(X_train, y_train)
        case "nb":
            model.fit(X_train, y_train, sample_weight=w_train)
        case "lr":
            model.fit(X_train, y_train, sample_weight=w_train)
    y_train_pred = model.predict(X_train)

    # Get Positive Indexes
    pos_ind = np.where(model.classes_ == dataset_train.favorable_label)[0][0]

    # Train
    dataset_train_pred = dataset_train.copy()
    dataset_train_pred.labels = y_train_pred

    # Valid
    dataset_valid_pred = dataset_valid.copy(deepcopy=True)
    X_valid, y_valid, w_valid, s_valid = splitter(args, dataset_valid_pred)
    dataset_valid_pred.scores = model.predict_proba(X_valid)[:, pos_ind].reshape(-1, 1)

    # Test
    dataset_test_pred = dataset_test.copy(deepcopy=True)
    X_test, y_test, w_test, s_test = splitter(args, dataset_test_pred)
    dataset_test_pred.scores = model.predict_proba(X_test)[:, pos_ind].reshape(-1, 1)

    best_class_threshold, class_threshold = optim_threshold(
        args, dataset_valid, dataset_valid_pred
    )

    bal, avg, disp = threshold(
        args,
        dataset_test,
        dataset_test_pred,
        best_class_threshold,
        class_threshold,
    )

    return best_class_threshold, class_threshold, bal, disp, avg
