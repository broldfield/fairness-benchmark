from fairness_benchmark.process.metrics import metrics
from fairness_benchmark.process.preprocessing import preprocessing
from fairness_benchmark.process.load_dataset import load_dataset
from fairness_benchmark.utils.loading import save_dataset, to_csv


def process(args):
    dataset = load_dataset(args)
    before_metrics = metrics(args, dataset, "Original")

    save_dataset(args, dataset, "Original")
    dataset_processed = preprocessing(args, dataset)
    after_metrics = metrics(args, dataset_processed, "Processed")

    save_dataset(args, dataset_processed, "Processed")
    to_csv(args, dataset_processed, "Processed")

    to_csv(args, dataset, "Original")

    return
