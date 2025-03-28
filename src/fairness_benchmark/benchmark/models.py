from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def load_model(args, dataset):
    match args.model.lower():
        case "nb":
            return CategoricalNB()
        case "lr":
            return LogisticRegression(max_iter=1000)
        case "mlp":
            return MLPClassifier()
        case _:
            return CategoricalNB()
