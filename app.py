from data import FetalHealthDataLoader
from evaluation import EvaluationMetrics
from model import DecisionTreeModel, KNNModel, LogisticRegressionModel


def main() -> None:
    loader = FetalHealthDataLoader(test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test, feature_names = loader.load()

    print("UCI Cardiotocography (id=193)")
    print(f"features: {len(feature_names)}")
    print(f"train: {X_train.shape}  test: {X_test.shape}")
    print(f"classes: {loader.class_names}")

    evaluator = EvaluationMetrics(n_classes=len(loader.class_names))
    for clf in (LogisticRegressionModel(), DecisionTreeModel(), KNNModel()):
        clf.train(X_train, y_train)
        result = evaluator.compute(
            clf.model_name,
            y_test,
            clf.predict(X_test),
            clf.predict_proba(X_test),
            class_names=loader.class_names,
        )
        print(f"\n{clf.model_name}")
        for name, value in result.to_dict().items():
            print(f"  {name}: {value}")
        print(result.classification_rep)


if __name__ == "__main__":
    main()
