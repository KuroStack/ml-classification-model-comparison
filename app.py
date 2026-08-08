from sklearn.linear_model import LogisticRegression

from data import FetalHealthDataLoader
from evaluation import EvaluationMetrics


def main() -> None:
    loader = FetalHealthDataLoader(test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test, feature_names = loader.load()

    print("UCI Cardiotocography (id=193)")
    print(f"features: {len(feature_names)}")
    print(f"train: {X_train.shape}  test: {X_test.shape}")
    print(f"classes: {loader.class_names}")

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    evaluator = EvaluationMetrics(n_classes=len(loader.class_names))
    result = evaluator.compute(
        "Logistic Regression",
        y_test,
        clf.predict(X_test),
        clf.predict_proba(X_test),
        class_names=loader.class_names,
    )
    print("\nLogistic Regression (test call)")
    for name, value in result.to_dict().items():
        print(f"  {name}: {value}")
    print(result.classification_rep)


if __name__ == "__main__":
    main()
