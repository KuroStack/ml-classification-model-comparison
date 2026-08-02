from data import FetalHealthDataLoader


def main() -> None:
    loader = FetalHealthDataLoader(test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test, feature_names = loader.load()

    print("UCI Cardiotocography (id=193)")
    print(f"features: {len(feature_names)}")
    print(f"train: {X_train.shape}  test: {X_test.shape}")
    print(f"classes: {loader.class_names}")
    for i, name in enumerate(loader.class_names):
        print(f"  {name}: train={(y_train == i).sum()}  test={(y_test == i).sum()}")


if __name__ == "__main__":
    main()
