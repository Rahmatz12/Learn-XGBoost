"""Train a small XGBoost classifier on scikit-learn's breast cancer dataset."""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main() -> None:
    dataset = load_breast_cancer()
    features_train, features_test, labels_train, labels_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42,
        stratify=dataset.target,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features_train, labels_train)

    predictions = model.predict(features_test)
    probabilities = model.predict_proba(features_test)[:, 1]
    print(f"Accuracy: {accuracy_score(labels_test, predictions):.3f}")
    print(f"ROC AUC:  {roc_auc_score(labels_test, probabilities):.3f}")


if __name__ == "__main__":
    main()