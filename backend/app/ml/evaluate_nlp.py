from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from app.ml.train_nlp import DATA_PATH, MODEL_PATH, load_dataset, train


def evaluate() -> None:
    if not Path(MODEL_PATH).exists():
        print("Model artifact not found. Training a model first.")
        train()

    data = load_dataset()
    _, test_texts, _, test_labels = train_test_split(
        data["text"],
        data["intent"],
        test_size=0.2,
        random_state=42,
        stratify=data["intent"],
    )
    model = joblib.load(MODEL_PATH)
    predictions = model.predict(test_texts)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    print(f"Dataset: {DATA_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(classification_report(test_labels, predictions, digits=4, zero_division=0))


if __name__ == "__main__":
    evaluate()
