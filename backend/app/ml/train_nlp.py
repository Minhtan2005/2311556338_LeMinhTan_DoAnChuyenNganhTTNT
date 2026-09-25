from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from app.services.nlp import tokenize_vietnamese


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "vietnamese_intents.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "intent_classifier.joblib"
MIN_INTENT_SAMPLES = 200


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    tokenizer=tokenize_vietnamese,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )


def load_dataset() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    required_columns = {"text", "intent"}
    if set(data.columns) != required_columns:
        raise ValueError(f"Dataset must contain columns: {sorted(required_columns)}")
    if len(data) < MIN_INTENT_SAMPLES:
        raise ValueError(f"Dataset must contain at least {MIN_INTENT_SAMPLES} Vietnamese intent samples.")
    return data


def train() -> Pipeline:
    data = load_dataset()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"],
        data["intent"],
        test_size=0.2,
        random_state=42,
        stratify=data["intent"],
    )
    model = build_pipeline()
    model.fit(train_texts, train_labels)
    predictions = model.predict(test_texts)
    print(classification_report(test_labels, predictions, digits=4))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved NLP intent model to {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
