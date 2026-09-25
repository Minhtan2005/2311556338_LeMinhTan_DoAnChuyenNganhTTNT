import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


logger = logging.getLogger(__name__)


INTENT_SAMPLES: list[tuple[str, str]] = [
    ("tìm đoạn có xe máy", "search_object"),
    ("cho tôi xem lúc có ô tô", "search_object"),
    ("tìm xe tải trong video", "search_object"),
    ("đoạn nào xuất hiện xe buýt", "search_object"),
    ("có người đi bộ không", "search_object"),
    ("tìm xe đi ngược chiều", "search_action"),
    ("phương tiện nào dừng lại", "search_action"),
    ("có ùn tắc giao thông không", "search_action"),
    ("xe vào vùng cấm lúc nào", "search_action"),
    ("xe tải xuất hiện lúc nào", "find_timestamp"),
    ("khi nào có xe máy", "find_timestamp"),
    ("timestamp của xe buýt", "find_timestamp"),
    ("ô tô xuất hiện thời điểm nào", "find_timestamp"),
    ("có bao nhiêu ô tô", "count_object"),
    ("đếm số xe máy", "count_object"),
    ("bao nhiêu xe tải trong video", "count_object"),
    ("số lượng xe buýt là bao nhiêu", "count_object"),
    ("tóm tắt video", "summarize_video"),
    ("hãy tổng hợp nội dung video", "summarize_video"),
    ("video này có gì đáng chú ý", "summarize_video"),
    ("tổng quan tình hình giao thông", "summarize_video"),
]


OBJECT_ALIASES = {
    "person": ["người", "người đi bộ", "pedestrian", "person", "man", "woman"],
    "car": ["ô tô", "oto", "xe hơi", "xe con", "car"],
    "motorcycle": ["xe máy", "mô tô", "motorbike", "motorcycle"],
    "truck": ["xe tải", "truck"],
    "bus": ["xe buýt", "xe bus", "bus"],
    "bicycle": ["xe đạp", "bicycle", "bike"],
    "traffic light": ["đèn giao thông", "đèn đỏ", "đèn xanh", "traffic light"],
    "backpack": ["ba lô", "balo", "backpack", "bag", "túi"],
}

ACTION_ALIASES = {
    "vehicle_appears": ["xuất hiện", "đi vào", "có mặt"],
    "vehicle_leaves": ["rời khỏi", "biến mất", "đi ra"],
    "wrong_way": ["ngược chiều", "đi ngược", "sai chiều"],
    "vehicle_stopped": ["dừng", "đứng yên", "đỗ"],
    "traffic_congestion": ["ùn tắc", "kẹt xe", "đông xe", "tắc đường"],
    "enter_restricted_area": ["vùng cấm", "vùng hạn chế", "khu vực cấm"],
    "standing": ["đứng", "standing", "stand"],
    "walking": ["đi bộ", "đi qua", "băng qua", "walking", "crossing"],
    "next_to": ["cạnh", "bên cạnh", "gần", "next to", "near"],
}

COLOR_ALIASES = {
    "red": ["đỏ", "màu đỏ", "red"],
    "blue": ["xanh dương", "màu xanh dương", "blue"],
    "green": ["xanh lá", "màu xanh lá", "green"],
    "white": ["trắng", "màu trắng", "white"],
    "black": ["đen", "màu đen", "black"],
    "yellow": ["vàng", "màu vàng", "yellow"],
    "gray": ["xám", "ghi", "gray", "grey"],
}

DIRECTION_ALIASES = {
    "left_to_right": ["trái sang phải", "từ trái qua phải"],
    "right_to_left": ["phải sang trái", "từ phải qua trái"],
    "top_to_bottom": ["trên xuống dưới", "từ trên xuống"],
    "bottom_to_top": ["dưới lên trên", "từ dưới lên"],
    "wrong_way": ["ngược chiều", "sai chiều"],
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def tokenize_vietnamese(text: str) -> list[str]:
    text = normalize_text(text)
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").split()
    except Exception:
        return text.split()


@dataclass(frozen=True)
class QueryUnderstanding:
    intent: str
    entities: dict[str, Any]


class VietnameseQueryParser:
    def __init__(self) -> None:
        self.model_path = Path(__file__).resolve().parents[1] / "ml" / "models" / "intent_classifier.joblib"
        self.intent_model: Pipeline = self._load_or_train_model()

    def _load_or_train_model(self) -> Pipeline:
        if self.model_path.exists():
            return joblib.load(self.model_path)

        logger.warning(
            "NLP intent model not found at %s. Training an in-memory fallback from CSV samples. "
            "Run `python scripts/train_nlp.py` from backend to create the persisted model.",
            self.model_path,
        )
        samples = self._load_dataset_samples()
        texts = [sample for sample, _ in samples]
        labels = [label for _, label in samples]
        model: Pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        tokenizer=tokenize_vietnamese,
                        token_pattern=None,
                        ngram_range=(1, 2),
                        min_df=1,
                    ),
                ),
                ("clf", LinearSVC(class_weight="balanced")),
            ]
        )
        model.fit(texts, labels)
        return model

    @staticmethod
    def _load_dataset_samples() -> list[tuple[str, str]]:
        dataset_path = Path(__file__).resolve().parents[1] / "ml" / "data" / "vietnamese_intents.csv"
        if not dataset_path.exists():
            return INTENT_SAMPLES

        samples: list[tuple[str, str]] = []
        with dataset_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = normalize_text(row.get("text", ""))
                intent = row.get("intent", "").strip()
                if text and intent:
                    samples.append((text, intent))
        return samples or INTENT_SAMPLES

    def parse(self, question: str) -> QueryUnderstanding:
        normalized = normalize_text(question)
        intent = str(self.intent_model.predict([normalized])[0])
        entities = self._extract_entities(normalized)

        if any(word in normalized for word in ["bao nhiêu", "đếm", "số lượng"]):
            intent = "count_object"
        elif any(word in normalized for word in ["tóm tắt", "tổng hợp", "tổng quan"]):
            intent = "summarize_video"
        elif any(word in normalized for word in ["khi nào", "lúc nào", "thời điểm", "timestamp"]):
            intent = "find_timestamp"
        elif entities.get("action"):
            intent = "search_action"
        elif entities.get("object"):
            intent = "search_object"

        return QueryUnderstanding(intent=intent, entities=entities)

    def _extract_entities(self, text: str) -> dict[str, Any]:
        entities: dict[str, Any] = {}
        entities["object"] = self._first_alias(text, OBJECT_ALIASES)
        entities["vehicle_type"] = entities["object"] if entities["object"] in {"car", "motorcycle", "truck", "bus", "bicycle"} else None
        entities["action"] = self._first_alias(text, ACTION_ALIASES)
        entities["color"] = self._first_alias(text, COLOR_ALIASES)
        entities["direction"] = self._first_alias(text, DIRECTION_ALIASES)
        entities["time"] = self._extract_time(text)
        return {key: value for key, value in entities.items() if value is not None}

    @staticmethod
    def _first_alias(text: str, aliases: dict[str, list[str]]) -> str | None:
        for canonical, words in aliases.items():
            if any(word in text for word in words):
                return canonical
        return None

    @staticmethod
    def _extract_time(text: str) -> dict[str, float] | None:
        range_match = re.search(r"từ\s+(\d+(?:[.,]\d+)?)\s*(?:giây|s)?\s+đến\s+(\d+(?:[.,]\d+)?)", text)
        if range_match:
            return {
                "start": float(range_match.group(1).replace(",", ".")),
                "end": float(range_match.group(2).replace(",", ".")),
            }

        second_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:giây|s)\b", text)
        if second_match:
            value = float(second_match.group(1).replace(",", "."))
            return {"start": max(value - 2, 0), "end": value + 2}
        return None


query_parser = VietnameseQueryParser()
