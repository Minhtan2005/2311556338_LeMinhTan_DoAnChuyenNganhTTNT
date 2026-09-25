from __future__ import annotations

import logging
import unicodedata
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)

MAX_QA_ANSWER_LENGTH = 100
INSUFFICIENT_EVIDENCE_ANSWER = "Không đủ dữ liệu thị giác để trả lời"

OBJECT_QA_ALIASES = {
    "car": ["car", "ô tô", "oto", "xe ô tô", "xe oto", "xe hơi", "xe con"],
    "person": ["person", "người", "nguoi", "người đi bộ", "nguoi di bo", "pedestrian", "man", "woman"],
    "motorcycle": ["motorcycle", "xe máy", "xe may", "mô tô", "mo to", "motorbike", "scooter"],
    "bicycle": ["bicycle", "xe đạp", "xe dap", "bike"],
    "bus": ["bus", "xe buýt", "xe buyt", "xe bus"],
    "truck": ["truck", "xe tải", "xe tai"],
}

OBJECT_QA_LABELS = {
    "car": "ô tô",
    "person": "người",
    "motorcycle": "xe máy",
    "bicycle": "xe đạp",
    "bus": "xe buýt",
    "truck": "xe tải",
}


@dataclass(frozen=True)
class VisualQAResult:
    answer: str
    confidence: float
    model: str
    latency_ms: float
    evidence: str | None = None
    is_insufficient_evidence: bool = False


class VisualQAService:
    def __init__(self) -> None:
        self.model_name = settings.gemini_model

    def answer_question(
        self,
        image_path: str | Path,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> VisualQAResult:
        """
        Visually answer question based strictly on the keyframe image.
        Guarantees answer length <= 100 characters.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            return VisualQAResult(
                answer="Hình ảnh không tồn tại",
                confidence=0.0,
                model="system",
                latency_ms=0.0,
                is_insufficient_evidence=True,
            )

        clean_question = question.strip()
        if not clean_question:
            return VisualQAResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                confidence=0.0,
                model="system",
                latency_ms=0.0,
                is_insufficient_evidence=True,
            )

        # 1. Attempt Gemini Multimodal Vision only when the separate vision flag is enabled.
        if settings.gemini_vision_enabled and settings.gemini_api_key:
            try:
                gemini_result = self._call_gemini_vision(img_path, clean_question, context)
                if not gemini_result.is_insufficient_evidence:
                    return gemini_result
                local_result = self._local_fallback(clean_question, context)
                if not local_result.is_insufficient_evidence:
                    return local_result
                return gemini_result
            except Exception as exc:
                logger.warning("Gemini Vision Q&A failed (%s). Falling back to safe response.", exc)

        # 2. Local fallback if Gemini is offline or fails
        return self._local_fallback(clean_question, context)

    def _call_gemini_vision(
        self,
        image_path: Path,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> VisualQAResult:
        from google import genai
        from google.genai import types

        t0 = time.perf_counter()
        image = Image.open(image_path).convert("RGB")

        client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_seconds * 1000)),
        )

        prompt = (
            "Bạn là trợ lý AI trả lời câu hỏi thị giác cho cuộc thi truy vấn video.\n"
            f"Câu hỏi: {question}\n\n"
            "Quy tắc trả lời:\n"
            "1. Trả lời trực tiếp, chính xác và ngắn gọn dựa HOÀN TOÀN vào hình ảnh được cung cấp.\n"
            "2. Độ dài câu trả lời BẮT BUỘC dưới 90 ký tự. Không lặp lại câu hỏi. Không giải thích dài dòng.\n"
            "3. Nếu đối tượng, hành động, hoặc chi tiết được hỏi KHÔNG nhìn thấy rõ ràng trong hình, "
            f"hãy trả lời đúng câu: '{INSUFFICIENT_EVIDENCE_ANSWER}'. Tuyệt đối không suy đoán hoặc bịa đặt thông tin.\n"
            "4. Nếu câu hỏi bằng tiếng Anh, trả lời bằng tiếng Anh. Nếu câu hỏi bằng tiếng Việt, trả lời bằng tiếng Việt."
        )

        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=150,
            ),
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        raw_text = (getattr(response, "text", "") or "").strip()

        if not raw_text:
            return VisualQAResult(
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                confidence=0.3,
                model=self.model_name,
                latency_ms=latency_ms,
                is_insufficient_evidence=True,
            )

        cleaned_answer = self._clean_and_truncate_answer(raw_text)
        is_insufficient = (
            INSUFFICIENT_EVIDENCE_ANSWER.lower() in cleaned_answer.lower()
            or "insufficient" in cleaned_answer.lower()
            or "không đủ" in cleaned_answer.lower()
            or "không thấy" in cleaned_answer.lower()
            or "không rõ" in cleaned_answer.lower()
        )
        if is_insufficient:
            cleaned_answer = INSUFFICIENT_EVIDENCE_ANSWER

        return VisualQAResult(
            answer=cleaned_answer,
            confidence=0.5 if is_insufficient else 0.95,
            model=self.model_name,
            latency_ms=round(latency_ms, 2),
            evidence=cleaned_answer,
            is_insufficient_evidence=is_insufficient,
        )

    def _local_fallback(
        self,
        question: str,
        context: dict[str, Any] | None = None,
    ) -> VisualQAResult:
        """Heuristic answer when Gemini Vision API is offline or unavailable."""
        if context:
            target_class = self._target_class_from_question(question)
            if target_class:
                detections = context.get("video_detections") if self._asks_whole_video(question) else context.get("detections")
                detections = list(detections or [])
                matched_count = self._count_unique_objects(detections, target_class)
                label = OBJECT_QA_LABELS[target_class]

                if self._asks_count(question):
                    scope = "trong video" if self._asks_whole_video(question) else "trong khung hình"
                    return VisualQAResult(
                        answer=self._clean_and_truncate_answer(f"Có {matched_count} {label} {scope}."),
                        confidence=0.85,
                        model="yolo_detection_fallback",
                        latency_ms=0.0,
                        evidence=(
                            f"YOLO detections: class={target_class}, count={matched_count}, "
                            f"scope={context.get('detection_scope', 'unknown')}"
                        ),
                    )

                if self._asks_presence(question):
                    answer = f"Có {label} trong khung hình." if matched_count > 0 else f"Không thấy {label} trong khung hình."
                    return VisualQAResult(
                        answer=self._clean_and_truncate_answer(answer),
                        confidence=0.8,
                        model="yolo_detection_fallback",
                        latency_ms=0.0,
                        evidence=(
                            f"YOLO detections: class={target_class}, count={matched_count}, "
                            f"scope={context.get('detection_scope', 'unknown')}"
                        ),
                    )

        return VisualQAResult(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            confidence=0.0,
            model="local_fallback",
            latency_ms=0.0,
            is_insufficient_evidence=True,
        )

    @staticmethod
    def _clean_and_truncate_answer(raw_text: str) -> str:
        """Sanitizes text and enforces strict <= 100 characters limit."""
        # Replace newlines, carriage returns, tabs
        cleaned = " ".join(raw_text.replace("\r", " ").replace("\n", " ").split())
        # Remove markdown quotes or code blocks if present
        if cleaned.startswith(('"', "'", "`")) and cleaned.endswith(('"', "'", "`")) and len(cleaned) > 2:
            cleaned = cleaned[1:-1].strip()

        # Enforce <= 100 chars
        if len(cleaned) > MAX_QA_ANSWER_LENGTH:
            cleaned = cleaned[:MAX_QA_ANSWER_LENGTH].rstrip()

        return cleaned or INSUFFICIENT_EVIDENCE_ANSWER

    @classmethod
    def _target_class_from_question(cls, question: str) -> str | None:
        normalized = cls._normalize_for_match(question)
        for class_name, aliases in OBJECT_QA_ALIASES.items():
            if any(cls._normalize_for_match(alias) in normalized for alias in aliases):
                return class_name
        return None

    @classmethod
    def _asks_count(cls, question: str) -> bool:
        normalized = cls._normalize_for_match(question)
        return any(token in normalized for token in ["bao nhieu", "so luong", "dem", "how many"])

    @classmethod
    def _asks_presence(cls, question: str) -> bool:
        normalized = cls._normalize_for_match(question)
        return any(token in normalized for token in ["co ", "khong", "co thay", "xuat hien"])

    @classmethod
    def _asks_whole_video(cls, question: str) -> bool:
        normalized = cls._normalize_for_match(question)
        return any(token in normalized for token in ["trong video", "toan bo", "ca video", "tong so", "duy nhat"])

    @staticmethod
    def _count_unique_objects(detections: list[Any], class_name: str) -> int:
        matched = [item for item in detections if str(getattr(item, "class_name", "")).lower() == class_name]
        track_ids = {int(track_id) for track_id in (getattr(item, "track_id", None) for item in matched) if track_id is not None}
        if track_ids:
            return len(track_ids)
        return len(matched)

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(char for char in text if unicodedata.category(char) != "Mn")
        return " ".join(text.split())


visual_qa_service = VisualQAService()
