from app.core.config import settings
from app.schemas import RetrievedSegment
from app.services.nlp import QueryUnderstanding


class NaturalAnswerService:
    def generate(
        self,
        *,
        question: str,
        understanding: QueryUnderstanding,
        facts: list[str],
        segments: list[RetrievedSegment],
    ) -> str:
        if not facts and not segments:
            return "Mình chưa tìm thấy bằng chứng phù hợp trong dữ liệu đã phân tích của video."

        deterministic = self._deterministic_answer(understanding, facts, segments)
        if not settings.enable_openai or not settings.openai_api_key:
            return deterministic

        try:
            from openai import OpenAI

            client = OpenAI(api_key=settings.openai_api_key)
            evidence = "\n".join(f"- {fact}" for fact in facts[:20])
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn chỉ diễn đạt câu trả lời tiếng Việt dựa trên evidence được cung cấp. "
                            "Không thêm thông tin ngoài evidence. Nếu evidence rỗng, nói không tìm thấy."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Câu hỏi: {question}\nIntent: {understanding.intent}\nEvidence:\n{evidence}",
                    },
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or deterministic
        except Exception:
            return deterministic

    @staticmethod
    def _deterministic_answer(
        understanding: QueryUnderstanding,
        facts: list[str],
        segments: list[RetrievedSegment],
    ) -> str:
        if understanding.intent == "count_object":
            return " ".join(facts) if facts else "Không có đủ dữ liệu để đếm đối tượng này."
        if understanding.intent == "summarize_video":
            if not facts:
                return "Video chưa có event nổi bật để tóm tắt."
            return "Tóm tắt từ pipeline phân tích: " + "; ".join(facts[:10]) + "."
        if segments:
            times = ", ".join(f"{segment.start_time:.1f}s-{segment.end_time:.1f}s" for segment in segments[:8])
            return f"Tìm thấy {len(segments)} đoạn phù hợp: {times}."
        return " ".join(facts[:5])


natural_answer_service = NaturalAnswerService()
