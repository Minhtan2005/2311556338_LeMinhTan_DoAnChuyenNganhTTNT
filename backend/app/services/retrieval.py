from dataclasses import dataclass

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import Detection, Event, TranscriptSegment
from app.schemas import RetrievedSegment
from app.services.nlp import QueryUnderstanding


@dataclass
class EvidenceDocument:
    start_time: float
    end_time: float
    label: str
    text: str
    confidence: float


class VideoRetrievalService:
    def retrieve(self, db: Session, video_id: str, question: str, understanding: QueryUnderstanding) -> tuple[list[RetrievedSegment], list[str]]:
        intent = understanding.intent
        entities = understanding.entities

        if intent == "summarize_video":
            return self._summarize(db, video_id)
        if intent == "count_object":
            return self._count_object(db, video_id, entities)
        if intent in {"search_object", "find_timestamp"} and entities.get("object"):
            return self._search_object(db, video_id, entities)
        if intent == "search_action" and entities.get("action"):
            return self._search_action(db, video_id, entities)

        docs = self._build_documents(db, video_id)
        return self._semantic_search(question, docs)

    def _search_object(self, db: Session, video_id: str, entities: dict) -> tuple[list[RetrievedSegment], list[str]]:
        object_name = entities["object"]
        rows = (
            db.query(
                Detection.track_id,
                func.min(Detection.timestamp).label("start"),
                func.max(Detection.timestamp).label("end"),
                func.max(Detection.confidence).label("confidence"),
            )
            .filter(Detection.video_id == video_id, Detection.class_name == object_name)
            .group_by(Detection.track_id)
            .order_by(func.min(Detection.timestamp))
            .limit(20)
            .all()
        )
        segments = [
            RetrievedSegment(
                start_time=float(row.start),
                end_time=float(row.end),
                label=f"{object_name} track #{row.track_id}" if row.track_id is not None else object_name,
                confidence=float(row.confidence or 1.0),
                evidence=f"{object_name} xuất hiện từ giây {row.start:.1f} đến {row.end:.1f}.",
            )
            for row in rows
        ]
        facts = [segment.evidence for segment in segments]
        return segments, facts

    def _search_action(self, db: Session, video_id: str, entities: dict) -> tuple[list[RetrievedSegment], list[str]]:
        action = entities["action"]
        query = db.query(Event).filter(Event.video_id == video_id, Event.event_type == action)
        if entities.get("object"):
            query = query.filter(Event.object_type == entities["object"])
        events = query.order_by(Event.timestamp_start).limit(20).all()
        segments = [self._event_to_segment(event) for event in events]
        return segments, [event.description for event in events]

    def _count_object(self, db: Session, video_id: str, entities: dict) -> tuple[list[RetrievedSegment], list[str]]:
        object_name = entities.get("object")
        if object_name:
            query = db.query(Detection.track_id).filter(Detection.video_id == video_id, Detection.class_name == object_name)
            distinct_tracks = {track_id for (track_id,) in query.all() if track_id is not None}
            if distinct_tracks:
                count = len(distinct_tracks)
            else:
                count = query.count()
            fact = f"Số lượng {object_name} ước tính: {count}."
            return [], [fact]

        rows = (
            db.query(Detection.class_name, Detection.track_id)
            .filter(Detection.video_id == video_id)
            .all()
        )
        counts: dict[str, set[int] | int] = {}
        for class_name, track_id in rows:
            if track_id is None:
                counts[class_name] = int(counts.get(class_name, 0)) + 1
            else:
                value = counts.setdefault(class_name, set())
                if isinstance(value, set):
                    value.add(track_id)

        facts = []
        for class_name, value in sorted(counts.items()):
            count = len(value) if isinstance(value, set) else value
            facts.append(f"{class_name}: {count}")
        return [], facts

    def _summarize(self, db: Session, video_id: str) -> tuple[list[RetrievedSegment], list[str]]:
        rows = (
            db.query(Event.event_type, Event.object_type, func.count(Event.id))
            .filter(Event.video_id == video_id)
            .group_by(Event.event_type, Event.object_type)
            .order_by(func.count(Event.id).desc())
            .all()
        )
        facts = [f"{event_type} - {object_type or 'all'}: {count}" for event_type, object_type, count in rows]
        recent_events = db.query(Event).filter(Event.video_id == video_id).order_by(Event.timestamp_start).limit(10).all()
        segments = [self._event_to_segment(event) for event in recent_events]
        return segments, facts

    def _build_documents(self, db: Session, video_id: str) -> list[EvidenceDocument]:
        events = db.query(Event).filter(Event.video_id == video_id).order_by(Event.timestamp_start).all()
        transcripts = db.query(TranscriptSegment).filter(TranscriptSegment.video_id == video_id).order_by(TranscriptSegment.start_time).all()
        docs = [
            EvidenceDocument(
                start_time=event.timestamp_start,
                end_time=event.timestamp_end or event.timestamp_start + 2,
                label=event.event_type,
                text=event.description,
                confidence=event.confidence,
            )
            for event in events
        ]
        docs.extend(
            EvidenceDocument(
                start_time=segment.start_time,
                end_time=segment.end_time,
                label="speech",
                text=segment.text,
                confidence=1.0,
            )
            for segment in transcripts
        )
        return docs

    def _semantic_search(self, question: str, docs: list[EvidenceDocument]) -> tuple[list[RetrievedSegment], list[str]]:
        if not docs:
            return [], []

        texts = [doc.text for doc in docs]
        if not settings.enable_faiss:
            selected = self._numpy_search(question, texts)
        else:
            selected = self._faiss_search(question, texts)

        segments = [
            RetrievedSegment(
                start_time=docs[index].start_time,
                end_time=docs[index].end_time,
                label=docs[index].label,
                confidence=max(score, docs[index].confidence),
                evidence=docs[index].text,
            )
            for index, score in selected
        ]
        return segments, [segment.evidence for segment in segments]

    def _faiss_search(self, question: str, texts: list[str]) -> list[tuple[int, float]]:
        try:
            import faiss
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(texts + [question]).toarray().astype("float32")
            doc_vectors = matrix[:-1]
            query_vector = matrix[-1:]
            faiss.normalize_L2(doc_vectors)
            faiss.normalize_L2(query_vector)
            index = faiss.IndexFlatIP(doc_vectors.shape[1])
            index.add(doc_vectors)
            scores, indices = index.search(query_vector, min(5, len(texts)))
            selected = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0 and score > 0]
        except Exception:
            selected = self._numpy_search(question, texts)
        return selected

    @staticmethod
    def _numpy_search(question: str, texts: list[str]) -> list[tuple[int, float]]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts + [question]).toarray()
        docs = matrix[:-1]
        query = matrix[-1]
        denom = np.linalg.norm(docs, axis=1) * max(np.linalg.norm(query), 1e-9)
        scores = np.divide(docs @ query, denom, out=np.zeros(len(docs)), where=denom > 0)
        top = np.argsort(scores)[::-1][:5]
        return [(int(index), float(scores[index])) for index in top if scores[index] > 0]

    @staticmethod
    def _event_to_segment(event: Event) -> RetrievedSegment:
        return RetrievedSegment(
            start_time=event.timestamp_start,
            end_time=event.timestamp_end or event.timestamp_start + 2,
            label=event.event_type,
            confidence=event.confidence,
            evidence=event.description,
        )


retrieval_service = VideoRetrievalService()
