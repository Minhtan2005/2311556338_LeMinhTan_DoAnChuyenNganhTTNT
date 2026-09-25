from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.api.videos import recover_processing_video_if_complete, to_video_out
from app.db.session import Base
from app.models import Detection, Event, Video


def make_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return TestingSessionLocal()


def test_completed_processing_video_is_recovered_without_deleting_artifacts() -> None:
    db = make_session()
    try:
        video = Video(
            id="video-1",
            original_filename="complete.mp4",
            stored_filename="complete.mp4",
            stored_path="uploads/complete.mp4",
            status="processing",
            progress_percent=100,
            duration_seconds=35.5,
            fps=24.0,
            total_frames=852,
            created_at=datetime.utcnow(),
        )
        db.add(video)
        db.add(
            Detection(
                video_id=video.id,
                frame_index=0,
                timestamp=0.0,
                track_id=54,
                class_name="car",
                confidence=0.9,
                x1=1,
                y1=2,
                x2=3,
                y2=4,
            )
        )
        db.add(
            Event(
                video_id=video.id,
                event_type="vehicle_enters",
                object_type="car",
                track_id=54,
                timestamp_start=0.0,
                confidence=0.9,
                description="car enters",
            )
        )
        db.commit()

        payload = to_video_out(video, db)

        assert payload.status == "completed"
        assert payload.progress_percent == 100
        assert payload.processed_at is not None
        assert payload.detection_count == 1
        assert payload.track_count == 1
        assert payload.event_count == 1
        assert db.query(Detection).filter(Detection.video_id == video.id).count() == 1
        assert db.query(Event).filter(Event.video_id == video.id).count() == 1
    finally:
        db.close()


def test_processing_video_below_complete_progress_is_not_recovered() -> None:
    db = make_session()
    try:
        video = Video(
            id="video-2",
            original_filename="running.mp4",
            stored_filename="running.mp4",
            stored_path="uploads/running.mp4",
            status="processing",
            progress_percent=99,
            duration_seconds=35.5,
            fps=24.0,
            total_frames=852,
            created_at=datetime.utcnow(),
        )
        db.add(video)
        db.commit()

        assert recover_processing_video_if_complete(video, db) is False
        payload = to_video_out(video, db)

        assert payload.status == "processing"
        assert payload.progress_percent == 99
        assert payload.processed_at is None
    finally:
        db.close()
