from app.services.events import EventDetector


def test_wrong_way_event_for_reverse_motion() -> None:
    detector = EventDetector(frame_width=1000, frame_height=600)

    detector.observe(
        video_id="v1",
        track_id=1,
        class_name="car",
        timestamp=0.0,
        confidence=0.9,
        bbox=(800, 100, 900, 200),
        active_vehicle_count=1,
    )
    events = detector.observe(
        video_id="v1",
        track_id=1,
        class_name="car",
        timestamp=2.0,
        confidence=0.9,
        bbox=(500, 100, 600, 200),
        active_vehicle_count=1,
    )

    assert any(event.event_type == "wrong_way" for event in events)


def test_finalize_emits_leave_event() -> None:
    detector = EventDetector(frame_width=1000, frame_height=600)
    detector.observe(
        video_id="v1",
        track_id=7,
        class_name="truck",
        timestamp=1.0,
        confidence=0.8,
        bbox=(100, 100, 180, 180),
        active_vehicle_count=1,
    )

    events = detector.finalize("v1")

    assert events[0].event_type == "vehicle_leaves"
    assert events[0].track_id == 7
