from dataclasses import dataclass, field
from math import hypot

from app.core.config import settings
from app.models import Event


VEHICLE_CLASSES = {"car", "motorcycle", "truck", "bus", "bicycle"}


@dataclass
class TrackedObjectState:
    track_id: int
    class_name: str
    first_seen: float
    last_seen: float
    first_center: tuple[float, float]
    last_center: tuple[float, float]
    confidences: list[float] = field(default_factory=list)
    appeared_emitted: bool = False
    stopped_emitted: bool = False
    wrong_way_emitted: bool = False
    restricted_emitted: bool = False
    low_movement_since: float | None = None


class EventDetector:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.tracks: dict[int, TrackedObjectState] = {}
        self._last_congestion_second: int | None = None
        self._appearance_buckets: set[tuple[str, float]] = set()
        self._leave_buckets: set[tuple[str, float]] = set()
        self._restricted_polygon = self._parse_polygon(settings.restricted_area_polygon)

    def observe(
        self,
        *,
        video_id: str,
        track_id: int | None,
        class_name: str,
        timestamp: float,
        confidence: float,
        bbox: tuple[float, float, float, float],
        active_vehicle_count: int,
    ) -> list[Event]:
        if track_id is None:
            return []

        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        events: list[Event] = []
        state = self.tracks.get(track_id)

        if state is None:
            state = TrackedObjectState(
                track_id=track_id,
                class_name=class_name,
                first_seen=timestamp,
                last_seen=timestamp,
                first_center=center,
                last_center=center,
                confidences=[confidence],
            )
            self.tracks[track_id] = state
            state.appeared_emitted = True
            appearance_key = (class_name, round(timestamp, 1))
            if appearance_key in self._appearance_buckets:
                return events
            self._appearance_buckets.add(appearance_key)
            events.append(
                self._event(
                    video_id,
                    "vehicle_appears" if class_name in VEHICLE_CLASSES else "object_appears",
                    class_name,
                    track_id,
                    timestamp,
                    timestamp,
                    confidence,
                    f"{self._vi_name(class_name)} track #{track_id} xuất hiện tại giây {timestamp:.1f}.",
                )
            )
            return events

        distance = hypot(center[0] - state.last_center[0], center[1] - state.last_center[1])
        if distance <= settings.stopped_max_movement_px:
            state.low_movement_since = state.low_movement_since or state.last_seen
        else:
            state.low_movement_since = None

        state.last_seen = timestamp
        state.last_center = center
        state.confidences.append(confidence)

        if class_name in VEHICLE_CLASSES:
            events.extend(self._detect_wrong_way(video_id, state, timestamp, confidence))
            events.extend(self._detect_stopped(video_id, state, timestamp, confidence))
            events.extend(self._detect_restricted_area(video_id, state, timestamp, confidence, center))
            events.extend(self._detect_congestion(video_id, timestamp, active_vehicle_count))

        return events

    def finalize(self, video_id: str) -> list[Event]:
        events: list[Event] = []
        for state in self.tracks.values():
            leave_key = (state.class_name, round(state.last_seen, 1))
            if leave_key in self._leave_buckets:
                continue
            self._leave_buckets.add(leave_key)
            mean_confidence = sum(state.confidences) / max(len(state.confidences), 1)
            events.append(
                self._event(
                    video_id,
                    "vehicle_leaves" if state.class_name in VEHICLE_CLASSES else "object_leaves",
                    state.class_name,
                    state.track_id,
                    state.last_seen,
                    state.last_seen,
                    mean_confidence,
                    f"{self._vi_name(state.class_name)} track #{state.track_id} rời khỏi khung hình tại giây {state.last_seen:.1f}.",
                )
            )
        return events

    def _detect_wrong_way(
        self,
        video_id: str,
        state: TrackedObjectState,
        timestamp: float,
        confidence: float,
    ) -> list[Event]:
        if state.wrong_way_emitted or timestamp - state.first_seen < 1.5:
            return []

        dx = state.last_center[0] - state.first_center[0]
        dy = state.last_center[1] - state.first_center[1]
        min_motion = max(self.frame_width, self.frame_height) * 0.08

        if hypot(dx, dy) < min_motion:
            return []

        expected = settings.expected_traffic_direction
        if expected not in {"left_to_right", "right_to_left", "top_to_bottom", "bottom_to_top"}:
            return []
        wrong_way = (
            (expected == "left_to_right" and dx < -min_motion)
            or (expected == "right_to_left" and dx > min_motion)
            or (expected == "top_to_bottom" and dy < -min_motion)
            or (expected == "bottom_to_top" and dy > min_motion)
        )
        if not wrong_way:
            return []

        state.wrong_way_emitted = True
        return [
            self._event(
                video_id,
                "wrong_way",
                state.class_name,
                state.track_id,
                state.first_seen,
                timestamp,
                confidence,
                f"{self._vi_name(state.class_name)} track #{state.track_id} có dấu hiệu đi ngược chiều.",
            )
        ]

    def _detect_stopped(
        self,
        video_id: str,
        state: TrackedObjectState,
        timestamp: float,
        confidence: float,
    ) -> list[Event]:
        if state.stopped_emitted or state.low_movement_since is None:
            return []
        if timestamp - state.low_movement_since < settings.stopped_min_seconds:
            return []

        state.stopped_emitted = True
        return [
            self._event(
                video_id,
                "vehicle_stopped",
                state.class_name,
                state.track_id,
                state.low_movement_since,
                timestamp,
                confidence,
                f"{self._vi_name(state.class_name)} track #{state.track_id} dừng lại từ giây {state.low_movement_since:.1f}.",
            )
        ]

    def _detect_restricted_area(
        self,
        video_id: str,
        state: TrackedObjectState,
        timestamp: float,
        confidence: float,
        center: tuple[float, float],
    ) -> list[Event]:
        if state.restricted_emitted or not self._restricted_polygon:
            return []
        in_area = self._point_in_polygon(center, self._restricted_polygon)
        if not in_area:
            return []

        state.restricted_emitted = True
        return [
            self._event(
                video_id,
                "enter_restricted_area",
                state.class_name,
                state.track_id,
                timestamp,
                timestamp,
                confidence,
                f"{self._vi_name(state.class_name)} track #{state.track_id} đi vào vùng hạn chế.",
            )
        ]

    def _detect_congestion(self, video_id: str, timestamp: float, active_vehicle_count: int) -> list[Event]:
        second = int(timestamp)
        if active_vehicle_count < settings.congestion_vehicle_threshold:
            return []
        if self._last_congestion_second is not None and second - self._last_congestion_second < 5:
            return []

        self._last_congestion_second = second
        return [
            self._event(
                video_id,
                "traffic_congestion",
                None,
                None,
                timestamp,
                timestamp,
                0.8,
                f"Có dấu hiệu ùn tắc tại giây {timestamp:.1f} với khoảng {active_vehicle_count} phương tiện trong khung hình.",
            )
        ]

    @staticmethod
    def _event(
        video_id: str,
        event_type: str,
        object_type: str | None,
        track_id: int | None,
        start: float,
        end: float | None,
        confidence: float,
        description: str,
    ) -> Event:
        return Event(
            video_id=video_id,
            event_type=event_type,
            object_type=object_type,
            track_id=track_id,
            timestamp_start=start,
            timestamp_end=end,
            confidence=float(confidence),
            description=description,
        )

    @staticmethod
    def _vi_name(class_name: str) -> str:
        names = {
            "person": "người",
            "car": "ô tô",
            "motorcycle": "xe máy",
            "truck": "xe tải",
            "bus": "xe buýt",
            "bicycle": "xe đạp",
            "traffic light": "đèn giao thông",
        }
        return names.get(class_name, class_name)

    def _parse_polygon(self, value: str) -> list[tuple[float, float]]:
        if not value.strip():
            return []
        points: list[tuple[float, float]] = []
        for raw_point in value.split(";"):
            parts = [part.strip() for part in raw_point.split(",")]
            if len(parts) != 2:
                return []
            try:
                x = float(parts[0])
                y = float(parts[1])
            except ValueError:
                return []
            if 0 <= x <= 1 and 0 <= y <= 1:
                x *= self.frame_width
                y *= self.frame_height
            points.append((x, y))
        return points if len(points) >= 3 else []

    @staticmethod
    def _point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
        x, y = point
        inside = False
        j = len(polygon) - 1
        for i, current in enumerate(polygon):
            xi, yi = current
            xj, yj = polygon[j]
            intersects = ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            )
            if intersects:
                inside = not inside
            j = i
        return inside
