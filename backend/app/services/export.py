from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.schemas import KISExportItem, QAExportItem, TRAKEExportItem


class ExportValidationError(ValueError):
    """Raised when submission data violates competition formatting rules."""
    pass


class CompetitionExportService:
    @staticmethod
    def sanitize_video_name(name: str) -> str:
        """Ensure video name is clean and lacks file extensions (e.g., 'L01_V001')."""
        if not name or not isinstance(name, str):
            raise ExportValidationError("video_name must be a non-empty string.")
        cleaned = Path(name.strip()).stem.strip()
        if not cleaned:
            raise ExportValidationError("video_name cannot be empty after trimming.")
        # Only allow alphanumeric, underscores, hyphens
        if not re.match(r"^[A-Za-z0-9_\-]+$", cleaned):
            raise ExportValidationError(f"Invalid video_name format '{cleaned}'. Must only contain letters, numbers, hyphens, and underscores.")
        return cleaned

    @staticmethod
    def validate_frame_id(frame_id: Any) -> int:
        """Validate and cast frame ID to a non-negative integer."""
        try:
            val = int(frame_id)
        except (TypeError, ValueError) as exc:
            raise ExportValidationError(f"frame_id '{frame_id}' must be an integer.") from exc
        if val < 0:
            raise ExportValidationError(f"frame_id must be non-negative, got {val}.")
        return val

    @classmethod
    def format_kis_row(cls, video_name: str, frame_id: int) -> str:
        """Output format: <video_name>,<frame_id>"""
        vname = cls.sanitize_video_name(video_name)
        fid = cls.validate_frame_id(frame_id)
        return f"{vname},{fid}"

    @classmethod
    def format_qa_row(cls, video_name: str, frame_id: int, answer: str) -> str:
        """Output format: <video_name>,<frame_id>,<answer>"""
        vname = cls.sanitize_video_name(video_name)
        fid = cls.validate_frame_id(frame_id)
        if not isinstance(answer, str) or not answer.strip():
            raise ExportValidationError("answer must be a non-empty string.")
        cleaned_answer = answer.strip().replace("\r", " ").replace("\n", " ")
        if "," in cleaned_answer:
            cleaned_answer = f'"{cleaned_answer}"'
        return f"{vname},{fid},{cleaned_answer}"

    @classmethod
    def format_trake_row(cls, video_name: str, frame_ids: list[int]) -> str:
        """Output format: <video_name>,<frame_event_1>,...,<frame_event_n>"""
        vname = cls.sanitize_video_name(video_name)
        if not isinstance(frame_ids, (list, tuple)) or len(frame_ids) < 2:
            raise ExportValidationError("TRAKE requires at least 2 event frame IDs.")
        
        validated_frames = [cls.validate_frame_id(fid) for fid in frame_ids]
        
        # Enforce strictly monotonic increasing frames: t(F_1) < t(F_2) < ... < t(F_n)
        for i in range(len(validated_frames) - 1):
            if validated_frames[i] >= validated_frames[i + 1]:
                raise ExportValidationError(
                    f"TRAKE frames must be strictly monotonically increasing. "
                    f"Frame at index {i} ({validated_frames[i]}) >= frame at index {i + 1} ({validated_frames[i + 1]})."
                )
        
        frames_str = ",".join(str(f) for f in validated_frames)
        return f"{vname},{frames_str}"

    def generate_csv(
        self,
        task_type: str,
        items: list[dict[str, Any]],
        save_file: bool = True,
        submission_id: str | None = None,
    ) -> tuple[str, list[str], Path | None]:
        """
        Generate submission CSV string, validated rows, and optionally write to disk.
        Returns: (csv_content, preview_rows, file_path_if_saved)
        """
        task_upper = task_type.upper()
        if task_upper not in {"KIS", "QA", "TRAKE"}:
            raise ExportValidationError(f"Unsupported task_type '{task_type}'. Must be KIS, QA, or TRAKE.")
        if not items:
            raise ExportValidationError("items list cannot be empty.")

        rows: list[str] = []
        for index, item in enumerate(items, start=1):
            try:
                if task_upper == "KIS":
                    vname = item.get("video_name") or item.get("video_id", "")
                    fid = item.get("frame_id")
                    rows.append(self.format_kis_row(vname, fid))
                elif task_upper == "QA":
                    vname = item.get("video_name") or item.get("video_id", "")
                    fid = item.get("frame_id")
                    ans = item.get("answer", "")
                    rows.append(self.format_qa_row(vname, fid, ans))
                elif task_upper == "TRAKE":
                    vname = item.get("video_name") or item.get("video_id", "")
                    fids = item.get("frame_ids") or item.get("frames", [])
                    rows.append(self.format_trake_row(vname, fids))
            except ExportValidationError as exc:
                raise ExportValidationError(f"Item #{index} validation failed: {exc}") from exc

        csv_content = "\n".join(rows) + "\n"

        saved_path: Path | None = None
        if save_file:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            sub_tag = f"_{submission_id}" if submission_id else ""
            filename = f"{task_upper}{sub_tag}_{timestamp}.csv"
            out_dir = Path(settings.export_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            saved_path = out_dir / filename
            saved_path.write_text(csv_content, encoding="utf-8")

        return csv_content, rows, saved_path


export_service = CompetitionExportService()
