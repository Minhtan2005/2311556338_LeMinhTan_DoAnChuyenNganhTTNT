from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings


ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def validate_video_filename(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video extension '{suffix}'. Allowed: {allowed}")


async def save_upload_file(file: UploadFile) -> tuple[str, str]:
    original_name = file.filename or "video.mp4"
    validate_video_filename(original_name)

    suffix = Path(original_name).suffix.lower()
    stored_name = f"{uuid4()}{suffix}"
    destination = Path(settings.upload_dir) / stored_name

    with destination.open("wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    return stored_name, str(destination)
