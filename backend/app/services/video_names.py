from pathlib import Path

from app.models import Video


def competition_video_name(video: Video) -> str:
    """Return the competition-compatible video identifier without extension."""
    source_name = video.original_filename or video.stored_filename or video.id
    return Path(source_name).stem
