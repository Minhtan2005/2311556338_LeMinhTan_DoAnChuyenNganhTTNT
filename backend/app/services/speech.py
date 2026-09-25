import subprocess
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models import TranscriptSegment, Video


class SpeechToTextService:
    def transcribe_video(self, db: Session, video: Video) -> None:
        audio_path = self._extract_audio(Path(video.stored_path))
        if audio_path is None:
            return

        try:
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError(
                    "Whisper is enabled but not installed. Install backend/requirements-optional.txt "
                    "or set ENABLE_WHISPER=false."
                ) from exc

            model = whisper.load_model(settings.whisper_model)
            result = model.transcribe(str(audio_path), language="vi", fp16=False)
            segments = [
                TranscriptSegment(
                    video_id=video.id,
                    start_time=float(segment["start"]),
                    end_time=float(segment["end"]),
                    text=str(segment["text"]).strip(),
                )
                for segment in result.get("segments", [])
                if str(segment.get("text", "")).strip()
            ]
            if segments:
                db.add_all(segments)
                db.commit()
        finally:
            audio_path.unlink(missing_ok=True)

    @staticmethod
    def _extract_audio(video_path: Path) -> Path | None:
        audio_path = video_path.with_suffix(".wav")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "FFmpeg is required for Whisper transcription but was not found in PATH. "
                "Install FFmpeg or set ENABLE_WHISPER=false."
            ) from exc
        if result.returncode != 0 or not audio_path.exists() or audio_path.stat().st_size == 0:
            audio_path.unlink(missing_ok=True)
            return None
        return audio_path


speech_to_text_service = SpeechToTextService()
