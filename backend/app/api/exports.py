from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from app.core.config import settings
from app.schemas import CompetitionExportRequest, CompetitionExportResponse
from app.services.export import ExportValidationError, export_service

router = APIRouter()


@router.post("/competition", response_model=CompetitionExportResponse)
def export_competition(payload: CompetitionExportRequest) -> CompetitionExportResponse:
    try:
        csv_content, preview_rows, saved_path = export_service.generate_csv(
            task_type=payload.task_type,
            items=payload.items,
            save_file=payload.save_file,
            submission_id=payload.submission_id,
        )
    except ExportValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    file_name = saved_path.name if saved_path else None
    download_url = f"/api/exports/download/{file_name}" if file_name else None

    return CompetitionExportResponse(
        task_type=payload.task_type.upper(),
        row_count=len(preview_rows),
        csv_content=csv_content,
        preview_rows=preview_rows[:20],
        file_name=file_name,
        download_url=download_url,
    )


@router.get("/download/{filename}")
def download_export_file(filename: str):
    safe_name = Path(filename).name
    target_path = Path(settings.export_dir) / safe_name
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail=f"Export file '{filename}' not found.")
    return FileResponse(
        path=str(target_path),
        media_type="text/csv",
        filename=safe_name,
    )


@router.get("/list")
def list_export_files() -> list[dict[str, str]]:
    out_dir = Path(settings.export_dir)
    if not out_dir.exists():
        return []
    files = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "filename": f.name,
            "download_url": f"/api/exports/download/{f.name}",
            "size_bytes": str(f.stat().st_size),
        }
        for f in files
    ]
