from __future__ import annotations

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.export import ExportValidationError, export_service


def test_format_kis_row_standard() -> None:
    row = export_service.format_kis_row("L01_V001", 124)
    assert row == "L01_V001,124"


def test_format_kis_row_strips_extension() -> None:
    row = export_service.format_kis_row("L02_V015.mp4", 550)
    assert row == "L02_V015,550"


def test_format_kis_row_invalid_frame_id() -> None:
    with pytest.raises(ExportValidationError):
        export_service.format_kis_row("L01_V001", -5)
    with pytest.raises(ExportValidationError):
        export_service.format_kis_row("L01_V001", "abc")  # type: ignore


def test_format_qa_row_standard() -> None:
    row = export_service.format_qa_row("L01_V003", 250, "red")
    assert row == "L01_V003,250,red"


def test_format_qa_row_quotes_commas() -> None:
    row = export_service.format_qa_row("L01_V003", 250, "black, white")
    assert row == 'L01_V003,250,"black, white"'


def test_format_qa_row_empty_answer() -> None:
    with pytest.raises(ExportValidationError):
        export_service.format_qa_row("L01_V003", 250, "   ")


def test_format_trake_row_standard() -> None:
    row = export_service.format_trake_row("L01_V005", [100, 250, 480])
    assert row == "L01_V005,100,250,480"


def test_format_trake_row_rejects_non_increasing() -> None:
    # equal
    with pytest.raises(ExportValidationError, match="monotonically increasing"):
        export_service.format_trake_row("L01_V005", [100, 100, 200])
    # decreasing
    with pytest.raises(ExportValidationError, match="monotonically increasing"):
        export_service.format_trake_row("L01_V005", [300, 200, 400])


def test_format_trake_row_requires_at_least_two_frames() -> None:
    with pytest.raises(ExportValidationError, match="at least 2 event frame IDs"):
        export_service.format_trake_row("L01_V005", [100])


def test_generate_csv_memory_and_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.config.settings.export_dir", str(tmp_path / "exports"))
    items = [
        {"video_name": "L01_V001", "frame_id": 10},
        {"video_name": "L01_V002.mp4", "frame_id": 20},
    ]
    csv_content, preview_rows, saved_path = export_service.generate_csv(
        task_type="KIS",
        items=items,
        save_file=True,
        submission_id="test_sub",
    )
    assert len(preview_rows) == 2
    assert preview_rows[0] == "L01_V001,10"
    assert preview_rows[1] == "L01_V002,20"
    assert csv_content == "L01_V001,10\nL01_V002,20\n"
    assert saved_path is not None
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == csv_content


def test_api_export_competition_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.config.settings.export_dir", str(tmp_path / "exports"))
    client = TestClient(app)

    payload = {
        "task_type": "KIS",
        "items": [
            {"video_name": "L01_V010", "frame_id": 150},
            {"video_name": "L01_V010", "frame_id": 300},
        ],
        "save_file": True,
        "submission_id": "kis_run",
    }
    response = client.post("/api/exports/competition", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["task_type"] == "KIS"
    assert data["row_count"] == 2
    assert data["preview_rows"] == ["L01_V010,150", "L01_V010,300"]
    assert data["download_url"] is not None

    # test download
    file_name = data["file_name"]
    dl_response = client.get(f"/api/exports/download/{file_name}")
    assert dl_response.status_code == 200
    assert dl_response.text.splitlines() == ["L01_V010,150", "L01_V010,300"]
