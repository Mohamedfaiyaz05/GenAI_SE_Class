import asyncio
from io import BytesIO
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import backend


def test_upload_saves_pdf_without_indexing(tmp_path, monkeypatch):
    vector_manager = SimpleNamespace(
        process_new_documents=Mock(),
    )
    monkeypatch.setattr(backend, "DATA_DIR", tmp_path)
    monkeypatch.setattr(backend, "vector_manager", vector_manager)
    upload = UploadFile(
        filename="new_agriculture.pdf",
        file=BytesIO(b"%PDF-test-content"),
    )

    result = asyncio.run(backend.upload_pdf(upload))

    assert result["status"] == "uploaded"
    assert (tmp_path / "new_agriculture.pdf").read_bytes() == b"%PDF-test-content"
    vector_manager.process_new_documents.assert_not_called()


def test_upload_rejects_existing_file_without_overwriting(tmp_path, monkeypatch):
    existing_file = tmp_path / "report.pdf"
    existing_file.write_bytes(b"original")
    monkeypatch.setattr(backend, "DATA_DIR", tmp_path)

    try:
        asyncio.run(
            backend.upload_pdf(
                UploadFile(filename="report.pdf", file=BytesIO(b"replacement"))
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("Expected an existing filename to be rejected.")

    assert existing_file.read_bytes() == b"original"


def test_process_upload_indexes_and_verifies_file(tmp_path, monkeypatch):
    vector_manager = SimpleNamespace(
        process_new_documents=Mock(return_value=object()),
        _load_registry=Mock(return_value={"new_agriculture.pdf"}),
    )
    monkeypatch.setattr(backend, "DATA_DIR", tmp_path)
    monkeypatch.setattr(backend, "vector_manager", vector_manager)
    (tmp_path / "new_agriculture.pdf").write_bytes(b"%PDF-test-content")

    result = backend.process_uploaded_pdf("new_agriculture.pdf")

    assert result["status"] == "success"
    vector_manager.process_new_documents.assert_called_once_with(
        backend.extractor,
        data_dir=str(tmp_path),
        filenames={"new_agriculture.pdf"},
    )


def test_process_upload_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "DATA_DIR", tmp_path)

    try:
        backend.process_uploaded_pdf("../outside.pdf")
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("Expected an unsafe PDF path to be rejected.")


def test_feedback_saves_rating_without_conversation_content(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "FEEDBACK_DIR", tmp_path)
    response_id = "a" * 32

    result = backend.save_response_feedback(
        response_id=response_id,
        rating="like",
    )
    updated_result = backend.save_response_feedback(
        response_id=response_id,
        rating="dislike",
    )

    saved_feedback = (tmp_path / f"{response_id}.json").read_text(encoding="utf-8")
    assert result == {"status": "saved", "rating": "like"}
    assert updated_result == {"status": "saved", "rating": "dislike"}
    assert '"rating": "dislike"' in saved_feedback
    assert f'"response_id": "{response_id}"' in saved_feedback
    assert "session_id" not in saved_feedback
    assert "question" not in saved_feedback
    assert "answer" not in saved_feedback


def test_feedback_rejects_invalid_rating_and_response_id():
    for response_id, rating in [
        ("../unsafe", "like"),
        ("a" * 32, "neutral"),
    ]:
        try:
            backend.save_response_feedback(
                response_id=response_id,
                rating=rating,
            )
        except HTTPException as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("Expected invalid feedback input to be rejected.")


def test_feedback_api_persists_a_valid_rating(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "FEEDBACK_DIR", tmp_path)
    response_id = "b" * 32

    response = TestClient(backend.app).post(
        "/api/feedback",
        data={"response_id": response_id, "rating": "like"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "saved", "rating": "like"}
    assert (tmp_path / f"{response_id}.json").is_file()
