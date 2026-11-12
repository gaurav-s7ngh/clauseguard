import io
import pytest
from app.models.contract import Contract


def test_upload_txt_file_success(client, sample_contract_text, db_session):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("services_agreement.txt", sample_contract_text.encode("utf-8"), "text/plain")}
    )
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "services_agreement.txt"
    assert data["status"] == "uploaded"

    doc_id = data["document_id"]
    contract = db_session.query(Contract).filter(Contract.id == doc_id).first()
    assert contract is not None
    assert contract.filename == "services_agreement.txt"
    assert contract.file_type == "txt"
    assert contract.raw_text == sample_contract_text
    assert contract.char_length == len(sample_contract_text)


def test_upload_docx_file_success(client, sample_docx_bytes, db_session):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("contract.docx", sample_docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
    )
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "contract.docx"
    assert data["status"] == "uploaded"

    doc_id = data["document_id"]
    contract = db_session.query(Contract).filter(Contract.id == doc_id).first()
    assert contract is not None
    assert contract.file_type == "docx"
    assert "LIMITATION OF LIABILITY" in contract.raw_text


def test_upload_pdf_file_success(client, sample_pdf_bytes, db_session):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("contract.pdf", sample_pdf_bytes, "application/pdf")}
    )
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["filename"] == "contract.pdf"
    assert data["status"] == "uploaded"

    doc_id = data["document_id"]
    contract = db_session.query(Contract).filter(Contract.id == doc_id).first()
    assert contract is not None
    assert contract.file_type == "pdf"
    assert "MASTER SERVICES AGREEMENT" in contract.raw_text


def test_upload_unsupported_file_extension(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("contract.exe", b"binary content", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_upload_empty_document(client):
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("empty.txt", b"   \n\n  ", "text/plain")}
    )
    assert response.status_code == 400
    assert "no readable text" in response.json()["detail"].lower()


def test_upload_oversized_file(client, monkeypatch):
    from app.core.config import settings
    # Temporarily set max file size to 100 bytes for testing
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_BYTES", 100)

    large_content = b"A" * 200
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("large.txt", large_content, "text/plain")}
    )
    assert response.status_code == 400
    assert "exceeds" in response.json()["detail"].lower()

