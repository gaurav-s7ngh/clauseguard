import pytest


def test_analyze_document_end_to_end(client, sample_contract_text):
    # 1. Upload contract
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("master_agreement.txt", sample_contract_text.encode("utf-8"), "text/plain")}
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 2. Check initial status
    status_res = client.get(f"/api/v1/status/{doc_id}")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "pending"

    # 3. Trigger analysis
    analyze_res = client.post(f"/api/v1/analyze/{doc_id}")
    assert analyze_res.status_code == 200
    assert analyze_res.json()["status"] == "completed"
    assert analyze_res.json()["progress"] == 100

    # 4. Check completed status
    status_after = client.get(f"/api/v1/status/{doc_id}")
    assert status_after.status_code == 200
    assert status_after.json()["status"] == "completed"

    # 5. Fetch full results
    results_res = client.get(f"/api/v1/results/{doc_id}")
    assert results_res.status_code == 200
    result_data = results_res.json()

    assert result_data["document_id"] == doc_id
    assert result_data["filename"] == "master_agreement.txt"
    assert result_data["document_text"] == sample_contract_text
    assert result_data["document_char_length"] == len(sample_contract_text)
    assert result_data["total_chunks"] >= 1
    assert result_data["total_extracted_clauses"] > 0

    clauses = result_data["clauses"]
    assert len(clauses) >= 5

    categories_detected = set(c["category"] for c in clauses)
    assert "liability_cap" in categories_detected
    assert "termination_for_convenience" in categories_detected
    assert "governing_law" in categories_detected

    # 6. Verify 100% exact substring match for character offsets
    for c in clauses:
        s = c["start_char"]
        e = c["end_char"]
        txt = c["clause_text"]
        assert sample_contract_text[s:e] == txt, f"Mismatch for {c['category']}: text slice does not equal clause_text"
        assert c["risk_level"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NEEDS_REVIEW"]
        assert len(c["plain_english_explanation"]) > 0
        assert len(c["risk_reasoning"]) > 0


def test_analyze_non_existent_document(client):
    res = client.post("/api/v1/analyze/doc_nonexistent_123")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_get_results_non_existent_document(client):
    res = client.get("/api/v1/results/doc_nonexistent_123")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_analysis_idempotency(client, sample_contract_text):
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("idempotency_test.txt", sample_contract_text.encode("utf-8"), "text/plain")}
    )
    doc_id = upload_res.json()["document_id"]

    # First run
    client.post(f"/api/v1/analyze/{doc_id}")
    res1 = client.get(f"/api/v1/results/{doc_id}").json()
    count1 = res1["total_extracted_clauses"]

    # Second run
    client.post(f"/api/v1/analyze/{doc_id}")
    res2 = client.get(f"/api/v1/results/{doc_id}").json()
    count2 = res2["total_extracted_clauses"]

    assert count1 == count2, "Analysis should be idempotent without duplicate clause rows"

