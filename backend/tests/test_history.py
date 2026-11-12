def test_empty_history(client):
    res = client.get("/api/v1/documents")
    assert res.status_code == 200
    assert res.json() == []


def test_history_aggregation(client, sample_contract_text):
    # Upload and analyze document
    upload_res = client.post(
        "/api/v1/documents/upload",
        files={"file": ("history_contract.txt", sample_contract_text.encode("utf-8"), "text/plain")}
    )
    doc_id = upload_res.json()["document_id"]
    client.post(f"/api/v1/analyze/{doc_id}")

    # Fetch history
    history_res = client.get("/api/v1/documents")
    assert history_res.status_code == 200
    items = history_res.json()
    assert len(items) == 1

    item = items[0]
    assert item["document_id"] == doc_id
    assert item["filename"] == "history_contract.txt"
    assert item["total_clauses"] > 0
    # Sum of counts equals total clauses
    total = (
        item["critical_count"]
        + item["high_count"]
        + item["medium_count"]
        + item["low_count"]
        + item["needs_review_count"]
    )
    assert total == item["total_clauses"]

