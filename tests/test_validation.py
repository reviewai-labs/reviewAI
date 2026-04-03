from __future__ import annotations


def test_analyze_review_rejects_blank_review_text(client):
    response = client.post(
        "/analyze-review",
        json={"review_text": "   ", "include_raw_outputs": True},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert body["error"]["details"][0]["field"] == "review_text"


def test_analyze_review_rejects_extra_fields(client):
    response = client.post(
        "/analyze-review",
        json={
            "review_text": "Valid review",
            "include_raw_outputs": True,
            "unexpected": "field",
        },
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
