from __future__ import annotations


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "reviewAI"
    assert body["model_version"] == "fake-two-stage"
    assert body["predictor_loaded"] is False
