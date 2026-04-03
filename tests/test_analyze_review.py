from __future__ import annotations


def assert_rating_shape(aspect_output: dict) -> None:
    rating = aspect_output["rating_suggested"]
    if aspect_output["present"]:
        assert isinstance(rating, float | int)
        assert 1.0 <= float(rating) <= 5.0
    else:
        assert rating is None


def test_analyze_review_response_shape(client):
    response = client.post(
        "/analyze-review",
        json={
            "review_text": "The sushi was excellent but the service was slow.",
            "include_raw_outputs": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "fake-two-stage"
    assert body["analysis_language"] is None
    assert set(body["semeval_raw_outputs"]) == {
        "food",
        "service",
        "price",
        "ambience",
        "anecdotes/miscellaneous",
    }
    assert set(body) == {
        "model_version",
        "analysis_language",
        "semeval_raw_outputs",
        "aspect_outputs",
    }
    assert set(body["aspect_outputs"]) == {"food", "ambiance", "service", "value"}
    assert set(body["aspect_outputs"]["food"]) == {
        "present",
        "polarity",
        "confidence",
        "rating_suggested",
    }
    assert body["aspect_outputs"]["service"]["polarity"] == "negative"
    assert_rating_shape(body["aspect_outputs"]["food"])
    assert_rating_shape(body["aspect_outputs"]["ambiance"])
    assert body["aspect_outputs"]["food"]["rating_suggested"] == 4.61
    assert body["aspect_outputs"]["ambiance"]["rating_suggested"] is None


def test_analyze_review_can_hide_raw_outputs(client):
    response = client.post(
        "/analyze-review",
        json={
            "review_text": "Great ambiance and fair prices.",
            "include_raw_outputs": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["semeval_raw_outputs"] is None
    assert set(body) == {
        "model_version",
        "analysis_language",
        "semeval_raw_outputs",
        "aspect_outputs",
    }
    assert body["aspect_outputs"]["ambiance"]["present"] is False
    assert body["aspect_outputs"]["ambiance"]["rating_suggested"] is None


def test_batch_endpoint_response_shape(client):
    response = client.post(
        "/analyze-reviews-batch",
        json={
            "reviews": [
                {"review_text": "The sushi was excellent but the service was slow."},
                {"review_text": "Great ambiance and fair prices."},
            ],
            "include_raw_outputs": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["model_version"] == "fake-two-stage"
    assert body["results"][1]["semeval_raw_outputs"] is None
    assert set(body["results"][0]) == {
        "model_version",
        "analysis_language",
        "semeval_raw_outputs",
        "aspect_outputs",
    }
    assert_rating_shape(body["results"][0]["aspect_outputs"]["food"])
    assert_rating_shape(body["results"][0]["aspect_outputs"]["value"])
