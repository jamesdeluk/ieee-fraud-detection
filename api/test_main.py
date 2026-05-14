from fastapi.testclient import TestClient

from .main import app, load_basic_features, load_holdout


client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_returns_both_models():
    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert set(response.json()["models"]) == {"basic", "full"}


def test_basic_features_returns_expected_count():
    response = client.get("/api/features/basic")

    assert response.status_code == 200
    assert response.json()["features_count"] == 14


def test_basic_features_include_numeric_sample_values():
    response = client.get("/api/features/basic")
    features = response.json()["features"]
    transaction_amount = next(feature for feature in features if feature["name"] == "TransactionAmt")

    assert response.status_code == 200
    assert len(transaction_amount["sample_values"]) > 0
    assert all(isinstance(value, float) for value in transaction_amount["sample_values"])


def test_transactions_paginate_by_default():
    response = client.get("/api/transactions")
    payload = response.json()

    assert response.status_code == 200
    assert payload["limit"] == 25
    assert len(payload["items"]) == 25
    assert payload["total"] > 25
    assert payload["breakdown"]["actual"]["fraud"] + payload["breakdown"]["actual"]["not_fraud"] == payload["total"]
    assert (
        payload["breakdown"]["basic_prediction"]["fraud"]
        + payload["breakdown"]["basic_prediction"]["not_fraud"]
        == payload["total"]
    )
    assert (
        payload["breakdown"]["full_prediction"]["fraud"]
        + payload["breakdown"]["full_prediction"]["not_fraud"]
        == payload["total"]
    )


def test_transactions_filter_by_basic_prediction():
    response = client.get("/api/transactions?basic_prediction=fraud&limit=100")
    payload = response.json()

    assert response.status_code == 200
    assert payload["total"] > 0
    for row in payload["items"]:
        assert row["basic_fraud_probability"] >= 0.5


def test_transactions_filter_by_full_prediction():
    response = client.get("/api/transactions?full_prediction=not_fraud&limit=100")
    payload = response.json()

    assert response.status_code == 200
    assert payload["total"] > 0
    for row in payload["items"]:
        assert row["full_fraud_probability"] < 0.5


def test_transactions_filter_by_selected_threshold():
    response = client.get("/api/transactions?basic_prediction=fraud&threshold=0.7&limit=100")
    payload = response.json()

    assert response.status_code == 200
    assert payload["total"] > 0
    assert payload["breakdown"]["basic_prediction"]["not_fraud"] == 0
    for row in payload["items"]:
        assert row["basic_fraud_probability"] >= 0.7


def test_transactions_filter_by_actual_basic_and_full_predictions():
    response = client.get(
        "/api/transactions?actual=fraud&basic_prediction=not_fraud&full_prediction=fraud&limit=100"
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["total"] > 0
    for row in payload["items"]:
        assert row["isFraud"] == 1
        assert row["basic_fraud_probability"] < 0.5
        assert row["full_fraud_probability"] >= 0.5


def test_transaction_detail_uses_transaction_id():
    holdout = load_holdout()
    transaction_id = int(holdout.iloc[0]["TransactionID"])

    response = client.get(f"/api/transactions/{transaction_id}")

    assert response.status_code == 200
    assert response.json()["TransactionID"] == transaction_id


def test_basic_prediction_accepts_valid_payload():
    payload = {}
    for feature in load_basic_features()["features"]:
        if feature["type"] == "numeric":
            payload[feature["name"]] = feature.get("median", 0)
        else:
            values = feature.get("sample_values") or ["missing"]
            payload[feature["name"]] = values[0]

    response = client.post("/api/predict/basic", json=payload)
    result = response.json()

    assert response.status_code == 200
    assert 0 <= result["fraud_probability"] <= 1
    assert result["prediction"] in [0, 1]
    assert result["risk_band"] in ["Low", "Watch", "Elevated", "High"]


def test_basic_prediction_rejects_unknown_fields():
    response = client.post("/api/predict/basic", json={"unexpected": "field"})

    assert response.status_code == 422
