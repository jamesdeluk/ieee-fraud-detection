from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"

BASIC_FEATURES_PATH = MODELS_DIR / "basic_features.json"
FULL_FEATURES_PATH = MODELS_DIR / "full_features.json"
METRICS_PATH = DATA_DIR / "model_metrics.json"
HOLDOUT_PATH = DATA_DIR / "holdout_demo.parquet"
BASIC_MODEL_PATH = MODELS_DIR / "basic.joblib"
LEGACY_BASIC_MODEL_PATH = MODELS_DIR / "core.joblib"

BASIC_PROBABILITY_COLUMN = "basic_fraud_probability"
FULL_PROBABILITY_COLUMN = "full_fraud_probability"
BASIC_PREDICTION_COLUMN = "basic_prediction"
FULL_PREDICTION_COLUMN = "full_prediction"
TARGET_COLUMN = "isFraud"
TRANSACTION_ID_COLUMN = "TransactionID"
TRANSACTION_TIME_COLUMN = "TransactionDT"
DEFAULT_THRESHOLD = 0.5


class BasicPredictionRequest(BaseModel):
    """Accepts a dynamic basic-model row while rejecting accidental extra fields."""

    model_config = ConfigDict(extra="allow")


def convert_categorical_to_string(data: pd.DataFrame) -> pd.DataFrame:
    """Matches the notebook transformer required by the saved joblib pipeline."""

    return data.astype("string").fillna("missing")


# The current model artefact was saved from a notebook where the function lived
# on __main__. Registering it here lets joblib resolve the existing artefact.
setattr(sys.modules["__main__"], "convert_categorical_to_string", convert_categorical_to_string)


app = FastAPI(
    title="Fraud Investigation Console API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_basic_features() -> dict[str, Any]:
    """Loads the basic feature contract used by the custom prediction form."""

    features = json.loads(BASIC_FEATURES_PATH.read_text())
    holdout = load_holdout()

    for feature in features["features"]:
        if feature["type"] != "numeric" or feature["name"] not in holdout:
            continue

        # Use deterministic holdout samples so frontend randomisation follows real row values
        # instead of drawing implausibly often from the extreme min/max range.
        sampled_values = (
            holdout[feature["name"]]
            .dropna()
            .sample(n=min(250, holdout[feature["name"]].dropna().shape[0]), random_state=42)
            .round(3)
            .tolist()
        )
        feature["sample_values"] = [json_safe_value(value) for value in sampled_values]

    return features


@lru_cache(maxsize=1)
def load_full_features() -> dict[str, Any]:
    """Loads the full feature contract used for model documentation."""

    return json.loads(FULL_FEATURES_PATH.read_text())


@lru_cache(maxsize=1)
def load_metrics() -> dict[str, Any]:
    """Loads precomputed model metrics from the modelling notebook."""

    return json.loads(METRICS_PATH.read_text())


@lru_cache(maxsize=1)
def load_holdout() -> pd.DataFrame:
    """Loads the pre-scored holdout data once for transaction exploration."""

    return pd.read_parquet(HOLDOUT_PATH)


@lru_cache(maxsize=1)
def load_basic_model() -> Any:
    """Loads the basic scikit-learn pipeline used for live custom checks."""

    # The notebook now exports basic.joblib; the fallback keeps existing local
    # checkouts usable until the notebook has been rerun.
    model_path = BASIC_MODEL_PATH if BASIC_MODEL_PATH.exists() else LEGACY_BASIC_MODEL_PATH
    return joblib.load(model_path)


def json_safe_value(value: Any) -> Any:
    """Converts pandas, NumPy, and floating null values into JSON-safe values."""

    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialises dataframe rows without leaking pandas-specific scalar types."""

    return [
        {key: json_safe_value(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def model_columns(model: Literal["basic", "full"]) -> tuple[str, str]:
    """Returns the prediction and probability columns for a selected model."""

    if model == "basic":
        return BASIC_PREDICTION_COLUMN, BASIC_PROBABILITY_COLUMN
    return FULL_PREDICTION_COLUMN, FULL_PROBABILITY_COLUMN


def apply_actual_filter(frame: pd.DataFrame, actual: str | None) -> pd.DataFrame:
    """Filters holdout rows by their true fraud label."""

    if actual is None:
        return frame
    if actual == "fraud":
        return frame[frame[TARGET_COLUMN] == 1]
    return frame[frame[TARGET_COLUMN] == 0]


def apply_prediction_filter(
    frame: pd.DataFrame,
    model: Literal["basic", "full"],
    prediction: str | None,
    threshold: float,
) -> pd.DataFrame:
    """Filters holdout rows by the selected model's thresholded probability."""

    if prediction is None:
        return frame

    _, probability_column = model_columns(model)
    predicts_fraud = frame[probability_column] >= threshold
    if prediction == "fraud":
        return frame[predicts_fraud]
    return frame[~predicts_fraud]


def apply_sort(frame: pd.DataFrame, sort: str | None) -> pd.DataFrame:
    """Sorts rows by model risk or model disagreement."""

    if sort == "basic_probability":
        return frame.sort_values(BASIC_PROBABILITY_COLUMN, ascending=False)
    if sort == "full_probability":
        return frame.sort_values(FULL_PROBABILITY_COLUMN, ascending=False)
    if sort == "disagreement":
        return frame.assign(
            _disagreement=(frame[FULL_PROBABILITY_COLUMN] - frame[BASIC_PROBABILITY_COLUMN]).abs()
        ).sort_values("_disagreement", ascending=False).drop(columns="_disagreement")
    return frame


def transaction_breakdown(frame: pd.DataFrame, threshold: float) -> dict[str, dict[str, int]]:
    """Counts actual and thresholded prediction labels across filtered rows."""

    basic_predicts_fraud = frame[BASIC_PROBABILITY_COLUMN] >= threshold
    full_predicts_fraud = frame[FULL_PROBABILITY_COLUMN] >= threshold

    return {
        "actual": {
            "fraud": int((frame[TARGET_COLUMN] == 1).sum()),
            "not_fraud": int((frame[TARGET_COLUMN] == 0).sum()),
        },
        "basic_prediction": {
            "fraud": int(basic_predicts_fraud.sum()),
            "not_fraud": int((~basic_predicts_fraud).sum()),
        },
        "full_prediction": {
            "fraud": int(full_predicts_fraud.sum()),
            "not_fraud": int((~full_predicts_fraud).sum()),
        },
    }


def risk_band(probability: float) -> str:
    """Maps a fraud probability into the product risk bands used in the UI."""

    if probability < 0.10:
        return "Low"
    if probability < 0.30:
        return "Watch"
    if probability < 0.50:
        return "Elevated"
    return "High"


def validate_basic_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Builds a single-row basic dataframe and validates custom input values."""

    features = load_basic_features()["features"]
    feature_names = {feature["name"] for feature in features}
    unknown_fields = sorted(set(payload) - feature_names)
    if unknown_fields:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown basic feature field(s): {', '.join(unknown_fields)}",
        )

    row: dict[str, Any] = {feature["name"]: pd.NA for feature in features}

    for feature in features:
        name = feature["name"]
        if name not in payload or payload[name] in ("", None):
            continue

        if feature["type"] == "numeric":
            try:
                row[name] = float(payload[name])
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"{name} must be numeric.",
                ) from exc
        else:
            row[name] = payload[name]

    return pd.DataFrame([row], columns=[feature["name"] for feature in features])


@app.get("/api/health")
def health() -> dict[str, str]:
    """Reports whether the API process is available."""

    return {"status": "ok"}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    """Returns precomputed model and dataset metrics."""

    return load_metrics()


@app.get("/api/features/basic")
def basic_features() -> dict[str, Any]:
    """Returns the basic model feature contract."""

    return load_basic_features()


@app.get("/api/features/full")
def full_features() -> dict[str, Any]:
    """Returns the full model feature contract."""

    return load_full_features()


@app.get("/api/transactions")
def transactions(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=100),
    threshold: float = Query(default=DEFAULT_THRESHOLD, ge=0, le=1),
    actual: Literal["fraud", "not_fraud"] | None = None,
    basic_prediction: Literal["fraud", "not_fraud"] | None = None,
    full_prediction: Literal["fraud", "not_fraud"] | None = None,
    sort: Literal["basic_probability", "full_probability", "disagreement"] | None = None,
) -> dict[str, Any]:
    """Returns paginated, filtered holdout transactions for the explorer."""

    frame = load_holdout()
    filtered = apply_actual_filter(frame, actual)
    filtered = apply_prediction_filter(filtered, "basic", basic_prediction, threshold)
    filtered = apply_prediction_filter(filtered, "full", full_prediction, threshold)
    breakdown = transaction_breakdown(filtered, threshold)
    filtered = apply_sort(filtered, sort)
    total = len(filtered)
    page = filtered.iloc[offset : offset + limit]

    return {
        "offset": offset,
        "limit": limit,
        "total": total,
        "breakdown": breakdown,
        "items": dataframe_records(page),
    }


@app.get("/api/transactions/{transaction_id}")
def transaction_detail(transaction_id: int) -> dict[str, Any]:
    """Returns one holdout transaction by TransactionID."""

    frame = load_holdout()
    match = frame[frame[TRANSACTION_ID_COLUMN] == transaction_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    return dataframe_records(match.head(1))[0]


@app.post("/api/predict/basic")
def predict_basic(payload: BasicPredictionRequest) -> dict[str, Any]:
    """Runs a live custom row through the basic model pipeline."""

    row = validate_basic_payload(payload.model_dump())
    probability = float(load_basic_model().predict_proba(row)[:, 1][0])
    prediction = int(probability >= DEFAULT_THRESHOLD)

    if not math.isfinite(probability):
        raise HTTPException(status_code=500, detail="Model returned a non-finite probability.")

    return {
        "fraud_probability": probability,
        "prediction": prediction,
        "risk_band": risk_band(probability),
        "threshold": DEFAULT_THRESHOLD,
    }
