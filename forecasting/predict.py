"""
Reusable forecast interface used by the backend API and the agent.

This module provides a single function — get_forecast() — that loads
a trained Prophet model for a product and returns a structured prediction
for the requested number of future days.

Design principle: this module contains no business logic and no database
writes.  It is a pure read-only interface.  All mutation (e.g., storing
forecasts) is the caller's responsibility.

Usage example:
    from forecasting.predict import get_forecast
    result = get_forecast(product_id=85, days_ahead=14)
    # result['dates'][0] → '2015-08-01'
    # result['yhat'][0]  → 6234.5
"""

import logging
from pathlib import Path
from typing import Any

import cmdstanpy
import numpy as np
import pandas as pd
from prophet.serialize import model_from_json

logger = logging.getLogger(__name__)

# Patch cmdstanpy.set_cmdstan_path so Prophet's internal model_from_dict call doesn't fail on invalid wheel path
_real_set_cmdstan_path = cmdstanpy.set_cmdstan_path


def _safe_set_cmdstan_path(path: str) -> None:
    try:
        _real_set_cmdstan_path(path)
    except ValueError:
        pass


cmdstanpy.set_cmdstan_path = _safe_set_cmdstan_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODELS_DIR = _PROJECT_ROOT / "forecasting" / "models" / "prophet"
_MAX_DAYS_AHEAD = 90
_MIN_DAYS_AHEAD = 1


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ModelNotFoundError(Exception):
    """
    Raised when no trained Prophet model exists for the requested product.

    Callers (the API, the agent) should catch this to return a user-friendly
    error rather than letting it propagate as an unhandled 500.
    """
    pass


class ForecastRangeError(ValueError):
    """
    Raised when days_ahead is outside the supported range [1, 90].
    """
    pass


# ---------------------------------------------------------------------------
# Model loading (cached in memory to avoid re-reading the JSON on every call)
# ---------------------------------------------------------------------------

_model_cache: dict[int, Any] = {}


def _load_model(product_id: int) -> Any:
    """
    Load and cache the Prophet model for a product.

    On first call the model is deserialized from its JSON file and stored
    in a module-level dict.  Subsequent calls for the same product_id
    return the cached object immediately.

    Parameters:
        product_id: The product whose model should be loaded.

    Returns:
        A deserialized Prophet model object.

    Raises:
        ModelNotFoundError: If the model file does not exist.
    """
    if product_id in _model_cache:
        return _model_cache[product_id]

    model_path = _MODELS_DIR / f"product_{product_id}.json"
    if not model_path.exists():
        raise ModelNotFoundError(
            f"No trained model found for product_id={product_id}. "
            f"Expected file: {model_path}"
        )

    logger.info("Loading Prophet model for product %d from %s.", product_id, model_path.name)
    with open(model_path, "r", encoding="utf-8") as fh:
        model = model_from_json(fh.read())

    _model_cache[product_id] = model
    return model


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_forecast(product_id: int, days_ahead: int) -> dict[str, Any]:
    """
    Generate a demand forecast for a product using its trained Prophet model.

    The forecast starts from the day after the model's training data ends
    and extends for the requested number of days.  Regressors (promo,
    state_holiday, school_holiday) are set to their conservative defaults:
    - promo = 0 (no active promotion)
    - state_holiday = 0 (no state holiday)
    - school_holiday = 0 (no school holiday)

    Callers that know about upcoming promotions or holidays can pass their
    own regressor DataFrames by using the model directly, but for routine
    inventory planning this conservative baseline is appropriate.

    Negative yhat values are clipped to 0 — daily sales cannot be negative.

    Parameters:
        product_id: The product to forecast.  A trained model must exist
            in forecasting/models/prophet/product_{product_id}.json.
        days_ahead: Number of future days to forecast.  Must be between
            1 and 90 (inclusive).

    Returns:
        A dictionary with the following keys:
        - "product_id"   (int): the requested product
        - "days_ahead"   (int): the requested horizon
        - "dates"        (list[str]): ISO-8601 date strings, one per day
        - "yhat"         (list[float]): point forecast (clipped ≥ 0)
        - "yhat_lower"   (list[float]): lower 80% confidence bound
        - "yhat_upper"   (list[float]): upper 80% confidence bound
        - "training_end" (str): the last date used during training (ISO-8601)

    Raises:
        ModelNotFoundError: If no trained model exists for product_id.
        ForecastRangeError: If days_ahead is outside [1, 90].
    """
    if not (_MIN_DAYS_AHEAD <= days_ahead <= _MAX_DAYS_AHEAD):
        raise ForecastRangeError(
            f"days_ahead must be between {_MIN_DAYS_AHEAD} and {_MAX_DAYS_AHEAD}; "
            f"got {days_ahead}."
        )

    model = _load_model(product_id)

    # Determine the last date the model was trained on.
    training_end: pd.Timestamp = model.history_dates.max()

    # Build a future DataFrame starting from the day after training ended.
    # make_future_dataframe(periods=N) includes the training period too,
    # so we slice off the training portion and take only the N future rows.
    future = model.make_future_dataframe(periods=days_ahead, freq="D", include_history=False)

    # Supply default regressor values for the future period.
    # Using 0 for all is the conservative choice: no promo, no holidays.
    future["promo"] = 0
    future["state_holiday"] = 0
    future["school_holiday"] = 0

    forecast = model.predict(future)

    # Clip negative predictions — Prophet can produce slightly negative
    # values for low-sales periods which are not meaningful here.
    yhat = np.clip(forecast["yhat"].values, 0, None)
    yhat_lower = np.clip(forecast["yhat_lower"].values, 0, None)
    yhat_upper = np.clip(forecast["yhat_upper"].values, 0, None)

    return {
        "product_id": product_id,
        "days_ahead": days_ahead,
        "dates": [d.strftime("%Y-%m-%d") for d in forecast["ds"].dt.date],
        "yhat": [round(float(v), 2) for v in yhat],
        "yhat_lower": [round(float(v), 2) for v in yhat_lower],
        "yhat_upper": [round(float(v), 2) for v in yhat_upper],
        "training_end": training_end.strftime("%Y-%m-%d"),
    }


def list_available_products() -> list[int]:
    """
    Return the list of product IDs for which a trained model exists.

    This is useful for the API to validate product_id before attempting
    to load a model.

    Returns:
        Sorted list of integer product IDs with model files present.
    """
    model_files = sorted(_MODELS_DIR.glob("product_*.json"))
    product_ids: list[int] = []
    for path in model_files:
        try:
            pid = int(path.stem.replace("product_", ""))
            product_ids.append(pid)
        except ValueError:
            logger.warning("Unexpected file in models dir: %s — skipping.", path.name)
    return product_ids
