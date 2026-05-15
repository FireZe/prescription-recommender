from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "ranking_model.joblib"


@lru_cache(maxsize=1)
def load_ranking_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def predict_candidate_adequacy(features: Dict[str, Any]) -> Optional[float]:
    """
    Devolve um score previsto de adequação terapêutica entre 0 e 1,
    usando o modelo supervisionado treinado offline.
    """
    model = load_ranking_model()

    if model is None:
        return None

    row = pd.DataFrame([features])

    prediction = float(model.predict(row)[0])
    prediction = max(0.0, min(1.0, prediction))

    return prediction