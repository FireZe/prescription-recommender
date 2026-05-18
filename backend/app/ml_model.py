from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "ranking_model.joblib"


CLASS_SCORE_WEIGHTS = {
    0: 0.00,  # não admissível
    1: 0.60,  # admissível com precaução
    2: 1.00,  # admissível
}


@lru_cache(maxsize=1)
def load_ranking_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


def score_from_class_probabilities(model, row: pd.DataFrame) -> Optional[float]:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)

    class_to_probability = {
        int(class_label): float(probability)
        for class_label, probability in zip(classes, probabilities)
    }

    score = 0.0

    for class_label, weight in CLASS_SCORE_WEIGHTS.items():
        score += class_to_probability.get(class_label, 0.0) * weight

    return max(0.0, min(1.0, float(score)))


def score_from_prediction(model, row: pd.DataFrame) -> Optional[float]:
    if not hasattr(model, "predict"):
        return None

    prediction = float(model.predict(row)[0])

    # Compatibilidade com o modelo antigo, que já devolvia score 0-1.
    if 0.0 <= prediction <= 1.0:
        return max(0.0, min(1.0, prediction))

    # Compatibilidade defensiva caso o modelo devolva diretamente uma classe 0/1/2.
    predicted_class = int(round(prediction))
    return CLASS_SCORE_WEIGHTS.get(predicted_class)


def predict_candidate_adequacy(features: Dict[str, Any]) -> Optional[float]:
    """
    Devolve um score previsto de adequação terapêutica entre 0 e 1.

    Se o modelo for classificador, converte probabilidades de classe em score:
    classe 0 = não admissível;
    classe 1 = admissível com precaução;
    classe 2 = admissível.

    Se o modelo antigo for regressor, mantém compatibilidade com predict().
    """
    model = load_ranking_model()

    if model is None:
        return None

    row = pd.DataFrame([features])

    probability_score = score_from_class_probabilities(model, row)

    if probability_score is not None:
        return probability_score

    return score_from_prediction(model, row)