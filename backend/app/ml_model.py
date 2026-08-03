from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

def _aligned_row(expected_features, features: Dict[str, Any]) -> pd.DataFrame:
    """Row só com as colunas que o modelo espera; categóricas (object) -> 'category'."""
    row = pd.DataFrame([features]).reindex(columns=list(expected_features))
    for col in row.columns:
        if row[col].dtype == object:
            row[col] = row[col].astype("category")
    return row

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "ranking_model.joblib"
LTR_MODEL_PATH  = BASE_DIR / "models" / "ltr_model.txt"
META_MODEL_PATH = BASE_DIR / "models" / "meta_learner.joblib"

CLASS_SCORE_WEIGHTS = {0: 0.00, 1: 1.00} #0= não admissível; 1= admissível

@lru_cache(maxsize=1)
def load_ranking_model():
    if not MODEL_PATH.exists():
        return None

    if not MODEL_PATH.exists():
        logger.warning("Modelo ML não encontrado em %s. Score ML desativado.", MODEL_PATH)
        return None

    logger.info("Modelo ML carregado: %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)

@lru_cache(maxsize=1)
def load_ltr_model():
    if not LTR_MODEL_PATH.exists():
        return None
    logger.info("Modelo LTR carregado: %s", LTR_MODEL_PATH)
    return lgb.Booster(model_file=str(LTR_MODEL_PATH))


@lru_cache(maxsize=1)
def load_meta_learner():
    if not META_MODEL_PATH.exists():
        return None
    logger.info("Meta-learner carregado: %s", META_MODEL_PATH)
    return joblib.load(META_MODEL_PATH)

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
        logger.warning("Modelo ML nao encontrado em %s. Score ML desativado.", MODEL_PATH)
        return None

    row = _aligned_row(model.feature_names_in_, features)
    probability_score = score_from_class_probabilities(model, row)
    if probability_score is not None:
        return probability_score
    return score_from_prediction(model, row)

def predict_candidate_proba(features: Dict[str, Any]) -> Optional[list]:
    """Devolve [prob_classe_0, prob_classe_1, prob_classe_2] ou None."""
    model = load_ranking_model()
    if model is None or not hasattr(model, "predict_proba"):
        return None
    row = _aligned_row(model.feature_names_in_, features)
    return model.predict_proba(row)[0].tolist()


_LTR_CATS = {"candidate", "candidate_class", "main_problem"}

def predict_ltr_score(features: Dict[str, Any]) -> Optional[float]:
    """Score de relevância LTR normalizado para [0,1]. Prevê sobre NumPy
    (categóricas convertidas em códigos) para evitar a validação pandas do LightGBM."""
    model = load_ltr_model()
    if model is None:
        return None

    names = model.feature_name()
    pc = list(model.pandas_categorical or [])           # categorias guardadas no modelo
    cat_order = [n for n in names if n in _LTR_CATS]      # ordem das categóricas
    cats_by_name = (
        {n: pc[i] for i, n in enumerate(cat_order)}
        if len(pc) == len(cat_order) else {}
    )

    vals = []
    for n in names:
        v = features.get(n)
        if n in _LTR_CATS:
            cats = cats_by_name.get(n, [])
            v = cats.index(v) if v in cats else -1        # código da categoria (-1 = desconhecida)
        vals.append(float(v) if v is not None else float("nan"))

    raw = float(model.predict(np.asarray([vals], dtype="float64"))[0])
    return max(0.0, min(1.0, (raw + 2) / 6))


def predict_combined_score(
    features: Dict[str, Any],
    heuristic_score: float,
) -> float:
    """
    Combina heurístico + GBT calibrado + LTR + meta-learner.

    Hierarquia de fallback:
      1. Meta-learner + LTR   (ambos disponíveis)
      2. Meta-learner apenas  (LTR indisponível)
      3. Heurístico + GBT + LTR  (meta-learner indisponível)
      4. Combinação clássica 70/30 (só GBT disponível)
      5. Score heurístico puro    (nenhum modelo disponível)
    """
    ml_proba  = predict_candidate_proba(features)
    ltr_score = predict_ltr_score(features)
    meta      = load_meta_learner()

    if meta is not None and ml_proba is not None:
        # proxy (compatível com meta treinado com 'heuristic_proxy')
        renal         = features.get("renal_status_score", 0)
        renal_caution = features.get("candidate_renal_caution", 0)
        same_class    = features.get("same_therapeutic_class", 0)
        sseg = max(0.0, 1.0
                   - 0.4 * float(renal == 2) * float(renal_caution)
                   - 0.2 * float(features.get("candidate_qt_risk", 0)) * float(features.get("has_qt_risk_medication", 0)))
        sctx = max(0.0, min(1.0, 0.5
                   + 0.35 * float(same_class)
                   - 0.1  * float(renal == 1) * float(renal_caution)
                   - 0.3  * float(renal == 2) * float(renal_caution)))
        ssim = max(0.0, min(1.0, 0.5 + 0.5 * float(same_class)))
        h_proxy = 0.45 * sseg + 0.30 * sctx + 0.20 * ssim

        # adapta-se: usa 'heuristic_score' (real) ou 'heuristic_proxy' conforme o meta carregado
        hcol = list(getattr(meta, "feature_names_in_", ["heuristic_proxy"]))[0]
        hval = heuristic_score if hcol == "heuristic_score" else h_proxy
        meta_X = pd.DataFrame(
            [[hval, ml_proba[1], ml_proba[0]]],
            columns=[hcol, "prob_class_1", "prob_class_0"],
        )
        meta_score = float(meta.predict_proba(meta_X)[0][1])
        if ltr_score is not None:
            return round(max(0.0, min(1.0, 0.20 * meta_score + 0.80 * ltr_score)), 3)
        return round(max(0.0, min(1.0, meta_score)), 3)

    # Fallback: GBT clássico
    ml_score = predict_candidate_adequacy(features)

    if ml_score is not None and ltr_score is not None:
        return round(max(0.0, min(1.0,
            0.50 * heuristic_score + 0.25 * ml_score + 0.25 * ltr_score)), 3)

    if ml_score is not None:
        return round(max(0.0, min(1.0,
            0.70 * heuristic_score + 0.30 * ml_score)), 3)

    if ltr_score is not None:
        return round(max(0.0, min(1.0,
            0.70 * heuristic_score + 0.30 * ltr_score)), 3)

    return heuristic_score