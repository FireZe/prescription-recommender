"""
Script de treino — Meta-learner (Stacking)
==========================================

Substitui o peso fixo (heurístico + ML) por uma LogisticRegression que
aprende os pesos óptimos a partir dos dados.

Em vez de treinar o seu próprio modelo base, CARREGA o modelo escolhido pelo
train_supervised_ranking_model.py (ranking_model.joblib) e acrescenta por
cima a camada de combinação — assim a comparação dos 4 modelos tem
consequência real e não há dois scripts a gravar o mesmo ficheiro.

Passos:
  1. Carrega o modelo base selecionado (ranking_model.joblib)
  2. Cross-validation 10-fold estratificada por grupos (admissão) para obter
     probabilidades ML out-of-fold, sem leakage
  3. Deriva um proxy do score heurístico a partir das features
  4. Treina LogisticRegression([heuristic_proxy, prob_ml]) -> label_class
     e avalia em holdout por grupo

Output: backend/models/meta_learner.joblib (NÃO altera o ranking_model.joblib)

Executa com:
    python backend/scripts/train_meta_learner.py
"""

from pathlib import Path
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_val_predict,
)

BASE_DIR = Path(os.getcwd()) / "backend"

MIMIC_DATA_PATH = BASE_DIR / "data" / "training_examples_mimic.csv"
MODEL_DIR = BASE_DIR / "models"
BASE_MODEL_PATH = MODEL_DIR / "ranking_model.joblib"
META_MODEL_PATH = MODEL_DIR / "meta_learner.joblib"
META_METRICS_PATH = MODEL_DIR / "meta_learner_metrics.json"

NUMERIC_FEATURES = [
    "age", "age_squared", "is_elderly", "is_female",
    "active_medication_count", "condition_count", "renal_status_score",
    "candidate_renal_caution", "candidate_qt_risk",
    "has_anticoagulant", "has_antiplatelet", "has_diuretic",
    "has_acei_or_arb", "has_qt_risk_medication",
    "same_therapeutic_class", "candidate_is_nsaid",
]

CATEGORICAL_FEATURES = [
    "candidate", "candidate_class", "original_medication",
    "original_class", "main_problem",
]

CLASS_COLUMN = "label_class"
N_SPLITS = 10

"""
def heuristic_proxy(df: pd.DataFrame) -> np.ndarray:
    Aproxima o score heurístico (Sseg, Sctx, Ssim) a partir das features.
    sseg = np.ones(len(df))
    sseg -= 0.4 * (df["renal_status_score"] == 2).astype(float) * df["candidate_renal_caution"]
    sseg -= 0.2 * df["candidate_qt_risk"] * df["has_qt_risk_medication"]
    sseg = sseg.clip(0.0, 1.0)

    sctx = np.full(len(df), 0.5)
    sctx += 0.35 * df["same_therapeutic_class"].astype(float)
    sctx -= 0.1 * (df["renal_status_score"] == 1).astype(float) * df["candidate_renal_caution"]
    sctx -= 0.3 * (df["renal_status_score"] == 2).astype(float) * df["candidate_renal_caution"]
    sctx = sctx.clip(0.0, 1.0)

    ssim = np.full(len(df), 0.5)
    ssim += 0.5 * df["same_therapeutic_class"].astype(float)
    ssim = ssim.clip(0.0, 1.0)

    proxy = 0.45 * sseg + 0.30 * sctx + 0.20 * ssim
    return proxy.values
"""

def train() -> None:
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo base não encontrado: {BASE_MODEL_PATH}\n"
            "Corre primeiro: python backend/scripts/train_supervised_ranking_model.py"
        )

    df = pd.read_csv(MIMIC_DATA_PATH)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df.dropna(subset=[CLASS_COLUMN]).copy()
    df[CLASS_COLUMN] = df[CLASS_COLUMN].astype(int)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    # Mesmas dtypes que o train_supervised usou (categóricas nativas)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y = df[CLASS_COLUMN]
    groups = df["subject_id"]

    print(f"Dataset: {len(df)} exemplos | base carregado: {BASE_MODEL_PATH.name}")
    print(f"Distribuição: {dict(y.value_counts().sort_index())}")

    base_pipeline = joblib.load(BASE_MODEL_PATH)

    # ── Passo 1: probabilidades out-of-fold, por grupo (sem leakage) ──
    print(f"\nProbabilidades ML via {N_SPLITS}-fold por grupo (admissão)...")
    t_cv = datetime.now()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    # n_jobs=1 de propósito: paralelizar folds copiava o dataset N vezes
    # e rebentava os 16GB de RAM. O HistGB já usa todos os cores internamente.
    ml_proba = cross_val_predict(
        clone(base_pipeline), X, y, cv=cv,
        method="predict_proba", groups=groups, n_jobs=1,
    )
    print(f"  ✓ concluída em {(datetime.now() - t_cv).total_seconds() / 60:.1f} min")

    # ── Passo 2 + 3: proxy heurístico e meta-features ──
    meta_X_df = pd.DataFrame({
        "heuristic_score": df["heuristic_score"].values,
        "prob_class_1": ml_proba[:, 1],
        "prob_class_0": ml_proba[:, 0],
    })

    # ── Passo 4: avaliar honestamente (holdout por grupo) ──
    tr_idx, te_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        .split(meta_X_df, y, groups)
    )
    meta_learner = LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs",
    )
    meta_learner.fit(meta_X_df.iloc[tr_idx], y.iloc[tr_idx])
    preds_te = meta_learner.predict(meta_X_df.iloc[te_idx])
    macro_f1 = f1_score(y.iloc[te_idx], preds_te, average="macro", zero_division=0)
    print(f"\nMeta-learner macro_f1 (holdout): {macro_f1:.4f}")
    print(classification_report(y.iloc[te_idx], preds_te, zero_division=0))

    # Re-treina com todos os dados para guardar
    meta_learner.fit(meta_X_df, y)
    print("Coeficientes aprendidos (peso de cada sinal):")
    for i, coefs in enumerate(meta_learner.coef_):
        print(f"  Classe {i}: heuristic_score={coefs[0]:.3f}, "
              f"prob1={coefs[1]:.3f}, prob0={coefs[2]:.3f}")

    # ── Passo 5: guardar SÓ o meta-learner ──
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(meta_learner, META_MODEL_PATH)

    metrics_payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(df)),
        "base_model_file": BASE_MODEL_PATH.name,
        "n_splits": N_SPLITS,
        "meta_learner_macro_f1_holdout": float(macro_f1),
        "meta_learner_coefficients": {
            f"class_{i}": {
                "heuristic_proxy": float(meta_learner.coef_[i][0]),
                "prob_class_1": float(meta_learner.coef_[i][1]),
                "prob_class_0": float(meta_learner.coef_[i][2]),
            }
            for i in range(len(meta_learner.coef_))
        },
    }
    with META_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"\nMeta-learner guardado em: {META_MODEL_PATH}")
    print(f"(ranking_model.joblib NÃO foi alterado — pertence ao train_supervised)")
    print(f"Métricas guardadas em:    {META_METRICS_PATH}")


if __name__ == "__main__":
    train()