"""
Contextual bandit (LinUCB) — aprendizagem online a partir do feedback clínico
=============================================================================

Proof-of-concept de uma camada de aprendizagem por reforço para refinar o
ranking com base nos DESFECHOS reais (recompensa). Implementa o LinUCB
(Li et al., 2010, "A Contextual-Bandit Approach to Personalized News Article
Recommendation"), com parâmetros partilhados:

  - contexto x  = vetor de features do par (doente, candidato) — build_ml_features
  - recompensa r = desfecho do training_examples_feedback.csv
                   (resolved=1 ; not_resolved/adverse_event=0)
  - atualização : A += x xᵀ ;  b += r·x ;  θ = A⁻¹ b
  - score UCB   : θ·x + α·sqrt(xᵀ A⁻¹ x)   (exploração vs aproveitamento)

ESTADO / LIMITAÇÃO (documentar na tese):
  Este componente é FUNCIONAL mas é um proof-of-concept: é treinado com os
  desfechos disponíveis (atualmente sintéticos/curados). Requer um volume
  suficiente de desfechos REAIS (>~50) para ter validade clínica, e só então
  deve ser integrado na inferência ao vivo. Aqui é treinado e guardado, NÃO
  ligado ao recommender.

Executa:
    python backend/scripts/train_bandit.py
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

import os
BASE_DIR = Path(os.getcwd()) / "backend"
FEEDBACK_CSV = BASE_DIR / "data" / "training_examples_feedback.csv"
MODEL_DIR = BASE_DIR / "models"
BANDIT_PATH = MODEL_DIR / "bandit_model.joblib"

NUMERIC_FEATURES = [
    "age", "age_squared", "is_elderly", "is_female",
    "active_medication_count", "condition_count", "renal_status_score",
    "candidate_renal_caution", "candidate_qt_risk",
    "has_anticoagulant", "has_antiplatelet", "has_diuretic",
    "has_acei_or_arb", "has_qt_risk_medication", "candidate_is_nsaid",
]
CATEGORICAL_FEATURES = ["candidate", "candidate_class", "main_problem"]
ALPHA = 1.0  # nível de exploração (UCB)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         NUMERIC_FEATURES),
    ])


def train() -> None:
    if not FEEDBACK_CSV.exists():
        print(f"Sem dataset de feedback em {FEEDBACK_CSV}. Corre build_feedback_training_dataset.py primeiro.")
        return

    df = pd.read_csv(FEEDBACK_CSV)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
    if df.empty:
        print("Dataset de feedback vazio."); return

    pre = build_preprocessor()
    X = pre.fit_transform(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    rewards = df["label_class"].astype(float).clip(0, 1).values  # resolved=1, resto=0

    d = X.shape[1]
    A = np.identity(d)
    b = np.zeros(d)

    # Atualização LinUCB sobre as observações de feedback
    for x, r in zip(X, rewards):
        A += np.outer(x, x)
        b += r * x

    A_inv = np.linalg.inv(A)
    theta = A_inv @ b

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "preprocessor": pre,
        "A": A, "A_inv": A_inv, "b": b, "theta": theta,
        "alpha": ALPHA, "d": d,
        "n_observations": int(len(rewards)),
        "reward_mean": float(rewards.mean()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, BANDIT_PATH)

    print(f"Bandit (LinUCB) treinado com {len(rewards)} desfechos | dim contexto={d} | "
          f"recompensa média={rewards.mean():.2f}")
    print(f"Guardado em: {BANDIT_PATH}")
    print("NOTA: proof-of-concept (dados curados); requer feedback real para validade clínica "
          "e só então deve ser ligado à inferência.")


if __name__ == "__main__":
    train()
