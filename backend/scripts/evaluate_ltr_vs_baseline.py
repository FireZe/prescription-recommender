"""
Script de avaliação — LTR vs. baseline de popularidade
======================================================

Contextualiza o NDCG do LambdaMART comparando-o com um baseline ingénuo:
ordenar os candidatos pela frequência global de prescrição (popularidade).
Usa o MESMO split por grupo (query_id = admissão, seed 42) do train_ltr,
para ser diretamente comparável.

Interpretação:
  - Se LTR >> baseline -> o modelo usa contexto clínico, não só popularidade.
  - Se LTR ~ baseline  -> o modelo está a decorar frequências.

Executa com:
    python backend/scripts/evaluate_ltr_vs_baseline.py
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score

BASE_DIR = Path(__file__).resolve().parents[1]
MIMIC_DATA_PATH = BASE_DIR / "data" / "training_examples_mimic.csv"
LTR_MODEL_PATH = BASE_DIR / "models" / "ltr_model.txt"

# Devem coincidir com as listas do train_ltr_model.py
NUMERIC_FEATURES = [
    "age", "age_squared", "is_elderly", "is_female",
    "condition_count", "renal_status_score",
    "candidate_renal_caution", "candidate_qt_risk",
    "has_anticoagulant", "has_antiplatelet", "has_diuretic",
    "has_acei_or_arb", "has_qt_risk_medication",
    "candidate_is_nsaid",
]
CATEGORICAL_FEATURES = ["candidate", "candidate_class", "main_problem"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def mean_ndcg(scores_col, df_test, k_list=(3, 5)) -> dict:
    """NDCG@k médio por query, ignorando grupos all-zero e singletons."""
    out = {k: [] for k in k_list}
    for _, g in df_test.groupby("query_id"):
        rel = g["label_class"].values.reshape(1, -1)
        sc = g[scores_col].values.reshape(1, -1)
        if rel.max() > 0 and rel.shape[1] > 1:
            for k in k_list:
                out[k].append(ndcg_score(rel, sc, k=k))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}


def main() -> None:
    df = pd.read_csv(MIMIC_DATA_PATH)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # Mesmo split por grupo do treino
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    _, test_idx = next(splitter.split(df, df["label_class"], df["query_id"]))
    train_df = df.drop(index=test_idx)
    test_df = df.loc[test_idx].copy()

    # ── Baseline: popularidade (base rate por candidato, calculada no treino) ──
    rate = train_df.groupby("candidate")["label_class"].mean()
    glob = train_df["label_class"].mean()
    test_df["pop_score"] = test_df["candidate"].map(rate).fillna(glob)
    base = mean_ndcg("pop_score", test_df)

    # ── LTR ──
    X_test = test_df[ALL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X_test[col] = X_test[col].astype("category")
    model = lgb.Booster(model_file=str(LTR_MODEL_PATH))
    test_df["ltr_score"] = model.predict(X_test)
    ltr = mean_ndcg("ltr_score", test_df)

    print(f"{'':24s}{'NDCG@3':>10s}{'NDCG@5':>10s}")
    print(f"{'Baseline popularidade':24s}{base[3]:>10.4f}{base[5]:>10.4f}")
    print(f"{'LTR (LambdaMART)':24s}{ltr[3]:>10.4f}{ltr[5]:>10.4f}")
    print(f"{'Ganho do LTR':24s}{ltr[3]-base[3]:>+10.4f}{ltr[5]-base[5]:>+10.4f}")


if __name__ == "__main__":
    main()
