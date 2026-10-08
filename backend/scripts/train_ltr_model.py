"""
Script de treino — Learning to Rank com LightGBM LambdaMART
===========================================================

Aprende a ORDENAR os candidatos por relevância clínica dentro de cada
contexto, em vez de os classificar isoladamente. Otimiza NDCG.

Cada "query" é o par (admissão, problema principal): o modelo
ordena os medicamentos prescritos vs. apenas indicados para esse doente.
Esta definição por admissão é a correta para LambdaMART e evita o limite de
10000 linhas por query do LightGBM.

Labels de relevância:
  0 = indicado mas não prescrito
  1 = prescrito pelo médico

Contexto no sistema híbrido: fornece o sinal score_ltr, complementar ao
score combinado (heurístico + ML) do meta-learner.

Executa com:
    python backend/scripts/train_ltr_model.py
"""

from pathlib import Path
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score

import os
BASE_DIR = Path(__file__).resolve().parents[1]

MIMIC_DATA_PATH = BASE_DIR / "data" / "training_examples_mimic.csv"
MODEL_DIR = BASE_DIR / "models"
LTR_MODEL_PATH = MODEL_DIR / "ltr_model.txt"
LTR_METRICS_PATH = MODEL_DIR / "ltr_model_metrics.json"

NUMERIC_FEATURES = [
    "age", "age_squared", "is_elderly", "is_female",
    "condition_count", "renal_status_score",
    "candidate_renal_caution", "candidate_qt_risk",
    "has_anticoagulant", "has_antiplatelet", "has_diuretic",
    "has_acei_or_arb", "has_qt_risk_medication",
    "candidate_is_nsaid",
    "active_medication_count",
]
CATEGORICAL_FEATURES = [
    "candidate", "candidate_class", "main_problem",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# LightGBM lambdarank tem um limite rígido de 10000 linhas por query.
# Mantemos uma margem de segurança.
MAX_QUERY_SIZE = 9000


def load_and_prepare() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega o dataset MIMIC e prepara:
    - X: features (numeric + categorical codificadas como category)
    - y: labels de relevância (0 = indicado mas não prescrito, 1 = prescrito)
    - query_ids: identificador do evento de ranking (1 admissão = 1 query)

    Cada query corresponde a UMA admissão hospitalar: o modelo aprende a
    ordenar os candidatos prescritos vs. apenas indicados para esse doente.
    Esta é a definição correta para LambdaMART (ao contrário de agregar
    milhares de doentes que partilham age/problema/medicação, o que criava
    queries gigantes e estourava o limite de 10000 linhas do LightGBM).
    """
    df = pd.read_csv(MIMIC_DATA_PATH)

    # Preenche colunas ausentes com 0 (compatibilidade)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    # query_id por admissão (gerado em extract_mimic_training_data.py).
    # Fallback para datasets antigos sem a coluna.
    if "query_id" not in df.columns:
        print("  [aviso] coluna 'query_id' ausente — a usar fallback "
              "(age, main_problem, original_medication). Re-corre o extract "
              "para queries por admissão.")
        df["query_id"] = df.groupby(
            ["age", "main_problem", "original_medication"]
        ).ngroup()

    # Query = (admissão, problema terapêutico): ordenar dentro do mesmo fim clínico
    df["query_id"] = df["query_id"].astype(str) + "|" + df["main_problem"].astype(str)

    # Proteção: divide qualquer query > MAX_QUERY_SIZE em sub-queries
    # (evita o erro "Number of rows exceeds upper limit of 10000").
    sub = df.groupby("query_id").cumcount() // MAX_QUERY_SIZE
    if sub.max() > 0:
        df["query_id"] = (
            df["query_id"].astype(str) + "_" + sub.astype(str)
        )
        df["query_id"] = df.groupby("query_id").ngroup()

    # Ordena por query_id para que grupos fiquem contíguos (requisito LightGBM)
    df = df.sort_values("query_id").reset_index(drop=True)

    X = df[ALL_FEATURES].copy()

    # LightGBM aceita categoricals nativas — muito mais eficiente que OneHotEncoder
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    y = df["label_class"].values
    query_ids = df["query_id"].values
    subject_ids = df["subject_id"].values
    return X, y, query_ids, subject_ids


def _slice(X, y, query_ids, subject_ids, idx):
    """Extrai um subconjunto mantendo as queries contiguas (requisito LightGBM)."""
    idx = np.sort(idx)                     # preserva a ordem por query_id
    Xs = X.iloc[idx].reset_index(drop=True)
    qs = query_ids[idx]
    _, groups = np.unique(qs, return_counts=True)
    return Xs, y[idx], qs, subject_ids[idx], groups


def split_train_valid_test(X, y, query_ids, subject_ids,
                           test_size=0.25, valid_size=0.10):
    """Split por DOENTE em treino / validacao / teste.

    A particao de validacao existe exclusivamente para o early stopping, de
    modo a que o numero de arvores NAO seja escolhido em funcao do conjunto de
    teste. A particao de teste e identica a obtida anteriormente (mesma
    semente e mesma proporcao), pelo que os resultados continuam comparaveis.
    """
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    dev_idx, test_idx = next(outer.split(X, y, groups=subject_ids))

    inner = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=42)
    rel_train, rel_valid = next(
        inner.split(X.iloc[dev_idx], y[dev_idx], groups=subject_ids[dev_idx])
    )

    return (_slice(X, y, query_ids, subject_ids, dev_idx[rel_train]),
            _slice(X, y, query_ids, subject_ids, dev_idx[rel_valid]),
            _slice(X, y, query_ids, subject_ids, test_idx))


def evaluate_ndcg(
    model: lgb.Booster,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    groups_test: np.ndarray,
) -> dict:
    """
    Calcula NDCG@3 e NDCG@5 manualmente, por grupo.
    """
    scores = model.predict(X_test)

    ndcg3_list = []
    ndcg5_list = []
    offset = 0

    for size in groups_test:
        true_rel = y_test[offset: offset + size].reshape(1, -1)
        pred_scores = scores[offset: offset + size].reshape(1, -1)

        if true_rel.max() > 0 and size > 1:  # ignora grupos todos-zero e grupos singleton
            ndcg3_list.append(ndcg_score(true_rel, pred_scores, k=3))
            ndcg5_list.append(ndcg_score(true_rel, pred_scores, k=5))

        offset += size

    return {
        "ndcg_at_3": float(np.mean(ndcg3_list)) if ndcg3_list else 0.0,
        "ndcg_at_5": float(np.mean(ndcg5_list)) if ndcg5_list else 0.0,
        "n_queries_evaluated": len(ndcg3_list),
    }


def train() -> None:
    X, y, query_ids, subject_ids = load_and_prepare()

    n_queries = len(np.unique(query_ids))
    print(f"Dataset: {len(X)} exemplos, {n_queries} queries (contextos clínicos)")
    print(f"Distribuição de labels: {dict(zip(*np.unique(y, return_counts=True)))}")

    (X_train, y_train, q_train, s_train, groups_train), \
        (X_valid, y_valid, q_valid, s_valid, groups_valid), \
        (X_test, y_test, q_test, s_test, groups_test) = split_train_valid_test(
            X, y, query_ids, subject_ids
        )
    print(f"Validacao: {len(X_valid)} exemplos, {len(groups_valid)} queries "
          f"(usada apenas para early stopping)")

    print(f"Train: {len(X_train)} exemplos, {len(groups_train)} queries")
    print(f"Test:  {len(X_test)} exemplos, {len(groups_test)} queries")

    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, group=groups_valid,
                             reference=train_data)

    params = {
        "objective":      "lambdarank",
        "metric":         "ndcg",
        "ndcg_eval_at":   [3, 5],
        "learning_rate":  0.05,
        "num_leaves":     31,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":   5,
        "verbose":        -1,
    }

    t0 = datetime.now()
    print(f"\nA treinar LambdaMART — iniciado às {t0.strftime('%H:%M:%S')}")
    print(f"(Máx. 500 rounds, early stopping a cada 30 sem melhoria no NDCG)")

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(LTR_MODEL_PATH))

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"✓ LTR concluído em {elapsed/60:.1f} min | Best iteration: {model.best_iteration}")

    ndcg_metrics = evaluate_ndcg(model, X_test, y_test, groups_test)

    metrics_payload = {
        "trained_at":      datetime.now(timezone.utc).isoformat(),
        "n_examples":      int(len(X)),
        "n_queries":       int(n_queries),
        "best_iteration":  int(model.best_iteration),
        **ndcg_metrics,
    }

    with LTR_METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    print(f"\nNDCG@3: {ndcg_metrics['ndcg_at_3']:.4f}")
    print(f"NDCG@5: {ndcg_metrics['ndcg_at_5']:.4f}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"\nModelo guardado em: {LTR_MODEL_PATH}")
    print(f"Métricas guardadas em: {LTR_METRICS_PATH}")


if __name__ == "__main__":
    train()