"""
Script de treino — Modelo de Ranking Supervisionado
====================================================

Contexto no sistema híbrido de recomendação:
  - Filtragem por conhecimento: motor de regras determinístico (rules_engine.py)
  - Filtragem por conteúdo: scoring heurístico (Sseg, Sctx, Ssim, Sfb)
  - Ranking supervisionado: este modelo

Treina e compara 4 classificadores (decision_tree, random_forest,
logistic_regression e HistGradientBoosting calibrado), seleciona o melhor
por macro-F1 e guarda-o em ranking_model.joblib. Cada candidato é pontuado
quanto à probabilidade de ser uma prescrição adequada ao contexto do doente.
Esta probabilidade é depois combinada com o score heurístico pelo
meta-learner (train_meta_learner.py), que aprende os pesos dinamicamente.

Validação sem leakage: o split treino/teste é feito por grupos (subject_id =
utente), garantindo que linhas do mesmo utente não caem em ambos os lados.
Categóricas tratadas nativamente pelo HistGB; ordinal nas árvores; OneHot +
scaling na regressão logística.

Classes (feedback implícito do MIMIC-IV):
  0 = indicado mas não prescrito pelo médico
  1 = prescrito pelo médico

Dataset: training_examples_mimic.csv (gerado por extract_mimic_training_data.py)

Executa com:
    python backend/scripts/train_supervised_ranking_model.py
"""

from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder

import os
BASE_DIR = Path(os.getcwd()) / "backend"

GOLDEN_DATA_PATH = BASE_DIR / "data" / "training_examples_reviewed.csv"
FEEDBACK_DATA_PATH = BASE_DIR / "data" / "training_examples_feedback.csv"
MIMIC_DATA_PATH = BASE_DIR / "data" / "training_examples_mimic.csv"
SILVER_DATA_PATH = BASE_DIR / "data" / "training_examples.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "ranking_model.joblib"
METRICS_PATH = MODEL_DIR / "ranking_model_metrics.json"


CLASS_COLUMN = "label_class"

NUMERIC_FEATURES = [
    "age", "age_squared", "is_elderly", "is_female",
    "active_medication_count", "condition_count", "renal_status_score",
    "candidate_renal_caution", "candidate_qt_risk",
    "has_anticoagulant", "has_antiplatelet", "has_diuretic",
    "has_acei_or_arb", "has_qt_risk_medication",
    "candidate_is_nsaid",
]
CATEGORICAL_FEATURES = [
    "candidate", "candidate_class", "main_problem",
]

CLASS_NAMES = {
    0: "0_nao_admissivel",
    1: "1_admissivel",
}


def load_training_dataset() -> tuple[pd.DataFrame, str]:
    frames = []
    sources = []

    #if GOLDEN_DATA_PATH.exists():
    #    df_golden = pd.read_csv(GOLDEN_DATA_PATH)
    #    frames.append(df_golden)
    #    sources.append(f"golden_reviewed ({len(df_golden)} linhas)")

    #if FEEDBACK_DATA_PATH.exists():
    #    df_feedback = pd.read_csv(FEEDBACK_DATA_PATH)
    #    frames.append(df_feedback)
    #    sources.append(f"feedback ({len(df_feedback)} linhas)")

    if MIMIC_DATA_PATH.exists():
        df_mimic = pd.read_csv(MIMIC_DATA_PATH)
        frames.append(df_mimic)
        sources.append(f"mimic_iv ({len(df_mimic)} linhas)")

    if frames:
        df_combined = pd.concat(frames, ignore_index=True)
        return df_combined, " + ".join(sources)

    #if SILVER_DATA_PATH.exists():
    #    df = pd.read_csv(SILVER_DATA_PATH)
    #    return df, "silver_fallback"

    raise FileNotFoundError(
        "Não foi encontrado dataset de treino. "
        "Esperado: training_examples_reviewed.csv ou training_examples.csv."
    )


def validate_dataset(df: pd.DataFrame, dataset_source: str) -> pd.DataFrame:
    # Preenche colunas numéricas ausentes com 0 (compatibilidade com datasets antigos)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [CLASS_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Faltam colunas no dataset de treino ({dataset_source}): {missing}"
        )

    df = df.dropna(subset=[CLASS_COLUMN]).copy()
    df[CLASS_COLUMN] = df[CLASS_COLUMN].astype(int)

    invalid_classes = sorted(set(df[CLASS_COLUMN].unique()) - {0, 1, 2})

    if invalid_classes:
        raise ValueError(
            f"O dataset contém classes inválidas: {invalid_classes}. "
            "Usa apenas 0, 1 ou 2."
        )

    class_counts = df[CLASS_COLUMN].value_counts().sort_index()

    if df[CLASS_COLUMN].nunique() < 2:
        raise ValueError(
            "O dataset precisa de pelo menos duas classes diferentes para treino."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Cada classe presente precisa de pelo menos 2 exemplos para "
            "train_test_split estratificado.\n"
            f"Distribuição atual:\n{class_counts}"
        )

    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
        ]
    )


def build_models() -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced",
            max_depth=6,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            verbose=1,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            class_weight="balanced",
            categorical_features="from_dtype",
            random_state=42,
        ),
    }

def make_json_serializable(value):
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value

def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    predictions = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    labels_present = sorted(
        int(label)
        for label in (set(y_test.unique()) | set(predictions))
    )

    report = classification_report(
        y_test,
        predictions,
        labels=labels_present,
        target_names=[CLASS_NAMES[label] for label in labels_present],
        zero_division=0,
        output_dict=True,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels_present,
    )

    return {
        "model_name": name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),        
        "labels": [int(label) for label in labels_present],
        "classification_report": make_json_serializable(report),
        "confusion_matrix": make_json_serializable(matrix),
    }


def train() -> None:
    df, dataset_source = load_training_dataset()
    df = validate_dataset(df, dataset_source)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    y = df[CLASS_COLUMN]
    
    groups = df["subject_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model_results = []

    models = build_models()
    n_models = len(models)

    for i, (name, estimator) in enumerate(models.items(), start=1):
        t0 = datetime.now()
        print(f"\n[{i}/{n_models}] A treinar: {name} — iniciado às {t0.strftime('%H:%M:%S')}")

        def make_pipeline(name, estimator):
            if name == "gradient_boosting":          # HistGB: categóricas nativas
                return Pipeline([("model", estimator)])
            if name in ("random_forest", "decision_tree"):   # árvores: ordinal basta
                pre = ColumnTransformer([
                    ("categorical", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                unknown_value=-1), CATEGORICAL_FEATURES),
                    ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
                ])
                return Pipeline([("preprocessor", pre), ("model", estimator)])
            return Pipeline([("preprocessor", build_preprocessor()),  # LogReg: OneHot+scale
                            ("model", estimator)])

        pipeline = make_pipeline(name, estimator)
        pipeline.fit(X_train, y_train)

        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  ✓ {name} concluído em {elapsed/60:.1f} min")

        metrics = evaluate_model(
            name=name,
            pipeline=pipeline,
            X_test=X_test,
            y_test=y_test,
        )

        model_results.append(
            {
                "name": name,
                "pipeline": pipeline,
                "metrics": metrics,
            }
        )

    model_results = sorted(
        model_results,
        key=lambda item: (
            item["metrics"]["pr_auc"],
            item["metrics"]["roc_auc"],
        ),
        reverse=True,
    )

    best = model_results[0]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_PATH)

    metrics_payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        #"dataset_path": str(
        #    GOLDEN_DATA_PATH if dataset_source.startswith("golden") else SILVER_DATA_PATH
        #),
        "dataset_path": str(MIMIC_DATA_PATH),
        "n_rows": int(len(df)),
        "class_distribution": {
            str(label): int(total)
            for label, total in df[CLASS_COLUMN].value_counts().sort_index().items()
        },
        "selected_model": best["name"],
        "model_results": [
            item["metrics"]
            for item in model_results
        ],
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            make_json_serializable(metrics_payload),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Dataset usado:", dataset_source)
    print("Linhas:", len(df))
    print()
    print("Distribuicao de classes:")
    print(df[CLASS_COLUMN].value_counts().sort_index())
    print()
    print("Resultados por modelo:")

    for item in model_results:
        metrics = item["metrics"]
        print(
            f"- {item['name']}: "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"roc_auc={metrics['roc_auc']:.4f}, "
            f"pr_auc={metrics['pr_auc']:.4f}"
        )

    print()
    print("Modelo selecionado:", best["name"])
    print(f"Modelo guardado em: {MODEL_PATH}")
    print(f"Metricas guardadas em: {METRICS_PATH}")


if __name__ == "__main__":
    train()
