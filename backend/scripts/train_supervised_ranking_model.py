from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
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
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parents[1]

REVIEWED_DATA_PATH = BASE_DIR / "data" / "training_examples_reviewed.csv"
SILVER_DATA_PATH = BASE_DIR / "data" / "training_examples.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "ranking_model.joblib"
METRICS_PATH = MODEL_DIR / "ranking_model_metrics.json"


CLASS_COLUMN = "label_class"

NUMERIC_FEATURES = [
    "age",
    "age_squared",
    "is_elderly",
    "active_medication_count",
    "condition_count",
    "renal_status_score",
    "candidate_renal_caution",
    "candidate_qt_risk",
    "has_anticoagulant",
    "has_antiplatelet",
    "has_diuretic",
    "has_acei_or_arb",
    "has_qt_risk_medication",
    "same_therapeutic_class",
    "candidate_is_nsaid",
]

CATEGORICAL_FEATURES = [
    "candidate",
    "candidate_class",
    "original_medication",
    "original_class",
    "main_problem",
]

CLASS_NAMES = {
    0: "0_nao_admissivel",
    1: "1_admissivel_com_precaucao",
    2: "2_admissivel",
}


def load_training_dataset() -> tuple[pd.DataFrame, str]:
    if REVIEWED_DATA_PATH.exists():
        df = pd.read_csv(REVIEWED_DATA_PATH)
        return df, "golden_reviewed"

    if SILVER_DATA_PATH.exists():
        df = pd.read_csv(SILVER_DATA_PATH)
        return df, "silver_fallback"

    raise FileNotFoundError(
        "Não foi encontrado dataset de treino. "
        "Esperado: training_examples_reviewed.csv ou training_examples.csv."
    )


def validate_dataset(df: pd.DataFrame, dataset_source: str) -> pd.DataFrame:
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
            ("numeric", "passthrough", NUMERIC_FEATURES),
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
        ),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            multi_class="auto",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    predictions = pipeline.predict(X_test)

    labels_present = sorted(set(y_test.unique()) | set(predictions))

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
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro", zero_division=0),
        "labels": labels_present,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def train() -> None:
    df, dataset_source = load_training_dataset()
    df = validate_dataset(df, dataset_source)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[CLASS_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model_results = []

    for name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

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
            item["metrics"]["macro_f1"],
            item["metrics"]["balanced_accuracy"],
        ),
        reverse=True,
    )

    best = model_results[0]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], MODEL_PATH)

    metrics_payload = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "dataset_path": str(
            REVIEWED_DATA_PATH if dataset_source == "golden_reviewed" else SILVER_DATA_PATH
        ),
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
        json.dump(metrics_payload, file, ensure_ascii=False, indent=2)

    print("Dataset usado:", dataset_source)
    print("Linhas:", len(df))
    print()
    print("Distribuição de classes:")
    print(df[CLASS_COLUMN].value_counts().sort_index())
    print()
    print("Resultados por modelo:")

    for item in model_results:
        metrics = item["metrics"]
        print(
            f"- {item['name']}: "
            f"macro_f1={metrics['macro_f1']:.4f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    print()
    print("Modelo selecionado:", best["name"])
    print(f"Modelo guardado em: {MODEL_PATH}")
    print(f"Métricas guardadas em: {METRICS_PATH}")


if __name__ == "__main__":
    train()