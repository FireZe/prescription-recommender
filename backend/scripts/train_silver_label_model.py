from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "training_examples.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "ranking_model.joblib"


TARGET_COLUMN = "adequacy_score"
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


def score_to_class(score: float) -> int:
    if score < 0.40:
        return 0  # não admissível
    if score < 0.75:
        return 1  # admissível com precaução
    return 2      # admissível


def train():
    df = pd.read_csv(DATA_PATH)

    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN, CLASS_COLUMN]
    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(f"Faltam colunas no training_examples.csv: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_score = df[TARGET_COLUMN]
    y_class = df[CLASS_COLUMN]

    X_train, X_test, y_train_score, y_test_score, y_train_class, y_test_class = train_test_split(
        X,
        y_score,
        y_class,
        test_size=0.25,
        random_state=42,
        stratify=y_class,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=5,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train_score)

    predictions_score = pipeline.predict(X_test).clip(0.0, 1.0)
    predictions_class = [score_to_class(score) for score in predictions_score]

    mae = mean_absolute_error(y_test_score, predictions_score)
    rmse = mean_squared_error(y_test_score, predictions_score) ** 0.5
    r2 = r2_score(y_test_score, predictions_score)

    print("Regression metrics:")
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))

    print()
    print("Classification metrics derived from predicted score:")
    print(classification_report(
        y_test_class,
        predictions_class,
        target_names=[
            "0_nao_admissivel",
            "1_admissivel_com_precaucao",
            "2_admissivel",
        ],
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(confusion_matrix(y_test_class, predictions_class))

    print()
    print("Target score distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    print()
    print("Target class distribution:")
    print(df[CLASS_COLUMN].value_counts().sort_index())

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print()
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()