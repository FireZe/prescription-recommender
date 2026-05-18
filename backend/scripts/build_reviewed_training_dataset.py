from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

REVIEW_INPUT_PATH = BASE_DIR / "data" / "training_candidates_for_review.csv"
REVIEWED_OUTPUT_PATH = BASE_DIR / "data" / "training_examples_reviewed.csv"


FEATURE_COLUMNS = [
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

CLASS_TO_DEFAULT_SCORE = {
    0: 0.00,  # não admissível
    1: 0.60,  # admissível com precaução
    2: 1.00,  # admissível
}


def build_reviewed_dataset() -> None:
    if not REVIEW_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Ficheiro não encontrado: {REVIEW_INPUT_PATH}. "
            "Executa primeiro: python scripts/export_training_candidates_for_review.py"
        )

    df = pd.read_csv(REVIEW_INPUT_PATH)

    required_columns = (
        FEATURE_COLUMNS
        + CATEGORICAL_FEATURES
        + [
            "gold_label_class",
            "gold_adequacy_score",
            "review_status",
        ]
    )

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Faltam colunas no ficheiro de revisão: {missing}"
        )

    df["review_status"] = df["review_status"].fillna("").astype(str).str.strip().str.lower()

    reviewed = df[df["review_status"] == "reviewed"].copy()

    if reviewed.empty:
        raise ValueError(
            "Não existem linhas com review_status='reviewed'. "
            "Preenche primeiro o ficheiro training_candidates_for_review.csv."
        )

    reviewed["gold_label_class"] = pd.to_numeric(
        reviewed["gold_label_class"],
        errors="coerce",
    )

    reviewed = reviewed.dropna(subset=["gold_label_class"]).copy()
    reviewed["gold_label_class"] = reviewed["gold_label_class"].astype(int)

    invalid_classes = sorted(
        set(reviewed["gold_label_class"].unique()) - {0, 1, 2}
    )

    if invalid_classes:
        raise ValueError(
            f"gold_label_class contém valores inválidos: {invalid_classes}. "
            "Usa apenas 0, 1 ou 2."
        )

    reviewed["gold_adequacy_score"] = pd.to_numeric(
        reviewed["gold_adequacy_score"],
        errors="coerce",
    )

    reviewed["adequacy_score"] = reviewed.apply(
        lambda row: (
            float(row["gold_adequacy_score"])
            if pd.notna(row["gold_adequacy_score"])
            else CLASS_TO_DEFAULT_SCORE[int(row["gold_label_class"])]
        ),
        axis=1,
    )

    reviewed["adequacy_score"] = reviewed["adequacy_score"].clip(0.0, 1.0)
    reviewed["label_class"] = reviewed["gold_label_class"]

    output_columns = FEATURE_COLUMNS + CATEGORICAL_FEATURES + [
        "adequacy_score",
        "label_class",
    ]

    output_df = reviewed[output_columns].copy()

    class_counts = output_df["label_class"].value_counts().sort_index()

    if output_df["label_class"].nunique() < 2:
        raise ValueError(
            "O dataset revisto precisa de pelo menos duas classes diferentes "
            "para treino supervisionado."
        )

    if class_counts.min() < 2:
        raise ValueError(
            "Cada classe presente deve ter pelo menos 2 exemplos para permitir "
            "divisão treino/teste estratificada. Distribuição atual:\n"
            f"{class_counts}"
        )

    REVIEWED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(REVIEWED_OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"Dataset revisto guardado em: {REVIEWED_OUTPUT_PATH}")
    print()
    print("Distribuição gold_label_class:")
    print(class_counts)
    print()
    print("Linhas:", len(output_df))
    print("Colunas:", list(output_df.columns))


if __name__ == "__main__":
    build_reviewed_dataset()