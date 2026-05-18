from pathlib import Path
import argparse

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

AUDIT_INPUT_PATH = BASE_DIR / "data" / "training_examples_audit.csv"
REVIEW_OUTPUT_PATH = BASE_DIR / "data" / "training_candidates_for_review.csv"


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

META_COLUMNS = [
    "case_id",
    "patient_id",
    "age",
    "sex",
    "renal_status",
    "conditions",
    "active_medications",
    "main_problem",
    "original_medication",
    "original_class",
    "candidate",
    "candidate_class",
    "max_alert_severity",
    "alerts",
]

SILVER_COLUMNS = [
    "silver_adequacy_score",
    "silver_label_class",
]

GOLD_COLUMNS = [
    "gold_adequacy_score",
    "gold_label_class",
    "review_status",
    "reviewer",
    "review_comment",
]


def unique_columns(columns: list[str]) -> list[str]:
    seen = set()
    result = []

    for column in columns:
        if column not in seen:
            seen.add(column)
            result.append(column)

    return result


def export_candidates(
    sample_per_class: int | None = None,
    limit: int | None = None,
    random_state: int = 42,
) -> None:
    if not AUDIT_INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Ficheiro não encontrado: {AUDIT_INPUT_PATH}. "
            "Executa primeiro: python scripts/generate_synthea_training_examples_v2.py"
        )

    df = pd.read_csv(AUDIT_INPUT_PATH)

    required_columns = [
        "adequacy_score",
        "label_class",
        "patient_id",
        "candidate",
        "original_medication",
    ]

    missing = [column for column in required_columns if column not in df.columns]

    if missing:
        raise ValueError(
            f"Faltam colunas em training_examples_audit.csv: {missing}"
        )

    df = df.copy()

    df["case_id"] = [
        f"REV-{index + 1:06d}"
        for index in range(len(df))
    ]

    df["silver_adequacy_score"] = df["adequacy_score"]
    df["silver_label_class"] = df["label_class"]

    df["gold_adequacy_score"] = ""
    df["gold_label_class"] = ""
    df["review_status"] = "pending"
    df["reviewer"] = ""
    df["review_comment"] = ""

    if sample_per_class is not None:
        sampled_parts = []

        for _, group in df.groupby("silver_label_class"):
            n = min(sample_per_class, len(group))
            sampled_parts.append(
                group.sample(n=n, random_state=random_state)
            )

        df = pd.concat(sampled_parts, ignore_index=True)
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    if limit is not None:
        df = df.head(limit).copy()

    ordered_columns = unique_columns(
        META_COLUMNS
        + SILVER_COLUMNS
        + GOLD_COLUMNS
        + FEATURE_COLUMNS
        + CATEGORICAL_FEATURES
    )

    missing_ordered = [column for column in ordered_columns if column not in df.columns]

    if missing_ordered:
        raise ValueError(
            f"Faltam colunas necessárias para revisão: {missing_ordered}"
        )

    output_df = df[ordered_columns]

    REVIEW_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(REVIEW_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Ficheiro para revisão humana/dev guardado em: {REVIEW_OUTPUT_PATH}")
    print()
    print("Distribuição silver_label_class:")
    print(output_df["silver_label_class"].value_counts().sort_index())
    print()
    print("Próximo passo:")
    print("1. Abrir o CSV no Excel/LibreOffice.")
    print("2. Preencher gold_label_class, gold_adequacy_score, review_status, reviewer e review_comment.")
    print("3. Guardar novamente como CSV UTF-8.")


def main():
    parser = argparse.ArgumentParser(
        description="Exporta candidatos de treino para revisão humana/dev."
    )

    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help="Número de exemplos a exportar por classe silver. Ex.: 50 gera até 150 exemplos.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o número total de exemplos exportados.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    export_candidates(
        sample_per_class=args.sample_per_class,
        limit=args.limit,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()