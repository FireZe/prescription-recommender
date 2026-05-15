import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Permite importar módulos app.* quando o script é executado a partir de backend/
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from app.schemas import PatientContext, MedicationLine, Alert
from app.data_loader import load_knowledge_base
from app.rules_engine import run_safety_checks
from app.normalization import normalize_medication_id


INDEX_PATH = BASE_DIR / "data" / "synthea_context_index.csv"
OUTPUT_PATH = BASE_DIR / "data" / "training_examples.csv"
AUDIT_OUTPUT_PATH = BASE_DIR / "data" / "training_examples_audit.csv"

BLOCKING_SEVERITIES = {"high", "critical"}

FEATURE_COLUMNS = [
    "age",
    "is_elderly",
    "renal_severe",
    "candidate_is_nsaid",
    "interaction_grave",
    "contraindication",
    "duplication",
]


def parse_json_list(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    return []


def choose_main_problem_for_original_medication(medication: Dict[str, Any]) -> str:
    """
    Escolhe um problema clínico aproximado para simular o contexto da prescrição.
    Isto não é diagnóstico real; serve apenas para gerar exemplos de treino coerentes
    com as indicações declaradas na base de conhecimento.
    """
    indications = medication.get("indications", [])

    preferred_order = [
        "pain",
        "inflammation",
        "fever",
        "infection",
        "hypertension",
        "heart_failure",
        "arrhythmia",
        "dyslipidemia",
        "cardiovascular_prevention",
        "antiplatelet",
        "anticoagulation",
        "depression",
        "anxiety",
    ]

    for problem in preferred_order:
        if problem in indications:
            return problem

    if indications:
        return indications[0]

    return "unspecified"


def build_patient_context_from_row(row: pd.Series, main_problem: str) -> PatientContext:
    sex = row.get("sex", "Other")
    if sex not in {"M", "F"}:
        sex = "Other"

    return PatientContext(
        patient_id=str(row["patient_id"]),
        age=int(row["age"]),
        sex=sex,
        conditions=parse_json_list(row.get("conditions_norm", "[]")),
        allergies=parse_json_list(row.get("allergies", "[]")),
        active_medications=parse_json_list(row.get("active_medications", "[]")),
        renal_status=row.get("renal_status", "normal") or "normal",
        main_problem=main_problem,
    )


def has_blocking_alert(alerts: List[Alert]) -> bool:
    return any(alert.severity in BLOCKING_SEVERITIES for alert in alerts)


def get_medication_class(medication_id: str, kb: Dict[str, Any]) -> str | None:
    medication = kb.get("medications", {}).get(medication_id)
    if not medication:
        return None

    return medication.get("therapeutic_class")


def has_therapeutic_duplication(
    candidate: str,
    patient: PatientContext,
    original_medication: str,
    kb: Dict[str, Any],
) -> int:
    """
    Verifica duplicação terapêutica entre o candidato e a medicação ativa do utente.

    Nota: o medicamento original é ignorado, porque aqui o candidato é tratado como
    alternativa/substituição e não como medicação adicional.
    """
    candidate_class = get_medication_class(candidate, kb)

    if not candidate_class:
        return 0

    for raw_medication in patient.active_medications:
        medication_id = normalize_medication_id(raw_medication)

        if not medication_id:
            continue

        if medication_id == candidate or medication_id == original_medication:
            continue

        medication_class = get_medication_class(medication_id, kb)

        if medication_class == candidate_class:
            return 1

    return 0


def build_features(
    candidate: str,
    original_medication: str,
    patient: PatientContext,
    candidate_alerts: List[Alert],
    kb: Dict[str, Any],
) -> Dict[str, int | float]:
    medication = kb["medications"][candidate]
    candidate_class = medication.get("therapeutic_class")

    interaction_grave = int(
        any(
            alert.type == "interaction" and alert.severity in {"high", "critical"}
            for alert in candidate_alerts
        )
    )

    contraindication = int(
        any(
            alert.type in {"contraindication", "renal_risk"}
            and alert.severity in {"high", "critical"}
            for alert in candidate_alerts
        )
    )

    return {
        "age": int(patient.age),
        "is_elderly": int(patient.age >= 65),
        "renal_severe": int(patient.renal_status == "severe_impairment"),
        "candidate_is_nsaid": int(candidate_class == "aine"),
        "interaction_grave": interaction_grave,
        "contraindication": contraindication,
        "duplication": has_therapeutic_duplication(
            candidate=candidate,
            patient=patient,
            original_medication=original_medication,
            kb=kb,
        ),
    }


def max_alert_severity(alerts: List[Alert]) -> str:
    rank = {
        "low": 1,
        "moderate": 2,
        "high": 3,
        "critical": 4,
    }

    if not alerts:
        return "none"

    return max((alert.severity for alert in alerts), key=lambda sev: rank.get(sev, 0))


def alerts_to_text(alerts: List[Alert]) -> str:
    if not alerts:
        return ""

    return " | ".join(
        f"{alert.type}:{alert.severity}:{alert.medication}:{alert.description}"
        for alert in alerts
    )


def balance_dataset(
    df: pd.DataFrame,
    max_positive_ratio: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evita que o dataset fique demasiado dominado por exemplos positivos.
    Mantém todos os negativos e limita positivos a uma razão máxima.
    """
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]

    if positives.empty or negatives.empty:
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    max_positives = int(len(negatives) * max_positive_ratio)

    if len(positives) > max_positives:
        positives = positives.sample(
            n=max_positives,
            random_state=random_state,
        )

    balanced = pd.concat([positives, negatives], ignore_index=True)
    return balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)


def generate_training_examples(
    max_patients: int | None = None,
    balance: bool = True,
    max_positive_ratio: float = 2.0,
) -> None:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Índice Synthea não encontrado: {INDEX_PATH}. "
            "Executa primeiro: python scripts/build_synthea_context_index.py"
        )

    kb = load_knowledge_base()
    synthea_index = pd.read_csv(INDEX_PATH)

    if max_patients is not None:
        synthea_index = synthea_index.head(max_patients)

    medications = kb.get("medications", {})

    rows = []
    audit_rows = []

    original_medications = [
        med_id
        for med_id, med in medications.items()
        if med.get("alternatives")
    ]

    for _, row in synthea_index.iterrows():
        for original_medication in original_medications:
            original_med = medications[original_medication]
            main_problem = choose_main_problem_for_original_medication(original_med)

            patient = build_patient_context_from_row(
                row=row,
                main_problem=main_problem,
            )

            alternatives = original_med.get("alternatives", [])

            for candidate in alternatives:
                if candidate not in medications:
                    continue

                candidate_prescription = [
                    MedicationLine(
                        medication=candidate,
                        dose=None,
                        frequency=None,
                        route=None,
                    )
                ]

                candidate_alerts = run_safety_checks(
                    patient=patient,
                    prescription=candidate_prescription,
                    kb=kb,
                )

                label = 0 if has_blocking_alert(candidate_alerts) else 1

                features = build_features(
                    candidate=candidate,
                    original_medication=original_medication,
                    patient=patient,
                    candidate_alerts=candidate_alerts,
                    kb=kb,
                )

                rows.append({
                    **features,
                    "label": label,
                })

                audit_rows.append({
                    "patient_id": patient.patient_id,
                    "age": patient.age,
                    "sex": patient.sex,
                    "renal_status": patient.renal_status,
                    "conditions": json.dumps(patient.conditions, ensure_ascii=False),
                    "active_medications": json.dumps(patient.active_medications, ensure_ascii=False),
                    "main_problem": patient.main_problem,
                    "original_medication": original_medication,
                    "candidate": candidate,
                    "max_alert_severity": max_alert_severity(candidate_alerts),
                    "alerts": alerts_to_text(candidate_alerts),
                    "label": label,
                    **features,
                })

    if not rows:
        raise RuntimeError("Nenhum exemplo de treino foi gerado.")

    df = pd.DataFrame(rows)

    # Garante a ordem das colunas esperada pelo train_silver_label_model.py
    df = df[FEATURE_COLUMNS + ["label"]]

    audit_df = pd.DataFrame(audit_rows)

    if balance:
        df = balance_dataset(
            df,
            max_positive_ratio=max_positive_ratio,
        )

        # Mantemos o ficheiro audit completo, sem balanceamento, para inspeção.
        # O ficheiro de treino é o dataset efetivamente balanceado.

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    audit_df.to_csv(AUDIT_OUTPUT_PATH, index=False)

    print(f"Training examples saved to: {OUTPUT_PATH}")
    print(f"Audit file saved to: {AUDIT_OUTPUT_PATH}")
    print()
    print("Training dataset:")
    print(df["label"].value_counts().rename(index={0: "label_0_inadequado", 1: "label_1_admissivel"}))
    print()
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print()
    print("Audit dataset rows:", len(audit_df))


def main():
    parser = argparse.ArgumentParser(
        description="Generate training_examples.csv from Synthea + knowledge_base + rules_engine."
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Número máximo de utentes Synthea a usar. Por defeito usa todos.",
    )

    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Desativa balanceamento entre labels positivos e negativos.",
    )

    parser.add_argument(
        "--max-positive-ratio",
        type=float,
        default=2.0,
        help="Razão máxima de positivos por negativo quando o balanceamento está ativo.",
    )

    args = parser.parse_args()

    generate_training_examples(
        max_patients=args.max_patients,
        balance=not args.no_balance,
        max_positive_ratio=args.max_positive_ratio,
    )


if __name__ == "__main__":
    main()