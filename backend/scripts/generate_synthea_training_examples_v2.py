import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from app.schemas import PatientContext, MedicationLine, Alert
from app.data_loader import load_knowledge_base
from app.rules_engine import run_safety_checks
from app.normalization import normalize_medication_id


INDEX_PATH = BASE_DIR / "data" / "synthea_context_index.csv"
OUTPUT_PATH = BASE_DIR / "data" / "training_examples.csv"
AUDIT_OUTPUT_PATH = BASE_DIR / "data" / "training_examples_audit.csv"


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


def get_medication_class(medication_id: str, kb: Dict[str, Any]) -> str | None:
    medication = kb.get("medications", {}).get(medication_id)

    if not medication:
        return None

    return medication.get("therapeutic_class")


def get_active_medication_ids(patient: PatientContext) -> List[str]:
    ids = []

    for raw_med in patient.active_medications:
        med_id = normalize_medication_id(raw_med)
        if med_id:
            ids.append(med_id)

    return ids


def has_active_class(patient: PatientContext, kb: Dict[str, Any], classes: set[str]) -> int:
    for med_id in get_active_medication_ids(patient):
        med_class = get_medication_class(med_id, kb)
        if med_class in classes:
            return 1

    return 0


def has_active_qt_risk(patient: PatientContext, kb: Dict[str, Any]) -> int:
    for med_id in get_active_medication_ids(patient):
        med = kb.get("medications", {}).get(med_id)
        if med and med.get("qt_risk"):
            return 1

    return 0


def renal_status_score(renal_status: str) -> int:
    mapping = {
        "normal": 0,
        "mild_impairment": 1,
        "severe_impairment": 2,
    }
    return mapping.get(renal_status, 0)


def max_alert_severity(alerts: List[Alert]) -> str:
    rank = {
        "none": 0,
        "low": 1,
        "moderate": 2,
        "high": 3,
        "critical": 4,
    }

    if not alerts:
        return "none"

    return max((alert.severity for alert in alerts), key=lambda sev: rank.get(sev, 0))


def adequacy_score_from_alerts(alerts: List[Alert]) -> float:
    """
    Silver score ordinal/contínuo derivado da camada de segurança clínica.
    Quanto mais grave o alerta, menor a adequação terapêutica.
    """
    severity = max_alert_severity(alerts)

    mapping = {
        "none": 1.00,
        "low": 0.85,
        "moderate": 0.60,
        "high": 0.20,
        "critical": 0.00,
    }

    return mapping.get(severity, 0.50)


def label_class_from_score(score: float) -> int:
    if score < 0.40:
        return 0  # não admissível
    if score < 0.75:
        return 1  # admissível com precaução
    return 2      # admissível


def alerts_to_text(alerts: List[Alert]) -> str:
    if not alerts:
        return ""

    return " | ".join(
        f"{alert.type}:{alert.severity}:{alert.medication}:{alert.description}"
        for alert in alerts
    )


def build_features(
    candidate: str,
    original_medication: str,
    patient: PatientContext,
    kb: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_obj = kb["medications"][candidate]
    original_obj = kb["medications"][original_medication]

    candidate_class = candidate_obj.get("therapeutic_class", "unknown")
    original_class = original_obj.get("therapeutic_class", "unknown")

    return {
        "age": int(patient.age),
        "age_squared": int(patient.age) ** 2,
        "is_elderly": int(patient.age >= 65),
        "active_medication_count": len(get_active_medication_ids(patient)),
        "condition_count": len(patient.conditions),
        "renal_status_score": renal_status_score(patient.renal_status),

        "candidate_renal_caution": int(bool(candidate_obj.get("renal_caution"))),
        "candidate_qt_risk": int(bool(candidate_obj.get("qt_risk"))),

        "has_anticoagulant": has_active_class(patient, kb, {"anticoagulante"}),
        "has_antiplatelet": has_active_class(patient, kb, {"antiagregante"}),
        "has_diuretic": has_active_class(patient, kb, {"diuretico_ansa", "diuretico_tiazidico"}),
        "has_acei_or_arb": has_active_class(patient, kb, {"ieca", "ara"}),
        "has_qt_risk_medication": has_active_qt_risk(patient, kb),

        "same_therapeutic_class": int(candidate_class == original_class),
        "candidate_is_nsaid": int(candidate_class == "aine"),

        "candidate": candidate,
        "candidate_class": candidate_class,
        "original_medication": original_medication,
        "original_class": original_class,
        "main_problem": patient.main_problem,
    }


def generate_training_examples(max_patients: int | None = None) -> None:
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

            patient = build_patient_context_from_row(row=row, main_problem=main_problem)

            for candidate in original_med.get("alternatives", []):
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

                adequacy_score = adequacy_score_from_alerts(candidate_alerts)
                label_class = label_class_from_score(adequacy_score)

                features = build_features(
                    candidate=candidate,
                    original_medication=original_medication,
                    patient=patient,
                    kb=kb,
                )

                rows.append({
                    **features,
                    "adequacy_score": adequacy_score,
                    "label_class": label_class,
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
                    "adequacy_score": adequacy_score,
                    "label_class": label_class,
                    **features,
                })

    if not rows:
        raise RuntimeError("Nenhum exemplo de treino foi gerado.")

    df = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audit_rows)

    ordered_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["adequacy_score", "label_class"]
    df = df[ordered_columns]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False)
    audit_df.to_csv(AUDIT_OUTPUT_PATH, index=False)

    print(f"Training examples saved to: {OUTPUT_PATH}")
    print(f"Audit file saved to: {AUDIT_OUTPUT_PATH}")
    print()
    print("Adequacy score distribution:")
    print(df["adequacy_score"].value_counts().sort_index())
    print()
    print("Label class distribution:")
    print(df["label_class"].value_counts().sort_index())
    print()
    print("Rows:", len(df))
    print("Columns:", list(df.columns))


def main():
    parser = argparse.ArgumentParser(
        description="Generate graded silver-label training examples from Synthea."
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Número máximo de utentes Synthea a usar. Por defeito usa todos.",
    )

    args = parser.parse_args()

    generate_training_examples(max_patients=args.max_patients)


if __name__ == "__main__":
    main()