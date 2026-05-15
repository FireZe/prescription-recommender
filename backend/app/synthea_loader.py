import json
from pathlib import Path
from typing import Optional

import pandas as pd

from app.schemas import PatientContext
from app.normalization import normalize_main_problem


BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_PATH = BASE_DIR / "data" / "synthea_context_index.csv"


def _load_index() -> pd.DataFrame:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Synthea context index not found: {INDEX_PATH}. "
            "Run: python scripts/build_synthea_context_index.py"
        )

    return pd.read_csv(INDEX_PATH)


def list_synthea_patients(
    limit: int = 20,
    adults_only: bool = False,
    with_active_medications: bool = False,
) -> list[dict]:
    df = _load_index()

    if adults_only:
        df = df[df["age"] >= 18]

    if with_active_medications:
        df = df[df["active_medications"].apply(lambda value: len(json.loads(value)) > 0)]

    df = df.head(limit)

    return [
        {
            "patient_id": row["patient_id"],
            "age": int(row["age"]),
            "sex": row["sex"],
            "renal_status": row["renal_status"],
            "main_problem_guess": row["main_problem_guess"],
            "active_medications": json.loads(row["active_medications"]),
        }
        for _, row in df.iterrows()
    ]


def get_synthea_patient_context(
    patient_id: str,
    main_problem: Optional[str] = None,
) -> PatientContext:
    df = _load_index()

    match = df[df["patient_id"] == patient_id]

    if match.empty:
        raise ValueError(f"Patient not found in Synthea index: {patient_id}")

    row = match.iloc[0]

    conditions = json.loads(row["conditions_norm"])
    allergies = json.loads(row["allergies"])
    active_medications = json.loads(row["active_medications"])

    return PatientContext(
        patient_id=row["patient_id"],
        age=int(row["age"]),
        sex=row["sex"] if row["sex"] in ["M", "F"] else "Other",
        conditions=conditions,
        allergies=allergies,
        active_medications=active_medications,
        renal_status=row["renal_status"],
        main_problem=normalize_main_problem(main_problem or row["main_problem_guess"] or "unspecified"),
    )