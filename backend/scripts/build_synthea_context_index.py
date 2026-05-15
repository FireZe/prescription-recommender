import json
from pathlib import Path
from datetime import date

import pandas as pd

# Permite importar app.normalization quando o script é executado a partir de backend/
import sys
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from app.normalization import (
    normalize_medication_id,
    normalize_condition_id,
    infer_main_problem,
)


SYNTHEA_DIR = BASE_DIR / "data" / "synthea"
OUTPUT_PATH = BASE_DIR / "data" / "synthea_context_index.csv"


EGFR_CODES = {"33914-3"}


def calculate_age(birthdate: str) -> int:
    born = pd.to_datetime(birthdate).date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def renal_status_from_egfr(value) -> str:
    try:
        egfr = float(value)
    except (TypeError, ValueError):
        return "normal"

    if egfr < 30:
        return "severe_impairment"

    if egfr < 60:
        return "mild_impairment"

    return "normal"

def json_unique_strings(values) -> str:
    cleaned = []

    for value in values:
        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        cleaned.append(value)

    return json.dumps(sorted(set(cleaned)), ensure_ascii=False)


def load_patients() -> pd.DataFrame:
    path = SYNTHEA_DIR / "patients.csv"
    df = pd.read_csv(path, usecols=["Id", "BIRTHDATE", "GENDER"])
    df["age"] = df["BIRTHDATE"].apply(calculate_age)
    df["sex"] = df["GENDER"].fillna("Other")
    return df.rename(columns={"Id": "patient_id"})


def aggregate_conditions() -> pd.DataFrame:
    path = SYNTHEA_DIR / "conditions.csv"
    df = pd.read_csv(path, usecols=["PATIENT", "DESCRIPTION", "STOP"])

    # Para protótipo, consideramos condições sem STOP como ativas.
    active = df[df["STOP"].isna()].copy()
    active["condition_norm"] = active["DESCRIPTION"].apply(normalize_condition_id)

    grouped = active.groupby("PATIENT").agg(
        conditions_raw=("DESCRIPTION", json_unique_strings),
        conditions_norm=("condition_norm", json_unique_strings)
    ).reset_index()

    return grouped.rename(columns={"PATIENT": "patient_id"})


def aggregate_allergies() -> pd.DataFrame:
    path = SYNTHEA_DIR / "allergies.csv"
    df = pd.read_csv(path, usecols=["PATIENT", "DESCRIPTION", "STOP"])

    active = df[df["STOP"].isna()].copy()
    active["allergy_norm"] = active["DESCRIPTION"].apply(lambda x: str(x).lower().strip())

    grouped = active.groupby("PATIENT").agg(
        allergies=("allergy_norm", json_unique_strings)
    ).reset_index()

    return grouped.rename(columns={"PATIENT": "patient_id"})


def aggregate_active_medications() -> pd.DataFrame:
    path = SYNTHEA_DIR / "medications.csv"
    df = pd.read_csv(path, usecols=["PATIENT", "DESCRIPTION", "STOP"])

    active = df[df["STOP"].isna()].copy()
    active["med_norm"] = active["DESCRIPTION"].apply(normalize_medication_id)

    grouped = active.groupby("PATIENT").agg(
        active_medications=("med_norm", json_unique_strings)
    ).reset_index()

    return grouped.rename(columns={"PATIENT": "patient_id"})


def extract_latest_egfr() -> pd.DataFrame:
    path = SYNTHEA_DIR / "observations.csv"

    rows = []

    usecols = ["DATE", "PATIENT", "CODE", "DESCRIPTION", "VALUE"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000):
        mask = (
            chunk["CODE"].astype(str).isin(EGFR_CODES)
            | chunk["DESCRIPTION"].astype(str).str.contains("Glomerular filtration rate", case=False, na=False)
        )

        egfr_chunk = chunk.loc[mask, ["DATE", "PATIENT", "VALUE"]].copy()

        if not egfr_chunk.empty:
            rows.append(egfr_chunk)

    if not rows:
        return pd.DataFrame(columns=["patient_id", "latest_egfr", "renal_status"])

    egfr = pd.concat(rows, ignore_index=True)
    egfr["DATE"] = pd.to_datetime(egfr["DATE"], errors="coerce")
    egfr["VALUE_NUM"] = pd.to_numeric(egfr["VALUE"], errors="coerce")

    egfr = egfr.dropna(subset=["DATE", "VALUE_NUM"])
    egfr = egfr.sort_values(["PATIENT", "DATE"])

    latest = egfr.groupby("PATIENT").tail(1).copy()
    latest["renal_status"] = latest["VALUE_NUM"].apply(renal_status_from_egfr)

    latest = latest.rename(columns={
        "PATIENT": "patient_id",
        "VALUE_NUM": "latest_egfr"
    })

    return latest[["patient_id", "latest_egfr", "renal_status"]]


def build_index():
    patients = load_patients()
    conditions = aggregate_conditions()
    allergies = aggregate_allergies()
    medications = aggregate_active_medications()
    egfr = extract_latest_egfr()

    df = patients.merge(conditions, on="patient_id", how="left")
    df = df.merge(allergies, on="patient_id", how="left")
    df = df.merge(medications, on="patient_id", how="left")
    df = df.merge(egfr, on="patient_id", how="left")

    df["conditions_raw"] = df["conditions_raw"].fillna("[]")
    df["conditions_norm"] = df["conditions_norm"].fillna("[]")
    df["allergies"] = df["allergies"].fillna("[]")
    df["active_medications"] = df["active_medications"].fillna("[]")
    df["renal_status"] = df["renal_status"].fillna("normal")

    df["main_problem_guess"] = df["conditions_norm"].apply(
        lambda value: infer_main_problem(json.loads(value))
    )

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved Synthea context index to {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(df.head(5)[[
        "patient_id",
        "age",
        "sex",
        "renal_status",
        "main_problem_guess",
        "active_medications"
    ]])


if __name__ == "__main__":
    build_index()