"""
Script: build_feedback_training_dataset.py

Extrai exemplos de treino a partir do feedback real dos utilizadores.
As labels são derivadas das decisões clínicas (accepted=2, rejected=0),
não do motor de regras — quebrando assim o problema do treino circular.

Executa com:
    python backend/scripts/build_feedback_training_dataset.py
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.normalization import normalize_medication_id
from app.data_loader import load_knowledge_base
from app.recommender import (
    build_ml_features,
    renal_status_score as renal_status_score_fn,
    get_active_medication_ids,
    has_active_class,
    has_active_qt_risk,
    get_medication_class,
)
from app.schemas import PatientContext, MedicationLine

DB_PATH = BASE_DIR / "data" / "prescription_data.db"
OUTPUT_PATH = BASE_DIR / "data" / "training_examples_feedback.csv"

DECISION_TO_CLASS = {
    "accepted": 2,    # admissível
    "rejected": 0,    # não admissível
    # "ignored" é descartado — sem sinal clínico claro
}


def extract_feedback_examples() -> pd.DataFrame:
    kb = load_knowledge_base()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            f.recommendation,
            f.decision,
            a.request_json
        FROM feedback f
        JOIN analyses a ON f.analysis_id = a.analysis_id
        WHERE f.decision IN ('accepted', 'rejected')
        AND f.recommendation IS NOT NULL
    """)
    rows = cursor.fetchall()
    conn.close()

    records = []

    for row in rows:
        try:
            request_data = json.loads(row["request_json"])
            patient_ctx = request_data.get("patient_context", {})
            prescription_data = request_data.get("prescription", [])

            patient = PatientContext(**patient_ctx)
            prescription = [MedicationLine(**p) for p in prescription_data]
            candidate = normalize_medication_id(row["recommendation"]) or row["recommendation"]

            if candidate not in kb.get("medications", {}):
                continue

            features = build_ml_features(
                candidate=candidate,
                patient=patient,
                prescription=prescription,
                candidate_alerts=[],
                kb=kb,
            )

            features["label_class"] = DECISION_TO_CLASS[row["decision"]]
            records.append(features)

        except Exception:
            continue

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = extract_feedback_examples()

    if df.empty:
        print("Sem exemplos de feedback suficientes para gerar dataset.")
    else:
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Dataset de feedback guardado: {OUTPUT_PATH} ({len(df)} linhas)")
        print(df["label_class"].value_counts().sort_index())