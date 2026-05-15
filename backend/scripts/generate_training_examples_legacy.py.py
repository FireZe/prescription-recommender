import json
import random
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = DATA_DIR / "training_examples.csv"


def generate_examples(n: int = 500):
    rows = []

    medications = ["ibuprofen", "naproxen", "paracetamol"]
    renal_statuses = ["normal", "mild_impairment", "severe_impairment"]
    age_groups = [35, 55, 72, 84]

    for _ in range(n):
        age = random.choice(age_groups)
        renal_status = random.choice(renal_statuses)
        candidate = random.choice(medications)

        is_nsaid = candidate in ["ibuprofen", "naproxen"]
        severe_renal = renal_status == "severe_impairment"

        interaction_grave = random.choice([0, 1]) if is_nsaid else 0
        contraindication = 1 if is_nsaid and severe_renal else 0
        duplication = random.choice([0, 1])

        # silver label simplificado
        if contraindication == 1 or interaction_grave == 1:
            label = 0
        else:
            label = 1

        rows.append({
            "age": age,
            "is_elderly": int(age >= 65),
            "renal_severe": int(severe_renal),
            "candidate_is_nsaid": int(is_nsaid),
            "interaction_grave": interaction_grave,
            "contraindication": contraindication,
            "duplication": duplication,
            "label": label
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} training examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_examples()