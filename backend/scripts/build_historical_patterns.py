"""
Script: build_historical_patterns.py

Lê a tabela `feedback` da base de dados SQLite e constrói o ficheiro
historical_patterns.json com base nas decisões dos utilizadores.

A chave é: "{main_problem}|{age_group}|{renal_group}"
O valor é: {medication_id: score_normalizado}

score_normalizado = accepted_count / (accepted_count + rejected_count)
(decisões "ignored" são ignoradas)

Executa com:
    python backend/scripts/build_historical_patterns.py
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "prescription_data.db"
OUTPUT_PATH = BASE_DIR / "data" / "historical_patterns.json"

# Mínimo de decisões (accepted+rejected) para o padrão ser considerado fiável
MIN_DECISIONS = 3


def build_historical_key(main_problem: str, age: int, renal_status: str) -> str:
    age_group = "elderly" if age >= 65 else "adult"
    renal_group = "renal_impairment" if renal_status != "normal" else "normal_renal"
    return f"{main_problem}|{age_group}|{renal_group}"


def main():
    if not DB_PATH.exists():
        print(f"Base de dados não encontrada: {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Junta feedback com a análise original para recuperar o contexto do doente
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

    # Contadores: key -> medication -> {"accepted": n, "rejected": n}
    counts = defaultdict(lambda: defaultdict(lambda: {"accepted": 0, "rejected": 0}))

    for row in rows:
        try:
            request_data = json.loads(row["request_json"])
            patient_ctx = request_data.get("patient_context", {})
            age = patient_ctx.get("age", 50)
            renal_status = patient_ctx.get("renal_status", "normal")
            main_problem = patient_ctx.get("main_problem", "unspecified")
            medication = row["recommendation"]
            decision = row["decision"]

            key = build_historical_key(main_problem, age, renal_status)
            counts[key][medication][decision] += 1

        except (json.JSONDecodeError, KeyError):
            continue

    # Constrói o dicionário de padrões históricos
    patterns = {}

    for key, medications in counts.items():
        patterns[key] = {}

        for medication, totals in medications.items():
            accepted = totals["accepted"]
            rejected = totals["rejected"]
            total = accepted + rejected

            if total < MIN_DECISIONS:
                continue  # Dados insuficientes para ser fiável

            score = round(accepted / total, 3)
            patterns[key][medication] = score

    # Só guarda chaves com pelo menos um medicamento com dados suficientes
    patterns = {k: v for k, v in patterns.items() if v}

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(patterns, f, ensure_ascii=False, indent=2)

    print(f"historical_patterns.json atualizado: {len(patterns)} chaves")
    for key, meds in patterns.items():
        print(f"  {key}: {meds}")


if __name__ == "__main__":
    main()