"""
Script: build_historical_patterns_mimic.py

Enriquece o historical_patterns.json com padrões de co-prescrição
extraídos do MIMIC-IV (training_examples_mimic.csv).

Para cada combinação (main_problem, age_group, renal_group),
calcula a frequência com que cada medicamento foi prescrito
de forma continuada (label=2 = admissível).

Só inclui padrões com pelo menos MIN_OBSERVATIONS observações.

Executa com:
    python backend/scripts/build_historical_patterns_mimic.py
"""

import json
from pathlib import Path
import pandas as pd

import os
BASE_DIR = Path(os.getcwd()) / "backend"
MIMIC_CSV   = BASE_DIR / "data" / "training_examples_mimic.csv"
OUTPUT_PATH = BASE_DIR / "data" / "historical_patterns.json"

MIN_OBSERVATIONS = 10   # mínimo de casos para o padrão ser fiável


def age_group(age: int) -> str:
    return "elderly" if age >= 65 else "adult"


def renal_group(renal_score: float) -> str:
    # renal_status_score: 0=normal, 1=mild, 2=severe
    return "renal_impairment" if renal_score > 0 else "normal_renal"


def build_key(row) -> str:
    return f"{row['main_problem']}|{age_group(row['age'])}|{renal_group(row['renal_status_score'])}"


def main():
    if not MIMIC_CSV.exists():
        print(f"Ficheiro não encontrado: {MIMIC_CSV}")
        return

    kb_path = BASE_DIR / "data" / "knowledge_base.json"
    with kb_path.open(encoding="utf-8") as f:
        kb = json.load(f)

    # Constrói mapa: medicamento → lista de indicações
    med_indications: dict[str, set] = {}
    for med_id, med_data in kb.get("medications", {}).items():
        med_indications[med_id] = set(med_data.get("indications", []))

    # Medicamentos de uso geral (permitidos em qualquer contexto)
    GENERAL_PURPOSE = {"paracetamol", "ibuprofen", "naproxen", "acetylsalicylic_acid"}

    df = pd.read_csv(MIMIC_CSV)
    print(f"MIMIC exemplos carregados: {len(df):,}")

    df["pattern_key"] = df.apply(build_key, axis=1)

    # Filtra: só mantém linhas onde o medicamento tem indicação para o main_problem
    # OU é de uso geral
    def is_indicated(row) -> bool:
        candidate = row["candidate"]
        main_prob = row["main_problem"]
        if candidate in GENERAL_PURPOSE:
            return True
        indications = med_indications.get(candidate, set())
        return main_prob in indications

    df_filtered = df[df.apply(is_indicated, axis=1)].copy()
    print(f"Após filtro de indicação KB: {len(df_filtered):,} exemplos")

    grouped = (
        df_filtered.groupby(["pattern_key", "candidate"])
        .agg(
            total=("label_class", "count"),
            admissible=("label_class", lambda x: (x == 2).sum()),
        )
        .reset_index()
    )

    # Vocabulário por chave (número de candidatos únicos antes do filtro de MIN_OBSERVATIONS)
    vocab_size = (
        df_filtered.groupby("pattern_key")["candidate"]
        .nunique()
        .rename("vocab_size")
        .reset_index()
    )

    grouped = grouped.merge(vocab_size, on="pattern_key")
    grouped = grouped[grouped["total"] >= MIN_OBSERVATIONS].copy()

    # Laplace smoothing: evita 0.0 para candidatos raros e suaviza frequências
    ALPHA = 1
    grouped["score"] = (
        (grouped["admissible"] + ALPHA) / (grouped["total"] + ALPHA * grouped["vocab_size"])
    ).round(3)

    patterns: dict = {}
    for _, row in grouped.iterrows():
        key = row["pattern_key"]
        if key not in patterns:
            patterns[key] = {}
        patterns[key][row["candidate"]] = float(row["score"])

    existing: dict = {}
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}

    merged = {**patterns}
    for key, meds in existing.items():
        if key not in merged:
            merged[key] = {}
        merged[key].update(meds)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\nPadrões históricos actualizados: {len(merged)} chaves")
    print(f"  - Do MIMIC (filtrado): {len(patterns)} chaves")
    print(f"  - Do feedback existente: {len(existing)} chaves")
    print(f"\nTop padrões:")
    for key, meds in sorted(merged.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {key}: {len(meds)} medicamentos → {list(meds.keys())}")
    print(f"\nGuardado em: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()