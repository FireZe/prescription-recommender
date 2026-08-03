"""
Script: extract_mimic_training_data.py  (versão 3)

Extrai exemplos de treino supervisionado a partir do MIMIC-IV.
Suporta ICD-9 e ICD-10. Totalmente vectorizado.

Executa com:
    python backend/scripts/extract_mimic_training_data.py
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np

import os
BASE_DIR = Path(os.getcwd()) / "backend"
sys.path.insert(0, str(BASE_DIR))

from app.normalization import normalize_medication_id
from app.data_loader import load_knowledge_base
from app.recommender import build_ml_features
from app.schemas import PatientContext, MedicationLine
from app.recommender import build_ml_features, score_candidate_base

MIMIC_DIR    = BASE_DIR / "data" / "mimic"
OUTPUT_PATH  = BASE_DIR / "data" / "training_examples_mimic.csv"
MAX_ADMISSIONS = 500_000

# ── ICD-9 — cobertura completa para todas as condições da KB ───────────────
ICD9_MAP = {
    # Hipertensão
    "401": "hypertension", "402": "hypertension", "403": "hypertension",
    "404": "hypertension", "405": "hypertension",
    # Insuficiência cardíaca
    "428": "heart_failure",
    # Enfarte do miocárdio / prevenção cardiovascular
    "410": "myocardial_infarction", "411": "myocardial_infarction",
    "412": "cardiovascular_prevention",
    "413": "cardiovascular_prevention",
    # Doença coronária crónica (aterosclerose, pós-stent, pós-bypass) — muito comum
    "414": "cardiovascular_prevention",
    # Status pós-procedimento cardiovascular
    "V451": "cardiovascular_prevention",
    "V458": "cardiovascular_prevention",
    # Fibrilhação auricular
    "4273": "atrial_fibrillation",
    # Outras arritmias
    "4270": "arrhythmia", "4271": "arrhythmia", "4272": "arrhythmia",
    "4274": "arrhythmia", "4275": "arrhythmia", "4276": "arrhythmia",
    "4279": "arrhythmia",
    # AVC
    "430": "stroke", "431": "stroke", "432": "stroke",
    "433": "stroke", "434": "stroke", "435": "stroke", "436": "stroke",
    # Diabetes
    "249": "diabetes", "250": "diabetes",
    # Tromboembolismo / anticoagulação
    "415": "thromboembolism_prevention",
    "451": "thromboembolism_prevention", "452": "thromboembolism_prevention",
    "453": "thromboembolism_prevention",
    # Uso crónico de anticoagulante (warfarin em doentes com FA crónica)
    "V5861": "thromboembolism_prevention",
    # Dislipidemia
    "272": "dyslipidemia",
    # Depressão
    "2962": "depression", "2963": "depression",
    "3004": "depression", "311": "depression",
    # Ansiedade
    "3000": "anxiety", "3001": "anxiety", "3009": "anxiety",
    "30002": "anxiety",
    # Dor neuropática
    "3570": "neuropathic_pain", "3571": "neuropathic_pain",
    "3572": "neuropathic_pain", "3576": "neuropathic_pain",
    "7292": "neuropathic_pain", "3379": "neuropathic_pain",
    # Enxaqueca / migraine
    "346": "migraine_prophylaxis",
    # OCD
    "3003": "ocd",
    # Dor
    "338": "pain", "719": "pain", "724": "pain",
    "729": "pain", "7840": "pain", "7890": "pain", "7809": "pain",
    # Febre
    "7806": "fever",
    # Inflamação / artrite
    "714": "inflammation", "720": "inflammation",
    "725": "inflammation", "726": "inflammation",
    "696": "inflammation", "7100": "inflammation",
    # Infecção
    "038": "infection",
    "041": "infection",
    "460": "infection", "461": "infection", "462": "infection",
    "463": "infection", "464": "infection", "465": "infection",
    "466": "infection",
    "480": "infection", "481": "infection", "482": "infection",
    "483": "infection", "484": "infection", "485": "infection",
    "486": "infection",
    "590": "infection", "595": "infection", "599": "infection",
    # Úlcera GI activa
    "531": "active_gi_ulcer", "532": "active_gi_ulcer",
    "533": "active_gi_ulcer", "534": "active_gi_ulcer",
    # Edema / retenção de fluidos
    "7823": "edema", "2766": "fluid_retention",
    # Congestão/edema pulmonar
    "514": "edema",
    "5185": "edema",
    # Insuficiência renal (renal_disease e renal_protection)
    "585": "renal_disease", "586": "renal_disease", "587": "renal_disease",
}
RENAL9_SEVERE = {"5854", "5855", "5856"}
RENAL9_MILD   = {"5851", "5852", "5853"}

# ── ICD-10 — cobertura completa para todas as condições da KB ──────────────
ICD10_MAP = {
    # Hipertensão
    "I10": "hypertension", "I11": "hypertension",
    "I12": "hypertension", "I13": "hypertension",
    # Insuficiência cardíaca
    "I50": "heart_failure",
    # Enfarte / prevenção cardiovascular
    "I21": "myocardial_infarction", "I22": "myocardial_infarction",
    "I20": "cardiovascular_prevention",
    "I25": "cardiovascular_prevention",   # toda a doença isquémica crónica (I252 está incluído)
    "I73": "cardiovascular_prevention",
    "Z951": "cardiovascular_prevention",  # bypass coronário (status pós-cirurgia)
    "Z955": "cardiovascular_prevention",  # angioplastia coronária
    # Fibrilhação auricular
    "I48": "atrial_fibrillation",
    # Outras arritmias
    "I47": "arrhythmia", "I49": "arrhythmia",
    # AVC
    "I60": "stroke", "I61": "stroke", "I62": "stroke",
    "I63": "stroke", "I64": "stroke", "I65": "stroke", "I66": "stroke",
    # Diabetes
    "E10": "diabetes", "E11": "diabetes",
    "E12": "diabetes", "E13": "diabetes",
    # Tromboembolismo / anticoagulação
    "I26": "thromboembolism_prevention",
    "I82": "thromboembolism_prevention",
    "I80": "thromboembolism_prevention",
    "Z7901": "thromboembolism_prevention",  # uso crónico de anticoagulantes (ICD-10)
    # Dislipidemia
    "E780": "dyslipidemia", "E781": "dyslipidemia",
    "E782": "dyslipidemia", "E784": "dyslipidemia",
    "E785": "dyslipidemia",
    # Depressão
    "F32": "depression", "F33": "depression", "F341": "depression",
    # Ansiedade
    "F40": "anxiety", "F410": "anxiety",
    "F411": "anxiety", "F412": "anxiety", "F419": "anxiety",
    # Dor neuropática
    "G62": "neuropathic_pain", "G54": "neuropathic_pain",
    "M544": "neuropathic_pain", "G89": "neuropathic_pain",
    # Enxaqueca / migraine
    "G43": "migraine_prophylaxis",
    # OCD
    "F42": "ocd",
    # Dor
    "R52": "pain", "M54": "pain", "M79": "pain",
    "R07": "pain", "M255": "pain",
    # Febre
    "R500": "fever", "R509": "fever",
    # Inflamação / artrite
    "M05": "inflammation", "M06": "inflammation",
    "M08": "inflammation", "M13": "inflammation",
    "M45": "inflammation", "L40": "inflammation",
    # Infecção
    "A41": "infection", "A49": "infection",
    "J00": "infection", "J02": "infection",
    "J04": "infection", "J06": "infection",
    "J15": "infection", "J18": "infection",
    "J20": "infection", "J22": "infection",
    "N10": "infection", "N39": "infection",
    # Úlcera GI activa
    "K25": "active_gi_ulcer", "K26": "active_gi_ulcer",
    "K27": "active_gi_ulcer", "K28": "active_gi_ulcer",
    # Edema / retenção de fluidos
    "R600": "edema", "R609": "edema",
    "E877": "fluid_retention",
    "J81": "edema",   # edema pulmonar
    # Insuficiência renal
    "N17": "renal_disease", "N18": "renal_disease", "N19": "renal_disease",
}
RENAL10_SEVERE = {"N173", "N174", "N175", "N183", "N184", "N185", "N186"}
RENAL10_MILD   = {"N171", "N172", "N181", "N182"}

CONDITION_PRIORITY = [
    "pain", "fever", "infection",
    "heart_failure", "myocardial_infarction", "atrial_fibrillation",
    "thromboembolism_prevention", "stroke",
    "hypertension", "cardiovascular_prevention",
    "diabetes", "renal_disease",
    "dyslipidemia", "edema", "fluid_retention",
    "active_gi_ulcer", "inflammation",
    "depression", "anxiety", "neuropathic_pain",
    "migraine_prophylaxis", "ocd",
]

def infer_main_problem(conditions: list) -> str:
    for p in CONDITION_PRIORITY:
        if p in conditions:
            return p
    return "unspecified"

def _map_icd9(code: str) -> tuple:
    if code in RENAL9_SEVERE:
        return "renal_disease", "severe_impairment"
    if code in RENAL9_MILD:
        return "renal_disease", "mild_impairment"
    for plen in (3, 4, 5):
        if code[:plen] in ICD9_MAP:
            return ICD9_MAP[code[:plen]], "normal"
    return None, "normal"


def _map_icd10(code: str) -> tuple:
    if code in RENAL10_SEVERE:
        return "renal_disease", "severe_impairment"
    if code in RENAL10_MILD:
        return "renal_disease", "mild_impairment"
    for plen in (3, 4):
        if code[:plen] in ICD10_MAP:
            return ICD10_MAP[code[:plen]], "normal"
    return None, "normal"


def build_diagnosis_index(diagnoses_df: pd.DataFrame) -> dict:
    """
    Mapeia cada código ICD único (não cada linha) → condição + renal.
    Com ~6M linhas mas apenas ~poucos milhares de códigos únicos,
    isto é muito mais rápido do que iterar linha a linha.
    """
    df = diagnoses_df.copy()
    df["code_clean"] = (
        df["icd_code"].astype(str)
        .str.replace(".", "", regex=False)
        .str.upper()
    )

    # Separa ICD-9 e ICD-10
    mask9  = df["icd_version"] == 9
    mask10 = df["icd_version"] == 10

    df9  = df[mask9].copy()
    df10 = df[mask10].copy()

    # Mapeia só os códigos únicos
    unique9  = df9["code_clean"].unique()
    unique10 = df10["code_clean"].unique()
    map9     = {c: _map_icd9(c)  for c in unique9}
    map10    = {c: _map_icd10(c) for c in unique10}

    df9["condition"]  = df9["code_clean"].map(lambda c: map9[c][0])
    df9["renal"]      = df9["code_clean"].map(lambda c: map9[c][1])
    df10["condition"] = df10["code_clean"].map(lambda c: map10[c][0])
    df10["renal"]     = df10["code_clean"].map(lambda c: map10[c][1])

    combined = pd.concat(
        [df9[["hadm_id", "condition", "renal"]],
         df10[["hadm_id", "condition", "renal"]]],
        ignore_index=True,
    )

    index = {}
    for hadm_id, grp in combined.groupby("hadm_id"):
        conditions  = list(grp["condition"].dropna().unique())
        renal_vals  = grp["renal"].tolist()
        if "severe_impairment" in renal_vals:
            renal_status = "severe_impairment"
        elif "mild_impairment" in renal_vals:
            renal_status = "mild_impairment"
        else:
            renal_status = "normal"
        index[int(hadm_id)] = {
            "conditions":   conditions,
            "renal_status": renal_status,
        }

    return index

def prepare_prescriptions(prescriptions_df: pd.DataFrame, kb_meds: set) -> pd.DataFrame:
    """
    Normaliza nomes de medicamentos e filtra para os que estão na KB.
    Labels são atribuídos pela KB em assign_kb_label(), não por duração.
    """
    unique_drugs = prescriptions_df["drug"].dropna().unique()
    print(f"  Nomes únicos de medicamentos: {len(unique_drugs):,}")
    norm_map = {d: normalize_medication_id(d) for d in unique_drugs}

    rx = prescriptions_df.copy()
    rx["drug_norm"] = rx["drug"].map(norm_map)
    rx = rx[rx["drug_norm"].isin(kb_meds)].copy()
    print(f"  Prescrições na KB: {len(rx):,}")

    return rx

def main():
    kb      = load_knowledge_base()
    kb_meds = set(kb.get("medications", {}).keys())

    print("A carregar MIMIC-IV...")

    patients_df = pd.read_csv(
        MIMIC_DIR / "patients.csv.gz", compression="gzip",
        usecols=["subject_id", "gender", "anchor_age"],
    ).set_index("subject_id")
    print(f"  patients: {len(patients_df):,} doentes")

    #admissions_df = pd.read_csv(
    #    MIMIC_DIR / "admissions.csv.gz", compression="gzip",
    #    usecols=["hadm_id", "hospital_expire_flag"],
    #).set_index("hadm_id")

    diagnoses_df = pd.read_csv(
        MIMIC_DIR / "diagnoses_icd.csv.gz", compression="gzip",
        usecols=["hadm_id", "icd_code", "icd_version"],
    )
    print(f"  diagnoses: {len(diagnoses_df):,} linhas → a indexar...")
    diag_index = build_diagnosis_index(diagnoses_df)
    # Guarda raw ICD codes por admissão para o log de diagnóstico
    del diagnoses_df
    print(f"  diagnoses index: {len(diag_index):,} admissões com diagnóstico")

    print("  A carregar prescriptions (aguarda ~2 min)...")
    prescriptions_df = pd.read_csv(
        MIMIC_DIR / "prescriptions.csv.gz", compression="gzip",
        usecols=["subject_id", "hadm_id", "drug"],
    )
    print(f"  prescriptions: {len(prescriptions_df):,} linhas → a filtrar...")
    rx = prepare_prescriptions(prescriptions_df, kb_meds)
    del prescriptions_df

    # Amostra de admissões
    all_hadm_ids = rx["hadm_id"].unique()
    if len(all_hadm_ids) > MAX_ADMISSIONS:
        rng = np.random.default_rng(42)
        sampled = rng.choice(all_hadm_ids, size=MAX_ADMISSIONS, replace=False)
        rx = rx[rx["hadm_id"].isin(sampled)]
        print(f"  Amostra: {MAX_ADMISSIONS:,} admissões de {len(all_hadm_ids):,}")

    # Pré-agrupa
    rx_by_hadm = {
        int(hadm_id): grp
        for hadm_id, grp in rx.groupby("hadm_id")
    }
    print(f"\nA gerar exemplos para {len(rx_by_hadm):,} admissões...")

    records = []
    skipped = 0

    import time
    t_start = time.time()
    total = len(rx_by_hadm)
    for i, (hadm_id, grp) in enumerate(rx_by_hadm.items(), 1):
        if i % 20000 == 0 or i == total:
            elapsed = time.time() - t_start
            rate = i / max(elapsed, 1e-9)
            eta = (total - i) / max(rate, 1e-9)
            print(f"  {i:,}/{total:,} admissões ({i/total*100:.0f}%) | "
                  f"decorrido {elapsed/60:.1f} min | ETA {eta/60:.1f} min", flush=True)
        subject_id = int(grp["subject_id"].iloc[0])

        if subject_id not in patients_df.index:
            skipped += 1
            continue

        pat_row      = patients_df.loc[subject_id]
        age          = int(pat_row["anchor_age"])
        sex          = "F" if pat_row["gender"] == "F" else "M"

        diag_info    = diag_index.get(hadm_id, {"conditions": [], "renal_status": "normal"})
        conditions   = diag_info["conditions"]
        renal_status = diag_info["renal_status"]
        main_problem = infer_main_problem(conditions)

        prescribed_set = set(grp["drug_norm"].drop_duplicates().tolist())
        all_active_meds = list(prescribed_set)

        # ── POSITIVOS: medicamentos prescritos pelo médico → label 2 ──────
        for candidate in prescribed_set:
            active_meds = [d for d in all_active_meds if d != candidate]

            med_info    = kb.get("medications", {}).get(candidate, {})
            indications = set(med_info.get("indications", []))
            matched     = [c for c in conditions if c in indications]
            effective_main_problem = (
                next((p for p in CONDITION_PRIORITY if p in matched), matched[0])
                if matched else main_problem
            )

            try:
                patient = PatientContext(
                    patient_id=str(subject_id),
                    age=age, sex=sex,
                    conditions=conditions, allergies=[],
                    active_medications=active_meds,
                    renal_status=renal_status,
                    main_problem=effective_main_problem,
                )
                features = build_ml_features(
                    candidate=candidate,
                    patient=patient,
                    prescription=[MedicationLine(medication=candidate)],
                    candidate_alerts=[], kb=kb,
                )
                features["label_class"] = 1   # prescrito pelo médico
                features["query_id"] = hadm_id  # 1 admissão = 1 evento de ranking
                features["subject_id"] = subject_id
                features["heuristic_score"] = score_candidate_base(
                    candidate, patient, candidate, kb
                )[0]
                records.append(features)
            except Exception:
                skipped += 1

        # ── NEGATIVOS: KB indica mas médico não prescreveu → label 0 ──────
        for candidate, med_info in kb.get("medications", {}).items():
            if candidate in prescribed_set:
                continue  # já é positivo

            indications = set(med_info.get("indications", []))
            matched     = [c for c in conditions if c in indications]
            if not matched:
                continue  # KB não indica para este doente → não gera negativo

            effective_main_problem = next(
                (p for p in CONDITION_PRIORITY if p in matched), matched[0]
            )

            try:
                patient = PatientContext(
                    patient_id=str(subject_id),
                    age=age, sex=sex,
                    conditions=conditions, allergies=[],
                    active_medications=all_active_meds,
                    renal_status=renal_status,
                    main_problem=effective_main_problem,
                )
                features = build_ml_features(
                    candidate=candidate,
                    patient=patient,
                    prescription=[MedicationLine(medication=candidate)],
                    candidate_alerts=[], kb=kb,
                )
                features["label_class"] = 0   # indicado mas não prescrito
                features["query_id"] = hadm_id  # 1 admissão = 1 evento de ranking
                features["subject_id"] = subject_id
                features["heuristic_score"] = score_candidate_base(
                    candidate, patient, candidate, kb
                )[0]
                records.append(features)
            except Exception:
                skipped += 1

    df = pd.DataFrame(records)

    if df.empty:
        print("\nNenhum exemplo gerado.")
        print("Verifica se os ficheiros estão em backend/data/mimic/")
        return

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nExemplos gerados : {len(df):,}")
    print(f"Ignorados        : {skipped:,}")

    # ── Log de diagnóstico ──────────────────────────────────────────────────
    print(f"\nExemplos gerados : {len(df):,}")
    print(f"Ignorados        : {skipped:,}")
    print("\nDistribuição de classes (implicit feedback):")
    print(df["label_class"].value_counts().sort_index())
    print(f"\nLabel 2 (prescritos): {(df['label_class']==2).sum():,}")
    print(f"Label 0 (não prescritos mas indicados): {(df['label_class']==0).sum():,}")
    print(f"\nGuardado em: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()