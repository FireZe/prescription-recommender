"""Análise de sensibilidade dos pesos heurísticos (Anexo E).
Corre:  python -m scripts.sensitivity_weights
"""
import os, json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
KB = json.loads((BASE / "data" / "knowledge_base.json").read_text(encoding="utf-8"))
hp_path = BASE / "data" / "historical_patterns.json"
HP = json.loads(hp_path.read_text(encoding="utf-8")) if hp_path.exists() else {}

from app.schemas import PatientContext, MedicationLine
from app.rules_engine import run_safety_checks
from app.recommender import recommend_alternatives

def P(**kw):
    base = dict(sex="M", conditions=[], allergies=[], active_medications=[],
                renal_status="normal")
    base.update(kw)
    return PatientContext(**base)

# Cenários com >= 2 candidatos admissíveis (para a ordem poder mudar)
CENARIOS = [
    ("dor + clopidogrel",      P(patient_id="S1", age=70, main_problem="pain",
                                 active_medications=["clopidogrel"]),      "ibuprofen"),
    ("dor + varfarina",        P(patient_id="S2", age=68, main_problem="pain",
                                 active_medications=["warfarin"]),         "ibuprofen"),
    ("dor + apixabano (renal)",P(patient_id="S3", age=75, main_problem="pain",
                                 renal_status="mild_impairment",
                                 active_medications=["apixaban"]),         "naproxen"),
    ("dor + AAS",              P(patient_id="S4", age=72, main_problem="pain",
                                 active_medications=["acetylsalicylic_acid"]),"ibuprofen"),
    ("inflamação + clopidogrel",P(patient_id="S5", age=66, main_problem="inflammation",
                                 active_medications=["clopidogrel"]),      "ibuprofen"),
    ("dor idoso + clopidogrel",P(patient_id="S6", age=84, main_problem="pain",
                                 renal_status="mild_impairment",
                                 active_medications=["clopidogrel"]),      "ibuprofen"),
]

CONFIGS = [("0.50","0.30","0.20"),  # baseline
           ("0.60","0.25","0.15"),
           ("0.40","0.40","0.20"),
           ("0.70","0.20","0.10"),
           ("0.34","0.33","0.33")]

def ordem(patient, med):
    rx = [MedicationLine(medication=med)]
    alerts = run_safety_checks(patient=patient, prescription=rx, kb=KB)
    recs = recommend_alternatives(patient, rx, alerts, KB, HP)
    return [r.medication for r in recs]

def set_w(w):
    os.environ["W_SEG"], os.environ["W_CTX"], os.environ["W_SIM"] = w

# baseline
set_w(CONFIGS[0])
base = {nome: ordem(p, med) for nome, p, med in CENARIOS}
print("Baseline (0.50/0.30/0.20):")
for nome, o in base.items():
    print(f"  {nome}: {o}")

print("\nSensibilidade:")
for w in CONFIGS[1:]:
    set_w(w)
    iguais_top1, iguais_ordem, mudou = 0, 0, []
    for nome, p, med in CENARIOS:
        o = ordem(p, med)
        if o and base[nome] and o[0] == base[nome][0]:
            iguais_top1 += 1
        if o == base[nome]:
            iguais_ordem += 1
        else:
            mudou.append(nome)
    n = len(CENARIOS)
    print(f"  {'/'.join(w)}: top-1 igual={iguais_top1}/{n} | ordem igual={iguais_ordem}/{n}"
          + (f" | mudou: {mudou}" if mudou else ""))