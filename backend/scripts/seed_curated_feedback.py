"""
Semeia desfechos curados (bootstrap para o bandit / re-treino com feedback)
===========================================================================

Limpa o histórico de testes (analyses/feedback/outcomes) e cria ~44 casos
curados, cada um com um desfecho ESPERADO fundamentado nas guidelines/RCM:
  - resolved        : prescrição apropriada (1ª linha / segura)        -> sinal +
  - adverse_event   : prescrição arriscada/contraindicada              -> sinal - forte
  - not_resolved    : prescrição subótima (sem dano, mas não ideal)    -> sinal -

Para cada caso chama POST /analyze (guarda o contexto) e POST /outcome.

LIMITAÇÃO (documentar na tese): estes desfechos são SINTÉTICOS, codificam a
expectativa segundo as normas — NÃO são desfechos reais de doentes. Servem de
arranque (cold-start) para o bandit; o feedback real dos clínicos refina depois.

Pré-requisito: backend a correr (uvicorn). Executa:
    python backend/scripts/seed_curated_feedback.py
"""

import json
import sqlite3
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
DB_PATH = Path("backend/data/prescription_feedback.db")
CLEAR_FIRST = True   # apaga o histórico de testes antes de semear


def _p(pid, age, sex, problem, renal="normal", conditions=None, allergies=None, active=None):
    return {"patient_id": pid, "age": age, "sex": sex, "main_problem": problem,
            "renal_status": renal, "conditions": conditions or [],
            "allergies": allergies or [], "active_medications": active or []}


# (paciente, medicamento prescrito/seguido, desfecho, justificação)
CURATED = [
    # ── RESOLVED: prescrição apropriada (1ª linha / segura) ──────────────────
    (_p("C01", 70, "F", "pain", active=["warfarin"]), "paracetamol", "resolved", "Analgésico seguro com anticoagulante (sem risco hemorrágico do AINE)."),
    (_p("C02", 68, "M", "pain", active=["clopidogrel"]), "paracetamol", "resolved", "Analgésico seguro com antiagregante."),
    (_p("C03", 75, "F", "pain", "severe_impairment", ["renal_disease"]), "paracetamol", "resolved", "Analgésico preferível em insuf. renal."),
    (_p("C04", 60, "M", "pain", conditions=["active_gi_ulcer"]), "paracetamol", "resolved", "Evita o AINE contraindicado por úlcera."),
    (_p("C05", 35, "M", "pain"), "ibuprofen", "resolved", "AINE adequado quando não há contraindicação."),
    (_p("C06", 28, "F", "inflammation"), "naproxen", "resolved", "AINE adequado para componente inflamatória sem risco."),
    (_p("C07", 45, "M", "infection"), "amoxicillin", "resolved", "Amoxicilina 1ª linha na infeção respiratória (Norma DGS Pneumonia)."),
    (_p("C08", 50, "F", "infection", allergies=["amoxicillin"]), "azithromycin", "resolved", "Macrólido como alternativa na alergia à penicilina."),
    (_p("C09", 62, "M", "infection", conditions=["infection"]), "amoxicillin_clavulanate", "resolved", "Amox/clavulânico em suspeita de resistência (Norma DGS Pneumonia)."),
    (_p("C10", 58, "M", "hypertension"), "ramipril", "resolved", "IECA 1ª linha na HTA (Norma DGS 026/2011)."),
    (_p("C11", 60, "F", "hypertension"), "losartan", "resolved", "ARA como alternativa ao IECA."),
    (_p("C12", 67, "M", "heart_failure", conditions=["heart_failure"]), "bisoprolol", "resolved", "Beta-bloqueante na IC."),
    (_p("C13", 70, "F", "heart_failure", conditions=["heart_failure"]), "carvedilol", "resolved", "BB vasodilatador preferível na IC (Norma 026/2011, ponto 8)."),
    (_p("C14", 64, "M", "dyslipidemia"), "atorvastatin", "resolved", "Estatina 1ª escolha na dislipidemia (Norma DGS Dislipidemias)."),
    (_p("C15", 59, "F", "dyslipidemia"), "simvastatin", "resolved", "Estatina equivalente."),
    (_p("C16", 40, "F", "depression"), "sertraline", "resolved", "ISRS 1ª linha (Norma DGS Depressão)."),
    (_p("C17", 44, "M", "depression"), "escitalopram", "resolved", "ISRS alternativo de 1ª linha."),
    (_p("C18", 55, "M", "diabetes"), "metformin", "resolved", "Metformina 1ª linha na DM2."),
    (_p("C19", 52, "F", "diabetes"), "gliclazide", "resolved", "Sulfonilureia como opção oral."),
    (_p("C20", 72, "M", "atrial_fibrillation", conditions=["atrial_fibrillation"]), "apixaban", "resolved", "DOAC para anticoagulação na FA."),
    (_p("C21", 66, "M", "cardiovascular_prevention"), "acetylsalicylic_acid", "resolved", "Antiagregante na prevenção CV."),
    (_p("C22", 50, "F", "pain"), "tramadol", "resolved", "Opióide fraco (Passo II) quando o não-opióide é insuficiente."),
    (_p("C23", 63, "M", "active_gi_ulcer", conditions=["active_gi_ulcer"]), "omeprazole", "resolved", "IBP para úlcera/gastroproteção."),
    (_p("C24", 30, "F", "asthma", conditions=["asthma"]), "salbutamol", "resolved", "SABA para alívio na asma."),
    (_p("C25", 55, "M", "asthma", conditions=["asthma"]), "budesonide_formoterol", "resolved", "ICS+LABA para controlo."),
    (_p("C26", 71, "F", "hypertension"), "nebivolol", "resolved", "BB vasodilatador na HTA do idoso."),

    # ── ADVERSE_EVENT: prescrição arriscada/contraindicada ───────────────────
    (_p("C27", 70, "F", "pain", active=["warfarin"]), "ibuprofen", "adverse_event", "AINE+anticoagulante → hemorragia GI."),
    (_p("C28", 68, "M", "pain", active=["clopidogrel"]), "ibuprofen", "adverse_event", "AINE+antiagregante → hemorragia."),
    (_p("C29", 75, "F", "pain", "severe_impairment", ["renal_disease"]), "ibuprofen", "adverse_event", "AINE em rim grave → lesão renal."),
    (_p("C30", 60, "M", "pain", conditions=["active_gi_ulcer"]), "naproxen", "adverse_event", "AINE contraindicado por úlcera ativa."),
    (_p("C31", 60, "M", "infection", active=["simvastatin"]), "clarithromycin", "adverse_event", "Sinvastatina+claritromicina → miopatia/rabdomiólise (crítico)."),
    (_p("C32", 69, "F", "infection", active=["warfarin"]), "clarithromycin", "adverse_event", "Macrólido potencia a varfarina → hemorragia."),
    (_p("C33", 70, "M", "arrhythmia", active=["warfarin"]), "amiodarone", "adverse_event", "Amiodarona potencia a varfarina → hemorragia."),
    (_p("C34", 72, "M", "arrhythmia", active=["bisoprolol"]), "amiodarone", "adverse_event", "BB+amiodarona → bradicardia grave."),
    (_p("C35", 75, "M", "diabetes", "severe_impairment", ["renal_disease"]), "metformin", "adverse_event", "Metformina em TFG<30 → acidose láctica."),
    (_p("C36", 40, "F", "pain", allergies=["ibuprofen"]), "ibuprofen", "adverse_event", "Prescrição de fármaco a que o doente é alérgico."),
    (_p("C37", 55, "F", "pain", active=["sertraline"]), "tramadol", "adverse_event", "Tramadol+ISRS → síndrome serotoninérgica."),
    (_p("C38", 66, "M", "atrial_fibrillation", active=["apixaban"]), "warfarin", "adverse_event", "Dois anticoagulantes → hemorragia."),
    (_p("C39", 50, "M", "depression", active=["amitriptyline"]), "sertraline", "adverse_event", "ISRS+tricíclico → síndrome serotoninérgica."),

    # ── NOT_RESOLVED: subótimo (sem dano, mas não ideal) ─────────────────────
    (_p("C40", 55, "F", "inflammation", conditions=["inflammation"]), "paracetamol", "not_resolved", "Paracetamol não tem efeito anti-inflamatório."),
    (_p("C41", 48, "M", "pain"), "paracetamol", "not_resolved", "Dor intensa pode exigir escalada além do paracetamol."),
    (_p("C42", 57, "M", "hypertension"), "furosemide", "not_resolved", "Diurético de ansa não é 1ª linha na HTA (tiazida/IECA seriam)."),
    (_p("C43", 60, "F", "infection"), "azithromycin", "not_resolved", "Macrólido não é 1ª linha na infeção respiratória (amoxicilina seria)."),
    (_p("C44", 45, "M", "anxiety"), "amitriptyline", "not_resolved", "Tricíclico não é 1ª linha na ansiedade (ISRS seria)."),

    (_p("C45", 62, "M", "hypertension", conditions=["diabetes"]), "ramipril", "resolved", "IECA preferível no diabético (renoproteção)."),
    (_p("C46", 82, "F", "hypertension", "mild_impairment", ["hypertension", "heart_failure"], active=["ramipril", "furosemide"]), "ibuprofen", "adverse_event", "Triple whammy: AINE+IECA+diurético → lesão renal."),
    (_p("C47", 60, "M", "dyslipidemia", active=["clarithromycin"]), "simvastatin", "adverse_event", "Sinvastatina+claritromicina → miopatia (crítico)."),
    (_p("C48", 50, "M", "depression", active=["amitriptyline"]), "escitalopram", "adverse_event", "ISRS+tricíclico → síndrome serotoninérgica."),
    (_p("C49", 70, "F", "atrial_fibrillation", conditions=["atrial_fibrillation"]), "warfarin", "resolved", "AVK apropriado para anticoagulação na FA."),
    (_p("C50", 72, "M", "infection", active=["amiodarone"]), "azithromycin", "adverse_event", "Macrólido+amiodarona → risco QT."),
]


def post(path, payload):
    req = urllib.request.Request(
        BASE_URL + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    if CLEAR_FIRST and DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        for t in ("outcomes", "feedback", "analyses"):
            conn.execute(f"DELETE FROM {t}")
        conn.commit(); conn.close()
        print("Histórico de testes limpo (analyses, feedback, outcomes).")

    counts = {"resolved": 0, "not_resolved": 0, "adverse_event": 0}
    errors = 0
    for i, (patient, med, outcome, _nota) in enumerate(CURATED, 1):
        try:
            a = post("/analyze", {"patient": patient, "prescription": [{"medication": med}]})
            post("/outcome", {"analysis_id": a["analysis_id"], "medication": med,
                              "outcome": outcome, "comment": "curado (guideline)"})
            counts[outcome] += 1
        except Exception as e:
            errors += 1
            print(f"  [erro {i}] {e}")

    print(f"\nSemeados {sum(counts.values())} desfechos curados | erros: {errors}")
    print(f"Distribuição: {counts}")


if __name__ == "__main__":
    main()
