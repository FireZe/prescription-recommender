"""Executa os cinco cenários do protocolo de avaliação de usabilidade e
imprime a saída do sistema, para conferência com a tabela do anexo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.data_loader import load_historical_patterns, load_knowledge_base
from app.normalization import normalize_main_problem
from app.recommender import build_recommendation_notes, recommend_alternatives
from app.rules_engine import run_safety_checks
from app.schemas import MedicationLine, PatientContext

CENARIOS = [
    (1, "Risco", "Interação medicamentosa crítica", {
        "patient_id": "F1", "age": 60, "sex": "M",
        "conditions": ["respiratory_infection"], "allergies": [],
        "active_medications": ["simvastatin"], "renal_status": "normal",
        "main_problem": "infection"}, "claritromicina"),
    (2, "Controlo", "Antibiótico sem reatividade cruzada", {
        "patient_id": "F2", "age": 55, "sex": "F",
        "conditions": ["respiratory_infection"], "allergies": ["penicilina"],
        "active_medications": [], "renal_status": "normal",
        "main_problem": "infection"}, "azitromicina"),
    (3, "Risco", "Alergia ao fármaco prescrito", {
        "patient_id": "F3", "age": 55, "sex": "F",
        "conditions": ["respiratory_infection"], "allergies": ["penicilina"],
        "active_medications": [], "renal_status": "normal",
        "main_problem": "infection"}, "amoxicilina"),
    (4, "Controlo", "Analgésico sem risco renal associado", {
        "patient_id": "F4", "age": 66, "sex": "M",
        "conditions": ["hypertension"], "allergies": [],
        "active_medications": ["enalapril"], "renal_status": "normal",
        "main_problem": "dor"}, "paracetamol"),
    (5, "Risco", "Duplicação terapêutica", {
        "patient_id": "F5", "age": 66, "sex": "F",
        "conditions": ["hypertension"], "allergies": [],
        "active_medications": ["metoprolol"], "renal_status": "normal",
        "main_problem": "hipertensão"}, "atenolol"),
]


def main() -> None:
    kb = load_knowledge_base()
    historical_patterns = load_historical_patterns()

    for numero, tipo, nome, perfil, medicamento in CENARIOS:
        perfil["main_problem"] = normalize_main_problem(perfil["main_problem"])
        paciente = PatientContext(**perfil)
        prescricao = [MedicationLine(medication=medicamento)]

        alertas = run_safety_checks(
            patient=paciente, prescription=prescricao, kb=kb
        )
        recomendacoes = recommend_alternatives(
            patient=paciente, prescription=prescricao, alerts=alertas,
            kb=kb, historical_patterns=historical_patterns
        )
        notas = build_recommendation_notes(
            patient=paciente, recommendations=recomendacoes,
            alerts=alertas, kb=kb
        )

        print(f"### Cenário {numero} [{tipo}] {nome}")
        print(f"    Prescrição: {medicamento}")

        if alertas:
            for alerta in alertas:
                print(f"    ALERTA [{alerta.severity}] {alerta.medication} "
                      f"| {alerta.rule_id}")
                print(f"      {alerta.description}")
        else:
            print("    SEM ALERTA (comportamento esperado num controlo)")

        print(f"    Recomendações: "
              f"{[r.display_name or r.medication for r in recomendacoes] or 'nenhuma'}")

        for nota in notas:
            print(f"    Nota: {nota['description']}")

        print()


if __name__ == "__main__":
    main()