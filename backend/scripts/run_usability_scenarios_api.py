"""Executa os cinco cenários do protocolo de usabilidade contra a API em execução,
para que a saída registada no anexo seja a que o utilizador obtém pela interface.

Requer o servidor a correr: uvicorn app.main:app
"""

import os

import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

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
    with httpx.Client(timeout=60.0) as client:
        for numero, tipo, nome, perfil, medicamento in CENARIOS:
            payload = {
                "patient": perfil,
                "prescription": [{"medication": medicamento}],
            }

            resposta = client.post(f"{API_URL}/analyze", json=payload)
            resposta.raise_for_status()
            dados = resposta.json()

            print(f"### Cenário {numero} [{tipo}] {nome}")
            print(f"    Prescrição: {medicamento}")
            print(f"    analysis_id: {dados['analysis_id']}")

            alertas = dados.get("alerts", [])

            if alertas:
                for alerta in alertas:
                    print(f"    ALERTA [{alerta['severity']}] "
                          f"{alerta.get('medication')} | {alerta.get('rule_id')}")
                    print(f"      {alerta['description']}")
            else:
                print("    SEM ALERTA (comportamento esperado num controlo)")

            recomendacoes = [
                r.get("display_name") or r["medication"]
                for r in dados.get("recommendations", [])
            ]
            print(f"    Recomendações: {recomendacoes or 'nenhuma'}")

            for nota in dados.get("recommendation_notes", []):
                print(f"    Nota: {nota['description']}")

            print()


if __name__ == "__main__":
    main()