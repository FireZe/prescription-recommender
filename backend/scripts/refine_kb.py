"""
Refinações de rigor ao knowledge_base.json
===========================================

(A) Remove o token inerte 'severe_renal_impairment' dos contraindicated_conditions
    (o risco renal é tratado pelo mecanismo renal_caution / renal_alert_severity,
    para não duplicar alertas). Marca a metformina como contraindicação renal
    crítica (TFG<30) via renal_alert_severity.
(C) Adiciona a regra de duplicação de anticoagulantes (prescrever dois AVK/DOAC/HBPM).

Nota (B): tokens como penicillin_hypersensitivity, sulfonamide_allergy,
severe_hepatic_impairment, etc., disparam quando a condição/alergia consta do
doente (inserção manual) — é o comportamento pretendido; ficam como estão.

Faz backup, aplica e valida o JSON. Idempotente.

Executa:
    python backend/scripts/refine_kb.py
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

KB_PATH = Path("backend/data/knowledge_base.json")

ANTICOAG_DUP_RULE = {
    "id": "anticoagulante_duplicacao",
    "match": "class_class",
    "class_a": "anticoagulante",
    "class_b": "anticoagulante",
    "severity": "high",
    "description": "A utilização concomitante de dois anticoagulantes (AVK, DOAC ou HBPM) aumenta marcadamente o risco hemorrágico e deve ser evitada, salvo indicação específica e monitorizada.",
}


def main() -> None:
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    meds = kb["medications"]
    rules = kb.setdefault("interaction_rules", [])

    bak = KB_PATH.with_suffix(f".bak_{datetime.now():%Y%m%d_%H%M%S}.json")
    shutil.copy(KB_PATH, bak)

    # (A) remover token renal inerte
    cleaned = []
    for mid, d in meds.items():
        cc = d.get("contraindicated_conditions", [])
        if "severe_renal_impairment" in cc:
            d["contraindicated_conditions"] = [c for c in cc if c != "severe_renal_impairment"]
            cleaned.append(mid)
    # metformina: contraindicação renal crítica
    if "metformin" in meds:
        meds["metformin"]["renal_alert_severity"] = "critical"

    # (C) regra de duplicação de anticoagulantes
    rule_added = False
    if ANTICOAG_DUP_RULE["id"] not in {r.get("id") for r in rules}:
        rules.append(ANTICOAG_DUP_RULE)
        rule_added = True

    text = json.dumps(kb, ensure_ascii=False, indent=2)
    json.loads(text)
    KB_PATH.write_text(text, encoding="utf-8")

    print(f"Backup: {bak.name}")
    print(f"(A) 'severe_renal_impairment' removido de ({len(cleaned)}): {cleaned}")
    print(f"    metformin.renal_alert_severity = critical")
    print(f"(C) regra anticoagulante_duplicacao adicionada: {rule_added}")
    print(f"Total: {len(meds)} medicamentos, {len(rules)} regras. JSON VÁLIDO.")


if __name__ == "__main__":
    main()
