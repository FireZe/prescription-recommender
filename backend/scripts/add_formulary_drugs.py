"""
Adiciona ao knowledge_base.json os fármacos da fase B (formulário PT alargado)
==============================================================================

Acrescenta 12 medicamentos comuns (cuidados primários + hospital), as suas
regras de interação e religa os `alternatives` dos fármacos existentes, tudo
fundamentado em normas da DGS / Orientações Infarmed / RCM (citação em
`alternatives_fonte` e nos comentários).

Faz backup do KB, aplica as alterações e valida o JSON no fim.
Idempotente: correr duas vezes não duplica.

Executa (na pasta do projeto):
    python backend/scripts/add_formulary_drugs.py
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

KB_PATH = Path("backend/data/knowledge_base.json")

# ── 12 novos medicamentos ───────────────────────────────────────────────────
NEW_MEDS = {
    # Diabetes (não havia nenhum fármaco na KB)
    "metformin": {
        "display_name": "Metformina", "active_substance": "metformina",
        "therapeutic_class": "biguanida", "indications": ["diabetes"],
        "contraindicated_conditions": ["severe_renal_impairment", "metabolic_acidosis", "severe_hepatic_impairment"],
        "renal_caution": True, "qt_risk": False,
        "alternatives": ["gliclazide"],
        "alternatives_fonte": "Norma DGS DM2 / Orientação Infarmed nº12 (CNFT): metformina é 1ª linha na DM2; sulfonilureia (gliclazida) como alternativa oral. RCM Metformina Basi, secção 4.3 (p.3): contraindicada em TFG<30 e acidose metabólica.",
    },
    "gliclazide": {
        "display_name": "Gliclazida", "active_substance": "gliclazida",
        "therapeutic_class": "sulfonilureia", "indications": ["diabetes"],
        "contraindicated_conditions": ["severe_renal_impairment", "severe_hepatic_impairment", "sulfonamide_allergy"],
        "renal_caution": True, "qt_risk": False,
        "alternatives": ["metformin"],
        "alternatives_fonte": "Norma DGS DM2: metformina preferível em 1ª linha; gliclazida (sulfonilureia) como opção. RCM Gliclazida Generis, secção 4.3 (p.2): contraind. insuf. renal/hepática grave, alergia a sulfamidas.",
    },
    # Infeção (só existiam macrólidos; amoxicilina é 1ª linha)
    "amoxicillin": {
        "display_name": "Amoxicilina", "active_substance": "amoxicilina",
        "therapeutic_class": "penicilina", "indications": ["infection"],
        "contraindicated_conditions": ["penicillin_hypersensitivity"],
        "renal_caution": True, "qt_risk": False,
        "alternatives": ["amoxicillin_clavulanate", "azithromycin", "clarithromycin"],
        "alternatives_fonte": "Norma DGS Pneumonia (p.1/p.5): amoxicilina é 1ª linha na infeção respiratória. RCM Amoxicilina Generis, secção 4.3 (p.5): contraind. hipersensibilidade a penicilinas/beta-lactâmicos.",
    },
    "amoxicillin_clavulanate": {
        "display_name": "Amoxicilina + ácido clavulânico", "active_substance": "amoxicilina+acido clavulanico",
        "therapeutic_class": "penicilina", "indications": ["infection"],
        "contraindicated_conditions": ["penicillin_hypersensitivity", "history_amoxiclav_hepatic_injury"],
        "renal_caution": True, "qt_risk": False,
        "alternatives": ["amoxicillin", "azithromycin", "clarithromycin"],
        "alternatives_fonte": "Norma DGS Pneumonia (p.6): amoxicilina/clavulânico em infeções com suspeita de resistência. RCM Amox+Clav Generis, secção 4.3 (p.4): contraind. hipersensibilidade a penicilinas e antecedente de lesão hepática por amox/clav.",
    },
    # Proteção gástrica / úlcera
    "omeprazole": {
        "display_name": "Omeprazol", "active_substance": "omeprazol",
        "therapeutic_class": "ibp", "indications": ["active_gi_ulcer", "gastric_protection"],
        "contraindicated_conditions": [],
        "renal_caution": False, "qt_risk": False,
        "alternatives": [],
        "alternatives_fonte": "Norma DGS Úlcera Péptica; RCM Omeprazol Generis, secção 4.3 (p.6). IBP para úlcera/gastroproteção; sem outro IBP no formulário atual (pantoprazol a considerar).",
    },
    # Dor — Passo II (opióide fraco)
    "tramadol": {
        "display_name": "Tramadol", "active_substance": "tramadol",
        "therapeutic_class": "opioide", "indications": ["pain"],
        "contraindicated_conditions": ["acute_intoxication", "uncontrolled_epilepsy", "maoi_use", "severe_respiratory_depression"],
        "renal_caution": True, "qt_risk": False,
        "alternatives": ["paracetamol"],
        "alternatives_fonte": "Escada analgésica OMS, Passo II (opióide fraco; IASP FS18 p.1). RCM Tramadol Generis, secção 4.3 (p.2): contraind. intoxicação aguda e IMAO. Risco serotoninérgico/convulsivo com ISRS e tricíclicos.",
    },
    # Anticoagulação — DOAC (alternativa real aos AVK) e HBPM
    "apixaban": {
        "display_name": "Apixabano", "active_substance": "apixabano",
        "therapeutic_class": "anticoagulante", "indications": ["anticoagulation", "thromboembolism_prevention", "atrial_fibrillation"],
        "contraindicated_conditions": ["active_bleeding", "severe_hepatic_impairment", "high_bleeding_risk"],
        "renal_caution": True, "qt_risk": False,
        "alternatives": ["warfarin", "acenocoumarol", "enoxaparin"],
        "alternatives_fonte": "RCM Apixabano (Eliquis), secção 4.3 (p.7): contraind. hemorragia ativa e doença hepática com coagulopatia. DOAC como alternativa aos AVK (menos interações alimentares/medicamentosas). Classe 'anticoagulante' herda a regra AINE+anticoagulante.",
    },
    "enoxaparin": {
        "display_name": "Enoxaparina", "active_substance": "enoxaparina sodica",
        "therapeutic_class": "anticoagulante", "indications": ["thromboembolism_prevention", "anticoagulation"],
        "contraindicated_conditions": ["active_bleeding", "heparin_induced_thrombocytopenia"],
        "renal_caution": True, "qt_risk": False,
        "alternatives": ["apixaban", "warfarin"],
        "alternatives_fonte": "Norma DGS Tromboembolismo; RCM Enoxaparina (Lovenox), secção 4.3 (p.9): contraind. hemorragia ativa e trombocitopenia imunomediada (HIT).",
    },
    # Depressão — outro ISRS
    "escitalopram": {
        "display_name": "Escitalopram", "active_substance": "escitalopram",
        "therapeutic_class": "isrs", "indications": ["depression", "anxiety"],
        "contraindicated_conditions": ["maoi_use", "qt_prolongation"],
        "renal_caution": False, "qt_risk": True,
        "alternatives": ["sertraline"],
        "alternatives_fonte": "Norma DGS Depressão (p.9: citalopram/sertralina entre 1ª linha). ISRS alternativo à sertralina; prolongamento do QT dose-dependente (qt_risk).",
    },
    # Beta-bloqueante vasodilatador
    "nebivolol": {
        "display_name": "Nebivolol", "active_substance": "nebivolol",
        "therapeutic_class": "beta_bloqueante", "indications": ["hypertension", "heart_failure"],
        "contraindicated_conditions": ["severe_hepatic_impairment", "acute_heart_failure", "bradycardia"],
        "renal_caution": False, "qt_risk": False,
        "alternatives": ["bisoprolol", "carvedilol", "metoprolol", "atenolol"],
        "alternatives_fonte": "Norma DGS 026/2011, ponto 8 (p.2): beta-bloqueante vasodilatador preferível em doença coronária/IC. RCM Nebivolol Generis, secção 4.3 (p.3).",
    },
    # Asma / DPOC
    "salbutamol": {
        "display_name": "Salbutamol", "active_substance": "salbutamol",
        "therapeutic_class": "agonista_beta2_curta", "indications": ["asthma", "bronchospasm"],
        "contraindicated_conditions": [],
        "renal_caution": False, "qt_risk": False,
        "alternatives": ["budesonide_formoterol"],
        "alternatives_fonte": "Norma DGS Asma/DPOC; RCM Salbutamol (Ventilan). Broncodilatador de curta ação (SABA) para alívio sintomático.",
    },
    "budesonide_formoterol": {
        "display_name": "Budesonida + formoterol", "active_substance": "budesonida+formoterol",
        "therapeutic_class": "corticoide_inalado_laba", "indications": ["asthma", "copd"],
        "contraindicated_conditions": [],
        "renal_caution": False, "qt_risk": False,
        "alternatives": ["salbutamol"],
        "alternatives_fonte": "Norma DGS Asma/DPOC; RCM Symbicort. Corticoide inalado + LABA para controlo de manutenção.",
    },
}

# ── Novas regras de interação ────────────────────────────────────────────────
NEW_RULES = [
    {"id": "ibp_clopidogrel_eficacia", "match": "class_drug", "class_a": "ibp", "medication_b": "clopidogrel",
     "severity": "moderate",
     "description": "Os inibidores da bomba de protões (ex.: omeprazol) podem reduzir a eficácia antiagregante do clopidogrel. Considerar IBP alternativo ou separação temporal. (Norma DGS Antiagregantes, p.11)"},
    {"id": "tramadol_isrs_serotoninergico", "match": "class_drug", "class_a": "isrs", "medication_b": "tramadol",
     "severity": "moderate",
     "description": "A associação de tramadol com ISRS aumenta o risco de síndrome serotoninérgica e de convulsões."},
    {"id": "tramadol_triciclico_serotoninergico", "match": "class_drug", "class_a": "antidepressivo_triciclico", "medication_b": "tramadol",
     "severity": "moderate",
     "description": "A associação de tramadol com antidepressivos tricíclicos aumenta o risco de síndrome serotoninérgica e de convulsões."},
]

# ── Religação de alternativas dos fármacos existentes ────────────────────────
ALT_UPDATES = {
    "warfarin":             ["acenocoumarol", "apixaban"],
    "acenocoumarol":        ["warfarin", "apixaban"],
    "azithromycin":         ["amoxicillin", "clarithromycin"],
    "clarithromycin":       ["amoxicillin", "azithromycin"],
    "sertraline":           ["escitalopram", "amitriptyline"],
    "amitriptyline":        ["sertraline", "escitalopram"],
    "metoprolol":           ["bisoprolol", "carvedilol", "atenolol", "nebivolol"],
    "bisoprolol":           ["carvedilol", "metoprolol", "atenolol", "nebivolol"],
    "carvedilol":           ["bisoprolol", "metoprolol", "atenolol", "nebivolol"],
    "atenolol":             ["bisoprolol", "carvedilol", "metoprolol", "nebivolol"],
    "ibuprofen":            ["paracetamol", "tramadol"],
    "naproxen":             ["paracetamol", "tramadol"],
    "paracetamol":          ["ibuprofen", "naproxen", "tramadol"],
}


def main() -> None:
    kb = json.loads(KB_PATH.read_text(encoding="utf-8"))
    meds = kb["medications"]
    rules = kb.setdefault("interaction_rules", [])

    # backup
    bak = KB_PATH.with_suffix(f".bak_{datetime.now():%Y%m%d_%H%M%S}.json")
    shutil.copy(KB_PATH, bak)

    added = []
    for mid, data in NEW_MEDS.items():
        if mid not in meds:
            meds[mid] = data
            added.append(mid)

    existing_rule_ids = {r.get("id") for r in rules}
    rules_added = []
    for r in NEW_RULES:
        if r["id"] not in existing_rule_ids:
            rules.append(r)
            rules_added.append(r["id"])

    relinked = []
    for mid, alts in ALT_UPDATES.items():
        if mid in meds:
            meds[mid]["alternatives"] = alts
            relinked.append(mid)

    # valida e grava
    text = json.dumps(kb, ensure_ascii=False, indent=2)
    json.loads(text)  # rebenta se inválido (antes de gravar)
    KB_PATH.write_text(text, encoding="utf-8")

    print(f"Backup: {bak.name}")
    print(f"Medicamentos adicionados ({len(added)}): {added}")
    print(f"Regras de interação adicionadas ({len(rules_added)}): {rules_added}")
    print(f"Alternativas religadas ({len(relinked)}): {relinked}")
    print(f"Total no formulário agora: {len(meds)} medicamentos, {len(rules)} regras.")
    print("JSON VÁLIDO — gravado com sucesso.")


if __name__ == "__main__":
    main()
