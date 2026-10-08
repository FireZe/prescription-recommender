"""Extrai a fonte documental já presente na descrição de cada regra para um campo
próprio, e produz um esqueleto CSV com as regras que ainda não a declaram."""

import csv
import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"
ESQUELETO = BASE_DIR / "data" / "regras_sem_fonte.csv"

PADRAO = re.compile(r"\((RCM|FT|Norma|Ficha)[^)]*\)")

with KB_PATH.open("r", encoding="utf-8") as ficheiro:
    kb = json.load(ficheiro)

extraidas = 0
em_falta = []

for regra in kb["interaction_rules"]:
    if regra.get("fonte"):
        continue

    correspondencia = PADRAO.search(regra.get("description", ""))

    if correspondencia:
        regra["fonte"] = correspondencia.group(0).strip("()")
        extraidas += 1
    else:
        em_falta.append(regra["id"])

KB_PATH.write_text(
    json.dumps(kb, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

with ESQUELETO.open("w", encoding="utf-8", newline="") as ficheiro:
    escritor = csv.writer(ficheiro)
    escritor.writerow(["id", "fonte", "documento"])
    for rule_id in em_falta:
        escritor.writerow([rule_id, "", ""])

print(f"Fontes extraídas da descrição: {extraidas}")
print(f"Regras por preencher: {len(em_falta)} (ver {ESQUELETO.name})")