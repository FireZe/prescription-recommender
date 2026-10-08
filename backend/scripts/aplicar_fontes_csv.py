"""Aplica à base de conhecimento as fontes preenchidas manualmente no CSV."""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"
import sys

CSV_PATH = BASE_DIR / "data" / (
    sys.argv[1] if len(sys.argv) > 1 else "regras_sem_fonte.csv"
)
PROVENANCE_DIR = BASE_DIR / "data" / "diretrizes_pdf"

with KB_PATH.open("r", encoding="utf-8") as ficheiro:
    kb = json.load(ficheiro)

por_id = {regra["id"]: regra for regra in kb["interaction_rules"]}
aplicadas = 0
avisos = []

with CSV_PATH.open("r", encoding="utf-8") as ficheiro:
    for linha in csv.DictReader(ficheiro):
        regra = por_id.get(linha["id"])

        if regra is None:
            avisos.append(f"Regra desconhecida: {linha['id']}")
            continue

        fonte = (linha.get("fonte") or "").strip()
        documento = (linha.get("documento") or "").strip()

        if not fonte:
            avisos.append(f"Sem fonte preenchida: {linha['id']}")
            continue

        regra["fonte"] = fonte

        if documento:
            if (PROVENANCE_DIR / documento).is_file():
                regra["documento"] = documento
            else:
                avisos.append(f"Documento inexistente ({linha['id']}): {documento}")

        aplicadas += 1

KB_PATH.write_text(
    json.dumps(kb, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"Regras atualizadas: {aplicadas}")
for aviso in avisos:
    print(" -", aviso)