"""
Limpa os dados de teste (analyses, feedback, outcomes) da base de dados.
Executar a partir da raiz do projeto:
    python backend/scripts/limpar_followupPendente.py
"""
import sqlite3
from pathlib import Path

DB = Path("backend/data/prescription_feedback.db")

con = sqlite3.connect(DB)
total = {}
for t in ("outcomes", "feedback", "analyses"):
    total[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    con.execute(f"DELETE FROM {t}")
con.commit()
con.close()
print(f"Apagados -> {total}")
