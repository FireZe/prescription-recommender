import os, json
from pathlib import Path
from app.schemas import PatientContext, MedicationLine
from app.rules_engine import run_safety_checks
from app import recommender
from scripts.run_frontend_tests import HANDCRAFTED   # (nome, patient_dict, rx, exp)
from app import ml_model, recommender
ml_model.predict_ltr_score = lambda f: None      # força fallback heurístico
recommender.predict_ltr_raw = lambda f: None

BASE = Path(__file__).resolve().parents[1]
kb   = json.loads((BASE/"data"/"knowledge_base.json").read_text(encoding="utf-8"))
hist = json.loads((BASE/"data"/"historical_patterns.json").read_text(encoding="utf-8"))

import urllib.request
def _get(url):
    with urllib.request.urlopen(url, timeout=60) as x:
        return json.loads(x.read().decode())

_valid = set(PatientContext.model_fields)
SCEN = []
for nome, pdict, rx, _exp in HANDCRAFTED:
    SCEN.append((nome, PatientContext(**pdict),
                 [MedicationLine(medication=m["medication"], dose=None, frequency=None, route=None) for m in rx]))
for sp in _get("http://127.0.0.1:8000/synthea/patients?limit=15&adults_only=true&with_active_medications=true")[:15]:
    d = {k: v for k, v in sp.items() if k in _valid}; d.setdefault("main_problem", "pain")
    SCEN.append((f"Synthea {sp['patient_id'][:8]}", PatientContext(**d),
                 [MedicationLine(medication="ibuprofen", dose=None, frequency=None, route=None)]))

def rank(w):
    os.environ["W_SEG"], os.environ["W_CTX"], os.environ["W_SIM"] = map(str, w)
    res = {}
    for nome, p, presc in SCEN:
        al = run_safety_checks(patient=p, prescription=presc, kb=kb)
        recs = recommender.recommend_alternatives(p, presc, al, kb, hist)
        res[nome] = [r.medication for r in recs]
    return res

default = (0.50, 0.30, 0.20); base = rank(default)
multi = [n for n in base if len(base[n]) >= 2]     # só estes têm ordem comparável
grid = [(a/10, b/10, c/10) for a in range(3, 8) for b in range(1, 6) for c in range(1, 5) if a + b + c == 10]

t1_ok = t1_tot = ord_ok = ord_tot = 0
for w in grid:
    r = rank(w)
    for n in base:
        if base[n] and r[n]:
            t1_tot += 1; t1_ok += base[n][0] == r[n][0]
        if n in multi and r[n]:
            ord_tot += 1; ord_ok += base[n] == r[n]
print(f"Cenários com recomendação: {sum(1 for n in base if base[n])} | com >=2 alternativas: {len(multi)}")
print(f"Combinações testadas: {len(grid)}")
print(f"Top-1 estável: {100*t1_ok/t1_tot:.1f}%")
if ord_tot: print(f"Ordem completa estável (>=2 alt): {100*ord_ok/ord_tot:.1f}%")

changed = {}
for w in grid:
    r = rank(w)
    for n in base:
        if base[n] != r[n]:
            changed.setdefault(n, 0); changed[n] += 1
print("Cenários com ordenação alterada nalguma combinação:", changed or "nenhum")

print("\n| w_seg | w_ctx | w_sim | Top-1 vs default (%) | Ordem vs default (%) |")
print("|--:|--:|--:|--:|--:|")
for w in grid:
    r = rank(w); t1=t1n=o=on=0
    for n in base:
        if base[n] and r[n]:
            t1n += 1; t1 += base[n][0] == r[n][0]
        if len(base[n]) >= 2 and r[n]:
            on += 1; o += base[n] == r[n]
    print(f"| {w[0]:.1f} | {w[1]:.1f} | {w[2]:.1f} | {100*t1/t1n:.1f} | {100*o/on if on else 100:.1f} |")