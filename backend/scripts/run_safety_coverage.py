"""
Cobertura de segurança: gera 1 caso por cada contraindicação, por cada fármaco
com precaução renal, e para alergia (direta + cruzada de classe), a partir da
knowledge_base.json. Prova que TODAS estas regras disparam o alerta esperado.
Requer o backend a correr. Não usa LLM.
Executa (na pasta backend):  python scripts\\run_safety_coverage.py
"""
import json, urllib.request
from datetime import datetime
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
DATA = Path(__file__).resolve().parents[1] / "data"
KB = json.loads((DATA / "knowledge_base.json").read_text(encoding="utf-8"))

def _p(pid, age, sex, problem, renal="normal", conditions=None, allergies=None, active=None):
    return {"patient_id": pid, "age": age, "sex": sex, "main_problem": problem,
            "renal_status": renal, "conditions": conditions or [],
            "allergies": allergies or [], "active_medications": active or []}
def _rx(*m): return [{"medication": x} for x in m]

def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())

def build_cases():
    meds = KB["medications"]; cases = []; n = 0
    # (a) contraindicações: 1 caso por (fármaco, condição)
    for mid, m in meds.items():
        for cond in m.get("contraindicated_conditions", []):
            n += 1
            sex = "F" if cond in ("pregnancy", "breastfeeding") else "M"
            cases.append(("contraindicacao", f"{mid} × {cond}",
                _p(f"CI{n:03d}", 60, sex, "pain", conditions=[cond]), _rx(mid),
                "contraindication"))
    # (b) precaução renal: 1 caso por fármaco com renal_caution
    for mid, m in meds.items():
        if m.get("renal_caution"):
            n += 1
            cases.append(("renal", f"{mid} (compromisso renal grave)",
                _p(f"RN{n:03d}", 75, "M", "pain", renal="severe_impairment"), _rx(mid),
                "renal_caution"))
    # (c) alergia direta + cruzada de classe
    cases.append(("alergia", "alergia direta (ibuprofeno)",
        _p("ALG1", 50, "F", "pain", allergies=["ibuprofen"]), _rx("ibuprofen"), "allergy_conflict"))
    cases.append(("alergia", "alergia cruzada de classe (AINE: alérgico a ibuprofeno, prescrito naproxeno)",
        _p("ALG2", 50, "F", "pain", allergies=["ibuprofen"]), _rx("naproxen"), "allergy_cross_class"))
    return cases

def run():
    cases = build_cases(); rows = []; ok = 0
    for cat, nome, patient, rx, expect in cases:
        try:
            resp = post("/analyze", {"patient": patient, "prescription": rx})
            ids = {a.get("rule_id") for a in resp.get("alerts", [])}
            passed = expect in ids
        except Exception as e:
            ids = {f"ERRO: {e}"}; passed = False
        ok += passed
        rows.append((cat, nome, expect, "OK" if passed else "FALHOU", sorted(x for x in ids if x)))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    L = [f"# Cobertura de segurança — {ts}", "",
         f"**Total:** {len(rows)} | **OK:** {ok} | **Falhas:** {len(rows)-ok}", ""]
    for cat, nome, exp, st, ids in rows:
        L.append(f"- [{st}] ({cat}) {nome} → esperado `{exp}` | alertas: {ids}")
    out = DATA / f"safety_coverage_report_{ts}.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"{ok}/{len(rows)} casos passaram. Relatório: {out}")

if __name__ == "__main__":
    run()