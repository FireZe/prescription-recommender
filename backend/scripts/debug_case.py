import json
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]
KB = json.loads((BASE/"data"/"knowledge_base.json").read_text(encoding="utf-8"))
hp = BASE/"data"/"historical_patterns.json"
HP = json.loads(hp.read_text(encoding="utf-8")) if hp.exists() else {}
from app.schemas import PatientContext, MedicationLine
from app.rules_engine import run_safety_checks
from app.recommender import recommend_alternatives
p = PatientContext(patient_id="DBG", age=70, sex="F", conditions=[], allergies=[],
                   active_medications=["clopidogrel"], renal_status="normal", main_problem="pain")
rx = [MedicationLine(medication="ibuprofen")]
alerts = run_safety_checks(patient=p, prescription=rx, kb=KB)
for r in recommend_alternatives(p, rx, alerts, KB, HP):
    print(f"{r.medication:12} final={r.score_final} combined={r.score_combined} "
          f"heur={r.score_heuristic} hist={r.secondary_historical_score} adm={r.admissibility_class}")