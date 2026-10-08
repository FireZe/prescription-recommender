"""
Bateria de testes ao sistema (regras + heurística + ML + LLM)
=============================================================

Duas partes:
  A) Casos hand-crafted (ground-truth): cobrem TODAS as regras de interação da KB,
     contraindicação, risco renal, alergia, duplicação, multi-fármaco, controlo.
  B) Casos Synthea: utentes sintéticos reais (com medicação ativa) — robustez.

Guarda um relatório Markdown (legível) + JSON (dados) em backend/data/.

Pré-requisitos:
  - backend a correr:  uvicorn app.main:app  (na pasta backend)
  - LLM (opcional): ollama serve + modelo. Se não, põe RUN_LLM = False.

Executa (na pasta backend):
    python scripts/run_frontend_tests.py
"""

import json
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
RUN_LLM = True          # False = muito mais rápido (não chama o ollama)
LLM_TIMEOUT = 120
N_SYNTHEA = 15          # nº de utentes Synthea a testar
OUT_DIR = Path(__file__).resolve().parents[1] / "data"


def _p(pid, age, sex, problem, renal="normal", conditions=None, allergies=None, active=None):
    return {"patient_id": pid, "age": age, "sex": sex, "main_problem": problem,
            "renal_status": renal, "conditions": conditions or [],
            "allergies": allergies or [], "active_medications": active or []}


def _rx(*meds):
    return [{"medication": m} for m in meds]


# ── A) Casos hand-crafted: cobrem todas as regras de interação + segurança ──
HANDCRAFTED = [
    ("AINE + antiagregante (hemorragia)", _p("H01",70,"F","pain",active=["clopidogrel"]), _rx("ibuprofen"),
     "Alerta alto; recomendar paracetamol."),
    ("AINE + anticoagulante (hemorragia)", _p("H02",68,"M","pain",active=["warfarin"]), _rx("naproxen"),
     "Alerta alto; alternativa não-AINE."),
    ("Duplicação AINE+AINE", _p("H03",50,"M","pain",active=["ibuprofen"]), _rx("naproxen"),
     "Alerta de duplicação terapêutica."),
    ("AINE + IECA (risco renal)", _p("H04",65,"M","pain",active=["ramipril"]), _rx("ibuprofen"),
     "Alerta moderado renal."),
    ("AINE + ARA (risco renal)", _p("H05",65,"F","pain",active=["losartan"]), _rx("ibuprofen"),
     "Alerta moderado renal."),
    ("AINE + diurético de ansa", _p("H06",72,"F","pain",active=["furosemide"]), _rx("ibuprofen"),
     "Alerta moderado renal/diurético."),
    ("AINE + tiazida", _p("H07",72,"M","pain",active=["hydrochlorothiazide"]), _rx("naproxen"),
     "Alerta moderado tiazida."),
    ("Triple whammy (AINE+IECA+diurético)", _p("H08",82,"F","hypertension","mild_impairment",
        ["hypertension","heart_failure"], active=["ramipril","furosemide"]), _rx("ibuprofen"),
     "Alerta alto triple whammy + os moderados."),
    ("Sinvastatina + claritromicina (CRÍTICO)", _p("H09",60,"M","infection",active=["simvastatin"]), _rx("clarithromycin"),
     "Interação crítica (miopatia/rabdomiólise)."),
    ("Estatina + macrólido", _p("H10",61,"F","infection",active=["atorvastatin"]), _rx("azithromycin"),
     "Alerta alto miopatia."),
    ("Varfarina + amiodarona", _p("H11",70,"M","arrhythmia",active=["warfarin"]), _rx("amiodarone"),
     "Alerta alto (potenciação anticoagulante)."),
    ("Macrólido + varfarina", _p("H12",69,"F","infection",active=["warfarin"]), _rx("clarithromycin"),
     "Alerta alto."),
    ("ISRS + anticoagulante", _p("H13",58,"F","depression",active=["warfarin"]), _rx("sertraline"),
     "Alerta moderado hemorragia."),
    ("ISRS + antiagregante", _p("H14",58,"M","depression",active=["clopidogrel"]), _rx("sertraline"),
     "Alerta moderado."),
    ("ISRS + AINE", _p("H15",45,"F","depression",active=["ibuprofen"]), _rx("sertraline"),
     "Alerta moderado."),
    ("ISRS + tricíclico (serotoninérgico)", _p("H16",50,"M","depression",active=["amitriptyline"]), _rx("sertraline"),
     "Alerta alto (síndrome serotoninérgica)."),
    ("Combinação QT (amiodarona + azitromicina)", _p("H17",72,"M","infection",
        conditions=["arrhythmia"], active=["amiodarone"]), _rx("azithromycin"),
     "Alerta alto QT."),
    ("Amiodarona + digoxina", _p("H18",74,"M","arrhythmia",active=["digoxin"]), _rx("amiodarone"),
     "Alerta alto (toxicidade digoxina)."),
    ("Diurético de ansa + digoxina", _p("H19",78,"F","heart_failure",active=["digoxin"]), _rx("furosemide"),
     "Alerta moderado (hipocaliemia)."),
    ("Beta-bloqueante + amiodarona (bradicardia)", _p("H20",70,"M","arrhythmia",active=["bisoprolol"]), _rx("amiodarone"),
     "Alerta alto bradicardia."),
    ("Duplicação beta-bloqueante", _p("H21",66,"M","hypertension",active=["metoprolol"]), _rx("atenolol"),
     "Alerta elevado de duplicação."),
    ("Contraindicação: úlcera GI ativa", _p("H22",60,"M","pain",conditions=["active_gi_ulcer"]), _rx("naproxen"),
     "Alerta crítico de contraindicação."),
    ("AINE em insuficiência renal grave", _p("H23",75,"F","pain","severe_impairment",["renal_disease"]), _rx("ibuprofen"),
     "Alerta renal; preferir paracetamol."),
    ("Alergia ao medicamento prescrito", _p("H24",40,"F","pain",allergies=["ibuprofen"]), _rx("ibuprofen"),
     "Alerta CRÍTICO de alergia (regressão da nova regra)."),
    ("Multi-fármaco (3 de uma vez)", _p("H25",70,"M","cardiovascular_prevention",
        active=["clopidogrel"]), _rx("ibuprofen","warfarin","simvastatin"),
     "Vários alertas em simultâneo."),
    ("Controlo (sem riscos)", _p("H26",30,"M","pain"), _rx("paracetamol"),
     "Sem alerta bloqueante."),
    # ── fase B: fármacos/interações novos ──
    ("Omeprazol + clopidogrel (eficácia)", _p("H27",68,"M","active_gi_ulcer",active=["clopidogrel"]), _rx("omeprazole"),
     "Alerta moderado: IBP reduz eficácia do clopidogrel."),
    ("Tramadol + ISRS (serotoninérgico)", _p("H28",55,"F","pain",active=["sertraline"]), _rx("tramadol"),
     "Alerta moderado serotoninérgico."),
    ("AINE + apixabano (DOAC, hemorragia)", _p("H29",70,"M","pain",active=["apixaban"]), _rx("ibuprofen"),
     "Alerta de hemorragia (DOAC herda regra AINE+anticoagulante)."),
    ("Macrólido+varfarina → amoxicilina alternativa", _p("H30",69,"F","infection",active=["warfarin"]), _rx("clarithromycin"),
     "Alerta varfarina+macrólido; amoxicilina deve surgir como alternativa segura."),
    ("Escitalopram + amiodarona (QT)", _p("H31",72,"M","depression",active=["amiodarone"]), _rx("escitalopram"),
     "Alerta QT (escitalopram qt_risk)."),
    ("Metformina em insuf. renal grave", _p("H32",75,"M","diabetes","severe_impairment",["renal_disease"]), _rx("metformin"),
     "Alerta renal (metformina contraindicada em TFG<30; via renal_caution)."),
    ("Infeção com amoxicilina (controlo)", _p("H33",40,"M","infection"), _rx("amoxicillin"),
     "Sem alerta; 1ª linha adequada."),
     ("Tiazida + digoxina (hipocaliemia)", _p("H34",76,"F","heart_failure",active=["digoxin"]), _rx("hydrochlorothiazide"),
    "Alerta moderado (hipocaliemia -> toxicidade digitálica)."),
    ("Tramadol + tricíclico (serotoninérgico)", _p("H35",60,"M","pain",active=["amitriptyline"]), _rx("tramadol"),
    "Alerta moderado serotoninérgico/convulsivo."),
    ("Duplicação de anticoagulantes", _p("H36",72,"M","atrial_fibrillation",active=["warfarin"]), _rx("apixaban"),
    "Alerta alto (duplicação de anticoagulação)."),
    ("Duplo bloqueio SRAA (IECA + ARA)", _p("H37",68,"M","hypertension",active=["enalapril"]), _rx("losartan"),
    "Alerta alto (IECA+ARA: hipercaliemia/lesão renal, desaconselhado)."),
    ("Benzodiazepina + opióide (depressão SNC)", _p("H38",70,"M","anxiety",active=["tramadol"]), _rx("mexazolam"), "Alerta alto (depressão SNC/respiratória).")
]


def post_json(path, payload, timeout=60):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE_URL + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_json(path, timeout=30):
    with urllib.request.urlopen(BASE_URL + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def add_llm(entry):
    if RUN_LLM and entry.get("analysis_id"):
        try:
            llm = post_json("/explain/llm", {"analysis_id": entry["analysis_id"]}, timeout=LLM_TIMEOUT)
            entry["llm_model"] = llm.get("model")
            entry["llm_explanation"] = llm.get("explanation", "")
        except Exception as e:
            entry["llm_explanation"] = f"(LLM indisponível: {e})"


def run():
    results = []

    # ── A) hand-crafted ──
    for i, (nome, patient, rx, exp) in enumerate(HANDCRAFTED, 1):
        print(f"[A {i}/{len(HANDCRAFTED)}] {nome}", flush=True)
        e = {"grupo": "hand-crafted", "nome": nome, "expectativa": exp,
             "patient": patient, "prescription": rx}
        try:
            resp = post_json("/analyze", {"patient": patient, "prescription": rx})
            e.update({k: resp.get(k) for k in
                      ("analysis_id", "alerts", "recommendations", "recommendation_notes", "explanation")})
            add_llm(e)
        except Exception as ex:
            e["error"] = str(ex)
        results.append(e)

    # ── B) Synthea ──
    try:
        patients = get_json(f"/synthea/patients?limit={N_SYNTHEA}&adults_only=true&with_active_medications=true")
        for i, sp in enumerate(patients[:N_SYNTHEA], 1):
            nome = f"Synthea {sp['patient_id'][:8]} ({sp['age']}a {sp['sex']}, {len(sp.get('active_medications',[]))} ativos)"
            print(f"[B {i}/{len(patients[:N_SYNTHEA])}] {nome}", flush=True)
            e = {"grupo": "synthea", "nome": nome,
                 "expectativa": "Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.",
                 "patient": sp, "prescription": _rx("ibuprofen")}
            try:
                resp = post_json("/analyze/synthea",
                                 {"patient_id": sp["patient_id"], "main_problem": "pain",
                                  "prescription": _rx("ibuprofen")})
                e.update({k: resp.get(k) for k in
                          ("analysis_id", "alerts", "recommendations", "recommendation_notes", "explanation")})
                add_llm(e)
            except Exception as ex:
                e["error"] = str(ex)
            results.append(e)
    except Exception as ex:
        print(f"  (Synthea indisponível: {ex})")

    # ── Relatório ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT_DIR / f"frontend_test_report_{ts}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    com_alerta = sum(1 for e in results if e.get("alerts"))
    sev = {}
    for e in results:
        for a in e.get("alerts") or []:
            sev[a.get("severity")] = sev.get(a.get("severity"), 0) + 1

    L = [f"# Relatório de testes — {ts}", "",
         f"**Total de casos:** {len(results)} | **com alerta:** {com_alerta} | "
         f"**alertas por severidade:** {sev}", ""]
    for e in results:
        L.append(f"## [{e['grupo']}] {e['nome']}")
        if "error" in e:
            L.append(f"**ERRO:** {e['error']}\n"); continue
        p = e["patient"]
        L.append(f"- **Doente:** {p.get('age')}a {p.get('sex')}, problema="
                 f"{p.get('main_problem') or p.get('main_problem_guess')}, renal={p.get('renal_status')}, "
                 f"ativos={p.get('active_medications') or '—'}, condições={p.get('conditions') or '—'}, "
                 f"alergias={p.get('allergies') or '—'}")
        L.append(f"- **Prescrição:** {', '.join(m['medication'] for m in e['prescription'])}")
        L.append(f"- **Expectativa:** {e['expectativa']}")
        if e.get("alerts"):
            L.append("- **Alertas:**")
            for a in e["alerts"]:
                L.append(f"    - [{a.get('severity')}] {a.get('description')} (regra: {a.get('rule_id') or '—'})")
        else:
            L.append("- **Alertas:** nenhum")
        if e.get("recommendations"):
            L.append("- **Recomendações (med — heurístico/combinado/final):**")
            for r in e["recommendations"]:
                L.append(f"    - {r['medication']}: {r['score_heuristic']:.3f}/{r['score_combined']:.3f}/{r['score_final']:.3f}")
        else:
            L.append("- **Recomendações:** nenhuma")
        if e.get("recommendation_notes"):
            L.append("- **Notas:**")
            for n in e["recommendation_notes"]:
                L.append(f"    - {n.get('description')}")
        if e.get("llm_explanation"):
            L.append(f"- **LLM ({e.get('llm_model','?')}):** {e['llm_explanation'].strip().replace(chr(10),' ')}")
        L.append("")

    md = OUT_DIR / f"frontend_test_report_{ts}.md"
    md.write_text("\n".join(L), encoding="utf-8")
    print(f"\n✓ {len(results)} casos | {com_alerta} com alerta | severidades {sev}")
    print(f"Relatórios:\n  {md}\n  {OUT_DIR / f'frontend_test_report_{ts}.json'}")


if __name__ == "__main__":
    run()
