"""Compara modelos LLM (Ollama) na tarefa de explicação, sobre os cenários hand-crafted.
Métricas automáticas por modelo: % com as 4 secções, % em PT-PT, % dentro do comprimento,
% sem tokens de raciocínio e latência média. Uso: python -m scripts.compare_llms"""
import re, json, time, urllib.request
from pathlib import Path
from statistics import mean
from scripts.run_frontend_tests import HANDCRAFTED

BACKEND = "http://127.0.0.1:8000"
OLLAMA  = "http://127.0.0.1:11434/api/generate"
MODELS  = ["qwen3:4b-instruct", "phi4-mini", "qwen2.5:3b", "llama3.2:3b", "phi3.5"]  # ajustar aos modelos

SECOES = ["problema identificado", "motivo do alerta", "motivo da recomendação", "limitações"]
PT_HINT = [" que ", " para ", " com ", " não ", " foi ", " uma ", " são "]

def post(url, payload, t=240):
    r = urllib.request.Request(url, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(r, timeout=t) as x:
        return json.loads(x.read().decode())

def build_prompt(analysis, pdict):
    alerts = "; ".join(f"[{a['severity']}] {a['description']}" for a in analysis.get("alerts", [])) or "nenhum"
    recs = ", ".join(r["medication"] for r in analysis.get("recommendations", [])) or "nenhuma"
    return ("És um assistente clínico. Explica em PORTUGUÊS DE PORTUGAL (pt-PT), de forma objetiva, "
            "em exatamente quatro secções numeradas: 1. Problema identificado; 2. Motivo do alerta; "
            "3. Motivo da recomendação; 4. Limitações. Não inventes fármacos nem interações.\n"
            f"Problema: {pdict.get('main_problem')}\nAlertas: {alerts}\nRecomendações: {recs}\n")

def validate(text):
    low = text.lower()
    sec = all(s in low for s in SECOES)
    pt = (sum(h in low for h in PT_HINT) >= 3) and not re.search(r"[\u4e00-\u9fff]", text)
    length = 200 <= len(text) <= 4000
    nothink = "<think>" not in low
    return sec, pt, length, nothink

def run():
    analyses = []
    for nome, pdict, rx, _exp in HANDCRAFTED:
        a = post(f"{BACKEND}/analyze", {"patient": pdict, "prescription": rx})
        analyses.append((nome, pdict, a))

    def _get(url, t=60):
        with urllib.request.urlopen(url, timeout=t) as x:
            return json.loads(x.read().decode())
    for sp in _get(f"{BACKEND}/synthea/patients?limit=15&adults_only=true&with_active_medications=true")[:15]:
        a = post(f"{BACKEND}/analyze/synthea",
                 {"patient_id": sp["patient_id"], "main_problem": "pain",
                  "prescription": [{"medication": "ibuprofen"}]})
        analyses.append((f"Synthea {sp['patient_id'][:8]}", {"main_problem": "pain"}, a))

    rows = []
    for m in MODELS:
        se = pt = ln = nt = 0; lat = []; n = len(analyses)
        for nome, pdict, a in analyses:
            prompt = build_prompt(a, pdict)
            t0 = time.time()
            try:
                out = post(OLLAMA, {"model": m, "prompt": prompt, "stream": False})["response"]
            except Exception as e:
                out = f"(erro: {e})"
            lat.append(time.time() - t0)
            s, p, l, k = validate(out); se += s; pt += p; ln += l; nt += k
        rows.append((m, 100*se/n, 100*pt/n, 100*ln/n, 100*nt/n, mean(lat)))

    rep = [f"# Comparação de modelos LLM — {time.strftime('%Y-%m-%d %H:%M')}", "",
           "| Modelo | 4 secções % | PT-PT % | Comprimento % | S/ <think> % | Latência média (s) |",
           "|---|--:|--:|--:|--:|--:|"]
    for m, a, b, c, e, la in rows:
        rep.append(f"| {m} | {a:.0f} | {b:.0f} | {c:.0f} | {e:.0f} | {la:.1f} |")
    Path("compare_llms_report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep))

if __name__ == "__main__":
    run()