"""Métricas automáticas de relatórios de explicações.
Uso: python scripts\\llm_report_metrics.py [caminho1.json caminho2.json ...]
Sem argumentos, procura todos os frontend_test_report_*.json."""
import sys, json, re, glob, os
SEC = ["problema identificado", "motivo do alerta", "motivo da recomendação", "limitações"]
PT  = [" que ", " para ", " não ", " foi ", " uma ", " com ", " são "]

def checks(t):
    low = t.lower()
    s = all(x in low for x in SEC)
    p = (sum(w in low for w in PT) >= 3) and not re.search(r"[\u0370-\uffff]", t)
    l = 200 <= len(t) <= 4000
    return s, p, l

paths = sys.argv[1:] or sorted(glob.glob("**/frontend_test_report_*.json", recursive=True), key=os.path.getmtime)
if not paths:
    print("Nenhum relatório encontrado. Indica o caminho: python scripts\\llm_report_metrics.py <ficheiro.json>")
    sys.exit(1)

for path in paths:
    d = json.load(open(path, encoding="utf-8")); n = len(d); S = P = L = 0
    print(f"\n===== {os.path.basename(path)} =====")
    print(f"{'Cenário':46} sec PT comp")
    for e in d:
        s, p, l = checks(e.get("llm_explanation") or ""); S += s; P += p; L += l
        print(f"{e['nome'][:46]:46} {'ok' if s else 'X'}  {'ok' if p else 'X'}  {'ok' if l else 'X'}")
    print(f"TOTAIS: secções {100*S/n:.0f}% | PT-PT {100*P/n:.0f}% | comprimento {100*L/n:.0f}%")