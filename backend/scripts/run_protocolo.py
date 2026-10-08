"""Executa os cenarios do protocolo de avaliacao clinica contra a API em
execucao e imprime tudo o que o medico vai ver na interface: alertas,
recomendacoes, notas e explicacao gerada pelo modelo de linguagem.

Requer o servidor a correr noutra janela:
    uvicorn app.main:app --host 127.0.0.1 --port 8000

Uso:
    python scripts/run_protocolo.py                  # tudo, sem LLM
    python scripts/run_protocolo.py --llm            # tudo, com explicacao LLM
    python scripts/run_protocolo.py --llm --so 3     # apenas o cenario 3
    python scripts/run_protocolo.py --llm --out relatorio.txt
"""

import argparse
import os
import sys

import httpx

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

ORDEM_SEVERIDADE = {"critical": 0, "high": 1, "moderate": 2, "low": 3}

ETIQUETA_SEVERIDADE = {
    "critical": "CRITICO",
    "high": "ELEVADO",
    "moderate": "MODERADO",
    "low": "BAIXO",
}

# Dois utentes polimedicados, quatro prescricoes. Em cada par, muda apenas o
# farmaco prescrito, mantendo-se o utente exatamente igual.
UTENTE_A = {
    "patient_id": "PROT-A",
    "age": 74,
    "sex": "M",
    "conditions": ["hipertensão", "diabetes tipo 2", "DPOC", "gota"],
    "allergies": ["penicilina"],
    "active_medications": [
        "enalapril", "hidroclorotiazida", "metformina",
        "omeprazol", "salbutamol", "alopurinol",
    ],
    "renal_status": "normal",
    "main_problem": "infeção",
}

UTENTE_B = {
    "patient_id": "PROT-B",
    "age": 78,
    "sex": "F",
    "conditions": ["hipertensão", "diabetes tipo 2", "osteoporose", "fibrilhação auricular"],
    "allergies": [],
    "active_medications": [
        "varfarina", "digoxina", "furosemida", "enalapril", "sinvastatina",
        "metformina", "omeprazol", "alendronato", "calcio + vitamina d",
    ],
    "renal_status": "normal",
    "main_problem": "dor",
}

UTENTE_C = {
    "patient_id": "PROT-C",
    "age": 68,
    "sex": "M",
    "conditions": ["hipertensão", "diabetes tipo 2", "dislipidemia", "refluxo gastroesofágico", "hiperuricemia"],
    "allergies": [],
    "active_medications": [
        "metoprolol", "enalapril", "sinvastatina", "metformina",
        "omeprazol", "alopurinol", "hidroclorotiazida",
    ],
    "renal_status": "normal",
    "main_problem": "hipertensão",
}

CENARIOS = [
    {
        "n": 1,
        "tipo": "Risco",
        "titulo": "Alergia ao farmaco prescrito, utente polimedicado",
        "utente": UTENTE_A,
        "medicamento": "amoxicilina",
        "dose": "500mg", "frequencia": "8/8h", "via": "oral",
        "esperado": "Alerta critico de alergia a penicilina, mais um moderado de "
                    "reacoes cutaneas com alopurinol. Recomendacao: Azitromicina, "
                    "seguida de Claritromicina.",
        "pergunta_llm": "Porque e que a azitromicina ficou em primeiro lugar?",
    },
    {
        "n": 2,
        "tipo": "Controlo",
        "titulo": "Mesmo utente, antibiotico sem reatividade cruzada",
        "utente": UTENTE_A,
        "medicamento": "azitromicina",
        "dose": "500mg", "frequencia": "1x/dia", "via": "oral",
        "esperado": "Nenhum alerta. O utente e o mesmo do cenario 1 e apenas o "
                    "antibiotico mudou.",
        "pergunta_llm": None,
    },
    {
        "n": 3,
        "tipo": "Risco",
        "titulo": "Anti-inflamatorio numa utente com nove farmacos ativos",
        "utente": UTENTE_B,
        "medicamento": "ibuprofeno",
        "dose": "400mg", "frequencia": "8/8h", "via": "oral",
        "esperado": "Nove alertas. Dois elevados decorrentes da nova prescricao "
                    "(AINE com anticoagulante; triple whammy), um elevado entre "
                    "farmacos ja ativos (calcio com digoxina) e seis moderados. "
                    "Recomendacao: Paracetamol, seguido de Tramadol.",
        "pergunta_llm": "Porque e que o paracetamol e mais seguro aqui do que o ibuprofeno?",
    },
    {
        "n": 4,
        "tipo": "Verificacao da recomendacao",
        "titulo": "Mesma utente, prescrita a alternativa recomendada",
        "utente": UTENTE_B,
        "medicamento": "paracetamol",
        "dose": "1000mg", "frequencia": "8/8h", "via": "oral",
        "esperado": "Tres alertas, TODOS entre farmacos ja ativos e nenhum "
                    "relacionado com a nova prescricao. Os seis alertas que o "
                    "ibuprofeno provocava desaparecem.",
        "pergunta_llm": None,
    },
    {
        "n": 5,
        "tipo": "Risco",
        "titulo": "Duplicacao terapeutica num hipertenso polimedicado",
        "utente": UTENTE_C,
        "medicamento": "atenolol",
        "dose": "50mg", "frequencia": "1x/dia", "via": "oral",
        "esperado": "Um alerta elevado de duplicacao de bloqueadores "
                    "beta-adrenergicos. Nenhuma alternativa proposta, porque a "
                    "conduta indicada e suspender um dos farmacos e nao "
                    "substituir. As notas ao profissional assinalam que o "
                    "enalapril e a hidroclorotiazida ja constam da medicacao "
                    "ativa com indicacao compativel.",
        "pergunta_llm": "Porque e que o sistema nao propoe aqui nenhuma alternativa?",
    },
]


def cabecalho(texto, char="="):
    return "\n" + char * 78 + "\n" + texto + "\n" + char * 78


def formatar_utente(u):
    linhas = [
        f"  Utente        : {u['age']} anos, sexo {u['sex']}, funcao renal {u['renal_status']}",
        f"  Problema      : {u['main_problem']}",
        f"  Condicoes     : {', '.join(u['conditions']) or '(nenhuma)'}",
        f"  Alergias      : {', '.join(u['allergies']) or '(nenhuma)'}",
        f"  Medicacao ativa ({len(u['active_medications'])} farmacos):",
    ]
    for med in u["active_medications"]:
        linhas.append(f"      - {med}")
    return "\n".join(linhas)


def formatar_alertas(alertas):
    if not alertas:
        return "  SEM ALERTAS."

    ordenados = sorted(
        alertas,
        key=lambda a: ORDEM_SEVERIDADE.get(a.get("severity"), 9),
    )

    novos = [a for a in ordenados if a.get("involves_prescribed_medication")]
    preexistentes = [a for a in ordenados if not a.get("involves_prescribed_medication")]

    linhas = [f"  {len(alertas)} alerta(s): {len(novos)} relacionado(s) com a nova "
              f"prescricao, {len(preexistentes)} entre farmacos ja ativos."]

    def bloco(titulo, lista):
        if not lista:
            return
        linhas.append("")
        linhas.append(f"  -- {titulo} --")
        for a in lista:
            sev = ETIQUETA_SEVERIDADE.get(a.get("severity"), a.get("severity"))
            linhas.append(f"  [{sev:8}] {a.get('medication', '')}")
            linhas.append(f"             {a.get('description', '')}")
            linhas.append(f"             (regra: {a.get('rule_id')}, tipo: {a.get('type')})")

    bloco("Decorrentes da prescricao em analise", novos)
    bloco("Ja presentes no regime do utente", preexistentes)

    return "\n".join(linhas)


def formatar_recomendacoes(recs, notas):
    if not recs:
        linhas = ["  Nenhuma alternativa proposta."]
    else:
        linhas = ["  Alternativas ordenadas:"]
        for i, r in enumerate(recs, start=1):
            nome = r.get("display_name") or r.get("medication")
            linhas.append(
                f"   {i}. {nome:20} score_final={r.get('score_final'):.3f} "
                f"({r.get('admissibility_class', '')})"
            )
            for motivo in (r.get("reasons") or [])[:3]:
                linhas.append(f"        . {motivo}")

    if notas:
        linhas.append("")
        linhas.append("  Notas ao profissional:")
        for nota in notas:
            if isinstance(nota, dict):
                texto = nota.get("description") or nota.get("type") or str(nota)
            else:
                texto = str(nota)
            linhas.append(f"   - {texto}")

    return "\n".join(linhas)


def executar(cenario, cliente, com_llm):
    saida = [cabecalho(
        f"CENARIO {cenario['n']} ({cenario['tipo']}): {cenario['titulo']}"
    )]
    saida.append(formatar_utente(cenario["utente"]))
    saida.append(f"  PRESCRICAO A TESTAR: {cenario['medicamento']} "
                 f"{cenario['dose']} {cenario['frequencia']} {cenario['via']}")
    saida.append("")
    saida.append(f"  Esperado: {cenario['esperado']}")

    payload = {
        "patient": cenario["utente"],
        "prescription": [{
            "medication": cenario["medicamento"],
            "dose": cenario["dose"],
            "frequency": cenario["frequencia"],
            "route": cenario["via"],
        }],
    }

    resposta = cliente.post(f"{API_URL}/analyze", json=payload, timeout=120)
    resposta.raise_for_status()
    dados = resposta.json()

    saida.append(cabecalho("ALERTAS", "-"))
    saida.append(formatar_alertas(dados.get("alerts") or []))

    saida.append(cabecalho("RECOMENDACOES", "-"))
    saida.append(formatar_recomendacoes(
        dados.get("recommendations") or [],
        dados.get("recommendation_notes") or [],
    ))

    if dados.get("explanation"):
        saida.append(cabecalho("EXPLICACAO DETERMINISTICA", "-"))
        saida.append("  " + str(dados["explanation"]))

    if com_llm:
        analysis_id = dados.get("analysis_id")
        saida.append(cabecalho("EXPLICACAO GERADA POR LLM", "-"))
        try:
            r = cliente.post(
                f"{API_URL}/explain/llm",
                json={"analysis_id": analysis_id},
                timeout=300,
            )
            r.raise_for_status()
            llm = r.json()
            saida.append(f"  Modelo: {llm.get('model')} "
                         f"(fallback: {llm.get('fallback_used')})")
            saida.append("  " + str(llm.get("explanation", "")))
            if llm.get("fallback_notice"):
                saida.append(f"  Aviso: {llm['fallback_notice']}")

            if cenario.get("pergunta_llm"):
                saida.append(cabecalho("PERGUNTA DE SEGUIMENTO", "-"))
                saida.append(f"  P: {cenario['pergunta_llm']}")
                rc = cliente.post(
                    f"{API_URL}/explain/llm/chat",
                    json={"analysis_id": analysis_id,
                          "question": cenario["pergunta_llm"]},
                    timeout=300,
                )
                rc.raise_for_status()
                chat = rc.json()
                saida.append(f"  R: {chat.get('answer') or chat.get('content') or chat}")
        except Exception as erro:
            saida.append(f"  [FALHOU] {type(erro).__name__}: {erro}")

    return "\n".join(saida)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true",
                        help="gerar tambem a explicacao do modelo de linguagem")
    parser.add_argument("--so", type=int, default=None,
                        help="executar apenas o cenario indicado")
    parser.add_argument("--out", default=None,
                        help="gravar o relatorio num ficheiro de texto")
    args = parser.parse_args()

    cenarios = CENARIOS
    if args.so is not None:
        cenarios = [c for c in CENARIOS if c["n"] == args.so]
        if not cenarios:
            print(f"Cenario {args.so} nao existe.")
            sys.exit(1)

    partes = [cabecalho("PROTOCOLO DE AVALIACAO CLINICA, SAIDA DO PROTOTIPO")]
    partes.append(f"API: {API_URL}")

    with httpx.Client() as cliente:
        try:
            cliente.get(f"{API_URL}/", timeout=10).raise_for_status()
        except Exception as erro:
            print(f"Nao foi possivel contactar a API em {API_URL}.")
            print("Arranca o servidor noutra janela com:")
            print("    uvicorn app.main:app --host 127.0.0.1 --port 8000")
            print(f"Erro: {type(erro).__name__}: {erro}")
            sys.exit(1)

        for cenario in cenarios:
            partes.append(executar(cenario, cliente, args.llm))

    relatorio = "\n".join(partes) + "\n"
    print(relatorio)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as ficheiro:
            ficheiro.write(relatorio)
        print(f"\nRelatorio gravado em {args.out}")


if __name__ == "__main__":
    main()
