import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import httpx


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")

KNOWLEDGE_BASE_PATH = Path(
    os.getenv(
        "KNOWLEDGE_BASE_PATH",
        Path(__file__).resolve().parents[1] / "data" / "knowledge_base.json",
    )
)

THERAPEUTIC_CLASS_LABELS = {
    "aine": "anti-inflamatório não esteroide (AINE)",
    "analgesico_antipiretico": "analgésico e antipirético",
    "antiagregante": "antiagregante plaquetário",
    "anticoagulante": "anticoagulante",
    "ieca": "inibidor da enzima de conversão da angiotensina (IECA)",
    "ara": "antagonista dos recetores da angiotensina II (ARA)",
    "diuretico_ansa": "diurético de ansa",
    "diuretico_tiazidico": "diurético tiazídico",
    "estatina": "estatina",
    "macrolido": "macrólido",
    "isrs": "inibidor seletivo da recaptação da serotonina (ISRS)",
    "antidepressivo_triciclico": "antidepressivo tricíclico",
    "digitalico": "digitálico",
    "antiarritmico_classe_iii": "antiarrítmico de classe III",
}

MAIN_PROBLEM_LABELS = {
    "pain": "dor",
    "fever": "febre",
    "inflammation": "inflamação",
    "infection": "infeção",
    "hypertension": "hipertensão",
    "diabetes": "diabetes",
    "heart_failure": "insuficiência cardíaca",
    "arrhythmia": "arritmia",
    "unspecified": "problema clínico não especificado",
}

RENAL_STATUS_LABELS = {
    "normal": "estado renal normal",
    "mild_impairment": "compromisso renal ligeiro/moderado",
    "severe_impairment": "compromisso renal grave",
}

SEVERITY_RANK = {
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

ORIGIN_LABELS = {
    "prescription_related": "Relacionado com a prescrição submetida",
    "active_medication_existing": "Pré-existente na medicação ativa",
    "combined_profile_risk": "Risco combinado do perfil clínico",
    "unknown": "Origem não classificada",
}

RULE_CATEGORY_LABELS = {
    "aine_antiagregante_hemorragia": "Interação AINE + antiagregante plaquetário",
    "aine_anticoagulante_hemorragia": "Interação AINE + anticoagulante",
    "aine_aine_duplicacao": "Duplicação terapêutica de AINEs",
    "aine_ieca_risco_renal": "Interação AINE + IECA",
    "aine_ara_risco_renal": "Interação AINE + ARA",
    "aine_diuretico_risco_renal": "Interação AINE + diurético",
    "aine_tiazida_risco_renal": "Interação AINE + diurético tiazídico",
    "sinvastatina_claritromicina_contraindicada": "Interação crítica sinvastatina + claritromicina",
    "estatina_macrolido_miopatia": "Interação estatina + macrólido",
    "varfarina_amiodarona": "Interação varfarina + amiodarona",
    "varfarina_macrolido": "Interação varfarina + macrólido",
    "sertralina_anticoagulante": "Interação ISRS + anticoagulante",
    "sertralina_antiagregante": "Interação ISRS + antiagregante",
    "sertralina_aine": "Interação ISRS + AINE",
    "serotoninergicos": "Associação de antidepressivos serotoninérgicos",
    "qt_risk_combination": "Associação de fármacos com risco de prolongamento QT",
    "amiodarona_digoxina": "Interação amiodarona + digoxina",
    "diuretico_digoxina_hipocaliemia": "Interação diurético de ansa + digoxina",
    "tiazida_digoxina_hipocaliemia": "Interação diurético tiazídico + digoxina",
    "triple_whammy": "Associação AINE + IECA/ARA + diurético",
}

EXPLANATION_STYLE_EXAMPLES = {
    "with_recommendation": """
Exemplo de estilo com recomendação:
1. Problema identificado
O utente apresenta [problema clínico]. Foi submetida prescrição de [medicamento prescrito], avaliada no contexto da medicação ativa.

2. Motivo do alerta
O sistema identificou [tipo de alerta] entre [medicamentos envolvidos]. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.

3. Motivo da recomendação
O sistema identificou [alternativa] como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.

4. Limitações
A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.
""".strip(),

    "with_symptomatic_recommendation": """
Exemplo de estilo com alternativa sintomática:
1. Problema identificado
O utente apresenta [inflamação/dor associada]. Foi submetida prescrição de [AINE], avaliada no contexto da medicação ativa.

2. Motivo do alerta
O sistema identificou [alerta relevante], associado à prescrição submetida e/ou à medicação ativa.

3. Motivo da recomendação
O sistema identificou [alternativa] como alternativa sintomática/analgésica. Esta alternativa não substitui o efeito anti-inflamatório do AINE, mas foi considerada admissível na base de conhecimento atual.

4. Limitações
A análise não avalia todas as opções clínicas possíveis e depende da base de conhecimento atual do protótipo.
""".strip(),

    "no_recommendation": """
Exemplo de estilo sem recomendação:
1. Problema identificado
O utente apresenta [problema clínico]. Foi submetida prescrição de [medicamento], tendo sido identificados alertas relevantes.

2. Motivo do alerta
O sistema identificou [alertas principais], com base nas regras implementadas e no contexto clínico consolidado.

3. Motivo da recomendação
O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.

4. Limitações
A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.
""".strip(),

    "no_recommendation_already_active": """
Exemplo de estilo sem nova recomendação porque alternativa já está ativa:
1. Problema identificado
O utente apresenta [problema clínico] e foi submetida prescrição de [medicamento], que originou alertas relevantes.

2. Motivo do alerta
O sistema identificou [alertas principais], relacionados com a prescrição submetida e/ou com a medicação ativa.

3. Motivo da recomendação
O protótipo não identificou uma nova alternativa admissível dentro da sua base de conhecimento atual. Uma alternativa candidata já consta da medicação ativa do utente, pelo que não foi apresentada como nova recomendação.

4. Limitações
A análise não avalia ajuste de dose, suspensão, manutenção terapêutica ou outras opções clínicas fora da base atual.
""".strip(),
}

def determine_explanation_scenario(
    patient_context: Optional[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    already_active_candidate_notes: list[dict[str, Any]],
) -> str:
    if recommendations:
        main_problem = None

        if patient_context:
            main_problem = patient_context.get("main_problem")

        for recommendation in recommendations:
            medication = recommendation.get("medication")
            clinical_reasons = recommendation.get("clinical_reasons", [])

            if (
                medication == "paracetamol"
                and main_problem == "inflammation"
                and any("sintomática" in reason.lower() for reason in clinical_reasons)
            ):
                return "with_symptomatic_recommendation"

        return "with_recommendation"

    if already_active_candidate_notes:
        return "no_recommendation_already_active"

    return "no_recommendation"

def get_highest_severity(current: str | None, candidate: str | None) -> str:
    if not current:
        return candidate or "low"

    if not candidate:
        return current

    return (
        candidate
        if SEVERITY_RANK.get(candidate, 0) > SEVERITY_RANK.get(current, 0)
        else current
    )


def get_alert_category(alert: dict[str, Any]) -> str:
    rule_id = alert.get("rule_id")

    if rule_id in RULE_CATEGORY_LABELS:
        return RULE_CATEGORY_LABELS[rule_id]

    if alert.get("type") == "renal_risk":
        return "Risco renal associado ao medicamento prescrito"

    if alert.get("type") == "contraindication":
        return "Contraindicação ou condição clínica relevante"

    if alert.get("type") == "unknown_medication":
        return "Medicamento não reconhecido"

    return alert.get("type") or "Alerta clínico"


def build_alert_group_key(alert: dict[str, Any]) -> tuple[str, str]:
    origin = alert.get("origin") or "unknown"
    rule_or_type = alert.get("rule_id") or alert.get("type") or alert.get("description") or "unknown"
    return origin, rule_or_type

@lru_cache(maxsize=1)
def load_knowledge_base_for_llm() -> dict[str, Any]:
    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"medications": {}}


def normalize_for_lookup(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip().lower()


def resolve_medication_id(raw_value: Any, kb: dict[str, Any]) -> Optional[str]:
    if raw_value is None:
        return None

    medications = kb.get("medications", {})
    raw_text = normalize_for_lookup(raw_value)

    if raw_text in medications:
        return raw_text

    for med_id, med_data in medications.items():
        candidates = {
            normalize_for_lookup(med_id),
            normalize_for_lookup(med_data.get("display_name")),
            normalize_for_lookup(med_data.get("active_substance")),
        }

        if raw_text in candidates:
            return med_id

    return None

def get_display_name_for_llm(raw_value: Any, kb: dict[str, Any]) -> str:
    med_id = resolve_medication_id(raw_value, kb)

    if med_id:
        return kb.get("medications", {}).get(med_id, {}).get("display_name", med_id)

    if raw_value is None:
        return ""

    return str(raw_value)

def build_medication_fact(
    med_id: str,
    kb: dict[str, Any],
) -> Optional[dict[str, Any]]:
    med_data = kb.get("medications", {}).get(med_id)

    if not med_data:
        return None

    therapeutic_class = med_data.get("therapeutic_class")

    return {
        "id": med_id,
        "display_name": med_data.get("display_name", med_id),
        "active_substance": med_data.get("active_substance"),
        "therapeutic_class": therapeutic_class,
        "therapeutic_class_label": THERAPEUTIC_CLASS_LABELS.get(
            therapeutic_class,
            therapeutic_class,
        ),
        "indications": med_data.get("indications", []),
        "renal_caution": bool(med_data.get("renal_caution", False)),
        "qt_risk": bool(med_data.get("qt_risk", False)),
    }


def collect_relevant_medication_ids(
    patient_context: Optional[dict[str, Any]],
    prescription: Optional[list[dict[str, Any]]],
    alerts: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    kb: dict[str, Any],
) -> list[str]:
    med_ids: list[str] = []

    def add_med(raw_value: Any) -> None:
        med_id = resolve_medication_id(raw_value, kb)

        if med_id and med_id not in med_ids:
            med_ids.append(med_id)

    if patient_context:
        for medication in patient_context.get("active_medications", []):
            add_med(medication)

    for line in prescription or []:
        add_med(line.get("medication"))

    for alert in alerts:
        for med_id in alert.get("medication_ids") or []:
            add_med(med_id)

    for recommendation in recommendations:
        add_med(recommendation.get("medication"))

    return med_ids


def build_controlled_clinical_dictionary(
    patient_context: Optional[dict[str, Any]],
    prescription: Optional[list[dict[str, Any]]],
    alerts: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> dict[str, Any]:
    kb = load_knowledge_base_for_llm()

    medication_ids = collect_relevant_medication_ids(
        patient_context=patient_context,
        prescription=prescription,
        alerts=alerts,
        recommendations=recommendations,
        kb=kb,
    )

    medication_facts = []

    for med_id in medication_ids:
        fact = build_medication_fact(med_id, kb)

        if fact:
            medication_facts.append(fact)

    main_problem = None
    renal_status = None

    if patient_context:
        main_problem = patient_context.get("main_problem")
        renal_status = patient_context.get("renal_status")

    return {
        "main_problem": {
            "code": main_problem,
            "label": MAIN_PROBLEM_LABELS.get(main_problem, main_problem),
        },
        "renal_status": {
            "code": renal_status,
            "label": RENAL_STATUS_LABELS.get(renal_status, renal_status),
        },
        "medications": medication_facts,
        "therapeutic_class_labels": THERAPEUTIC_CLASS_LABELS,
        "instruction": (
            "Usa este dicionário como fonte controlada para classes terapêuticas, "
            "indicações e nomes dos medicamentos. Não inventes classes ou indicações."
        ),
    }

def build_already_active_candidate_notes(
    patient_context: Optional[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Identifica medicamentos já presentes na medicação ativa que podem explicar
    a ausência de nova recomendação.

    Isto não afirma que o medicamento é clinicamente suficiente.
    Apenas melhora a transparência: uma alternativa candidata pode não ser
    proposta porque já consta da medicação ativa do utente.
    """
    if not patient_context:
        return []

    # Se já existem recomendações, não precisamos desta nota.
    if recommendations:
        return []

    kb = load_knowledge_base_for_llm()

    main_problem = patient_context.get("main_problem")
    active_medications = patient_context.get("active_medications", [])

    notes = []

    for raw_medication in active_medications:
        med_id = resolve_medication_id(raw_medication, kb)

        if not med_id:
            continue

        fact = build_medication_fact(med_id, kb)

        if not fact:
            continue

        indications = fact.get("indications", [])

        if main_problem in indications:
            notes.append({
                "medication": fact.get("display_name", med_id),
                "medication_id": med_id,
                "reason": (
                    "Este medicamento já consta da medicação ativa do utente "
                    "e tem indicação compatível com o problema clínico principal "
                    "na base de conhecimento atual."
                ),
            })

        elif main_problem == "inflammation" and med_id == "paracetamol":
            notes.append({
                "medication": fact.get("display_name", med_id),
                "medication_id": med_id,
                "reason": (
                    "Paracetamol já consta da medicação ativa do utente. "
                    "Pode ser considerado apenas como alternativa sintomática/analgésica, "
                    "não como substituto anti-inflamatório equivalente."
                ),
            })

    return notes

def sanitize_alerts_for_llm(alerts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Agrupa alertas antes de os enviar ao LLM.

    Objetivos:
    - separar alertas causados pela prescrição submetida de alertas pré-existentes;
    - reduzir repetições;
    - traduzir regras técnicas em categorias clínicas mais claras;
    - evitar que o LLM repita pares redundantes ou interprete tudo como erro da nova prescrição.
    """
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for alert in alerts:
        origin = alert.get("origin") or "unknown"
        key = build_alert_group_key(alert)

        medication = alert.get("medication")
        medication_ids = alert.get("medication_ids") or []

        if key not in grouped:
            grouped[key] = {
                "origin": origin,
                "origin_label": ORIGIN_LABELS.get(origin, "Origem não classificada"),
                "category": get_alert_category(alert),
                "type": alert.get("type"),
                "rule_id": alert.get("rule_id"),
                "severity": alert.get("severity"),
                "medications": [],
                "medication_ids": [],
                "description": alert.get("description"),
                "involves_prescribed_medication": bool(
                    alert.get("involves_prescribed_medication")
                ),
                "involves_active_medication": bool(
                    alert.get("involves_active_medication")
                ),
            }

        group = grouped[key]

        group["severity"] = get_highest_severity(
            group.get("severity"),
            alert.get("severity"),
        )

        if medication and medication not in group["medications"]:
            group["medications"].append(medication)

        for med_id in medication_ids:
            if med_id not in group["medication_ids"]:
                group["medication_ids"].append(med_id)

        group["involves_prescribed_medication"] = (
            group["involves_prescribed_medication"]
            or bool(alert.get("involves_prescribed_medication"))
        )

        group["involves_active_medication"] = (
            group["involves_active_medication"]
            or bool(alert.get("involves_active_medication"))
        )

    origin_order = {
        "prescription_related": 1,
        "combined_profile_risk": 2,
        "active_medication_existing": 3,
        "unknown": 4,
    }

    return sorted(
        grouped.values(),
        key=lambda item: (
            origin_order.get(item.get("origin"), 99),
            -SEVERITY_RANK.get(item.get("severity"), 0),
            item.get("category") or "",
        ),
    )

def get_recommendation_status(recommendation: dict[str, Any]) -> str:
    score = recommendation.get("score_final")

    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = None

    reasons = recommendation.get("reasons", [])

    has_precaution_reason = any(
        "precaução" in str(reason).lower()
        or "compromisso renal grave" in str(reason).lower()
        for reason in reasons
    )

    if score_value is not None and score_value < 0.5:
        return "alternativa_com_precaucao"

    if has_precaution_reason:
        return "alternativa_com_precaucao"

    return "alternativa_sugerida"


def get_recommendation_status_label(status: str) -> str:
    labels = {
        "alternativa_sugerida": "Alternativa sugerida",
        "alternativa_com_precaucao": "Alternativa com precaução",
    }

    return labels.get(status, "Alternativa sugerida")

def sanitize_recommendations_for_llm(
    recommendations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    kb = load_knowledge_base_for_llm()
    sanitized = []

    for recommendation in recommendations:
        reasons = recommendation.get("reasons", [])

        clinical_reasons = [
            reason
            for reason in reasons
            if "score" not in str(reason).lower()
        ]

        status = get_recommendation_status(recommendation)

        sanitized.append({
            "medication": get_display_name_for_llm(
                recommendation.get("medication"),
                kb,
            ),
            "medication_id": resolve_medication_id(
                recommendation.get("medication"),
                kb,
            ),
            "score_final": recommendation.get("score_final"),
            "recommendation_status": status,
            "recommendation_status_label": get_recommendation_status_label(status),
            "clinical_reasons": clinical_reasons,
        })

    return sanitized

def sanitize_patient_context_for_llm(
    patient_context: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not patient_context:
        return None

    kb = load_knowledge_base_for_llm()

    return {
        "patient_id": patient_context.get("patient_id"),
        "age": patient_context.get("age"),
        "sex": patient_context.get("sex"),
        "renal_status": patient_context.get("renal_status"),
        "main_problem": patient_context.get("main_problem"),
        "conditions": patient_context.get("conditions", []),
        "allergies": patient_context.get("allergies", []),
        "active_medications": [
            get_display_name_for_llm(medication, kb)
            for medication in patient_context.get("active_medications", [])
        ],
    }

def build_explanation_prompt(
    analysis: dict[str, Any],
    user_question: Optional[str] = None,
) -> str:
    request_data = analysis["request_json"]
    alerts = analysis["alerts_json"]
    recommendations = analysis["recommendations_json"]
    explanation = analysis["explanation"]

    patient_context = request_data.get("patient_context")
    prescription = request_data.get("prescription")
    original_request = request_data.get("original_request", request_data)

    sanitized_alerts = sanitize_alerts_for_llm(alerts)
    sanitized_recommendations = sanitize_recommendations_for_llm(recommendations)
    has_recommendations = bool(sanitized_recommendations)
    stored_recommendation_notes = request_data.get("recommendation_notes")

    if stored_recommendation_notes is not None:
        already_active_candidate_notes = stored_recommendation_notes
    else:
        already_active_candidate_notes = build_already_active_candidate_notes(
            patient_context=patient_context,
            recommendations=sanitized_recommendations,
        )
    explanation_scenario = determine_explanation_scenario(
        patient_context=patient_context,
        recommendations=sanitized_recommendations,
        already_active_candidate_notes=already_active_candidate_notes,
    )

    payload = {
        "fonte_dos_dados": request_data.get("source", analysis.get("source")),
        "contexto_clinico_consolidado": sanitize_patient_context_for_llm(patient_context),
        "prescricao_analisada": prescription,
        "pedido_original": original_request,
        "dicionario_clinico_controlado": build_controlled_clinical_dictionary(
            patient_context=patient_context,
            prescription=prescription,
            alerts=alerts,
            recommendations=recommendations,
        ),
        "alertas_identificados": sanitized_alerts,
        "recomendacoes": sanitized_recommendations,
        "tem_recomendacoes": has_recommendations,
        "notas_alternativas_ja_ativas": already_active_candidate_notes,
        "explicacao_do_sistema": explanation,
        "pergunta_do_utilizador": user_question,
        "cenario_explicacao": explanation_scenario,
        "exemplo_de_estilo": EXPLANATION_STYLE_EXAMPLES[explanation_scenario],
    }

    return f"""
És uma camada de explicação textual de um protótipo académico de apoio à decisão em prescrição medicamentosa.
A tua função é explicar, em linguagem natural, os resultados já produzidos pelo sistema.
Não deves fazer inferência clínica nova, prescrever, corrigir o sistema ou acrescentar informação externa.

Regras de linguagem:
- Responde sempre em português de Portugal.
- Usa "utente" em vez de "paciente".
- Usa "medicamento" ou "fármaco"; evita "droga".
- Usa "estado renal" em vez de "status renal".
- Escreve "um AINE" e não "uma AINE".
- Não uses "sua"; usa "a sua" ou reformula.
- Não escrevas "a análise se baseia"; escreve "a análise baseia-se".
- Não uses português do Brasil, como "está sendo".

Regras clínicas:
- Usa prioritariamente "contexto_clinico_consolidado" para descrever o contexto do utente.
- Usa o campo "dicionario_clinico_controlado" como fonte controlada para classes terapêuticas, indicações e nomes dos medicamentos.
- Não inventes classes terapêuticas, indicações, diagnósticos, contraindicações ou mecanismos farmacológicos.
- Não acrescentes mecanismos farmacológicos que não estejam explicitamente no alerta.
- Não transformes risco de hemorragia em risco trombótico.
- Distingue alertas "Relacionado com a prescrição submetida" de alertas "Pré-existente na medicação ativa".
- Não apresentes alertas pré-existentes como se tivessem sido causados pela nova prescrição.
- O campo "alertas_identificados" já está agrupado por origem e categoria clínica; não recries pares redundantes.
- Quando houver duplicação terapêutica de AINEs, resume como utilização concomitante de dois AINEs.
- Quando houver AINE + IECA/ARA + diurético, descreve como risco combinado envolvendo AINE, IECA/ARA e diurético. Se o dicionário identificar o fármaco como ARA, não o chames IECA.
- "medicação ativa" refere-se apenas aos medicamentos em contexto_clinico_consolidado.active_medications.
- "prescrição submetida" refere-se apenas aos medicamentos em prescricao_analisada.
- Não digas que o medicamento prescrito é medicação ativa, salvo se ele também estiver explicitamente em active_medications.
- Quando uma interação envolve um medicamento ativo e um medicamento prescrito, escreve: "o medicamento prescrito foi avaliado no contexto da medicação ativa do utente".
- Não menciones "dicionário clínico controlado" na resposta final. Usa "base de conhecimento atual do protótipo".
- Se a classe terapêutica for "ara", escreve "antagonista dos recetores da angiotensina II (ARA)".
- Não escrevas "inibidor do recetor da angiotensina".
- Só menciones diurético se existir um medicamento de classe diuretico_ansa ou diuretico_tiazidico nos alertas ou no dicionário clínico controlado.
- Só menciones triple whammy se existir uma regra com rule_id = "triple_whammy".

Regras sobre recomendações:
- Se "tem_recomendacoes" for verdadeiro, explica apenas as recomendações presentes no campo "recomendacoes".
- Se "tem_recomendacoes" for falso e "notas_alternativas_ja_ativas" estiver vazio, escreve que o protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.
- Se "tem_recomendacoes" for falso e "notas_alternativas_ja_ativas" tiver elementos, explica que não foi identificada nova alternativa admissível e que uma alternativa candidata já consta da medicação ativa do utente.
- Se "tem_recomendacoes" for falso, não uses as expressões "esta alternativa", "mesmo alerta para esta alternativa", "alternativa recomendada" ou "opção sugerida".
- Nunca uses "seguro", "sem risco", "sem interação", "não apresenta interação" ou "segurança clínica confirmada".
- Quando justificares uma alternativa existente, podes escrever: "Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa."
- Se a recomendação for paracetamol e o problema principal for inflamação, descreve-o apenas como alternativa sintomática/analgésica, não como substituto anti-inflamatório equivalente.
- Usa "exemplo_de_estilo" apenas como molde linguístico e estrutural.
- Não copies medicamentos, diagnósticos ou alertas do exemplo se não estiverem nos dados da análise.
- Adapta o texto ao campo "cenario_explicacao".
- Se "cenario_explicacao" for "no_recommendation_already_active", menciona que uma alternativa candidata já consta da medicação ativa do utente.
- Se "notas_alternativas_ja_ativas" tiver elementos, podes usá-los para explicar a ausência de nova recomendação.
- Se a recomendação tiver "recommendation_status" igual a "alternativa_com_precaucao", descreve-a como "alternativa com precaução" e não como recomendação forte.
- Se a recomendação tiver score baixo, não uses linguagem demasiado confiante. Escreve que requer validação clínica.
- Se existirem alertas com origin = "active_medication_existing", apresenta-os como alertas pré-existentes e não como causados pela prescrição submetida.
- Só digas que "todos os alertas estão relacionados com a prescrição submetida" se todos os alertas tiverem origin = "prescription_related" ou "combined_profile_risk".

Limitações:
- Explica que a análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual.
- Não apresentes a explicação como decisão clínica final.
- Não menciones regras internas do prompt nem expliques aquilo que evitaste fazer.

Dados da análise:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Produz uma explicação curta, com no máximo 100 palavras por secção:

1. Problema identificado
2. Motivo do alerta
3. Motivo da recomendação
4. Limitações

Não ultrapasses 350 palavras no total.
Não uses Markdown excessivo. Não uses linguagem promocional.
""".strip()

def clean_llm_output(text: str) -> str:
    replacements = {
        # Termos gerais
        "paciente": "utente",
        "Paciente": "Utente",
        "droga": "medicamento",
        "Drogas": "Medicamentos",
        "drogas": "medicamentos",
        "detectou": "detetou",
        "Detectou": "Detetou",

        # Português do Brasil / formulações menos naturais em PT-PT
        "status renal": "estado renal",
        "Status renal": "Estado renal",
        "está sendo": "está a ser",
        "está sendo prescrito": "foi prescrito",
        "está recebendo": "está a receber",
        "em uma": "numa",
        "em um": "num",
        "a análise se baseia": "a análise baseia-se",
        "A análise se baseia": "A análise baseia-se",
        "afectar": "afetar",

        # Segurança excessivamente forte
        "alternativa terapêutica segura": "alternativa admissível",
        "alternativas terapêuticas seguras": "alternativas admissíveis",
        "opção terapêutica segura": "alternativa admissível",
        "opções terapêuticas seguras": "alternativas admissíveis",
        "alternativa segura": "alternativa admissível",
        "alternativas seguras": "alternativas admissíveis",
        "opção segura": "opção admissível",
        "opções seguras": "opções admissíveis",
        "Não foi possível identificar uma alternativa segura": (
            "O protótipo não identificou uma alternativa admissível"
        ),

        # Interações — evitar afirmações absolutas
        "não apresenta interação clínica significativa": (
            "na base de conhecimento atual do protótipo não foi identificado o mesmo alerta"
        ),
        "não apresenta interação": (
            "na base de conhecimento atual do protótipo não foi identificado o mesmo alerta"
        ),
        "sem interação": (
            "sem o mesmo alerta identificado na base de conhecimento atual do protótipo"
        ),
        "sem risco": "sem o mesmo alerta identificado",

        # Confusão entre medicação ativa e prescrição submetida
        "ambos ativos no utente": "avaliados no contexto clínico do utente",
        "ambos presentes na prescrição ativa e na nova prescrição": (
            "avaliados no contexto da medicação ativa e da prescrição submetida"
        ),
        "prescrição ativa": "medicação ativa",

        # Termos internos que não devem aparecer ao utilizador
        "dicionário clínico controlado": "base de conhecimento atual do protótipo",

        # Terminologia farmacológica
        "inibidor do recetor da angiotensina": (
            "antagonista dos recetores da angiotensina II"
        ),
    }

    cleaned = text

    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    # Substituições de nomes de medicamentos apenas quando aparecem como palavra isolada.
    # Evita transformar "ibuprofeno" em "ibuprofenoo" ou "naproxeno" em "naproxenoo".
    medication_name_patterns = {
        r"\bibuprofen\b": "ibuprofeno",
        r"\bIbuprofen\b": "Ibuprofeno",
        r"\bnaproxen\b": "naproxeno",
        r"\bNaproxen\b": "Naproxeno",
        r"\bsimvastatin\b": "sinvastatina",
        r"\bSimvastatin\b": "Sinvastatina",
        r"\bazithromycin\b": "azitromicina",
        r"\bAzithromycin\b": "Azitromicina",
        r"\bclarithromycin\b": "claritromicina",
        r"\bClarithromycin\b": "Claritromicina",
    }

    for pattern, replacement in medication_name_patterns.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    # Rede de segurança para erros já gerados pelo pós-processamento anterior.
    accidental_terms = {
        "ibuprofenoo": "ibuprofeno",
        "Ibuprofenoo": "Ibuprofeno",
        "naproxenoo": "naproxeno",
        "Naproxenoo": "Naproxeno",
        "simvastatino": "sinvastatina",
        "Simvastatino": "Sinvastatina",
    }

    for source, target in accidental_terms.items():
        cleaned = cleaned.replace(source, target)

    cleaned = re.sub(
        r"O risco de hemorragia não foi transformado em risco trombótico\.?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"Não foram avaliadas outras opções clínicas compatíveis",
        "Foram avaliadas apenas as alternativas existentes na base de conhecimento atual do protótipo",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa\.\s*O sistema não identificou uma alternativa admissível dentro da sua base de conhecimento atual\.",
        "O sistema não identificou uma alternativa admissível dentro da sua base de conhecimento atual.",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"interação entre AINE e diurético tiazídico entre ([^\.]+)",
        r"interação entre \1",
        cleaned,
        flags=re.IGNORECASE,
    )

    return cleaned.strip()

def contains_cjk_characters(text: str) -> bool:
    """
    Deteta caracteres chineses, japoneses ou coreanos, incluindo extensões CJK.
    Se aparecerem, a resposta deve ser rejeitada e substituída por retry/fallback.
    """
    cjk_ranges = [
        (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
        (0x4E00, 0x9FFF),    # CJK Unified Ideographs
        (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
        (0x3040, 0x30FF),    # Hiragana + Katakana
        (0xAC00, 0xD7AF),    # Hangul
        (0x20000, 0x2A6DF),  # CJK Extension B
        (0x2A700, 0x2B73F),  # CJK Extension C
        (0x2B740, 0x2B81F),  # CJK Extension D
        (0x2B820, 0x2CEAF),  # CJK Extension E/F
        (0x2CEB0, 0x2EBEF),  # CJK Extension F/G
        (0x30000, 0x3134F),  # CJK Extension G/H
    ]

    return any(
        start <= ord(char) <= end
        for char in text
        for start, end in cjk_ranges
    )

def contains_garbled_or_unwanted_language(text: str) -> bool:
    """
    Deteta artefactos linguísticos ou tokens estranhos que não devem aparecer
    numa explicação clínica apresentada ao utilizador.
    """
    forbidden_patterns = [
        r"afferentes",
        r"aferrentes",
        r"\bpatient\b",
        r"\bPaciente\b",
        r"\bestá sendo\b",
        r"\bestá recebendo\b",
        r"[A-Za-zÀ-ÿ]\.[A-Za-zÀ-ÿ]",  # exemplo: "clopidogrel.afferentes"
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in forbidden_patterns
    )

def contains_forbidden_clinical_phrases(text: str) -> bool:
    forbidden_patterns = [
        r"\bfoi validada clinicamente\b",
        r"\bfoi clinicamente validada\b",
        r"\bvalidada clinicamente devido\b",
        r"\bsegurança clínica confirmada\b",
        r"\bsem risco\b",
        r"\bsem interação\b",
        r"\bnão apresenta interação\b",
        r"\bnão apresenta risco\b",
        r"\bfoi validada\b",
        r"\bfoi validado\b",
        r"\bvalidada com base\b",
        r"\bvalidado com base\b",
        r"\balternativa foi validada\b",
        r"\brecomendação foi validada\b",
        r"\btodos os alertas estão relacionados com a prescrição submetida\b",
        r"\btodas relacionadas com a prescrição submetida\b",
        r"\btodos estão relacionados com a prescrição submetida\b",
    ]

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in forbidden_patterns
    )


def has_required_explanation_structure(text: str) -> bool:
    required_sections = [
        "problema identificado",
        "motivo do alerta",
        "motivo da recomendação",
        "limitações",
    ]

    normalized = text.lower()

    found = sum(1 for section in required_sections if section in normalized)

    return found >= 3


def is_valid_llm_explanation(text: str) -> bool:
    if not text or len(text.strip()) < 80:
        return False

    if contains_cjk_characters(text):
        return False

    if contains_garbled_or_unwanted_language(text):
        return False

    if contains_forbidden_clinical_phrases(text):
        return False

    if not has_required_explanation_structure(text):
        return False

    return True


def build_retry_prompt(original_prompt: str) -> str:
    return f"""
{original_prompt}

INSTRUÇÃO FINAL OBRIGATÓRIA:
- Responde exclusivamente em português de Portugal.
- Não uses chinês, inglês, espanhol ou português do Brasil.
- Nunca escrevas que uma alternativa foi validada, foi validada clinicamente ou foi validada com base nos dados. O sistema apenas sugere ou sinaliza alternativas admissíveis/com precaução.
- Se uma alternativa exigir validação, escreve: "requer validação clínica".
- Mantém exatamente as quatro secções pedidas.
""".strip()

def format_medication_list(items: list[str]) -> str:
    if not items:
        return ""

    if len(items) == 1:
        return items[0]

    return ", ".join(items[:-1]) + " e " + items[-1]


def build_deterministic_explanation(
    analysis: dict[str, Any],
) -> str:
    request_data = analysis["request_json"]
    alerts = analysis["alerts_json"]
    recommendations = analysis["recommendations_json"]

    patient_context = request_data.get("patient_context") or {}
    prescription = request_data.get("prescription") or []

    sanitized_alerts = sanitize_alerts_for_llm(alerts)
    sanitized_recommendations = sanitize_recommendations_for_llm(recommendations)

    main_problem = patient_context.get("main_problem")
    main_problem_label = MAIN_PROBLEM_LABELS.get(main_problem, main_problem or "problema clínico não especificado")

    prescribed_names = []
    kb = load_knowledge_base_for_llm()

    for line in prescription:
        prescribed_names.append(
            get_display_name_for_llm(line.get("medication"), kb)
        )

    prescribed_text = format_medication_list(prescribed_names) or "a prescrição submetida"

    prescription_related = [
        alert for alert in sanitized_alerts
        if alert.get("origin") == "prescription_related"
    ]

    existing_alerts = [
        alert for alert in sanitized_alerts
        if alert.get("origin") == "active_medication_existing"
    ]

    alert_parts = []

    if prescription_related:
        cats = []
        for alert in prescription_related:
            label = alert.get("category") or "alerta clínico"
            meds = format_medication_list(alert.get("medications") or [])
            if meds:
                cats.append(f"{label} ({meds})")
            else:
                cats.append(label)

        alert_parts.append(
            "Foram identificados alertas relacionados com a prescrição submetida: "
            + "; ".join(cats)
            + "."
        )

    if existing_alerts:
        cats = []
        for alert in existing_alerts:
            label = alert.get("category") or "alerta clínico"
            meds = format_medication_list(alert.get("medications") or [])
            if meds:
                cats.append(f"{label} ({meds})")
            else:
                cats.append(label)

        alert_parts.append(
            "Foram também identificados alertas pré-existentes na medicação ativa: "
            + "; ".join(cats)
            + "."
        )

    if not alert_parts:
        alert_parts.append("Não foram identificados alertas clínicos relevantes no âmbito atual do protótipo.")

    if sanitized_recommendations:
        rec_texts = []

        for rec in sanitized_recommendations:
            medication = rec.get("medication")
            status = rec.get("recommendation_status_label", "Alternativa sugerida")

            if rec.get("recommendation_status") == "alternativa_com_precaucao":
                rec_texts.append(
                    f"{medication} foi identificado como alternativa com precaução e requer validação clínica."
                )
            else:
                rec_texts.append(
                    f"{medication} foi identificado como alternativa sugerida na base de conhecimento atual."
                )

        recommendation_text = " ".join(rec_texts)

    else:
        stored_notes = request_data.get("recommendation_notes") or []

        if stored_notes:
            note_texts = []
            for note in stored_notes:
                medication = note.get("medication")
                description = note.get("description") or note.get("reason")
                if medication and description:
                    note_texts.append(f"{medication}: {description}")
                elif description:
                    note_texts.append(description)

            recommendation_text = (
                "O protótipo não identificou uma nova alternativa admissível dentro da sua base de conhecimento atual. "
                + " ".join(note_texts)
            )
        else:
            recommendation_text = (
                "O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual."
            )

    return f"""
1. Problema identificado
O utente apresenta {main_problem_label}. Foi submetida prescrição de {prescribed_text}, avaliada no contexto da medicação ativa.

2. Motivo do alerta
{" ".join(alert_parts)}

3. Motivo da recomendação
{recommendation_text}

4. Limitações
A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual do protótipo. Não substitui validação clínica.
""".strip()

def generate_llm_explanation(
    analysis: dict[str, Any],
    user_question: Optional[str] = None,
) -> dict[str, str]:
    prompt = build_explanation_prompt(
        analysis=analysis,
        user_question=user_question,
    )

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"

    def call_ollama(current_prompt: str) -> str:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": current_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 650,
                "num_ctx": 4096,
            },
        }

        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()

        data = response.json()
        return clean_llm_output(data.get("response", "").strip())

    try:
        text = call_ollama(prompt)

        if not is_valid_llm_explanation(text):
            retry_prompt = build_retry_prompt(prompt)
            text = call_ollama(retry_prompt)

        if not is_valid_llm_explanation(text):
            fallback = build_deterministic_explanation(analysis)
            return {
                "model": f"{OLLAMA_MODEL} + fallback determinístico",
                "explanation": fallback,
            }

    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Não foi possível ligar ao Ollama. Confirma se o Ollama está instalado e ativo."
        ) from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"Ollama devolveu erro HTTP: {exc.response.status_code}."
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            "O modelo demorou demasiado tempo a responder."
        ) from exc

    if not text:
        fallback = build_deterministic_explanation(analysis)
        return {
            "model": f"{OLLAMA_MODEL} + fallback determinístico",
            "explanation": fallback,
        }

    return {
        "model": OLLAMA_MODEL,
        "explanation": text,
    }