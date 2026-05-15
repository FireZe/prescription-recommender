from typing import List, Dict, Any, Tuple

from app.schemas import PatientContext, MedicationLine, Alert, Recommendation
from app.normalization import normalize_medication_id, normalize_condition_id
from app.rules_engine import run_safety_checks
from app.ml_model import predict_candidate_adequacy

BLOCKING_SEVERITIES = {"critical", "high"}
RECOMMENDATION_TRIGGER_SEVERITIES = {"moderate", "high", "critical"}
NOTE_EXCLUSION_SEVERITIES = {"moderate", "high", "critical"}

PENALTY_BY_SEVERITY = {
    "moderate": 0.20,
    "low": 0.05,
}

SYMPTOMATIC_ANALGESIC_REASON = (
    "Alternativa sintomática/analgésica; não substitui o efeito anti-inflamatório do AINE."
)

RENAL_CAUTION_REASON = (
    "Requer precaução em compromisso renal grave; requer validação clínica."
)


def append_unique_reason(reasons: List[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)

def generate_candidates(
    prescription: List[MedicationLine],
    kb: Dict[str, Any],
) -> List[str]:
    candidates = set()

    prescribed_normalized = {
        normalize_medication_id(line.medication) or line.medication.lower()
        for line in prescription
    }

    for line in prescription:
        med_id = normalize_medication_id(line.medication) or line.medication.lower()
        med = kb["medications"].get(med_id)

        if not med:
            continue

        for alternative in med.get("alternatives", []):
            if alternative in kb["medications"] and alternative not in prescribed_normalized:
                candidates.add(alternative)

    return list(candidates)


def build_historical_key(patient: PatientContext) -> str:
    age_group = "elderly" if patient.age >= 65 else "adult"
    renal_group = "renal_impairment" if patient.renal_status != "normal" else "normal_renal"
    return f"{patient.main_problem}|{age_group}|{renal_group}"


def get_historical_score(
    medication: str,
    patient: PatientContext,
    historical_patterns: Dict[str, Any],
) -> float:
    key = build_historical_key(patient)
    return historical_patterns.get(key, {}).get(medication, 0.0)


def evaluate_candidate_safety(
    candidate: str,
    patient: PatientContext,
    kb: Dict[str, Any],
) -> List[Alert]:
    """
    Reapplies the clinical safety layer to a candidate medication.
    This ensures that recommendations remain subordinated to deterministic safety rules.
    """
    candidate_prescription = [
        MedicationLine(
            medication=candidate,
            dose=None,
            frequency=None,
            route=None,
        )
    ]

    return run_safety_checks(
        patient=patient,
        prescription=candidate_prescription,
        kb=kb,
    )


def has_blocking_alert(alerts: List[Alert]) -> bool:
    return any(alert.severity in BLOCKING_SEVERITIES for alert in alerts)


def penalty_from_candidate_alerts(alerts: List[Alert]) -> float:
    penalty = 0.0

    for alert in alerts:
        penalty += PENALTY_BY_SEVERITY.get(alert.severity, 0.0)

    return min(penalty, 0.5)

def get_medication_class(medication_id: str, kb: Dict[str, Any]) -> str | None:
    medication = kb.get("medications", {}).get(medication_id)

    if not medication:
        return None

    return medication.get("therapeutic_class")


def has_therapeutic_duplication(
    candidate: str,
    patient: PatientContext,
    prescription: List[MedicationLine],
    kb: Dict[str, Any],
) -> int:
    candidate_class = get_medication_class(candidate, kb)

    if not candidate_class:
        return 0

    active_and_prescribed = list(patient.active_medications)

    for line in prescription:
        active_and_prescribed.append(line.medication)

    for raw_medication in active_and_prescribed:
        medication_id = normalize_medication_id(raw_medication)

        if not medication_id:
            continue

        if medication_id == candidate:
            continue

        medication_class = get_medication_class(medication_id, kb)

        if medication_class == candidate_class:
            return 1

    return 0


def renal_status_score(renal_status: str) -> int:
    mapping = {
        "normal": 0,
        "mild_impairment": 1,
        "severe_impairment": 2,
    }
    return mapping.get(renal_status, 0)


def get_active_medication_ids(patient: PatientContext) -> List[str]:
    ids = []

    for raw_med in patient.active_medications:
        med_id = normalize_medication_id(raw_med)
        if med_id:
            ids.append(med_id)

    return ids


def has_active_class(patient: PatientContext, kb: Dict[str, Any], classes: set[str]) -> int:
    for med_id in get_active_medication_ids(patient):
        med_class = get_medication_class(med_id, kb)
        if med_class in classes:
            return 1

    return 0


def has_active_qt_risk(patient: PatientContext, kb: Dict[str, Any]) -> int:
    for med_id in get_active_medication_ids(patient):
        med = kb.get("medications", {}).get(med_id)
        if med and med.get("qt_risk"):
            return 1

    return 0


def build_ml_features(
    candidate: str,
    patient: PatientContext,
    prescription: List[MedicationLine],
    candidate_alerts: List[Alert],
    kb: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_obj = kb["medications"][candidate]
    candidate_class = candidate_obj.get("therapeutic_class", "unknown")

    original_medication = (
        normalize_medication_id(prescription[0].medication)
        or prescription[0].medication.lower()
    )

    original_obj = kb["medications"].get(original_medication, {})
    original_class = original_obj.get("therapeutic_class", "unknown")

    return {
        "age": int(patient.age),
        "age_squared": int(patient.age) ** 2,
        "is_elderly": int(patient.age >= 65),
        "active_medication_count": len(get_active_medication_ids(patient)),
        "condition_count": len(patient.conditions),
        "renal_status_score": renal_status_score(patient.renal_status),

        "candidate_renal_caution": int(bool(candidate_obj.get("renal_caution"))),
        "candidate_qt_risk": int(bool(candidate_obj.get("qt_risk"))),

        "has_anticoagulant": has_active_class(patient, kb, {"anticoagulante"}),
        "has_antiplatelet": has_active_class(patient, kb, {"antiagregante"}),
        "has_diuretic": has_active_class(patient, kb, {"diuretico_ansa", "diuretico_tiazidico"}),
        "has_acei_or_arb": has_active_class(patient, kb, {"ieca", "ara"}),
        "has_qt_risk_medication": has_active_qt_risk(patient, kb),

        "same_therapeutic_class": int(candidate_class == original_class),
        "candidate_is_nsaid": int(candidate_class == "aine"),

        "candidate": candidate,
        "candidate_class": candidate_class,
        "original_medication": original_medication,
        "original_class": original_class,
        "main_problem": patient.main_problem,
    }


def combine_heuristic_and_ml_score(
    heuristic_score: float,
    ml_score: float | None,
    ml_weight: float = 0.30,
) -> float:
    if ml_score is None:
        return heuristic_score

    final_score = ((1 - ml_weight) * heuristic_score) + (ml_weight * ml_score)
    return round(max(0.0, min(1.0, final_score)), 3)

def is_symptomatic_analgesic_fallback(
    candidate: str,
    patient: PatientContext,
    original_medication: str,
    kb: Dict[str, Any],
) -> bool:
    """
    Identifica o caso em que o sistema recomenda paracetamol perante um problema
    de inflamação após prescrição inicial de AINE.

    Neste cenário, o paracetamol pode ser uma alternativa sintomática para dor,
    mas não deve ser apresentado como substituto anti-inflamatório equivalente.
    """
    if candidate != "paracetamol":
        return False

    if patient.main_problem != "inflammation":
        return False

    original = kb.get("medications", {}).get(original_medication)

    if not original:
        return False

    return original.get("therapeutic_class") == "aine"

def score_candidate_base(
    candidate: str,
    patient: PatientContext,
    original_medication: str,
    kb: Dict[str, Any],
    feedback_score: float = 0.0,
) -> Tuple[float, List[str]]:
    med = kb["medications"][candidate]
    original = kb["medications"].get(original_medication)

    reasons = []

    has_severe_renal_caution = (
        patient.renal_status == "severe_impairment"
        and bool(med.get("renal_caution"))
    )

    # Sseg: segurança clínica
    sseg = 1.0

    allergies = [a.lower() for a in patient.allergies]

    if candidate in allergies or med["active_substance"].lower() in allergies:
        sseg = 0.0
        reasons.append("Conflito com alergia conhecida do utente.")

    if has_severe_renal_caution:
        sseg -= 0.4
        append_unique_reason(reasons, RENAL_CAUTION_REASON)

    if any(c in patient.conditions for c in med.get("contraindicated_conditions", [])):
        sseg = 0.0
        reasons.append("Contraindicado para uma ou mais condições clínicas do utente.")

    sseg = max(0.0, min(1.0, sseg))

    # Sctx: adequação contextual
    sctx = 0.5

    if patient.main_problem in med.get("indications", []):
        sctx += 0.4
        reasons.append("Adequado ao problema clínico principal.")

    if patient.renal_status == "mild_impairment" and med.get("renal_caution"):
        sctx -= 0.1
        reasons.append(
            "Adequação contextual ligeiramente reduzida devido a compromisso renal ligeiro/moderado."
        )

    if has_severe_renal_caution:
        sctx -= 0.3

    sctx = max(0.0, min(1.0, sctx))

    # Ssim: proximidade terapêutica
    ssim = 0.5

    if original and original["therapeutic_class"] == med["therapeutic_class"]:
        ssim = 1.0
        reasons.append(
            "Pertence à mesma classe terapêutica do medicamento inicialmente prescrito."
        )
    elif original and patient.main_problem in med.get("indications", []):
        ssim = 0.7
        reasons.append(
            "Pertence a uma classe diferente, mas apresenta finalidade terapêutica compatível."
        )

    # Nuance clínica: paracetamol em contexto de inflamação após AINE
    if is_symptomatic_analgesic_fallback(
        candidate=candidate,
        patient=patient,
        original_medication=original_medication,
        kb=kb,
    ):
        sctx = min(sctx, 0.45)
        ssim = min(ssim, 0.45)

        if SYMPTOMATIC_ANALGESIC_REASON not in reasons:
            reasons.append(SYMPTOMATIC_ANALGESIC_REASON)

    # Sfb: feedback histórico, ainda placeholder
    sfb = feedback_score

    # Pesos heurísticos iniciais
    w1, w2, w3, w4 = 0.45, 0.30, 0.20, 0.05

    score = (w1 * sseg) + (w2 * sctx) + (w3 * ssim) + (w4 * sfb)
    score = round(max(0.0, min(1.0, score)), 3)

    if not reasons:
        reasons.append(
            "Alternativa considerada admissível pela base de conhecimento atual do protótipo."
        )

    return score, reasons


def apply_secondary_historical_refinement(
    recommendations: List[Recommendation],
    epsilon: float = 0.05,
    lambda_hist: float = 0.05,
) -> List[Recommendation]:
    """
    Applies historical-pattern refinement only when two or more alternatives
    have very similar base scores. If there is no tie/proximity group,
    score_final remains equal to score_base.
    """
    if len(recommendations) < 2:
        return recommendations

    recommendations = sorted(recommendations, key=lambda r: r.score_base, reverse=True)

    refined = []
    i = 0

    while i < len(recommendations):
        current = recommendations[i]
        group = [current]
        j = i + 1

        while j < len(recommendations):
            if abs(current.score_base - recommendations[j].score_base) <= epsilon:
                group.append(recommendations[j])
                j += 1
            else:
                break

        if len(group) > 1:
            for rec in group:
                hist = rec.secondary_historical_score or 0.0
                rec.score_final = round(rec.score_base + lambda_hist * hist, 3)

            refined.extend(sorted(group, key=lambda r: r.score_final, reverse=True))
        else:
            current.score_final = current.score_base
            refined.append(current)

        i = j

    return refined

def is_candidate_already_active(
    candidate: str,
    patient: PatientContext,
) -> bool:
    return candidate in set(get_active_medication_ids(patient))

def medication_in_relevant_alerts(
    medication_id: str,
    alerts: List[Alert] | None,
) -> bool:
    """
    Verifica se um medicamento ativo está envolvido em alertas clinicamente relevantes.

    Se estiver envolvido em alertas moderados, elevados ou críticos, não deve ser
    apresentado como potencial alternativa já ativa, porque isso pode confundir
    o utilizador.
    """
    if not alerts:
        return False

    for alert in alerts:
        if alert.severity not in NOTE_EXCLUSION_SEVERITIES:
            continue

        if medication_id in (alert.medication_ids or []):
            return True

    return False

def build_recommendation_notes(
    patient: PatientContext,
    recommendations: List[Recommendation],
    kb: Dict[str, Any],
    alerts: List[Alert] | None = None,
) -> List[dict]:
    """
    Gera notas explicativas quando não há nova recomendação.

    Exemplo:
    - uma alternativa candidata já consta da medicação ativa;
    - paracetamol já está ativo e poderia ser apenas opção sintomática/analgésica.
    """
    if recommendations:
        return []

    notes = []
    active_medication_ids = set(get_active_medication_ids(patient))
    main_problem = normalize_condition_id(patient.main_problem)

    for med_id in active_medication_ids:
        med = kb.get("medications", {}).get(med_id)

        if not med:
            continue

        if medication_in_relevant_alerts(med_id, alerts):
            continue

        display_name = med.get("display_name", med_id)
        indications = med.get("indications", [])

        if main_problem in indications:
            notes.append({
                "type": "already_active_candidate",
                "medication": display_name,
                "description": (
                    f"{display_name} já consta da medicação ativa do utente e tem "
                    "indicação compatível com o problema clínico principal na base "
                    "de conhecimento atual do protótipo. Por esse motivo, não foi "
                    "apresentado como nova alternativa."
                ),
            })

        elif main_problem == "inflammation" and med_id == "paracetamol":
            notes.append({
                "type": "already_active_symptomatic_candidate",
                "medication": display_name,
                "description": (
                    "Paracetamol já consta da medicação ativa do utente. Pode ser "
                    "considerado apenas como alternativa sintomática/analgésica, "
                    "não como substituto anti-inflamatório equivalente."
                ),
            })

    return notes

def is_prescription_related_alert(alert: Alert) -> bool:
    return alert.origin in {"prescription_related", "combined_profile_risk"}


def risk_signature(alert: Alert) -> tuple:
    """
    Agrupa riscos por tipo/regra/descrição, para comparar se um candidato
    mantém o mesmo tipo de problema que originou o alerta inicial.
    """
    return (
        alert.type,
        alert.rule_id or "",
        alert.description,
    )


def has_moderate_or_higher_alert(alerts: List[Alert]) -> bool:
    return any(alert.severity in {"moderate", "high", "critical"} for alert in alerts)


def clinically_relevant_alerts(alerts: List[Alert]) -> List[Alert]:
    return [
        alert for alert in alerts
        if alert.severity in {"moderate", "high", "critical"}
        and alert.origin != "active_medication_existing"
    ]

def recommend_alternatives(
    patient: PatientContext,
    prescription: List[MedicationLine],
    alerts: List[Alert],
    kb: Dict[str, Any],
    historical_patterns: Dict[str, Any],
) -> List[Recommendation]:
    if not prescription:
        return []

    relevant_alerts = [
        alert for alert in alerts
        if alert.severity in RECOMMENDATION_TRIGGER_SEVERITIES
        and is_prescription_related_alert(alert)
    ]

    if not relevant_alerts:
        return []

    original_risk_signatures = {
        risk_signature(alert)
        for alert in relevant_alerts
    }

    candidates = generate_candidates(prescription, kb)

    original_medication = (
        normalize_medication_id(prescription[0].medication)
        or prescription[0].medication.lower()
    )

    evaluated_candidates = []

    for candidate in candidates:
        # Não recomendar medicamento que já está na medicação ativa.
        if is_candidate_already_active(candidate, patient):
            continue

        candidate_alerts = evaluate_candidate_safety(
            candidate=candidate,
            patient=patient,
            kb=kb,
        )

        candidate_relevant_alerts = clinically_relevant_alerts(candidate_alerts)

        # Barreira determinística: candidatos com alerta high/critical não entram.
        if has_blocking_alert(candidate_relevant_alerts):
            continue

        candidate_risk_signatures = {
            risk_signature(alert)
            for alert in candidate_relevant_alerts
        }

        evaluated_candidates.append(
            {
                "candidate": candidate,
                "candidate_alerts": candidate_relevant_alerts,
                "same_risk_as_original": bool(
                    candidate_risk_signatures & original_risk_signatures
                ),
            }
        )

    if not evaluated_candidates:
        return []

    # Se existir pelo menos uma alternativa sem alertas moderados/relevantes,
    # elimina alternativas que mantêm alertas moderados.
    has_clean_candidate = any(
        not item["candidate_alerts"]
        for item in evaluated_candidates
    )

    if has_clean_candidate:
        evaluated_candidates = [
            item for item in evaluated_candidates
            if not item["candidate_alerts"]
        ]

    # Se ainda houver candidatos com o mesmo tipo de risco que a prescrição original
    # e também houver candidatos sem esse mesmo risco, remove os que mantêm o problema.
    has_candidate_without_same_risk = any(
        not item["same_risk_as_original"]
        for item in evaluated_candidates
    )

    if has_candidate_without_same_risk:
        evaluated_candidates = [
            item for item in evaluated_candidates
            if not item["same_risk_as_original"]
        ]

    recommendations = []

    for item in evaluated_candidates:
        candidate = item["candidate"]
        candidate_alerts = item["candidate_alerts"]

        score_base, reasons = score_candidate_base(
            candidate=candidate,
            patient=patient,
            original_medication=original_medication,
            kb=kb,
        )

        penalty = penalty_from_candidate_alerts(candidate_alerts)

        if penalty > 0:
            score_base = round(max(0.0, score_base - penalty), 3)

            for alert in candidate_alerts:
                if (
                    alert.rule_id == "renal_caution"
                    and "compromisso renal grave" in alert.description.lower()
                    and any("compromisso renal grave" in reason.lower() for reason in reasons)
                ):
                    continue

                append_unique_reason(
                    reasons,
                    f"Precaução no candidato: {alert.description}",
                )

        ml_features = build_ml_features(
            candidate=candidate,
            patient=patient,
            prescription=prescription,
            candidate_alerts=candidate_alerts,
            kb=kb,
        )

        ml_score = predict_candidate_adequacy(ml_features)

        if ml_score is not None:
            score_base = combine_heuristic_and_ml_score(
                heuristic_score=score_base,
                ml_score=ml_score,
                ml_weight=0.30,
            )

            # Não acrescentar score técnico às razões clínicas visíveis.
            # O modelo continua a influenciar o ranking, mas não polui a interface.

        historical_score = get_historical_score(candidate, patient, historical_patterns)

        recommendations.append(
            Recommendation(
                medication=candidate,
                score_base=score_base,
                score_final=score_base,
                reasons=reasons,
                secondary_historical_score=historical_score,
            )
        )

    recommendations = apply_secondary_historical_refinement(recommendations)

    return sorted(recommendations, key=lambda r: r.score_final, reverse=True)