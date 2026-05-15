from typing import List, Dict, Any, Optional, Tuple

from app.schemas import PatientContext, MedicationLine, Alert
from app.normalization import normalize_medication_id


def get_medication(kb: Dict[str, Any], med_id: str) -> Optional[Dict[str, Any]]:
    return kb.get("medications", {}).get(med_id)


def get_display_name(kb: Dict[str, Any], med_id: str) -> str:
    med = get_medication(kb, med_id)
    if not med:
        return med_id
    return med.get("display_name", med_id)


def get_class(kb: Dict[str, Any], med_id: str) -> Optional[str]:
    med = get_medication(kb, med_id)
    if not med:
        return None
    return med.get("therapeutic_class")


def normalize_prescription(prescription: List[MedicationLine]) -> List[str]:
    normalized = []

    for line in prescription:
        med_id = normalize_medication_id(line.medication)
        if med_id:
            normalized.append(med_id)

    return normalized


def normalize_active_medications(patient: PatientContext) -> List[str]:
    normalized = []

    for med in patient.active_medications:
        med_id = normalize_medication_id(med)
        if med_id:
            normalized.append(med_id)

    return normalized


def build_alert(
    alert_type: str,
    severity: str,
    medication: str,
    description: str,
    origin: str = "unknown",
    involves_prescribed_medication: bool = False,
    involves_active_medication: bool = False,
    rule_id: Optional[str] = None,
    medication_ids: Optional[List[str]] = None,
) -> Alert:
    return Alert(
        type=alert_type,
        severity=severity,
        medication=medication,
        description=description,
        origin=origin,
        involves_prescribed_medication=involves_prescribed_medication,
        involves_active_medication=involves_active_medication,
        rule_id=rule_id,
        medication_ids=medication_ids or [],
    )


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted([a, b]))

def classify_pair_origin(
    med_a: str,
    med_b: str,
    active_set: set[str],
    prescribed_set: set[str],
) -> str:
    a_prescribed = med_a in prescribed_set
    b_prescribed = med_b in prescribed_set
    a_active = med_a in active_set
    b_active = med_b in active_set

    if a_prescribed or b_prescribed:
        return "prescription_related"

    if a_active and b_active:
        return "active_medication_existing"

    return "combined_profile_risk"


def alert_involves_prescription(
    medication_ids: List[str],
    prescribed_set: set[str],
) -> bool:
    return any(med_id in prescribed_set for med_id in medication_ids)


def alert_involves_active_medication(
    medication_ids: List[str],
    active_set: set[str],
) -> bool:
    return any(med_id in active_set for med_id in medication_ids)

def has_condition(patient: PatientContext, condition: str) -> bool:
    return condition in patient.conditions or patient.renal_status == condition


def check_contraindications(
    patient: PatientContext,
    prescribed_medications: List[str],
    kb: Dict[str, Any],
) -> List[Alert]:
    alerts = []

    for med_id in prescribed_medications:
        med = get_medication(kb, med_id)
        if not med:
            continue

        for condition in med.get("contraindicated_conditions", []):
            if has_condition(patient, condition):
                alerts.append(
                    build_alert(
                        alert_type="contraindication",
                        severity="critical",
                        medication=get_display_name(kb, med_id),
                        description=(
                            f"{get_display_name(kb, med_id)} está contraindicado ou deve ser evitado "
                            f"neste contexto clínico devido à condição clínica identificada: {condition}."
                        ),
                        origin="prescription_related",
                        involves_prescribed_medication=True,
                        involves_active_medication=False,
                        medication_ids=[med_id],
                    )
                )

        if patient.renal_status == "severe_impairment" and med.get("renal_caution"):
            renal_alert_severity = med.get("renal_alert_severity", "high")

            alerts.append(
                build_alert(
                    alert_type="renal_risk",
                    severity=renal_alert_severity,
                    medication=get_display_name(kb, med_id),
                    description=(
                        f"{get_display_name(kb, med_id)} requer precaução acrescida em doentes "
                        "com compromisso renal grave."
                    ),
                    origin="prescription_related",
                    involves_prescribed_medication=True,
                    involves_active_medication=False,
                    rule_id="renal_caution",
                    medication_ids=[med_id],
                )
            )

    return alerts


def rule_matches_drug_drug(rule: Dict[str, Any], med_a: str, med_b: str) -> bool:
    return pair_key(rule.get("medication_a"), rule.get("medication_b")) == pair_key(med_a, med_b)


def rule_matches_class_class(
    rule: Dict[str, Any],
    class_a: Optional[str],
    class_b: Optional[str],
) -> bool:
    if not class_a or not class_b:
        return False

    return pair_key(rule.get("class_a"), rule.get("class_b")) == pair_key(class_a, class_b)


def rule_matches_class_drug(
    rule: Dict[str, Any],
    med_a: str,
    med_b: str,
    class_a: Optional[str],
    class_b: Optional[str],
) -> bool:
    rule_class = rule.get("class_a")
    rule_med = rule.get("medication_b")

    return (
        (class_a == rule_class and med_b == rule_med)
        or
        (class_b == rule_class and med_a == rule_med)
    )


def is_qt_risk(kb: Dict[str, Any], med_id: str) -> bool:
    med = get_medication(kb, med_id)
    return bool(med and med.get("qt_risk"))


def severity_rank(severity: str) -> int:
    ranks = {
        "low": 1,
        "moderate": 2,
        "high": 3,
        "critical": 4,
    }
    return ranks.get(severity, 0)


def specificity_rank(match_type: str) -> int:
    ranks = {
        "qt_qt": 1,
        "class_class": 2,
        "class_drug": 3,
        "drug_drug": 4,
    }
    return ranks.get(match_type, 0)


def check_interactions(
    medication_ids: List[str],
    kb: Dict[str, Any],
    active_medications: Optional[List[str]] = None,
    prescribed_medications: Optional[List[str]] = None,
) -> List[Alert]:
    rules = kb.get("interaction_rules", [])
    unique_meds = list(dict.fromkeys(medication_ids))
    active_set = set(active_medications or [])
    prescribed_set = set(prescribed_medications or [])

    best_alert_by_pair = {}

    for i in range(len(unique_meds)):
        for j in range(i + 1, len(unique_meds)):
            med_a = unique_meds[i]
            med_b = unique_meds[j]

            med_a_obj = get_medication(kb, med_a)
            med_b_obj = get_medication(kb, med_b)

            if not med_a_obj or not med_b_obj:
                continue

            class_a = get_class(kb, med_a)
            class_b = get_class(kb, med_b)
            pair = pair_key(med_a, med_b)

            for rule in rules:
                match_type = rule.get("match")
                matched = False

                if match_type == "drug_drug":
                    matched = rule_matches_drug_drug(rule, med_a, med_b)

                elif match_type == "class_class":
                    matched = rule_matches_class_class(rule, class_a, class_b)

                    if matched and med_a == med_b:
                        matched = False

                elif match_type == "class_drug":
                    matched = rule_matches_class_drug(rule, med_a, med_b, class_a, class_b)

                elif match_type == "qt_qt":
                    matched = is_qt_risk(kb, med_a) and is_qt_risk(kb, med_b)

                if not matched:
                    continue

                candidate = {
                    "rule": rule,
                    "med_a": med_a,
                    "med_b": med_b,
                    "match_type": match_type,
                    "severity": rule.get("severity", "moderate"),
                }

                current = best_alert_by_pair.get(pair)

                if current is None:
                    best_alert_by_pair[pair] = candidate
                    continue

                candidate_score = (
                    severity_rank(candidate["severity"]),
                    specificity_rank(candidate["match_type"]),
                )

                current_score = (
                    severity_rank(current["severity"]),
                    specificity_rank(current["match_type"]),
                )

                if candidate_score > current_score:
                    best_alert_by_pair[pair] = candidate

    alerts = []

    for candidate in best_alert_by_pair.values():
        rule = candidate["rule"]
        med_a = candidate["med_a"]
        med_b = candidate["med_b"]

        medication_ids = [med_a, med_b]
        origin = classify_pair_origin(
            med_a=med_a,
            med_b=med_b,
            active_set=active_set,
            prescribed_set=prescribed_set,
        )

        alerts.append(
            build_alert(
                alert_type="interaction",
                severity=rule.get("severity", "moderate"),
                medication=f"{get_display_name(kb, med_a)} + {get_display_name(kb, med_b)}",
                description=rule.get("description", "Interação medicamentosa potencial identificada."),
                origin=origin,
                involves_prescribed_medication=alert_involves_prescription(
                    medication_ids, prescribed_set
                ),
                involves_active_medication=alert_involves_active_medication(
                    medication_ids, active_set
                ),
                rule_id=rule.get("id"),
                medication_ids=medication_ids,
            )
        )

    return alerts


def check_triple_whammy(
    active_medications: List[str],
    prescribed_medications: List[str],
    kb: Dict[str, Any],
) -> List[Alert]:
    medication_ids = list(dict.fromkeys(active_medications + prescribed_medications))

    classes = {get_class(kb, med_id) for med_id in medication_ids}

    has_aine = "aine" in classes
    has_sraa = "ieca" in classes or "ara" in classes
    has_diuretic = "diuretico_ansa" in classes or "diuretico_tiazidico" in classes

    if not (has_aine and has_sraa and has_diuretic):
        return []

    prescribed_set = set(prescribed_medications)
    active_set = set(active_medications)

    involved_meds = [
        med_id
        for med_id in medication_ids
        if get_class(kb, med_id) in {
            "aine",
            "ieca",
            "ara",
            "diuretico_ansa",
            "diuretico_tiazidico",
        }
    ]

    origin = (
        "prescription_related"
        if any(med_id in prescribed_set for med_id in involved_meds)
        else "active_medication_existing"
    )

    return [
        build_alert(
            alert_type="interaction",
            severity="high",
            medication="AINE + IECA/ARA + diurético",
            description=(
                "Associação com risco aumentado de deterioração da função renal: "
                "AINE em combinação com inibidor da ECA ou antagonista dos recetores "
                "da angiotensina II e diurético. Recomenda-se evitar a associação ou "
                "monitorizar função renal, hidratação e eletrólitos."
            ),
            origin=origin,
            involves_prescribed_medication=any(
                med_id in prescribed_set for med_id in involved_meds
            ),
            involves_active_medication=any(
                med_id in active_set for med_id in involved_meds
            ),
            rule_id="triple_whammy",
            medication_ids=involved_meds,
        )
    ]


def check_unknown_medications(
    prescription: List[MedicationLine],
) -> List[Alert]:
    alerts = []

    for line in prescription:
        med_id = normalize_medication_id(line.medication)

        if not med_id:
            alerts.append(
                build_alert(
                    alert_type="unknown_medication",
                    severity="low",
                    medication=line.medication,
                    description=(
                        "O medicamento não foi reconhecido pela base de conhecimento do protótipo. "
                        "A análise automática pode estar incompleta."
                    ),
                    origin="prescription_related",
                    involves_prescribed_medication=True,
                    involves_active_medication=False,
                    medication_ids=[],
                )
            )

    return alerts

def run_safety_checks(
    patient: PatientContext,
    prescription: List[MedicationLine],
    kb: Dict[str, Any],
) -> List[Alert]:
    prescribed_medications = normalize_prescription(prescription)
    active_medications = normalize_active_medications(patient)

    all_medications = active_medications + prescribed_medications

    alerts = []

    alerts.extend(check_unknown_medications(prescription))
    alerts.extend(check_contraindications(patient, prescribed_medications, kb))

    alerts.extend(
        check_interactions(
            medication_ids=all_medications,
            kb=kb,
            active_medications=active_medications,
            prescribed_medications=prescribed_medications,
        )
    )

    alerts.extend(
        check_triple_whammy(
            active_medications=active_medications,
            prescribed_medications=prescribed_medications,
            kb=kb,
        )
    )

    return alerts