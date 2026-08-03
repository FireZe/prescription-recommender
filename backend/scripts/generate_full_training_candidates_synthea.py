import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import pandas as pd

import os
BASE_DIR = Path(os.getcwd()) / "backend"
sys.path.append(str(BASE_DIR))

from app.schemas import PatientContext, MedicationLine, Alert
from app.data_loader import load_knowledge_base
from app.rules_engine import run_safety_checks
from app.normalization import normalize_medication_id, normalize_main_problem


INDEX_PATH = BASE_DIR / "data" / "synthea_context_index.csv"

TRAINING_OUTPUT_PATH = BASE_DIR / "data" / "training_examples.csv"
FULL_TRAINING_OUTPUT_PATH = BASE_DIR / "data" / "training_examples_full_synthea.csv"
AUDIT_OUTPUT_PATH = BASE_DIR / "data" / "training_examples_audit.csv"
REVIEW_OUTPUT_PATH = BASE_DIR / "data" / "training_candidates_for_review.csv"


NUMERIC_FEATURES = [
    "age",
    "age_squared",
    "is_elderly",
    "active_medication_count",
    "condition_count",
    "renal_status_score",
    "candidate_renal_caution",
    "candidate_qt_risk",
    "has_anticoagulant",
    "has_antiplatelet",
    "has_diuretic",
    "has_acei_or_arb",
    "has_qt_risk_medication",
    "same_therapeutic_class",
    "candidate_is_nsaid",
]

CATEGORICAL_FEATURES = [
    "candidate",
    "candidate_class",
    "original_medication",
    "original_class",
    "main_problem",
]

REVIEW_GOLD_COLUMNS = [
    "gold_adequacy_score",
    "gold_label_class",
    "review_status",
    "reviewer",
    "review_comment",
]

SEVERITY_RANK = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "critical": 4,
}

BASE_SCORE_BY_SEVERITY = {
    "none": 1.00,
    "low": 0.85,
    "moderate": 0.60,
    "high": 0.20,
    "critical": 0.00,
}

PROBLEM_TO_ORIGINAL_PRIORITY = {
    "pain": ["ibuprofen", "naproxen", "paracetamol"],
    "inflammation": ["ibuprofen", "naproxen"],
    "hypertension": [
        "hydrochlorothiazide",
        "ramipril",
        "losartan",
        "enalapril",
        "valsartan",
        "furosemide",
    ],
    "infection": ["clarithromycin", "azithromycin"],
    "heart_failure": ["ramipril", "enalapril", "losartan", "valsartan", "furosemide", "digoxin"],
    "dyslipidemia": ["simvastatin", "atorvastatin"],
    "cardiovascular_prevention": ["clopidogrel", "acetylsalicylic_acid", "simvastatin", "atorvastatin"],
    "antiplatelet": ["clopidogrel", "acetylsalicylic_acid"],
    "anticoagulation": ["warfarin"],
    "depression": ["sertraline", "amitriptyline"],
    "anxiety": ["sertraline"],
    "arrhythmia": ["amiodarone", "digoxin"],
}

IMPORTANT_RISK_PROBES = [
    "ibuprofen",
    "naproxen",
    "paracetamol",
    "clarithromycin",
    "azithromycin",
    "sertraline",
    "amitriptyline",
    "warfarin",
    "amiodarone",
    "digoxin",
    "hydrochlorothiazide",
    "furosemide",
    "ramipril",
    "losartan",
]


def parse_json_list(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    return []


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def stable_rng(patient_id: str, original_medication: str, seed: int) -> random.Random:
    key = f"{seed}|{patient_id}|{original_medication}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    integer_seed = int(digest[:12], 16)
    return random.Random(integer_seed)


def get_medication_class(medication_id: str, kb: Dict[str, Any]) -> str | None:
    medication = kb.get("medications", {}).get(medication_id)
    if not medication:
        return None
    return medication.get("therapeutic_class")


def get_active_medication_ids(patient: PatientContext) -> List[str]:
    ids = []

    for raw_medication in patient.active_medications:
        medication_id = normalize_medication_id(raw_medication)
        if medication_id:
            ids.append(medication_id)

    return list(dict.fromkeys(ids))


def has_active_class(patient: PatientContext, kb: Dict[str, Any], classes: Set[str]) -> int:
    for medication_id in get_active_medication_ids(patient):
        if get_medication_class(medication_id, kb) in classes:
            return 1

    return 0


def has_active_qt_risk(patient: PatientContext, kb: Dict[str, Any]) -> int:
    for medication_id in get_active_medication_ids(patient):
        medication = kb.get("medications", {}).get(medication_id)
        if medication and medication.get("qt_risk"):
            return 1

    return 0


def renal_status_score(renal_status: str) -> int:
    return {
        "normal": 0,
        "mild_impairment": 1,
        "severe_impairment": 2,
    }.get(renal_status, 0)


def build_patient_context_from_row(row: pd.Series) -> PatientContext:
    sex = row.get("sex", "Other")
    if sex not in {"M", "F"}:
        sex = "Other"

    main_problem_raw = row.get("main_problem_guess", "unspecified") or "unspecified"
    main_problem = normalize_main_problem(str(main_problem_raw))

    return PatientContext(
        patient_id=str(row["patient_id"]),
        age=int(row["age"]),
        sex=sex,
        conditions=parse_json_list(row.get("conditions_norm", "[]")),
        allergies=parse_json_list(row.get("allergies", "[]")),
        active_medications=parse_json_list(row.get("active_medications", "[]")),
        renal_status=row.get("renal_status", "normal") or "normal",
        main_problem=main_problem,
    )


def medication_has_indication(
    medication_id: str,
    main_problem: str,
    kb: Dict[str, Any],
) -> bool:
    medication = kb.get("medications", {}).get(medication_id)
    if not medication:
        return False

    return main_problem in medication.get("indications", [])


def is_symptomatic_paracetamol_for_inflammation(
    candidate: str,
    main_problem: str,
) -> bool:
    return candidate == "paracetamol" and main_problem == "inflammation"


def choose_original_scenarios(
    patient: PatientContext,
    kb: Dict[str, Any],
    max_originals_per_patient: int,
) -> List[dict]:
    medications = kb.get("medications", {})
    scenarios: List[dict] = []

    def add_scenario(original_medication: str, scenario_type: str) -> None:
        if original_medication not in medications:
            return

        if any(item["original_medication"] == original_medication for item in scenarios):
            return

        scenarios.append(
            {
                "original_medication": original_medication,
                "scenario_type": scenario_type,
            }
        )

    for medication_id in PROBLEM_TO_ORIGINAL_PRIORITY.get(patient.main_problem, []):
        add_scenario(medication_id, "problem_original")

    for active_medication in get_active_medication_ids(patient):
        medication = medications.get(active_medication)

        if not medication:
            continue

        if medication.get("alternatives"):
            add_scenario(active_medication, "active_medication_original")

    if not scenarios:
        for medication_id, medication in medications.items():
            if medication.get("alternatives"):
                add_scenario(medication_id, "fallback_original")
            if len(scenarios) >= max_originals_per_patient:
                break

    return scenarios[:max_originals_per_patient]


def add_candidate(
    candidates: Dict[str, Set[str]],
    medication_id: str,
    source: str,
    kb: Dict[str, Any],
) -> None:
    if medication_id not in kb.get("medications", {}):
        return

    candidates.setdefault(medication_id, set()).add(source)


def get_problem_compatible_candidates(
    main_problem: str,
    kb: Dict[str, Any],
) -> List[str]:
    if main_problem in {"unspecified", "", None}:
        return []

    result = []

    for medication_id, medication in kb.get("medications", {}).items():
        if main_problem in medication.get("indications", []):
            result.append(medication_id)

    if main_problem == "inflammation" and "paracetamol" in kb.get("medications", {}):
        # Paracetamol é mantido como alternativa sintomática/analgésica, não como
        # substituto anti-inflamatório equivalente.
        result.append("paracetamol")

    return list(dict.fromkeys(result))


def get_active_alternative_candidates(
    patient: PatientContext,
    kb: Dict[str, Any],
) -> List[str]:
    candidates = []

    for active_medication in get_active_medication_ids(patient):
        medication = kb.get("medications", {}).get(active_medication)

        if not medication:
            continue

        for alternative in medication.get("alternatives", []):
            if alternative in kb.get("medications", {}):
                candidates.append(alternative)

    return list(dict.fromkeys(candidates))


def get_risk_probe_candidates(
    patient: PatientContext,
    kb: Dict[str, Any],
) -> List[str]:
    probes = set()

    active_ids = get_active_medication_ids(patient)
    active_classes = {
        get_medication_class(medication_id, kb)
        for medication_id in active_ids
    }

    if patient.renal_status == "severe_impairment":
        for medication_id, medication in kb.get("medications", {}).items():
            if medication.get("renal_caution"):
                probes.add(medication_id)

    if "antiagregante" in active_classes or "anticoagulante" in active_classes:
        probes.update({"ibuprofen", "naproxen", "sertraline"})

    if (
        "diuretico_ansa" in active_classes
        or "diuretico_tiazidico" in active_classes
        or "ieca" in active_classes
        or "ara" in active_classes
    ):
        probes.update({"ibuprofen", "naproxen"})

    if "estatina" in active_classes:
        probes.update({"clarithromycin", "azithromycin"})

    if "aine" in active_classes:
        probes.update({"ibuprofen", "naproxen"})

    if "digitalico" in active_classes:
        probes.update({"amiodarone", "furosemide", "hydrochlorothiazide"})

    if "anticoagulante" in active_classes:
        probes.update({"amiodarone", "clarithromycin", "azithromycin", "sertraline", "ibuprofen", "naproxen"})

    if has_active_qt_risk(patient, kb):
        for medication_id, medication in kb.get("medications", {}).items():
            if medication.get("qt_risk"):
                probes.add(medication_id)

    probes.update(
        medication_id
        for medication_id in IMPORTANT_RISK_PROBES
        if medication_id in kb.get("medications", {})
    )

    return list(dict.fromkeys(sorted(probes)))


def get_random_candidates(
    kb: Dict[str, Any],
    rng: random.Random,
    n: int,
    excluded: Iterable[str],
) -> List[str]:
    excluded_set = set(excluded)
    available = [
        medication_id
        for medication_id in kb.get("medications", {})
        if medication_id not in excluded_set
    ]

    if n <= 0 or not available:
        return []

    n = min(n, len(available))
    return rng.sample(available, n)


def build_hybrid_candidate_set(
    patient: PatientContext,
    original_medication: str,
    kb: Dict[str, Any],
    random_candidates_per_scenario: int,
    seed: int,
) -> Dict[str, Set[str]]:
    candidates: Dict[str, Set[str]] = {}
    medications = kb.get("medications", {})
    original = medications.get(original_medication, {})

    for candidate in get_problem_compatible_candidates(patient.main_problem, kb):
        add_candidate(candidates, candidate, "problem_compatible", kb)

    for candidate in original.get("alternatives", []):
        add_candidate(candidates, candidate, "original_alternative", kb)

    for candidate in get_active_alternative_candidates(patient, kb):
        add_candidate(candidates, candidate, "active_medication_alternative", kb)

    for candidate in get_risk_probe_candidates(patient, kb):
        add_candidate(candidates, candidate, "risk_probe", kb)

    rng = stable_rng(patient.patient_id, original_medication, seed)
    random_candidates = get_random_candidates(
        kb=kb,
        rng=rng,
        n=random_candidates_per_scenario,
        excluded=set(candidates) | {original_medication},
    )

    for candidate in random_candidates:
        add_candidate(candidates, candidate, "random_diversity", kb)

    candidates.pop(original_medication, None)

    return candidates


def relevant_alerts_for_candidate(alerts: List[Alert]) -> List[Alert]:
    return [
        alert for alert in alerts
        if alert.origin != "active_medication_existing"
    ]


def max_alert_severity(alerts: List[Alert]) -> str:
    if not alerts:
        return "none"

    return max(
        (alert.severity for alert in alerts),
        key=lambda severity: SEVERITY_RANK.get(severity, 0),
    )


def alerts_to_text(alerts: List[Alert]) -> str:
    if not alerts:
        return ""

    return " | ".join(
        (
            f"{alert.type}:{alert.severity}:"
            f"{alert.origin}:"
            f"{alert.rule_id or 'no_rule'}:"
            f"{alert.medication}:"
            f"{alert.description}"
        )
        for alert in alerts
    )


def alert_rule_ids(alerts: List[Alert]) -> str:
    rule_ids = [
        alert.rule_id
        for alert in alerts
        if alert.rule_id
    ]

    return "|".join(sorted(set(rule_ids)))


def alert_origins(alerts: List[Alert]) -> str:
    origins = [
        alert.origin
        for alert in alerts
        if alert.origin
    ]

    return "|".join(sorted(set(origins)))


def alert_severity_counts(alerts: List[Alert]) -> dict:
    counts = {
        "alert_count": len(alerts),
        "critical_alert_count": 0,
        "high_alert_count": 0,
        "moderate_alert_count": 0,
        "low_alert_count": 0,
        "renal_alert_count": 0,
        "interaction_alert_count": 0,
        "duplication_alert_count": 0,
    }

    for alert in alerts:
        if alert.severity == "critical":
            counts["critical_alert_count"] += 1
        elif alert.severity == "high":
            counts["high_alert_count"] += 1
        elif alert.severity == "moderate":
            counts["moderate_alert_count"] += 1
        elif alert.severity == "low":
            counts["low_alert_count"] += 1

        if alert.type == "renal_risk":
            counts["renal_alert_count"] += 1

        if alert.type == "interaction":
            counts["interaction_alert_count"] += 1

        if alert.rule_id == "aine_aine_duplicacao":
            counts["duplication_alert_count"] += 1

    return counts


def risk_signatures(alerts: List[Alert]) -> Set[tuple]:
    signatures = set()

    for alert in alerts:
        if alert.severity not in {"moderate", "high", "critical"}:
            continue

        signatures.add(
            (
                alert.type,
                alert.rule_id or "",
                alert.description,
            )
        )

    return signatures


def label_class_from_score(score: float) -> int:
    if score < 0.40:
        return 0
    if score < 0.75:
        return 1
    return 2


def build_silver_label(
    patient: PatientContext,
    candidate: str,
    original_medication: str,
    candidate_sources: Set[str],
    original_alerts: List[Alert],
    candidate_alerts: List[Alert],
    kb: Dict[str, Any],
) -> tuple[float, int, str]:
    medication = kb["medications"][candidate]
    main_problem = patient.main_problem
    candidate_is_already_active = candidate in set(get_active_medication_ids(patient))

    relevant_candidate_alerts = relevant_alerts_for_candidate(candidate_alerts)
    severity = max_alert_severity(relevant_candidate_alerts)
    score = BASE_SCORE_BY_SEVERITY.get(severity, 0.50)
    reasons = []

    if relevant_candidate_alerts:
        reasons.append(f"Alertas relevantes no candidato: gravidade máxima {severity}.")
    else:
        reasons.append("Sem alertas relevantes diretamente atribuídos ao candidato.")

    if candidate_is_already_active:
        score = min(score, 0.20)
        reasons.append("Não é nova alternativa: medicamento já consta da medicação ativa.")

    has_indication = medication_has_indication(candidate, main_problem, kb)

    if has_indication:
        reasons.append("Indicação compatível com o problema clínico principal.")
    elif is_symptomatic_paracetamol_for_inflammation(candidate, main_problem):
        score = min(score, 0.60)
        reasons.append(
            "Paracetamol em inflamação: alternativa sintomática/analgésica, não anti-inflamatória equivalente."
        )
    elif main_problem not in {"unspecified", "", None}:
        score = min(score, 0.25)
        reasons.append("Sem indicação compatível com o problema clínico principal.")

    if (
        patient.renal_status == "severe_impairment"
        and medication.get("renal_caution")
        and severity not in {"high", "critical"}
    ):
        score = min(score, 0.60)
        reasons.append("Requer precaução por compromisso renal grave.")

    shared_risk = bool(
        risk_signatures(relevant_alerts_for_candidate(original_alerts))
        & risk_signatures(relevant_candidate_alerts)
    )

    if shared_risk:
        score = min(score, 0.55)
        reasons.append("Mantém pelo menos um tipo de risco identificado no medicamento original.")

    if "risk_probe" in candidate_sources and not has_indication and severity == "none":
        score = min(score, 0.35)
        reasons.append("Candidato de risco/controlo negativo sem indicação contextual.")

    score = round(max(0.0, min(1.0, score)), 3)
    label_class = label_class_from_score(score)

    return score, label_class, " ".join(reasons)


def build_features(
    candidate: str,
    original_medication: str,
    patient: PatientContext,
    kb: Dict[str, Any],
) -> Dict[str, Any]:
    candidate_obj = kb["medications"][candidate]
    original_obj = kb["medications"][original_medication]

    candidate_class = candidate_obj.get("therapeutic_class", "unknown")
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


def build_review_columns(df: pd.DataFrame) -> pd.DataFrame:
    review_df = df.copy()

    review_df["case_id"] = [
        f"FULL-SYN-{index + 1:06d}"
        for index in range(len(review_df))
    ]

    review_df["silver_adequacy_score"] = review_df["adequacy_score"]
    review_df["silver_label_class"] = review_df["label_class"]

    review_df["gold_adequacy_score"] = ""
    review_df["gold_label_class"] = ""
    review_df["review_status"] = "pending"
    review_df["reviewer"] = ""
    review_df["review_comment"] = ""

    meta_columns = [
        "case_id",
        "patient_id",
        "age",
        "sex",
        "renal_status",
        "conditions",
        "active_medications",
        "main_problem",
        "scenario_type",
        "candidate_sources",
        "original_medication",
        "original_class",
        "candidate",
        "candidate_class",
        "candidate_already_active",
        "candidate_has_problem_indication",
        "max_alert_severity",
        "relevant_max_alert_severity",
        "alert_rule_ids",
        "alert_origins",
        "alerts",
        "silver_reason",
        "silver_adequacy_score",
        "silver_label_class",
    ]

    audit_extra_columns = [
        "alert_count",
        "critical_alert_count",
        "high_alert_count",
        "moderate_alert_count",
        "low_alert_count",
        "renal_alert_count",
        "interaction_alert_count",
        "duplication_alert_count",
    ]

    ordered_columns = (
        meta_columns
        + REVIEW_GOLD_COLUMNS
        + audit_extra_columns
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    seen = set()
    unique_ordered_columns = []
    for column in ordered_columns:
        if column not in seen and column in review_df.columns:
            seen.add(column)
            unique_ordered_columns.append(column)

    return review_df[unique_ordered_columns]


def maybe_sample_per_class(
    df: pd.DataFrame,
    sample_per_class: int | None,
    random_state: int,
) -> pd.DataFrame:
    if sample_per_class is None:
        return df

    sampled_parts = []

    for _, group in df.groupby("label_class"):
        n = min(sample_per_class, len(group))
        sampled_parts.append(group.sample(n=n, random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


def generate_full_training_candidates(
    max_patients: int | None = None,
    adults_only: bool = False,
    with_active_medications: bool = True,
    include_unspecified: bool = False,
    random_candidates_per_scenario: int = 3,
    max_originals_per_patient: int = 3,
    sample_per_class: int | None = None,
    limit: int | None = None,
    random_state: int = 42,
) -> None:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Índice Synthea não encontrado: {INDEX_PATH}. "
            "Executa primeiro: python scripts/build_synthea_context_index.py"
        )

    kb = load_knowledge_base()
    synthea_index = pd.read_csv(INDEX_PATH)

    if adults_only:
        synthea_index = synthea_index[synthea_index["age"] >= 18].copy()

    if with_active_medications:
        synthea_index = synthea_index[
            synthea_index["active_medications"].apply(
                lambda value: len(parse_json_list(value)) > 0
            )
        ].copy()

    if not include_unspecified:
        synthea_index = synthea_index[
            synthea_index["main_problem_guess"].fillna("").astype(str).str.lower() != "unspecified"
        ].copy()

    if max_patients is not None:
        synthea_index = synthea_index.head(max_patients).copy()

    rows = []

    for _, raw_row in synthea_index.iterrows():
        patient = build_patient_context_from_row(raw_row)
        original_scenarios = choose_original_scenarios(
            patient=patient,
            kb=kb,
            max_originals_per_patient=max_originals_per_patient,
        )

        for scenario in original_scenarios:
            original_medication = scenario["original_medication"]
            original_prescription = [
                MedicationLine(
                    medication=original_medication,
                    dose=None,
                    frequency=None,
                    route=None,
                )
            ]

            original_alerts = run_safety_checks(
                patient=patient,
                prescription=original_prescription,
                kb=kb,
            )

            candidate_map = build_hybrid_candidate_set(
                patient=patient,
                original_medication=original_medication,
                kb=kb,
                random_candidates_per_scenario=random_candidates_per_scenario,
                seed=random_state,
            )

            for candidate, candidate_sources in candidate_map.items():
                candidate_prescription = [
                    MedicationLine(
                        medication=candidate,
                        dose=None,
                        frequency=None,
                        route=None,
                    )
                ]

                candidate_alerts = run_safety_checks(
                    patient=patient,
                    prescription=candidate_prescription,
                    kb=kb,
                )

                relevant_alerts = relevant_alerts_for_candidate(candidate_alerts)

                adequacy_score, label_class, silver_reason = build_silver_label(
                    patient=patient,
                    candidate=candidate,
                    original_medication=original_medication,
                    candidate_sources=candidate_sources,
                    original_alerts=original_alerts,
                    candidate_alerts=candidate_alerts,
                    kb=kb,
                )

                features = build_features(
                    candidate=candidate,
                    original_medication=original_medication,
                    patient=patient,
                    kb=kb,
                )

                row = {
                    "patient_id": patient.patient_id,
                    "sex": patient.sex,
                    "renal_status": patient.renal_status,
                    "conditions": json_dumps(patient.conditions),
                    "active_medications": json_dumps(patient.active_medications),
                    "scenario_type": scenario["scenario_type"],
                    "candidate_sources": "|".join(sorted(candidate_sources)),
                    "candidate_already_active": int(candidate in set(get_active_medication_ids(patient))),
                    "candidate_has_problem_indication": int(
                        medication_has_indication(candidate, patient.main_problem, kb)
                    ),
                    "max_alert_severity": max_alert_severity(candidate_alerts),
                    "relevant_max_alert_severity": max_alert_severity(relevant_alerts),
                    "alert_rule_ids": alert_rule_ids(candidate_alerts),
                    "alert_origins": alert_origins(candidate_alerts),
                    "alerts": alerts_to_text(candidate_alerts),
                    "silver_reason": silver_reason,
                    "adequacy_score": adequacy_score,
                    "label_class": label_class,
                    **alert_severity_counts(candidate_alerts),
                    **features,
                }

                rows.append(row)

    if not rows:
        raise RuntimeError("Nenhum candidato foi gerado. Revê filtros e dados Synthea.")

    full_df = pd.DataFrame(rows)

    full_df = full_df.drop_duplicates(
        subset=[
            "patient_id",
            "main_problem",
            "original_medication",
            "candidate",
        ],
        keep="first",
    ).reset_index(drop=True)

    if limit is not None:
        full_df = full_df.head(limit).copy()

    review_base_df = maybe_sample_per_class(
        df=full_df,
        sample_per_class=sample_per_class,
        random_state=random_state,
    )

    training_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [
        "adequacy_score",
        "label_class",
    ]

    training_df = full_df[training_columns].copy()
    audit_df = full_df.copy()
    review_df = build_review_columns(review_base_df)

    TRAINING_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    training_df.to_csv(TRAINING_OUTPUT_PATH, index=False, encoding="utf-8")
    training_df.to_csv(FULL_TRAINING_OUTPUT_PATH, index=False, encoding="utf-8")
    audit_df.to_csv(AUDIT_OUTPUT_PATH, index=False, encoding="utf-8")
    review_df.to_csv(REVIEW_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Geração concluída.")
    print()
    print(f"Dataset silver para treino guardado em: {TRAINING_OUTPUT_PATH}")
    print(f"Cópia completa guardada em: {FULL_TRAINING_OUTPUT_PATH}")
    print(f"Dataset de auditoria guardado em: {AUDIT_OUTPUT_PATH}")
    print(f"CSV para revisão humana/dev guardado em: {REVIEW_OUTPUT_PATH}")
    print()
    print("Pacientes usados:", synthea_index["patient_id"].nunique())
    print("Cenários/candidatos gerados:", len(full_df))
    print("Linhas no CSV de revisão:", len(review_df))
    print()
    print("Distribuição silver_label_class:")
    print(full_df["label_class"].value_counts().sort_index())
    print()
    print("Distribuição por main_problem:")
    print(full_df["main_problem"].value_counts().sort_index())
    print()
    print("Distribuição por scenario_type:")
    print(full_df["scenario_type"].value_counts().sort_index())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera candidatos de treino Synthea em modo híbrido: "
            "compatíveis com problema, alternativas ativas, probes de risco e diversidade aleatória."
        )
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Número máximo de utentes Synthea a usar. Por defeito usa todos após filtros.",
    )

    parser.add_argument(
        "--include-children",
        action="store_true",
        help="Inclui menores. Por defeito, usa apenas adultos.",
    )

    parser.add_argument(
        "--include-without-active-medications",
        action="store_true",
        help="Inclui utentes sem medicação ativa. Por defeito, usa apenas utentes com medicação ativa.",
    )

    parser.add_argument(
        "--include-unspecified",
        action="store_true",
        help="Inclui utentes com main_problem_guess='unspecified'. Por defeito, exclui.",
    )

    parser.add_argument(
        "--random-candidates-per-scenario",
        type=int,
        default=3,
        help="Número de candidatos aleatórios por cenário para diversidade.",
    )

    parser.add_argument(
        "--max-originals-per-patient",
        type=int,
        default=3,
        help="Número máximo de medicamentos originais simulados por utente.",
    )

    parser.add_argument(
        "--sample-per-class",
        type=int,
        default=None,
        help=(
            "Opcional: gera o CSV de revisão com amostra balanceada por classe. "
            "O dataset completo continua a ser guardado em training_examples_audit.csv."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Opcional: limita o número total de linhas geradas antes da amostragem.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    generate_full_training_candidates(
        max_patients=args.max_patients,
        adults_only=not args.include_children,
        with_active_medications=not args.include_without_active_medications,
        include_unspecified=args.include_unspecified,
        random_candidates_per_scenario=args.random_candidates_per_scenario,
        max_originals_per_patient=args.max_originals_per_patient,
        sample_per_class=args.sample_per_class,
        limit=args.limit,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
