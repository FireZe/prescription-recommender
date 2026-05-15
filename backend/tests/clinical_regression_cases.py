from __future__ import annotations

import re
from typing import Any


CLINICAL_CASES = [
    {
        "id": "T01",
        "name": "AINE + antiagregante",
        "payload": {
            "patient_id": "a71a4da3-c7b3-1c86-c972-a3ca0c4eac88",
            "main_problem": "dor",
            "prescription": [
                {
                    "medication": "ibuprofeno",
                    "dose": "400mg",
                    "frequency": "8/8h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "aine_antiagregante_hemorragia",
                "severity": "high",
                "origin": "prescription_related",
            }
        ],
        "expected_recommendations": ["paracetamol"],
        "forbidden_recommendations": ["naproxen", "naproxeno"],
        "expected_notes_contains": [],
    },
    {
        "id": "T02",
        "name": "AINE + antiagregante + compromisso renal grave",
        "payload": {
            "patient_id": "8e6a93c0-2e78-30d9-a564-51cb8a61dc32",
            "main_problem": "dor",
            "prescription": [
                {
                    "medication": "naproxeno",
                    "dose": "250mg",
                    "frequency": "12/12h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "renal_caution",
                "severity": "high",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_antiagregante_hemorragia",
                "severity": "high",
                "origin": "prescription_related",
            },
        ],
        "expected_recommendations": [],
        "forbidden_recommendations": ["paracetamol"],
        "expected_notes_contains": ["Paracetamol já consta da medicação ativa"],
    },
    {
        "id": "T03",
        "name": "AINE + IECA + duplicação de AINE",
        "payload": {
            "patient_id": "119aedc5-bbb8-b25e-45fc-fcb88fe90698",
            "main_problem": "inflamação",
            "prescription": [
                {
                    "medication": "ibuprofeno",
                    "dose": "400mg",
                    "frequency": "8/8h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "aine_aine_duplicacao",
                "severity": "high",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_ieca_risco_renal",
                "severity": "moderate",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_ieca_risco_renal",
                "severity": "moderate",
                "origin": "active_medication_existing",
            },
        ],
        "expected_recommendations": ["paracetamol"],
        "forbidden_recommendations": ["naproxen", "naproxeno"],
        "expected_notes_contains": [],
    },
    {
        "id": "T04",
        "name": "AINE + diurético tiazídico",
        "payload": {
            "patient_id": "75089c8b-a95c-2147-d330-18dddf28d5bb",
            "main_problem": "dor",
            "prescription": [
                {
                    "medication": "ibuprofeno",
                    "dose": "400mg",
                    "frequency": "8/8h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "aine_tiazida_risco_renal",
                "severity": "moderate",
                "origin": "prescription_related",
            }
        ],
        "expected_recommendations": ["paracetamol"],
        "forbidden_recommendations": ["naproxen", "naproxeno"],
        "expected_notes_contains": [],
    },
    {
        "id": "T05",
        "name": "Sinvastatina + claritromicina",
        "payload": {
            "patient_id": "70e4cf5c-85aa-8fc7-0bc1-b13c7b2f8567",
            "main_problem": "infeção",
            "prescription": [
                {
                    "medication": "claritromicina",
                    "dose": "500mg",
                    "frequency": "12/12h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "sinvastatina_claritromicina_contraindicada",
                "severity": "critical",
                "origin": "prescription_related",
            }
        ],
        "expected_recommendations": [],
        "forbidden_recommendations": ["azitromicina", "azithromycin"],
        "expected_notes_contains": [],
    },
    {
        "id": "T06",
        "name": "Sinvastatina + azitromicina",
        "payload": {
            "patient_id": "b01206ca-8306-5b75-d827-e3631625dab4",
            "main_problem": "infeção",
            "prescription": [
                {
                    "medication": "azitromicina",
                    "dose": "500mg",
                    "frequency": "1x/dia",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "estatina_macrolido_miopatia",
                "severity": "high",
                "origin": "prescription_related",
            }
        ],
        "expected_recommendations": [],
        "forbidden_recommendations": ["claritromicina", "clarithromycin"],
        "expected_notes_contains": [],
    },
    {
        "id": "T07",
        "name": "AINE + antiagregante + diurético + compromisso renal grave",
        "payload": {
            "patient_id": "217a7c06-2d5e-99aa-0c39-f6126ed5b266",
            "main_problem": "inflamação",
            "prescription": [
                {
                    "medication": "ibuprofeno",
                    "dose": "400mg",
                    "frequency": "8/8h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "renal_caution",
                "severity": "high",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_antiagregante_hemorragia",
                "severity": "high",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_tiazida_risco_renal",
                "severity": "moderate",
                "origin": "prescription_related",
            },
        ],
        "expected_recommendations": ["paracetamol"],
        "forbidden_recommendations": ["naproxen", "naproxeno"],
        "expected_notes_contains": [],
    },
    {
        "id": "T08",
        "name": "Triple whammy: AINE + ARA + diurético",
        "payload": {
            "patient_id": "1353b50a-5b6f-dcfe-683d-277a0f5bafc0",
            "main_problem": "inflamação",
            "prescription": [
                {
                    "medication": "ibuprofeno",
                    "dose": "400mg",
                    "frequency": "8/8h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "renal_caution",
                "severity": "high",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_diuretico_risco_renal",
                "severity": "moderate",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_ara_risco_renal",
                "severity": "moderate",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_aine_duplicacao",
                "severity": "high",
                "origin": "prescription_related",
            },
            {
                "rule_id": "triple_whammy",
                "severity": "high",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_diuretico_risco_renal",
                "severity": "moderate",
                "origin": "active_medication_existing",
            },
            {
                "rule_id": "aine_ara_risco_renal",
                "severity": "moderate",
                "origin": "active_medication_existing",
            },
        ],
        "expected_recommendations": ["paracetamol"],
        "forbidden_recommendations": ["naproxen", "naproxeno"],
        "expected_notes_contains": [],
    },
    {
        "id": "T09",
        "name": "AINE em compromisso renal grave sem antiagregante",
        "payload": {
            "patient_id": "78f968ff-8874-d84b-4933-0ae02723b7d5",
            "main_problem": "dor",
            "prescription": [
                {
                    "medication": "ibuprofeno",
                    "dose": "400mg",
                    "frequency": "8/8h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "renal_caution",
                "severity": "high",
                "origin": "prescription_related",
            }
        ],
        "expected_recommendations": [],
        "forbidden_recommendations": ["paracetamol"],
        "expected_notes_contains": ["Paracetamol já consta da medicação ativa"],
    },
    {
        "id": "T10",
        "name": "AINE + clopidogrel + AINE ativo",
        "payload": {
            "patient_id": "ca1ed690-165c-ea53-85c2-1f6d87823b17",
            "main_problem": "inflamação",
            "prescription": [
                {
                    "medication": "ibuprofeno",
                    "dose": "400mg",
                    "frequency": "8/8h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "aine_antiagregante_hemorragia",
                "severity": "high",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_aine_duplicacao",
                "severity": "high",
                "origin": "prescription_related",
            },
            {
                "rule_id": "aine_antiagregante_hemorragia",
                "severity": "high",
                "origin": "active_medication_existing",
            },
        ],
        "expected_recommendations": ["paracetamol"],
        "forbidden_recommendations": ["naproxen", "naproxeno"],
        "expected_notes_contains": [],
    },
]


FORBIDDEN_LLM_PATTERNS = [
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]",
    r"\bfoi validada clinicamente\b",
    r"\bvalidada clinicamente\b",
    r"\bsegurança clínica confirmada\b",
    r"\bsem risco\b",
    r"\bsem interação\b",
    r"\bnão apresenta interação\b",
]


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def get_recommendation_names(result: dict[str, Any]) -> list[str]:
    return [
        normalize_text(item.get("medication"))
        for item in result.get("recommendations", [])
    ]


def get_recommendation_notes_text(result: dict[str, Any]) -> str:
    notes = result.get("recommendation_notes") or []
    return "\n".join(
        str(note.get("description") or note.get("reason") or note)
        for note in notes
    )


def has_expected_alert(
    alerts: list[dict[str, Any]],
    *,
    rule_id: str,
    severity: str | None = None,
    origin: str | None = None,
) -> bool:
    for alert in alerts:
        if alert.get("rule_id") != rule_id:
            continue

        if severity and alert.get("severity") != severity:
            continue

        if origin and alert.get("origin") != origin:
            continue

        return True

    return False


def validate_analysis_result(case: dict[str, Any], result: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    alerts = result.get("alerts", [])
    recommendations = get_recommendation_names(result)
    notes_text = get_recommendation_notes_text(result)

    for expected in case.get("expected_rules", []):
        if not has_expected_alert(
            alerts,
            rule_id=expected["rule_id"],
            severity=expected.get("severity"),
            origin=expected.get("origin"),
        ):
            failures.append(
                "Alerta esperado não encontrado: "
                f"rule_id={expected['rule_id']}, "
                f"severity={expected.get('severity')}, "
                f"origin={expected.get('origin')}"
            )

    for expected_rec in case.get("expected_recommendations", []):
        if normalize_text(expected_rec) not in recommendations:
            failures.append(f"Recomendação esperada não encontrada: {expected_rec}")

    for forbidden_rec in case.get("forbidden_recommendations", []):
        if normalize_text(forbidden_rec) in recommendations:
            failures.append(f"Recomendação proibida encontrada: {forbidden_rec}")

    for expected_note in case.get("expected_notes_contains", []):
        if normalize_text(expected_note) not in normalize_text(notes_text):
            failures.append(f"Nota esperada não encontrada: {expected_note}")

    return failures


def validate_llm_explanation(text: str) -> list[str]:
    failures: list[str] = []

    if len(text.strip()) < 80:
        failures.append("Explicação LLM demasiado curta ou vazia.")

    required_sections = [
        "problema identificado",
        "motivo do alerta",
        "motivo da recomendação",
        "limitações",
    ]

    lower_text = text.lower()

    for section in required_sections:
        if section not in lower_text:
            failures.append(f"Secção ausente na explicação LLM: {section}")

    for pattern in FORBIDDEN_LLM_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            failures.append(f"Expressão proibida encontrada no LLM: {pattern}")

    return failures