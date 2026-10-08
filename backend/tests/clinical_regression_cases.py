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
    {
        "id": "T11",
        "name": "Colchicina + claritromicina: alerta crítico específico suprime o genérico",
        "payload": {
            "patient_id": "119aedc5-bbb8-b25e-45fc-fcb88fe90698",
            "main_problem": "gota",
            "prescription": [
                {
                    "medication": "colchicina",
                    "dose": "0.5mg",
                    "frequency": "12/12h",
                    "route": "oral",
                },
                {
                    "medication": "claritromicina",
                    "dose": "500mg",
                    "frequency": "12/12h",
                    "route": "oral",
                },
            ],
        },
        "expected_rules": [
            {
                "rule_id": "colchicina_claritromicina_toxicidade",
                "severity": "critical",
                "origin": "prescription_related",
            }
        ],
        # A regra de classe cobre o mesmo par e é menos específica: o motor
        # emite um único alerta por par, pelo que esta tem de ser suprimida.
        "forbidden_rules": [
            {"rule_id": "colchicina_macrolido_toxicidade"},
        ],
        "expected_recommendations": [],
        "forbidden_recommendations": ["clarithromycin", "claritromicina"],
        "expected_notes_contains": [],
    },
    {
        "id": "T12",
        "name": "Fluoxetina + metoprolol (contraindicação por inibição do CYP2D6)",
        "payload": {
            "patient_id": "70e4cf5c-85aa-8fc7-0bc1-b13c7b2f8567",
            "main_problem": "depressão",
            "prescription": [
                {
                    "medication": "fluoxetina",
                    "dose": "20mg",
                    "frequency": "24/24h",
                    "route": "oral",
                },
                {
                    "medication": "metoprolol",
                    "dose": "50mg",
                    "frequency": "12/12h",
                    "route": "oral",
                },
            ],
        },
        "expected_rules": [
            {
                "rule_id": "fluoxetina_metoprolol_contraindicada",
                "severity": "critical",
                "origin": "prescription_related",
            }
        ],
        "expected_recommendations": [],
        "forbidden_recommendations": ["fluoxetine", "fluoxetina"],
        "expected_notes_contains": [],
    },
    {
        "id": "T13",
        "name": "Omeprazol com clopidogrel ativo (substituição por pantoprazol)",
        "payload": {
            "patient_id": "ca1ed690-165c-ea53-85c2-1f6d87823b17",
            "main_problem": "úlcera",
            "prescription": [
                {
                    "medication": "omeprazol",
                    "dose": "20mg",
                    "frequency": "24/24h",
                    "route": "oral",
                }
            ],
        },
        "expected_rules": [
            {
                "rule_id": "ibp_clopidogrel_eficacia",
                "severity": "moderate",
                "origin": "prescription_related",
            }
        ],
        "expected_recommendations": ["pantoprazole"],
        "forbidden_recommendations": ["omeprazole", "omeprazol"],
        "expected_notes_contains": [],
    },
        {
        "id": "T14",
        "name": "Colchicina + azitromicina (regra de classe macrólido-colchicina)",
        "payload": {
            "patient_id": "119aedc5-bbb8-b25e-45fc-fcb88fe90698",
            "main_problem": "gota",
            "prescription": [
                {
                    "medication": "colchicina",
                    "dose": "0.5mg",
                    "frequency": "12/12h",
                    "route": "oral",
                },
                {
                    "medication": "azitromicina",
                    "dose": "500mg",
                    "frequency": "1x/dia",
                    "route": "oral",
                },
            ],
        },
        "expected_rules": [
            {
                "rule_id": "colchicina_macrolido_toxicidade",
                "severity": "high",
                "origin": "prescription_related",
            }
        ],
        "expected_recommendations": [],
        "forbidden_recommendations": [],
        "expected_notes_contains": [],
    },
]


FORBIDDEN_LLM_PATTERNS = [
    # Alfabetos não latinos: grego, cirílico, hebraico, árabe, devanágari,
    # tailandês, japonês, chinês, coreano.
    r"[\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF]",

    # Frases clinicamente proibidas
    r"\bfoi validada clinicamente\b",
    r"\bvalidada clinicamente\b",
    r"\bsegurança clínica confirmada\b",
    r"\bsem risco\b",
    r"\bsem interação\b",
    r"\bnão apresenta interação\b",

    # Inglês explícito
    r"\bthe patient\b",
    r"\bpatient\b",
    r"\bprescribed\b",
    r"\brecommendation\b",
    r"\bactive medication\b",
    r"\bbleeding risk\b",
]

def case_has_mixed_alert_origins(case: dict[str, Any]) -> bool:
    origins = {
        expected.get("origin")
        for expected in case.get("expected_rules", [])
    }

    return (
        "prescription_related" in origins
        and "active_medication_existing" in origins
    )

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

    for forbidden in case.get("forbidden_rules", []):
        if has_expected_alert(
            alerts,
            rule_id=forbidden["rule_id"],
            severity=forbidden.get("severity"),
            origin=forbidden.get("origin"),
        ):
            failures.append(
                "Alerta redundante presente — deveria ter sido suprimido pela "
                "desduplicação por par de fármacos: "
                f"rule_id={forbidden['rule_id']}"
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


def validate_llm_explanation(
    text: str,
    case: dict[str, Any] | None = None,
) -> list[str]:
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

    if case and case_has_mixed_alert_origins(case):
        has_prescription_reference = (
            "prescrição submetida" in lower_text
            or "relacionado com a prescrição" in lower_text
            or "relacionados com a prescrição" in lower_text
        )

        has_existing_reference = (
            "pré-existente" in lower_text
            or "pre-existente" in lower_text
            or "medicação ativa" in lower_text
        )

        if not has_prescription_reference or not has_existing_reference:
            failures.append(
                "A explicação LLM não separa claramente alertas relacionados "
                "com a prescrição submetida e alertas pré-existentes na medicação ativa."
            )

    return failures