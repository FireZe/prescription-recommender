from __future__ import annotations

import os

import httpx
import pytest

from tests.clinical_regression_cases import (
    CLINICAL_CASES,
    validate_analysis_result,
    validate_llm_explanation,
)


BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


@pytest.mark.parametrize("case", CLINICAL_CASES, ids=[case["id"] for case in CLINICAL_CASES])
def test_clinical_scenario_analysis(case: dict) -> None:
    response = httpx.post(
        f"{BASE_URL}/analyze/synthea",
        json=case["payload"],
        timeout=60.0,
    )

    assert response.status_code == 200, response.text

    result = response.json()
    failures = validate_analysis_result(case, result)

    assert failures == []


@pytest.mark.parametrize("case", CLINICAL_CASES, ids=[case["id"] for case in CLINICAL_CASES])
def test_clinical_scenario_llm_explanation_structure(case: dict) -> None:
    analysis_response = httpx.post(
        f"{BASE_URL}/analyze/synthea",
        json=case["payload"],
        timeout=60.0,
    )

    assert analysis_response.status_code == 200, analysis_response.text

    analysis_result = analysis_response.json()
    analysis_id = analysis_result.get("analysis_id")

    assert analysis_id, "A resposta da análise não devolveu analysis_id."

    llm_response = httpx.post(
        f"{BASE_URL}/explain/llm",
        json={
            "analysis_id": analysis_id,
            "user_question": "Explica de forma objetiva os alertas e recomendações deste caso.",
        },
        timeout=240.0,
    )

    assert llm_response.status_code == 200, llm_response.text

    llm_result = llm_response.json()
    explanation = llm_result.get("explanation", "")

    failures = validate_llm_explanation(explanation)

    assert failures == []