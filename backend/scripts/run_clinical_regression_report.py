from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx


BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

from tests.clinical_regression_cases import (  # noqa: E402
    CLINICAL_CASES,
    validate_analysis_result,
    validate_llm_explanation,
)


BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
REPORTS_DIR = BACKEND_ROOT / "reports"


def format_alert(alert: dict[str, Any]) -> str:
    return (
        f"- [{alert.get('origin', 'unknown')}] "
        f"{alert.get('medication', '—')} | "
        f"severity={alert.get('severity', '—')} | "
        f"type={alert.get('type', '—')} | "
        f"rule_id={alert.get('rule_id', '—')}\n"
        f"  {alert.get('description', '')}"
    )


def format_recommendation(rec: dict[str, Any]) -> str:
    reasons = rec.get("reasons", [])
    reasons_text = "\n".join(f"  - {reason}" for reason in reasons) if reasons else "  - Sem razões clínicas."

    status = (
        rec.get("recommendation_status")
        or rec.get("status")
        or rec.get("label")
        or "—"
    )

    return (
        f"- {rec.get('medication', '—')} | "
        f"score_final={rec.get('score_final', '—')} | "
        f"status={status}\n"
        f"{reasons_text}"
    )


def format_notes(result: dict[str, Any]) -> str:
    notes = result.get("recommendation_notes") or []

    if not notes:
        return "Sem notas adicionais."

    lines = []

    for note in notes:
        medication = note.get("medication", "—")
        description = note.get("description") or note.get("reason") or str(note)
        lines.append(f"- {medication}: {description}")

    return "\n".join(lines)


def call_analysis(case: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(
        f"{BASE_URL}/analyze/synthea",
        json=case["payload"],
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def call_llm_explanation(analysis_id: str) -> dict[str, Any]:
    response = httpx.post(
        f"{BASE_URL}/explain/llm",
        json={
            "analysis_id": analysis_id,
            "user_question": "Explica de forma objetiva os alertas e recomendações deste caso.",
        },
        timeout=240.0,
    )
    response.raise_for_status()
    return response.json()


def run_report(with_llm: bool) -> tuple[str, dict[str, Any]]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    raw_results: dict[str, Any] = {
        "generated_at": now,
        "base_url": BASE_URL,
        "with_llm": with_llm,
        "cases": [],
    }

    total = len(CLINICAL_CASES)
    passed = 0

    lines.append("RELATÓRIO DE TESTES CLÍNICOS DO PROTÓTIPO")
    lines.append("=" * 70)
    lines.append(f"Gerado em: {now}")
    lines.append(f"API: {BASE_URL}")
    lines.append(f"LLM incluído: {'sim' if with_llm else 'não'}")
    lines.append("")

    for case in CLINICAL_CASES:
        case_id = case["id"]
        case_name = case["name"]

        lines.append("=" * 70)
        lines.append(f"{case_id} — {case_name}")
        lines.append("=" * 70)
        lines.append("Pedido:")
        lines.append(json.dumps(case["payload"], ensure_ascii=False, indent=2))
        lines.append("")

        case_result: dict[str, Any] = {
            "id": case_id,
            "name": case_name,
            "payload": case["payload"],
            "analysis": None,
            "llm": None,
            "failures": [],
        }

        try:
            analysis = call_analysis(case)
            case_result["analysis"] = analysis

            analysis_failures = validate_analysis_result(case, analysis)
            case_result["failures"].extend(analysis_failures)

            lines.append(f"Analysis ID: {analysis.get('analysis_id', '—')}")
            lines.append("")

            lines.append("Alertas:")
            alerts = analysis.get("alerts", [])

            if alerts:
                for alert in alerts:
                    lines.append(format_alert(alert))
            else:
                lines.append("Sem alertas identificados.")

            lines.append("")
            lines.append("Recomendações:")
            recommendations = analysis.get("recommendations", [])

            if recommendations:
                for rec in recommendations:
                    lines.append(format_recommendation(rec))
            else:
                lines.append("Sem recomendações.")

            lines.append("")
            lines.append("Notas sobre recomendações:")
            lines.append(format_notes(analysis))
            lines.append("")

            if with_llm:
                analysis_id = analysis.get("analysis_id")

                if not analysis_id:
                    failure = "Não foi devolvido analysis_id; não foi possível chamar o LLM."
                    case_result["failures"].append(failure)
                    lines.append(f"Explicação LLM: {failure}")
                else:
                    llm_result = call_llm_explanation(analysis_id)
                    case_result["llm"] = llm_result

                    explanation = llm_result.get("explanation", "")
                    llm_failures = validate_llm_explanation(explanation)
                    case_result["failures"].extend(llm_failures)

                    lines.append("Explicação LLM:")
                    lines.append(f"Modelo: {llm_result.get('model', '—')}")
                    lines.append(explanation)
                    lines.append("")

            if case_result["failures"]:
                lines.append("Estado: FALHOU")
                lines.append("Falhas:")
                for failure in case_result["failures"]:
                    lines.append(f"- {failure}")
            else:
                lines.append("Estado: OK")
                passed += 1

        except Exception as exc:
            failure = f"Erro ao executar o caso: {exc}"
            case_result["failures"].append(failure)
            lines.append("Estado: ERRO")
            lines.append(failure)

        raw_results["cases"].append(case_result)
        lines.append("")

    lines.append("=" * 70)
    lines.append("RESUMO")
    lines.append("=" * 70)
    lines.append(f"Casos OK: {passed}/{total}")
    lines.append(f"Casos com falha/erro: {total - passed}/{total}")

    return "\n".join(lines), raw_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Inclui chamada ao endpoint /explain/llm para cada caso.",
    )
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = REPORTS_DIR / f"clinical_regression_report_{timestamp}.txt"
    json_path = REPORTS_DIR / f"clinical_regression_report_{timestamp}.json"

    report_text, raw_results = run_report(with_llm=args.with_llm)

    txt_path.write_text(report_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(raw_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Relatório TXT guardado em: {txt_path}")
    print(f"Relatório JSON guardado em: {json_path}")

    failed_cases = [
        case for case in raw_results["cases"]
        if case.get("failures")
    ]

    if failed_cases:
        print(f"Resultado: {len(failed_cases)} caso(s) com falha.")
        sys.exit(1)

    print("Resultado: todos os casos passaram.")


if __name__ == "__main__":
    main()