"""Gestão do conhecimento clínico (RF08).

Permite ao profissional de saúde inspecionar e corrigir o conhecimento aplicado
pelo sistema sem alterar diretamente o ficheiro knowledge_base.json. As alterações
ficam numa camada de sobreposição (knowledge_base_extensions.json) e só são aceites
se a suite de regressão clínica continuar a passar na totalidade.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
EXTENSIONS_PATH = DATA_DIR / "knowledge_base_extensions.json"
PROVENANCE_DIR = DATA_DIR / "diretrizes_pdf"

EXTENSIONS_SCHEMA_VERSION = 1

VALID_SEVERITIES = {"low", "moderate", "high", "critical"}
VALID_MATCH_TYPES = {"drug_drug", "class_drug", "class_class", "qt_qt"}

# Campos de gestão que não devem chegar ao motor de segurança.
ADMIN_METADATA_FIELDS = {
    "fonte",
    "documento",
    "autor",
    "created_at",
    "justificacao",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_extensions() -> dict[str, Any]:
    return {
        "schema_version": EXTENSIONS_SCHEMA_VERSION,
        "updated_at": None,
        "added_rules": [],
        "modified_rules": {},
        "disabled_rules": {},
        "medication_overrides": {},
    }


def load_base_knowledge_base() -> dict[str, Any]:
    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_extensions() -> dict[str, Any]:
    if not EXTENSIONS_PATH.exists():
        return empty_extensions()

    with EXTENSIONS_PATH.open("r", encoding="utf-8") as file:
        stored = json.load(file)

    extensions = empty_extensions()

    for key in extensions:
        if key in stored:
            extensions[key] = stored[key]

    return extensions


def save_extensions(extensions: dict[str, Any]) -> None:
    extensions["schema_version"] = EXTENSIONS_SCHEMA_VERSION
    extensions["updated_at"] = utc_now_iso()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTENSIONS_PATH.exists():
        backup = EXTENSIONS_PATH.with_name("knowledge_base_extensions.json.bak")
        backup.write_text(
            EXTENSIONS_PATH.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    temp = EXTENSIONS_PATH.with_name("knowledge_base_extensions.json.tmp")

    with temp.open("w", encoding="utf-8") as file:
        json.dump(extensions, file, ensure_ascii=False, indent=2)

    temp.replace(EXTENSIONS_PATH)


def apply_extensions(
    kb: dict[str, Any],
    extensions: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Devolve uma cópia da base de conhecimento com a camada de gestão aplicada."""
    if extensions is None:
        extensions = load_extensions()

    patched = copy.deepcopy(kb)

    disabled = extensions.get("disabled_rules", {}) or {}
    modified = extensions.get("modified_rules", {}) or {}

    rules: list[dict[str, Any]] = []

    for rule in patched.get("interaction_rules", []):
        rule_id = rule.get("id")

        if rule_id in disabled:
            continue

        override = modified.get(rule_id)

        if override:
            for field in ("severity", "description"):
                if override.get(field):
                    rule[field] = override[field]

            rule["origem"] = "regra_base_modificada"

        rules.append(rule)

    for added in extensions.get("added_rules", []) or []:
        if added.get("id") in disabled:
            continue

        rule = {
            key: value
            for key, value in added.items()
            if key not in ADMIN_METADATA_FIELDS
        }
        rule["origem"] = "regra_adicionada"
        rules.append(rule)

    patched["interaction_rules"] = rules

    medications = patched.setdefault("medications", {})

    for med_id, override in (extensions.get("medication_overrides", {}) or {}).items():
        medication = medications.get(med_id)

        if medication is None:
            continue

        conditions = medication.setdefault("contraindicated_conditions", [])

        for condition in override.get("contraindicated_conditions_add", []) or []:
            if condition not in conditions:
                conditions.append(condition)

        for condition in override.get("contraindicated_conditions_remove", []) or []:
            if condition in conditions:
                conditions.remove(condition)

        if "renal_caution" in override:
            medication["renal_caution"] = bool(override["renal_caution"])

        if "renal_alert_severity" in override:
            medication["renal_alert_severity"] = override["renal_alert_severity"]

        if "qt_risk" in override:
            medication["qt_risk"] = bool(override["qt_risk"])

    return patched


MAX_PROVENANCE_BYTES = 20 * 1024 * 1024

ALLOWED_DOCUMENT_CHARACTERS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,_-+()"
)


def sanitize_document_name(raw_name: str) -> str:
    """Reduz o nome do ficheiro ao nome base e a caracteres seguros."""
    name = Path(str(raw_name or "")).name.strip()
    name = "".join(
        character if character in ALLOWED_DOCUMENT_CHARACTERS else "_"
        for character in name
    )

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("._ ")


def provenance_document_path(document_name: str) -> Optional[Path]:
    """Caminho do documento, ou None se o nome sair da pasta de proveniência."""
    safe_name = sanitize_document_name(document_name)

    if not safe_name:
        return None

    path = (PROVENANCE_DIR / safe_name).resolve()

    if PROVENANCE_DIR.resolve() not in path.parents:
        return None

    return path


def provenance_document_exists(document_name: str) -> bool:
    path = provenance_document_path(document_name)

    return bool(path and path.is_file())


def list_provenance_documents() -> list[dict[str, Any]]:
    if not PROVENANCE_DIR.exists():
        return []

    documents = []

    for path in sorted(PROVENANCE_DIR.glob("*.pdf")):
        documents.append(
            {
                "documento": path.name,
                "size_bytes": path.stat().st_size,
                "updated_at": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )

    return documents


def store_provenance_document(filename: str, content: bytes) -> dict[str, Any]:
    """Guarda um PDF de proveniência sem substituir documentos já existentes."""
    safe_name = sanitize_document_name(filename)

    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Só são aceites documentos em formato PDF.")

    if not content:
        raise ValueError("O ficheiro submetido está vazio.")

    if len(content) > MAX_PROVENANCE_BYTES:
        raise ValueError(
            f"O ficheiro excede o limite de {MAX_PROVENANCE_BYTES // (1024 * 1024)} MB."
        )

    if not content.startswith(b"%PDF"):
        raise ValueError("O conteúdo do ficheiro não corresponde a um PDF válido.")

    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)

    stem = safe_name[:-4]
    candidate = PROVENANCE_DIR / safe_name
    counter = 1

    while candidate.exists():
        candidate = PROVENANCE_DIR / f"{stem}_{counter}.pdf"
        counter += 1

    candidate.write_bytes(content)

    return {
        "documento": candidate.name,
        "size_bytes": len(content),
        "substituiu_existente": False,
    }


def known_therapeutic_classes(kb: dict[str, Any]) -> set[str]:
    return {
        medication.get("therapeutic_class")
        for medication in kb.get("medications", {}).values()
        if medication.get("therapeutic_class")
    }


def known_conditions(kb: dict[str, Any]) -> set[str]:
    conditions: set[str] = set()

    for medication in kb.get("medications", {}).values():
        conditions.update(medication.get("contraindicated_conditions", []) or [])

    return conditions


def existing_rule_ids(kb: dict[str, Any], extensions: dict[str, Any]) -> set[str]:
    ids = {rule.get("id") for rule in kb.get("interaction_rules", [])}
    ids.update(rule.get("id") for rule in extensions.get("added_rules", []) or [])
    return {rule_id for rule_id in ids if rule_id}


def validate_rule_payload(
    rule: dict[str, Any],
    kb: dict[str, Any],
    taken_ids: set[str],
) -> list[str]:
    """Validação estrutural e referencial de uma regra antes da porta de regressão."""
    errors: list[str] = []

    rule_id = str(rule.get("id") or "").strip()

    if not rule_id:
        errors.append("A regra tem de ter um identificador (id).")
    elif rule_id in taken_ids:
        errors.append(f"Já existe uma regra com o identificador '{rule_id}'.")
    elif not rule_id.replace("_", "").isalnum():
        errors.append("O identificador só pode conter letras, dígitos e underscore.")

    match_type = rule.get("match")

    if match_type not in VALID_MATCH_TYPES:
        errors.append(
            "O tipo de correspondência tem de ser um de: "
            + ", ".join(sorted(VALID_MATCH_TYPES))
            + "."
        )

    if rule.get("severity") not in VALID_SEVERITIES:
        errors.append(
            "A severidade tem de ser uma de: " + ", ".join(sorted(VALID_SEVERITIES)) + "."
        )

    description = str(rule.get("description") or "").strip()

    if len(description) < 20:
        errors.append("A descrição clínica tem de ter pelo menos 20 caracteres.")

    if not str(rule.get("fonte") or "").strip():
        errors.append("É obrigatório indicar a fonte documental da regra.")

    documento = str(rule.get("documento") or "").strip()

    if documento and not provenance_document_exists(documento):
        errors.append(
            f"O documento de proveniência '{documento}' não se encontra no repositório. "
            "Carregue o ficheiro antes de o associar à regra."
        )

    medications = kb.get("medications", {})
    classes = known_therapeutic_classes(kb)

    def check_medication(field: str) -> Optional[str]:
        value = rule.get(field)

        if not value:
            errors.append(f"O campo '{field}' é obrigatório para este tipo de regra.")
            return None

        if value not in medications:
            errors.append(
                f"O medicamento '{value}' não existe na base de conhecimento."
            )
            return None

        return value

    def check_class(field: str) -> Optional[str]:
        value = rule.get(field)

        if not value:
            errors.append(f"O campo '{field}' é obrigatório para este tipo de regra.")
            return None

        if value not in classes:
            errors.append(
                f"A classe terapêutica '{value}' não existe na base de conhecimento."
            )
            return None

        return value

    if match_type == "drug_drug":
        med_a = check_medication("medication_a")
        med_b = check_medication("medication_b")

        if med_a and med_b and med_a == med_b:
            errors.append("Uma regra medicamento-medicamento exige dois fármacos distintos.")

    elif match_type == "class_class":
        class_a = check_class("class_a")
        class_b = check_class("class_b")

        if class_a and class_b and class_a == class_b:
            errors.append(
                "Uma regra classe-classe entre a mesma classe corresponde a duplicação "
                "terapêutica e deve usar o mecanismo próprio."
            )

    elif match_type == "class_drug":
        check_class("class_a")
        check_medication("medication_b")

    return errors


def validate_medication_override(
    med_id: str,
    override: dict[str, Any],
    kb: dict[str, Any],
) -> list[str]:
    errors: list[str] = []

    if med_id not in kb.get("medications", {}):
        errors.append(f"O medicamento '{med_id}' não existe na base de conhecimento.")
        return errors

    conditions = known_conditions(kb)

    for condition in override.get("contraindicated_conditions_add", []) or []:
        if condition not in conditions:
            errors.append(
                f"A condição clínica '{condition}' não é reconhecida pelo sistema. "
                "Use um identificador já existente na base de conhecimento."
            )

    severity = override.get("renal_alert_severity")

    if severity is not None and severity not in VALID_SEVERITIES:
        errors.append(
            "A severidade do alerta renal tem de ser uma de: "
            + ", ".join(sorted(VALID_SEVERITIES))
            + "."
        )

    if not str(override.get("fonte") or "").strip():
        errors.append("É obrigatório indicar a fonte documental da alteração.")

    documento = str(override.get("documento") or "").strip()

    if documento and not provenance_document_exists(documento):
        errors.append(
            f"O documento de proveniência '{documento}' não se encontra no repositório. "
            "Carregue o ficheiro antes de o associar à alteração."
        )

    return errors


def as_plain_dict(item: Any) -> dict[str, Any]:
    """Serializa modelos Pydantic e dicionários com a mesma interface.

    As notas de recomendação são produzidas como dicionários, ao contrário dos
    alertas e das recomendações, que são modelos Pydantic.
    """
    if hasattr(item, "model_dump"):
        return item.model_dump()

    if hasattr(item, "dict"):
        return item.dict()

    return item


def run_regression_gate(candidate_kb: dict[str, Any]) -> dict[str, Any]:
    """Executa a suite de regressão clínica sobre uma base de conhecimento candidata.

    Nenhuma alteração ao conhecimento é persistida se algum cenário deixar de passar.
    """
    from app.data_loader import load_historical_patterns
    from app.normalization import normalize_main_problem
    from app.recommender import build_recommendation_notes, recommend_alternatives
    from app.rules_engine import run_safety_checks
    from app.schemas import MedicationLine
    from app.synthea_loader import get_synthea_patient_context
    from tests.clinical_regression_cases import (
        CLINICAL_CASES,
        validate_analysis_result,
    )

    historical_patterns = load_historical_patterns()
    cases: list[dict[str, Any]] = []

    for case in CLINICAL_CASES:
        payload = case["payload"]

        try:
            patient = get_synthea_patient_context(
                patient_id=payload["patient_id"],
                main_problem=payload.get("main_problem"),
            )
            patient = patient.model_copy(
                update={"main_problem": normalize_main_problem(patient.main_problem)}
            )

            prescription = [MedicationLine(**line) for line in payload["prescription"]]

            alerts = run_safety_checks(
                patient=patient,
                prescription=prescription,
                kb=candidate_kb,
            )

            recommendations = recommend_alternatives(
                patient=patient,
                prescription=prescription,
                alerts=alerts,
                kb=candidate_kb,
                historical_patterns=historical_patterns,
            )

            notes = build_recommendation_notes(
                patient=patient,
                recommendations=recommendations,
                alerts=alerts,
                kb=candidate_kb,
            )

            result = {
                "alerts": [as_plain_dict(alert) for alert in alerts],
                "recommendations": [as_plain_dict(item) for item in recommendations],
                "recommendation_notes": [as_plain_dict(note) for note in notes],
            }

            failures = validate_analysis_result(case, result)

        except Exception as error:  # noqa: BLE001
            failures = [f"Erro ao executar o cenário: {error}"]

        cases.append(
            {
                "id": case["id"],
                "name": case["name"],
                "passed": not failures,
                "failures": failures,
            }
        )

    failed = [item for item in cases if not item["passed"]]

    return {
        "total": len(cases),
        "failed": len(failed),
        "passed": not failed,
        "cases": cases,
    }


def evaluate_candidate_extensions(extensions: dict[str, Any]) -> dict[str, Any]:
    """Aplica as extensões candidatas e corre a porta de regressão."""
    candidate_kb = apply_extensions(load_base_knowledge_base(), extensions)
    report = run_regression_gate(candidate_kb)

    report["rules_total"] = len(candidate_kb.get("interaction_rules", []))
    report["medications_total"] = len(candidate_kb.get("medications", {}))

    return report


def summarize_extensions(extensions: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if extensions is None:
        extensions = load_extensions()

    return {
        "schema_version": extensions.get("schema_version"),
        "updated_at": extensions.get("updated_at"),
        "added_rules": len(extensions.get("added_rules", []) or []),
        "modified_rules": len(extensions.get("modified_rules", {}) or {}),
        "disabled_rules": len(extensions.get("disabled_rules", {}) or {}),
        "medication_overrides": len(extensions.get("medication_overrides", {}) or {}),
    }



def empty_report() -> dict[str, Any]:
    return {
        "passed": False,
        "total": 0,
        "failed": 0,
        "cases": [],
        "rules_total": 0,
        "medications_total": 0,
    }


def rejection(
    message: str,
    errors: Optional[list[str]] = None,
    report: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "applied": False,
        "change_id": None,
        "message": message,
        "errors": errors or [],
        "validation": report or empty_report(),
    }


def commit_extension_change(
    change_type: str,
    target_id: str,
    mutate: Callable[[dict[str, Any]], None],
    payload: dict[str, Any],
    author: Optional[str] = None,
    justification: Optional[str] = None,
    source_reference: Optional[str] = None,
    source_document: Optional[str] = None,
) -> dict[str, Any]:
    """Aplica uma alteração candidata e só a persiste se a regressão clínica passar."""
    from app.database import save_kb_change

    candidate = copy.deepcopy(load_extensions())
    mutate(candidate)

    report = evaluate_candidate_extensions(candidate)

    if not report["passed"]:
        failed_ids = ", ".join(
            case["id"] for case in report["cases"] if not case["passed"]
        )

        return rejection(
            message=(
                "A alteração não foi gravada porque faria falhar a validação de segurança "
                f"do sistema nos seguintes cenários clínicos: {failed_ids}."
            ),
            report=report,
        )

    save_extensions(candidate)

    change_id = str(uuid4())

    save_kb_change(
        change_id=change_id,
        change_type=change_type,
        target_id=target_id,
        payload=payload,
        validation=report,
        author=author,
        justification=justification,
        source_reference=source_reference,
        source_document=source_document,
    )

    return {
        "applied": True,
        "change_id": change_id,
        "message": (
            "Alteração gravada. Os "
            f"{report['total']} cenários de regressão clínica continuam a passar."
        ),
        "errors": [],
        "validation": report,
    }


def add_rule(rule: dict[str, Any], author: Optional[str] = None) -> dict[str, Any]:
    kb = load_base_knowledge_base()
    extensions = load_extensions()

    errors = validate_rule_payload(rule, kb, existing_rule_ids(kb, extensions))

    if errors:
        return rejection(
            message="A regra não passou a validação estrutural e não foi gravada.",
            errors=errors,
        )

    entry = {key: value for key, value in rule.items() if value is not None}
    entry["autor"] = author or entry.get("autor")
    entry["created_at"] = utc_now_iso()

    def mutate(candidate: dict[str, Any]) -> None:
        candidate["added_rules"].append(entry)

    return commit_extension_change(
        change_type="rule_added",
        target_id=entry["id"],
        mutate=mutate,
        payload=entry,
        author=entry.get("autor"),
        source_reference=entry.get("fonte"),
        source_document=entry.get("documento"),
    )


def modify_rule(
    rule_id: str,
    changes: dict[str, Any],
    author: Optional[str] = None,
) -> dict[str, Any]:
    kb = load_base_knowledge_base()
    extensions = load_extensions()

    base_ids = {rule.get("id") for rule in kb.get("interaction_rules", [])}
    added_ids = {rule.get("id") for rule in extensions.get("added_rules", []) or []}

    if rule_id not in base_ids and rule_id not in added_ids:
        return rejection(
            message=f"A regra '{rule_id}' não existe na base de conhecimento.",
            errors=["Identificador de regra desconhecido."],
        )

    severity = changes.get("severity")
    description = changes.get("description")
    errors: list[str] = []

    if severity is None and description is None:
        errors.append("Indique pelo menos a severidade ou a descrição a alterar.")

    if severity is not None and severity not in VALID_SEVERITIES:
        errors.append(
            "A severidade tem de ser uma de: " + ", ".join(sorted(VALID_SEVERITIES)) + "."
        )

    if description is not None and len(description.strip()) < 20:
        errors.append("A descrição clínica tem de ter pelo menos 20 caracteres.")

    if not str(changes.get("fonte") or "").strip():
        errors.append("É obrigatório indicar a fonte documental da alteração.")

    if errors:
        return rejection(
            message="A alteração não passou a validação estrutural e não foi gravada.",
            errors=errors,
        )

    def mutate(candidate: dict[str, Any]) -> None:
        if rule_id in added_ids:
            for entry in candidate["added_rules"]:
                if entry.get("id") != rule_id:
                    continue

                if severity is not None:
                    entry["severity"] = severity

                if description is not None:
                    entry["description"] = description

                entry["fonte"] = changes.get("fonte")
                entry["documento"] = changes.get("documento")
                entry["autor"] = author or changes.get("autor")
                entry["updated_at"] = utc_now_iso()

            return

        override = dict(candidate["modified_rules"].get(rule_id) or {})

        if severity is not None:
            override["severity"] = severity

        if description is not None:
            override["description"] = description

        override["fonte"] = changes.get("fonte")
        override["documento"] = changes.get("documento")
        override["autor"] = author or changes.get("autor")
        override["justificacao"] = changes.get("justificacao")
        override["updated_at"] = utc_now_iso()

        candidate["modified_rules"][rule_id] = override

    return commit_extension_change(
        change_type="rule_modified",
        target_id=rule_id,
        mutate=mutate,
        payload=changes,
        author=author or changes.get("autor"),
        justification=changes.get("justificacao"),
        source_reference=changes.get("fonte"),
        source_document=changes.get("documento"),
    )


def disable_rule(
    rule_id: str,
    justification: str,
    author: Optional[str] = None,
    source_reference: Optional[str] = None,
    source_document: Optional[str] = None,
) -> dict[str, Any]:
    kb = load_base_knowledge_base()
    extensions = load_extensions()

    if rule_id not in existing_rule_ids(kb, extensions):
        return rejection(
            message=f"A regra '{rule_id}' não existe na base de conhecimento.",
            errors=["Identificador de regra desconhecido."],
        )

    if rule_id in (extensions.get("disabled_rules", {}) or {}):
        return rejection(
            message=f"A regra '{rule_id}' já se encontra desativada.",
            errors=["Regra já desativada."],
        )

    if len(str(justification or "").strip()) < 20:
        return rejection(
            message="A desativação não foi gravada.",
            errors=["A justificação clínica tem de ter pelo menos 20 caracteres."],
        )

    entry = {
        "justificacao": justification.strip(),
        "autor": author,
        "fonte": source_reference,
        "documento": source_document,
        "created_at": utc_now_iso(),
    }

    def mutate(candidate: dict[str, Any]) -> None:
        candidate["disabled_rules"][rule_id] = entry

    return commit_extension_change(
        change_type="rule_disabled",
        target_id=rule_id,
        mutate=mutate,
        payload=entry,
        author=author,
        justification=entry["justificacao"],
        source_reference=source_reference,
        source_document=source_document,
    )


def enable_rule(rule_id: str, author: Optional[str] = None) -> dict[str, Any]:
    extensions = load_extensions()

    if rule_id not in (extensions.get("disabled_rules", {}) or {}):
        return rejection(
            message=f"A regra '{rule_id}' não se encontra desativada.",
            errors=["Regra não desativada."],
        )

    def mutate(candidate: dict[str, Any]) -> None:
        candidate["disabled_rules"].pop(rule_id, None)

    return commit_extension_change(
        change_type="rule_reenabled",
        target_id=rule_id,
        mutate=mutate,
        payload={"reativada_em": utc_now_iso()},
        author=author,
    )


def override_medication(
    med_id: str,
    override: dict[str, Any],
    author: Optional[str] = None,
) -> dict[str, Any]:
    kb = load_base_knowledge_base()

    errors = validate_medication_override(med_id, override, kb)

    if errors:
        return rejection(
            message="A alteração não passou a validação estrutural e não foi gravada.",
            errors=errors,
        )

    def mutate(candidate: dict[str, Any]) -> None:
        stored = dict(candidate["medication_overrides"].get(med_id) or {})

        added = list(stored.get("contraindicated_conditions_add", []) or [])
        removed = list(stored.get("contraindicated_conditions_remove", []) or [])

        for condition in override.get("contraindicated_conditions_add", []) or []:
            if condition not in added:
                added.append(condition)

            if condition in removed:
                removed.remove(condition)

        for condition in override.get("contraindicated_conditions_remove", []) or []:
            if condition not in removed:
                removed.append(condition)

            if condition in added:
                added.remove(condition)

        stored["contraindicated_conditions_add"] = added
        stored["contraindicated_conditions_remove"] = removed

        for field in ("renal_caution", "renal_alert_severity", "qt_risk"):
            if override.get(field) is not None:
                stored[field] = override[field]

        stored["fonte"] = override.get("fonte")
        stored["documento"] = override.get("documento")
        stored["autor"] = author or override.get("autor")
        stored["justificacao"] = override.get("justificacao")
        stored["updated_at"] = utc_now_iso()

        candidate["medication_overrides"][med_id] = stored

    return commit_extension_change(
        change_type="medication_override",
        target_id=med_id,
        mutate=mutate,
        payload=override,
        author=author or override.get("autor"),
        justification=override.get("justificacao"),
        source_reference=override.get("fonte"),
        source_document=override.get("documento"),
    )


def remove_added_rule(rule_id: str, author: Optional[str] = None) -> dict[str, Any]:
    """Remove uma regra criada no painel. As regras da base original nunca são
    apagadas, apenas desativadas com justificação clínica registada."""
    extensions = load_extensions()

    added_ids = {rule.get("id") for rule in extensions.get("added_rules", []) or []}

    if rule_id not in added_ids:
        return rejection(
            message=(
                f"A regra '{rule_id}' não pertence à camada de gestão do conhecimento. "
                "As regras da base original só podem ser desativadas com justificação."
            ),
            errors=["Regra não removível."],
        )

    def mutate(candidate: dict[str, Any]) -> None:
        candidate["added_rules"] = [
            rule
            for rule in candidate["added_rules"]
            if rule.get("id") != rule_id
        ]
        candidate["disabled_rules"].pop(rule_id, None)

    return commit_extension_change(
        change_type="rule_removed",
        target_id=rule_id,
        mutate=mutate,
        payload={"removida_em": utc_now_iso()},
        author=author,
    )


def list_rules(include_disabled: bool = True) -> list[dict[str, Any]]:
    """Vista de consulta: cada regra com a sua origem, estado e proveniência."""
    kb = load_base_knowledge_base()
    extensions = load_extensions()

    disabled = extensions.get("disabled_rules", {}) or {}
    modified = extensions.get("modified_rules", {}) or {}

    items: list[dict[str, Any]] = []

    for rule in kb.get("interaction_rules", []):
        rule_id = rule.get("id")
        override = modified.get(rule_id) or {}

        item = dict(rule)
        item["severity"] = override.get("severity") or rule.get("severity")
        item["description"] = override.get("description") or rule.get("description")
        item["origem"] = "base_modificada" if override else "base"
        item["enabled"] = rule_id not in disabled
        item["fonte"] = override.get("fonte") or rule.get("fonte")
        item["documento"] = override.get("documento") or rule.get("documento")
        item["justificacao_desativacao"] = disabled.get(rule_id, {}).get("justificacao")
        items.append(item)

    for added in extensions.get("added_rules", []) or []:
        rule_id = added.get("id")

        item = dict(added)
        item["origem"] = "adicionada"
        item["enabled"] = rule_id not in disabled
        item["justificacao_desativacao"] = disabled.get(rule_id, {}).get("justificacao")
        items.append(item)

    if not include_disabled:
        items = [item for item in items if item["enabled"]]

    return items


def list_medications() -> list[dict[str, Any]]:
    base = load_base_knowledge_base()
    extensions = load_extensions()
    overrides = extensions.get("medication_overrides", {}) or {}
    effective = apply_extensions(base, extensions).get("medications", {})

    items: list[dict[str, Any]] = []

    for med_id, medication in sorted(effective.items()):
        override = overrides.get(med_id) or {}

        item = dict(medication)
        item["id"] = med_id
        item["modificado"] = bool(override)
        item["fonte_alteracao"] = override.get("fonte")
        item["documento_alteracao"] = override.get("documento")
        items.append(item)

    return items


def load_knowledge_base_state() -> dict[str, Any]:
    """Conhecimento em vigor, ou seja, base mais camada de gestão."""
    return apply_extensions(load_base_knowledge_base())


def get_catalog() -> dict[str, Any]:
    """Vocabulário aceite pelo motor, para preenchimento dos campos da interface."""
    kb = apply_extensions(load_base_knowledge_base())

    return {
        "medications": [
            {"id": med_id, "display_name": medication.get("display_name", med_id)}
            for med_id, medication in sorted(kb.get("medications", {}).items())
        ],
        "therapeutic_classes": sorted(known_therapeutic_classes(kb)),
        "conditions": sorted(known_conditions(kb)),
        "severities": sorted(VALID_SEVERITIES),
        "match_types": sorted(VALID_MATCH_TYPES),
    }