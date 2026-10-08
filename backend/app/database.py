import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "prescription_feedback.db"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                analysis_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                request_json TEXT NOT NULL,
                alerts_json TEXT NOT NULL,
                recommendations_json TEXT NOT NULL,
                explanation TEXT NOT NULL,
                response_time_ms REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                medication TEXT,
                recommendation TEXT,
                decision TEXT NOT NULL CHECK(decision IN ('accepted', 'rejected', 'ignored')),
                comment TEXT,
                user_alternative_json TEXT,
                user_alternative_justification TEXT,
                alternative_evaluation_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                outcome_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                patient_id TEXT NOT NULL,
                medication TEXT,
                outcome TEXT NOT NULL CHECK(outcome IN ('resolved', 'not_resolved', 'adverse_event')),
                comment TEXT,
                follow_up_days INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id)
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS kb_changes (
                change_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                change_type TEXT NOT NULL CHECK(change_type IN (
                    'rule_added',
                    'rule_modified',
                    'rule_disabled',
                    'rule_removed',
                    'rule_reenabled',
                    'medication_override'
                )),
                target_id TEXT NOT NULL,
                author TEXT,
                justification TEXT,
                source_reference TEXT,
                source_document TEXT,
                payload_json TEXT NOT NULL,
                validation_json TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_messages (
                message_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                model TEXT,
                fallback_used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_llm_messages_analysis
            ON llm_messages (analysis_id, turn_index)
        """)

        conn.commit()


def save_analysis(
    analysis_id: str,
    patient_id: str,
    source: str,
    request_data: dict[str, Any],
    alerts: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    explanation: str,
    response_time_ms: Optional[float] = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO analyses (
                analysis_id,
                patient_id,
                source,
                created_at,
                request_json,
                alerts_json,
                recommendations_json,
                explanation,
                response_time_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                patient_id,
                source,
                utc_now_iso(),
                json.dumps(request_data, ensure_ascii=False),
                json.dumps(alerts, ensure_ascii=False),
                json.dumps(recommendations, ensure_ascii=False),
                explanation,
                response_time_ms,
            ),
        )
        conn.commit()


def get_analysis(analysis_id: str) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()

    if row is None:
        return None

    result = dict(row)
    result["request_json"] = json.loads(result["request_json"])
    result["alerts_json"] = json.loads(result["alerts_json"])
    result["recommendations_json"] = json.loads(result["recommendations_json"])

    return result


def save_feedback(
    feedback_id: str,
    analysis_id: str,
    patient_id: str,
    medication: Optional[str],
    recommendation: Optional[str],
    decision: str,
    comment: Optional[str],
    user_alternative: Optional[dict[str, Any]],
    user_alternative_justification: Optional[str],
    alternative_evaluation: Optional[dict[str, Any]],
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                feedback_id,
                analysis_id,
                patient_id,
                medication,
                recommendation,
                decision,
                comment,
                user_alternative_json,
                user_alternative_justification,
                alternative_evaluation_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                analysis_id,
                patient_id,
                medication,
                recommendation,
                decision,
                comment,
                json.dumps(user_alternative, ensure_ascii=False) if user_alternative else None,
                user_alternative_justification,
                json.dumps(alternative_evaluation, ensure_ascii=False) if alternative_evaluation else None,
                utc_now_iso(),
            ),
        )
        conn.commit()


def save_outcome(
    outcome_id: str,
    analysis_id: str,
    patient_id: str,
    medication: Optional[str],
    outcome: str,
    comment: Optional[str],
    analysis_created_at: Optional[str] = None,
) -> None:
    follow_up_days = None
    if analysis_created_at:
        try:
            created = datetime.fromisoformat(analysis_created_at)
            follow_up_days = (datetime.now(timezone.utc) - created).days
        except Exception:
            follow_up_days = None

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO outcomes (
                outcome_id, analysis_id, patient_id, medication,
                outcome, comment, follow_up_days, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome_id, analysis_id, patient_id, medication,
                outcome, comment, follow_up_days, utc_now_iso(),
            ),
        )
        conn.commit()


def get_pending_followups(min_days: int = 14, limit: int = 50) -> list[dict[str, Any]]:
    """Análises com >= min_days e ainda sem desfecho registado (lembretes de follow-up)."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT a.analysis_id, a.patient_id, a.created_at
            FROM analyses a
            LEFT JOIN outcomes o ON a.analysis_id = o.analysis_id
            WHERE o.outcome_id IS NULL
              AND julianday('now') - julianday(a.created_at) >= ?
              AND a.patient_id NOT LIKE 'H%'
            ORDER BY a.created_at DESC
            LIMIT ?
            """,
            (min_days, limit),
        ).fetchall()

    items = []
    for r in rows:
        days_elapsed = None
        try:
            created = datetime.fromisoformat(r["created_at"])
            days_elapsed = (datetime.now(timezone.utc) - created).days
        except Exception:
            days_elapsed = None
        items.append({
            "analysis_id": r["analysis_id"],
            "patient_id": r["patient_id"],
            "created_at": r["created_at"],
            "days_elapsed": days_elapsed if days_elapsed is not None else 0,
        })
    return items


def get_metrics() -> dict[str, Any]:
    with get_connection() as conn:
        total_analyses = conn.execute(
            "SELECT COUNT(*) AS total FROM analyses"
        ).fetchone()["total"]

        total_feedback = conn.execute(
            "SELECT COUNT(*) AS total FROM feedback"
        ).fetchone()["total"]

        feedback_by_decision_rows = conn.execute(
            """
            SELECT decision, COUNT(*) AS total
            FROM feedback
            GROUP BY decision
            """
        ).fetchall()

        avg_response_time = conn.execute(
            """
            SELECT AVG(response_time_ms) AS avg_response_time_ms
            FROM analyses
            WHERE response_time_ms IS NOT NULL
            """
        ).fetchone()["avg_response_time_ms"]

        analyses_rows = conn.execute(
            "SELECT alerts_json, recommendations_json FROM analyses"
        ).fetchall()

    feedback_by_decision = {
        row["decision"]: row["total"]
        for row in feedback_by_decision_rows
    }

    alerts_by_severity: dict[str, int] = {}
    alerts_by_type: dict[str, int] = {}
    total_recommendations = 0
    analyses_with_recommendations = 0

    for row in analyses_rows:
        alerts = json.loads(row["alerts_json"])
        recommendations = json.loads(row["recommendations_json"])

        if recommendations:
            analyses_with_recommendations += 1

        total_recommendations += len(recommendations)

        for alert in alerts:
            severity = alert.get("severity", "unknown")
            alert_type = alert.get("type", "unknown")

            alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1
            alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1

    accepted = feedback_by_decision.get("accepted", 0)
    rejected = feedback_by_decision.get("rejected", 0)
    ignored = feedback_by_decision.get("ignored", 0)

    acceptance_rate = accepted / total_feedback if total_feedback else 0.0

    return {
        "total_analyses": total_analyses,
        "total_feedback": total_feedback,
        "feedback_by_decision": {
            "accepted": accepted,
            "rejected": rejected,
            "ignored": ignored,
        },
        "acceptance_rate": round(acceptance_rate, 3),
        "alerts_by_severity": alerts_by_severity,
        "alerts_by_type": alerts_by_type,
        "total_recommendations": total_recommendations,
        "analyses_with_recommendations": analyses_with_recommendations,
        "average_response_time_ms": round(avg_response_time, 2) if avg_response_time else None,
    }


def save_kb_change(
    change_id: str,
    change_type: str,
    target_id: str,
    payload: dict[str, Any],
    validation: dict[str, Any],
    author: Optional[str] = None,
    justification: Optional[str] = None,
    source_reference: Optional[str] = None,
    source_document: Optional[str] = None,
) -> None:
    """Regista uma alteração ao conhecimento clínico, com fonte e prova de validação."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO kb_changes (
                change_id,
                created_at,
                change_type,
                target_id,
                author,
                justification,
                source_reference,
                source_document,
                payload_json,
                validation_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                change_id,
                utc_now_iso(),
                change_type,
                target_id,
                author,
                justification,
                source_reference,
                source_document,
                json.dumps(payload, ensure_ascii=False),
                json.dumps(validation, ensure_ascii=False),
            ),
        )
        conn.commit()


def get_kb_changes(limit: int = 100) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM kb_changes
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    changes = []

    for row in rows:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json"))
        item["validation"] = json.loads(item.pop("validation_json"))
        changes.append(item)

    return changes


def get_next_turn_index(analysis_id: str) -> int:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT MAX(turn_index) AS last_index FROM llm_messages WHERE analysis_id = ?",
            (analysis_id,),
        ).fetchone()

    last_index = row["last_index"] if row else None

    return 0 if last_index is None else last_index + 1


def save_llm_message(
    message_id: str,
    analysis_id: str,
    role: str,
    content: str,
    turn_index: Optional[int] = None,
    model: Optional[str] = None,
    fallback_used: bool = False,
) -> int:
    """Guarda um turno da conversa e devolve o índice atribuído."""
    if turn_index is None:
        turn_index = get_next_turn_index(analysis_id)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO llm_messages (
                message_id,
                analysis_id,
                turn_index,
                role,
                content,
                model,
                fallback_used,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                analysis_id,
                turn_index,
                role,
                content,
                model,
                1 if fallback_used else 0,
                utc_now_iso(),
            ),
        )
        conn.commit()

    return turn_index


def get_llm_messages(analysis_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT message_id, analysis_id, turn_index, role, content, model,
                   fallback_used, created_at
            FROM llm_messages
            WHERE analysis_id = ?
            ORDER BY turn_index ASC
            LIMIT ?
            """,
            (analysis_id, limit),
        ).fetchall()

    messages = []

    for row in rows:
        item = dict(row)
        item["fallback_used"] = bool(item["fallback_used"])
        messages.append(item)

    return messages