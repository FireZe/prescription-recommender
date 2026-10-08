from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class PatientContext(BaseModel):
    patient_id: str
    age: int
    sex: Literal["M", "F", "Other"]
    conditions: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    active_medications: List[str] = Field(default_factory=list)
    renal_status: Literal["normal", "mild_impairment", "severe_impairment"] = "normal"
    main_problem: str


class MedicationLine(BaseModel):
    medication: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None


class PrescriptionRequest(BaseModel):
    patient: PatientContext
    prescription: List[MedicationLine]


class Alert(BaseModel):
    type: str
    severity: Literal["low", "moderate", "high", "critical"]
    medication: Optional[str] = None
    description: str

    # Origem do alerta
    origin: Literal[
        "prescription_related",
        "active_medication_existing",
        "combined_profile_risk",
        "unknown"
    ] = "unknown"

    involves_prescribed_medication: bool = False
    involves_active_medication: bool = False

    # Campos técnicos úteis para auditoria, LLM e agrupamento
    rule_id: Optional[str] = None
    medication_ids: List[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    medication: str
    display_name: Optional[str] = None
    score_heuristic: float        # score heurístico puro (Sseg, Sctx, Ssim, Sfb)
    score_combined: float         # score do modelo de ordenação (LTR); degrada para o heurístico se indisponível
    score_final: float            # após refinamento histórico secundário
    reasons: List[str]
    secondary_historical_score: Optional[float] = None
    admissibility_class: Optional[str] = None

class RecommendationNote(BaseModel):
    type: Literal[
        "already_active_candidate",
        "already_active_symptomatic_candidate",
        "therapeutic_duplication",
    ]
    medication: Optional[str] = None
    description: str

class CandidateAdmissibility(BaseModel):
    candidate: str
    admissibility_class: str

class AnalyzeResponse(BaseModel):
    analysis_id: str
    alerts: List[Alert]
    recommendations: List[Recommendation]
    recommendation_notes: List[RecommendationNote] = Field(default_factory=list)
    explanation: str
    candidate_admissibility: List[CandidateAdmissibility] = Field(default_factory=list)

class SyntheaAnalyzeRequest(BaseModel):
    patient_id: str
    main_problem: Optional[str] = None
    prescription: List[MedicationLine]

class AlternativeEvaluation(BaseModel):
    status: Literal["not_provided", "unknown_medication", "evaluated"]
    alerts: List[Alert] = Field(default_factory=list)
    message: str


class FeedbackRequest(BaseModel):
    analysis_id: str
    patient_id: str
    medication: Optional[str] = None
    recommendation: Optional[str] = None
    decision: Literal["accepted", "rejected", "ignored"]
    comment: Optional[str] = None
    user_alternative: Optional[MedicationLine] = None
    user_alternative_justification: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    analysis_id: str
    saved: bool
    alternative_evaluation: Optional[AlternativeEvaluation] = None

class OutcomeRequest(BaseModel):
    analysis_id: str
    medication: Optional[str] = None
    outcome: Literal["resolved", "not_resolved", "adverse_event"]
    comment: Optional[str] = None


class OutcomeResponse(BaseModel):
    outcome_id: str
    analysis_id: str
    saved: bool


class PendingFollowupItem(BaseModel):
    analysis_id: str
    patient_id: str
    created_at: str
    days_elapsed: int


class LLMExplanationRequest(BaseModel):
    analysis_id: str
    user_question: Optional[str] = None


class LLMExplanationResponse(BaseModel):
    analysis_id: str
    model: str
    explanation: str
    fallback_used: bool = False
    fallback_notice: Optional[str] = None



# --- Gestão do conhecimento clínico (RF08) ---

Severity = Literal["low", "moderate", "high", "critical"]
MatchType = Literal["drug_drug", "class_drug", "class_class", "qt_qt"]


class KnowledgeRuleInput(BaseModel):
    id: str
    match: MatchType
    severity: Severity
    description: str
    class_a: Optional[str] = None
    class_b: Optional[str] = None
    medication_a: Optional[str] = None
    medication_b: Optional[str] = None
    fonte: str
    documento: Optional[str] = None
    autor: Optional[str] = None


class KnowledgeRuleUpdate(BaseModel):
    severity: Optional[Severity] = None
    description: Optional[str] = None
    fonte: str
    documento: Optional[str] = None
    autor: Optional[str] = None
    justificacao: Optional[str] = None


class KnowledgeRuleDisable(BaseModel):
    justificacao: str
    fonte: Optional[str] = None
    documento: Optional[str] = None
    autor: Optional[str] = None


class MedicationOverrideInput(BaseModel):
    contraindicated_conditions_add: List[str] = Field(default_factory=list)
    contraindicated_conditions_remove: List[str] = Field(default_factory=list)
    renal_caution: Optional[bool] = None
    renal_alert_severity: Optional[Severity] = None
    qt_risk: Optional[bool] = None
    fonte: str
    documento: Optional[str] = None
    autor: Optional[str] = None
    justificacao: Optional[str] = None


class RegressionCaseResult(BaseModel):
    id: str
    name: str
    passed: bool
    failures: List[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    passed: bool
    total: int
    failed: int
    rules_total: int = 0
    medications_total: int = 0
    cases: List[RegressionCaseResult] = Field(default_factory=list)


class KnowledgeChangeResponse(BaseModel):
    applied: bool
    change_id: Optional[str] = None
    message: str
    errors: List[str] = Field(default_factory=list)
    validation: ValidationReport


class KnowledgeBaseSummary(BaseModel):
    rules_base: int
    rules_ativas: int
    medications: int
    extensions: dict



class LLMChatRequest(BaseModel):
    analysis_id: str
    question: str = Field(min_length=3, max_length=500)


class LLMChatMessage(BaseModel):
    turn_index: int
    role: Literal["user", "assistant"]
    content: str
    model: Optional[str] = None
    fallback_used: bool = False
    created_at: str


class LLMChatResponse(BaseModel):
    analysis_id: str
    model: str
    answer: str
    fallback_used: bool = False
    fallback_notice: Optional[str] = None
    history: List[LLMChatMessage] = Field(default_factory=list)