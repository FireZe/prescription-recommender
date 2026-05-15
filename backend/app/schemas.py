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
    score_base: float
    score_final: float
    reasons: List[str]
    secondary_historical_score: Optional[float] = None

class RecommendationNote(BaseModel):
    type: Literal[
        "already_active_candidate",
        "already_active_symptomatic_candidate"
    ]
    medication: Optional[str] = None
    description: str

class AnalyzeResponse(BaseModel):
    analysis_id: str
    alerts: List[Alert]
    recommendations: List[Recommendation]
    recommendation_notes: List[RecommendationNote] = Field(default_factory=list)
    explanation: str

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

class LLMExplanationRequest(BaseModel):
    analysis_id: str
    user_question: Optional[str] = None


class LLMExplanationResponse(BaseModel):
    analysis_id: str
    model: str
    explanation: str