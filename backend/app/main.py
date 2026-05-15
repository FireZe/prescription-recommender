from fastapi import FastAPI, HTTPException
import time
from uuid import uuid4
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    PrescriptionRequest,
    AnalyzeResponse,
    SyntheaAnalyzeRequest,
    FeedbackRequest,
    FeedbackResponse,
    AlternativeEvaluation,
    LLMExplanationRequest,
    LLMExplanationResponse,
)
from app.database import (
    init_db,
    save_analysis,
    save_feedback,
    get_metrics,
    get_analysis,
)
from app.normalization import normalize_medication_id, normalize_main_problem
from app.data_loader import load_knowledge_base, load_historical_patterns
from app.rules_engine import run_safety_checks
from app.recommender import recommend_alternatives, build_recommendation_notes
from app.synthea_loader import (
    list_synthea_patients,
    get_synthea_patient_context,
)
from app.llm_explainer import generate_llm_explanation


app = FastAPI(
    title="Prescription Safety and Recommendation Prototype",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    init_db()

def to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()

def normalize_patient_context(patient_context):
    normalized_main_problem = normalize_main_problem(patient_context.main_problem)

    if hasattr(patient_context, "model_copy"):
        return patient_context.model_copy(
            update={"main_problem": normalized_main_problem}
        )

    return patient_context.copy(
        update={"main_problem": normalized_main_problem}
    )

def evaluate_user_alternative(
    feedback: FeedbackRequest,
    kb,
) -> AlternativeEvaluation | None:
    if feedback.user_alternative is None:
        return AlternativeEvaluation(
            status="not_provided",
            alerts=[],
            message="Não foi submetida alternativa terapêutica pelo profissional de saúde.",
        )

    medication_id = normalize_medication_id(feedback.user_alternative.medication)

    if medication_id is None or medication_id not in kb.get("medications", {}):
        return AlternativeEvaluation(
            status="unknown_medication",
            alerts=[],
            message=(
                "A alternativa indicada pelo profissional não existe na base de conhecimento "
                "atual do protótipo. A decisão foi registada para análise futura, mas o sistema "
                "não consegue avaliar automaticamente esta alternativa."
            ),
        )

    analysis = get_analysis(feedback.analysis_id)

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Análise não encontrada. Não é possível associar o feedback ao resultado original.",
        )

    request_data = analysis["request_json"]

    from app.schemas import PatientContext

    if "patient_context" in request_data:
        patient_context = PatientContext(**request_data["patient_context"])

    elif "patient" in request_data:
        patient_context = PatientContext(**request_data["patient"])

    elif "patient_id" in request_data:
        patient_context = get_synthea_patient_context(
            patient_id=request_data["patient_id"],
            main_problem=request_data.get("main_problem"),
        )

    elif "original_request" in request_data and "patient_id" in request_data["original_request"]:
        original_request = request_data["original_request"]

        patient_context = get_synthea_patient_context(
            patient_id=original_request["patient_id"],
            main_problem=original_request.get("main_problem"),
        )

    else:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível reconstruir o contexto clínico da análise original.",
        )

    alerts = run_safety_checks(
        patient=patient_context,
        prescription=[feedback.user_alternative],
        kb=kb,
    )

    blocking_alerts = [
        alert for alert in alerts
        if alert.severity in {"high", "critical"}
    ]

    if blocking_alerts:
        message = (
            "A alternativa indicada pelo profissional foi avaliada pelo motor de segurança "
            "e originou alertas de gravidade elevada ou crítica. A decisão clínica é registada, "
            "mas a alternativa deve ser revista antes de ser considerada segura."
        )
    elif alerts:
        message = (
            "A alternativa indicada pelo profissional foi avaliada pelo motor de segurança "
            "e originou apenas alertas de baixa ou moderada gravidade."
        )
    else:
        message = (
            "A alternativa indicada pelo profissional foi avaliada pelo motor de segurança "
            "e não originou alertas conhecidos na base de conhecimento atual."
        )

    return AlternativeEvaluation(
        status="evaluated",
        alerts=alerts,
        message=message,
    )

def list_to_dict(items):
    return [to_dict(item) for item in items]

def build_stored_analysis_request(
    source: str,
    original_request,
    patient_context,
    prescription,
    recommendation_notes=None,
) -> dict:
    return {
        "source": source,
        "original_request": to_dict(original_request),
        "patient_context": to_dict(patient_context),
        "prescription": list_to_dict(prescription),
        "recommendation_notes": recommendation_notes or [],
    }

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Prescription recommendation prototype is running."
    }


@app.get("/synthea/patients")
def get_synthea_patients(
    limit: int = 20,
    adults_only: bool = False,
    with_active_medications: bool = False,
):
    return list_synthea_patients(
        limit=limit,
        adults_only=adults_only,
        with_active_medications=with_active_medications,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_prescription(request: PrescriptionRequest):
    start_time = time.perf_counter()

    kb = load_knowledge_base()
    historical_patterns = load_historical_patterns()
    patient_context = normalize_patient_context(request.patient)

    alerts = run_safety_checks(
        patient=patient_context,
        prescription=request.prescription,
        kb=kb,
    )

    recommendations = recommend_alternatives(
        patient=patient_context,
        prescription=request.prescription,
        alerts=alerts,
        kb=kb,
        historical_patterns=historical_patterns,
    )

    recommendation_notes = build_recommendation_notes(
        patient=patient_context,
        recommendations=recommendations,
        alerts=alerts,
        kb=kb,
    )

    if recommendations:
        explanation = (
            "A prescrição foi analisada através de regras determinísticas de segurança clínica. "
            "As alternativas candidatas foram geradas a partir da base de conhecimento farmacológico "
            "e ordenadas de acordo com segurança clínica, adequação contextual, proximidade terapêutica "
            "e, quando aplicável, padrões observados em dados clínicos."
        )
    else:
        explanation = (
            "A prescrição foi analisada através de regras determinísticas de segurança clínica. "
            "Não foi identificada uma alternativa terapêutica admissível dentro da base de conhecimento "
            "atual do protótipo. Recomenda-se revisão clínica da prescrição e/ou validação por "
            "farmacêutico clínico."
        )

    analysis_id = str(uuid4())
    response_time_ms = (time.perf_counter() - start_time) * 1000

    save_analysis(
        analysis_id=analysis_id,
        patient_id=patient_context.patient_id,
        source="manual",
        request_data=build_stored_analysis_request(
            source="manual",
            original_request=request,
            patient_context=patient_context,
            prescription=request.prescription,
            recommendation_notes=recommendation_notes,
        ),
        alerts=list_to_dict(alerts),
        recommendations=list_to_dict(recommendations),
        explanation=explanation,
        response_time_ms=response_time_ms,
    )

    return AnalyzeResponse(
        analysis_id=analysis_id,
        alerts=alerts,
        recommendations=recommendations,
        recommendation_notes=recommendation_notes,
        explanation=explanation,
    )

@app.post("/analyze/synthea", response_model=AnalyzeResponse)
def analyze_synthea_prescription(request: SyntheaAnalyzeRequest):
    start_time = time.perf_counter()

    kb = load_knowledge_base()
    historical_patterns = load_historical_patterns()

    try:
        patient_context = get_synthea_patient_context(
            patient_id=request.patient_id,
            main_problem=request.main_problem,
        )
        patient_context = normalize_patient_context(patient_context)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))

    alerts = run_safety_checks(
        patient=patient_context,
        prescription=request.prescription,
        kb=kb,
    )

    recommendations = recommend_alternatives(
        patient=patient_context,
        prescription=request.prescription,
        alerts=alerts,
        kb=kb,
        historical_patterns=historical_patterns,
    )

    recommendation_notes = build_recommendation_notes(
        patient=patient_context,
        recommendations=recommendations,
        alerts=alerts,
        kb=kb,
    )

    if recommendations:
        explanation = (
            "A prescrição foi analisada através de regras determinísticas de segurança clínica. "
            "As alternativas candidatas foram geradas a partir da base de conhecimento farmacológico "
            "e ordenadas de acordo com segurança clínica, adequação contextual, proximidade terapêutica "
            "e, quando aplicável, padrões observados em dados clínicos."
        )
    else:
        explanation = (
            "A prescrição foi analisada através de regras determinísticas de segurança clínica. "
            "Não foi identificada uma alternativa terapêutica admissível dentro da base de conhecimento "
            "atual do protótipo. Recomenda-se revisão clínica da prescrição e/ou validação por "
            "farmacêutico clínico."
        )

    analysis_id = str(uuid4())
    response_time_ms = (time.perf_counter() - start_time) * 1000

    save_analysis(
        analysis_id=analysis_id,
        patient_id=patient_context.patient_id,
        source="synthea",
        request_data=build_stored_analysis_request(
            source="synthea",
            original_request=request,
            patient_context=patient_context,
            prescription=request.prescription,
            recommendation_notes=recommendation_notes,
        ),
        alerts=list_to_dict(alerts),
        recommendations=list_to_dict(recommendations),
        explanation=explanation,
        response_time_ms=response_time_ms,
    )

    return AnalyzeResponse(
        analysis_id=analysis_id,
        alerts=alerts,
        recommendations=recommendations,
        recommendation_notes=recommendation_notes,
        explanation=explanation,
    )

@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    analysis = get_analysis(request.analysis_id)

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Análise não encontrada. O feedback deve estar associado a um analysis_id válido.",
        )

    kb = load_knowledge_base()

    alternative_evaluation = evaluate_user_alternative(
        feedback=request,
        kb=kb,
    )

    feedback_id = str(uuid4())

    save_feedback(
        feedback_id=feedback_id,
        analysis_id=request.analysis_id,
        patient_id=request.patient_id,
        medication=request.medication,
        recommendation=request.recommendation,
        decision=request.decision,
        comment=request.comment,
        user_alternative=to_dict(request.user_alternative) if request.user_alternative else None,
        user_alternative_justification=request.user_alternative_justification,
        alternative_evaluation=to_dict(alternative_evaluation) if alternative_evaluation else None,
    )

    return FeedbackResponse(
        feedback_id=feedback_id,
        analysis_id=request.analysis_id,
        saved=True,
        alternative_evaluation=alternative_evaluation,
    )

@app.get("/metrics")
def metrics():
    return get_metrics()

@app.post("/explain/llm", response_model=LLMExplanationResponse)
def explain_analysis_with_llm(request: LLMExplanationRequest):
    analysis = get_analysis(request.analysis_id)

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Análise não encontrada. Não é possível gerar explicação LLM sem analysis_id válido.",
        )

    try:
        result = generate_llm_explanation(
            analysis=analysis,
            user_question=request.user_question,
        )
    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    return LLMExplanationResponse(
        analysis_id=request.analysis_id,
        model=result["model"],
        explanation=result["explanation"],
    )

