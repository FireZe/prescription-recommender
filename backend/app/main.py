from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import time
from uuid import uuid4
from contextlib import asynccontextmanager
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
    OutcomeRequest,
    OutcomeResponse,
    PendingFollowupItem,
    KnowledgeRuleInput,
    KnowledgeRuleUpdate,
    KnowledgeRuleDisable,
    MedicationOverrideInput,
    KnowledgeChangeResponse,
    KnowledgeBaseSummary,
    LLMChatRequest,
    LLMChatResponse,
)
from app.database import (
    init_db,
    get_kb_changes,
    get_llm_messages,
    save_llm_message,
    save_analysis,
    save_feedback,
    save_outcome,
    get_pending_followups,
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
from app.llm_explainer import generate_llm_explanation, generate_llm_followup
import logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Prescription Safety and Recommendation Prototype",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    recommendations, candidate_admissibility = recommend_alternatives(
        patient=patient_context,
        prescription=request.prescription,
        alerts=alerts,
        kb=kb,
        historical_patterns=historical_patterns,
        return_admissibility=True,
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
        candidate_admissibility=candidate_admissibility,
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

@app.post("/outcome", response_model=OutcomeResponse)
def submit_outcome(request: OutcomeRequest):
    """Regista o desfecho clínico (longitudinal) de uma análise: o médico,
    dias/semanas depois, indica se a prescrição resolveu, não resolveu ou
    causou reação adversa."""
    analysis = get_analysis(request.analysis_id)
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Análise não encontrada. O desfecho deve estar associado a um analysis_id válido.",
        )

    outcome_id = str(uuid4())
    save_outcome(
        outcome_id=outcome_id,
        analysis_id=request.analysis_id,
        patient_id=analysis["patient_id"],
        medication=request.medication,
        outcome=request.outcome,
        comment=request.comment,
        analysis_created_at=analysis.get("created_at"),
    )

    return OutcomeResponse(
        outcome_id=outcome_id,
        analysis_id=request.analysis_id,
        saved=True,
    )


@app.get("/outcomes/pending", response_model=list[PendingFollowupItem])
def pending_followups(min_days: int = 14):
    """Análises com pelo menos `min_days` dias e ainda sem desfecho registado
    — serve de lembrete de follow-up para o médico."""
    return get_pending_followups(min_days=min_days)


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

    if not get_llm_messages(request.analysis_id, limit=1):
        save_llm_message(
            message_id=str(uuid4()),
            analysis_id=request.analysis_id,
            role="assistant",
            content=result["explanation"],
            model=result["model"],
            fallback_used=bool(result.get("fallback_used", False)),
        )

    return LLMExplanationResponse(
        analysis_id=request.analysis_id,
        model=result["model"],
        explanation=result["explanation"],
        fallback_used=bool(result.get("fallback_used", False)),
        fallback_notice=result.get("fallback_notice"),
    )



# --- Gestão do conhecimento clínico (RF08) ---

@app.get("/kb/summary", response_model=KnowledgeBaseSummary)
def knowledge_base_summary():
    from app.knowledge_admin import (
        load_base_knowledge_base,
        load_knowledge_base_state,
        summarize_extensions,
    )

    base = load_base_knowledge_base()
    effective = load_knowledge_base_state()

    return KnowledgeBaseSummary(
        rules_base=len(base.get("interaction_rules", [])),
        rules_ativas=len(effective.get("interaction_rules", [])),
        medications=len(effective.get("medications", {})),
        extensions=summarize_extensions(),
    )


@app.get("/kb/rules")
def knowledge_base_rules(include_disabled: bool = True):
    from app.knowledge_admin import list_rules

    return list_rules(include_disabled=include_disabled)


@app.get("/kb/medications")
def knowledge_base_medications():
    from app.knowledge_admin import list_medications

    return list_medications()


@app.get("/kb/catalog")
def knowledge_base_catalog():
    from app.knowledge_admin import get_catalog

    return get_catalog()


@app.get("/kb/changes")
def knowledge_base_changes(limit: int = 100):
    return get_kb_changes(limit=limit)


@app.post("/kb/rules", response_model=KnowledgeChangeResponse)
def create_knowledge_rule(request: KnowledgeRuleInput):
    from app.knowledge_admin import add_rule

    return add_rule(to_dict(request), author=request.autor)


@app.patch("/kb/rules/{rule_id}", response_model=KnowledgeChangeResponse)
def update_knowledge_rule(rule_id: str, request: KnowledgeRuleUpdate):
    from app.knowledge_admin import modify_rule

    return modify_rule(rule_id, to_dict(request), author=request.autor)


@app.post("/kb/rules/{rule_id}/disable", response_model=KnowledgeChangeResponse)
def disable_knowledge_rule(rule_id: str, request: KnowledgeRuleDisable):
    from app.knowledge_admin import disable_rule

    return disable_rule(
        rule_id=rule_id,
        justification=request.justificacao,
        author=request.autor,
        source_reference=request.fonte,
        source_document=request.documento,
    )


@app.delete("/kb/rules/{rule_id}", response_model=KnowledgeChangeResponse)
def delete_knowledge_rule(rule_id: str, autor: str | None = None):
    from app.knowledge_admin import remove_added_rule

    return remove_added_rule(rule_id, author=autor)


@app.post("/kb/rules/{rule_id}/enable", response_model=KnowledgeChangeResponse)
def enable_knowledge_rule(rule_id: str, autor: str | None = None):
    from app.knowledge_admin import enable_rule

    return enable_rule(rule_id, author=autor)


@app.patch("/kb/medications/{medication_id}", response_model=KnowledgeChangeResponse)
def update_knowledge_medication(medication_id: str, request: MedicationOverrideInput):
    from app.knowledge_admin import override_medication

    return override_medication(medication_id, to_dict(request), author=request.autor)


@app.get("/kb/provenance")
def list_knowledge_provenance():
    from app.knowledge_admin import list_provenance_documents

    return list_provenance_documents()


@app.post("/kb/provenance")
async def upload_knowledge_provenance(file: UploadFile = File(...)):
    """Carrega o documento (RCM, norma da DGS, artigo) que fundamenta uma alteração."""
    from app.knowledge_admin import store_provenance_document

    content = await file.read()

    try:
        return store_provenance_document(
            filename=file.filename or "documento.pdf",
            content=content,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/kb/provenance/{document_name}")
def get_knowledge_provenance(document_name: str):
    from app.knowledge_admin import provenance_document_path

    path = provenance_document_path(document_name)

    if path is None or not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Documento de proveniência não encontrado.",
        )

    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.post("/kb/validate")
def validate_knowledge_base():
    """Corre a suite de regressão clínica sobre o conhecimento em vigor."""
    from app.knowledge_admin import evaluate_candidate_extensions, load_extensions

    return evaluate_candidate_extensions(load_extensions())



@app.get("/explain/llm/chat/{analysis_id}")
def get_llm_chat_history(analysis_id: str):
    return get_llm_messages(analysis_id)


@app.post("/explain/llm/chat", response_model=LLMChatResponse)
def continue_llm_conversation(request: LLMChatRequest):
    """Permite ao profissional de saúde colocar questões de seguimento sobre uma
    análise já produzida, mantendo o contexto da conversa."""
    analysis = get_analysis(request.analysis_id)

    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="Análise não encontrada. Não é possível continuar a conversa.",
        )

    history = get_llm_messages(request.analysis_id)

    try:
        result = generate_llm_followup(
            analysis=analysis,
            history=history,
            question=request.question,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error))

    save_llm_message(
        message_id=str(uuid4()),
        analysis_id=request.analysis_id,
        role="user",
        content=request.question.strip(),
    )

    save_llm_message(
        message_id=str(uuid4()),
        analysis_id=request.analysis_id,
        role="assistant",
        content=result["answer"],
        model=result["model"],
        fallback_used=result["fallback_used"],
    )

    return LLMChatResponse(
        analysis_id=request.analysis_id,
        model=result["model"],
        answer=result["answer"],
        fallback_used=result["fallback_used"],
        fallback_notice=result.get("fallback_notice"),
        history=get_llm_messages(request.analysis_id),
    )