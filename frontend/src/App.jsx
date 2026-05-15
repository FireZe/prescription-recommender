import { useMemo, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const EXAMPLE_PATIENT = {
  patient_id: "T001",
  age: 70,
  sex: "F",
  conditions: "",
  allergies: "",
  active_medications: "clopidogrel 75 mg oral tablet",
  renal_status: "normal",
  main_problem: "pain",
};

const EXAMPLE_PRESCRIPTION = {
  medication: "Ibuprofen 400 MG Oral Tablet [Ibu]",
  dose: "400mg",
  frequency: "8/8h",
  route: "oral",
};

const EMPTY_PATIENT = {
  patient_id: "",
  age: "",
  sex: "",
  conditions: "",
  allergies: "",
  active_medications: "",
  renal_status: "",
  main_problem: "",
};

const EMPTY_PRESCRIPTION = {
  medication: "",
  dose: "",
  frequency: "",
  route: "",
};

function parseList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function severityLabel(severity) {
  const labels = {
    low: "Baixa",
    moderate: "Moderada",
    high: "Elevada",
    critical: "Crítica",
  };

  return labels[severity] || severity;
}

function originLabel(origin) {
  const labels = {
    prescription_related: "Relacionado com a prescrição submetida",
    active_medication_existing: "Pré-existente na medicação ativa",
    combined_profile_risk: "Risco combinado do perfil clínico",
    unknown: "Origem não classificada",
  };

  return labels[origin] || "Origem não classificada";
}

function groupAlertsByOrigin(alerts = []) {
  return alerts.reduce((groups, alert) => {
    const origin = alert.origin || "unknown";

    if (!groups[origin]) {
      groups[origin] = [];
    }

    groups[origin].push(alert);
    return groups;
  }, {});
}

function isTechnicalReason(reason) {
  return reason.toLowerCase().includes("score supervisionado");
}

function isPrecautionReason(reason) {
  const normalized = reason.toLowerCase();

  return (
    normalized.includes("precaução") ||
    normalized.includes("compromisso renal grave") ||
    normalized.includes("validado clinicamente")
  );
}

function recommendationStatus(rec) {
  const score = Number(rec.score_final ?? 0);
  const reasons = rec.reasons || [];

  if (score < 0.5 || reasons.some(isPrecautionReason)) {
    return {
      label: "Alternativa com precaução",
      className: "recommendation recommendation-caution",
    };
  }

  return {
    label: "Alternativa sugerida",
    className: "recommendation recommendation-suggested",
  };
}

function visibleClinicalReasons(reasons = []) {
  const result = [];

  for (const reason of reasons) {
    if (isTechnicalReason(reason)) continue;

    const lower = reason.toLowerCase();

    const isDuplicateRenalReason =
      lower.includes("adequação contextual reduzida devido a compromisso renal grave") ||
      lower.includes("precaução no candidato: paracetamol requer precaução acrescida");

    if (isDuplicateRenalReason) continue;

    if (!result.includes(reason)) {
      result.push(reason);
    }
  }

  return result;
}

function decisionLabel(decision) {
  const labels = {
    accepted: "Aceite",
    rejected: "Rejeitada",
    ignored: "Ignorada",
  };

  return labels[decision] || decision;
}

export default function App() {
  const [patient, setPatient] = useState(EMPTY_PATIENT);
  const [prescription, setPrescription] = useState(EMPTY_PRESCRIPTION);
  const [analysisMode, setAnalysisMode] = useState("synthea");

  const [analysisResult, setAnalysisResult] = useState(null);
  const [feedbackResult, setFeedbackResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  const [feedback, setFeedback] = useState({
    decision: "accepted",
    recommendation: "",
    comment: "",
    userAlternativeMedication: "",
    userAlternativeDose: "",
    userAlternativeFrequency: "",
    userAlternativeRoute: "",
    userAlternativeJustification: "",
  });

  const [loadingAnalysis, setLoadingAnalysis] = useState(false);
  const [loadingFeedback, setLoadingFeedback] = useState(false);
  const [error, setError] = useState("");
  
  const [llmExplanation, setLlmExplanation] = useState(null);
  const [loadingLlmExplanation, setLoadingLlmExplanation] = useState(false);

  const selectedRecommendation = useMemo(() => {
    if (!analysisResult?.recommendations?.length) return "";
    return analysisResult.recommendations[0].medication;
  }, [analysisResult]);


  function updatePatientField(field, value) {
    setPatient((current) => ({
      ...current,
      [field]: value,
    }));
  }

  function updatePrescriptionField(field, value) {
    setPrescription((current) => ({
      ...current,
      [field]: value,
    }));
  }

  function updateFeedbackField(field, value) {
    setFeedback((current) => ({
      ...current,
      [field]: value,
    }));
  }

  function loadExampleCase() {
    setAnalysisMode("manual");
    setPatient(EXAMPLE_PATIENT);
    setPrescription(EXAMPLE_PRESCRIPTION);
    setLlmExplanation(null);
    setAnalysisResult(null);
    setFeedbackResult(null);
    setError("");
  }

  async function analyzePrescription() {
    setLoadingAnalysis(true);
    setError("");
    setFeedbackResult(null);
    setLlmExplanation(null);

    const prescriptionPayload = [
      {
        medication: prescription.medication,
        dose: prescription.dose || null,
        frequency: prescription.frequency || null,
        route: prescription.route || null,
      },
    ];

    let endpoint = `${API_URL}/analyze`;
    let payload;

    if (!prescription.medication.trim()) {
      setError("Indica o medicamento a analisar.");
      setLoadingAnalysis(false);
      return;
    }

    if (analysisMode === "synthea") {
      if (!patient.patient_id.trim()) {
        setError("No modo Synthea, indica o identificador do utente.");
        setLoadingAnalysis(false);
        return;
      }

      endpoint = `${API_URL}/analyze/synthea`;

      payload = {
        patient_id: patient.patient_id,
        main_problem: patient.main_problem || null,
        prescription: prescriptionPayload,
      };
    } else {
      if (!patient.patient_id.trim()) {
        setError("Indica o identificador do utente.");
        setLoadingAnalysis(false);
        return;
      }

      if (!patient.age || !patient.sex || !patient.renal_status || !patient.main_problem) {
        setError("No modo manual, preenche idade, sexo, estado renal e problema clínico principal.");
        setLoadingAnalysis(false);
        return;
      }

      payload = {
        patient: {
          patient_id: patient.patient_id,
          age: Number(patient.age),
          sex: patient.sex,
          conditions: parseList(patient.conditions),
          allergies: parseList(patient.allergies),
          active_medications: parseList(patient.active_medications),
          renal_status: patient.renal_status,
          main_problem: patient.main_problem,
        },
        prescription: prescriptionPayload,
      };
    }

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Erro na análise: ${response.status} ${detail}`);
      }

      const data = await response.json();
      setAnalysisResult(data);

      setFeedback((current) => ({
        ...current,
        recommendation: data.recommendations?.[0]?.medication || "",
        comment: "",
        userAlternativeMedication: "",
        userAlternativeDose: "",
        userAlternativeFrequency: "",
        userAlternativeRoute: "",
        userAlternativeJustification: "",
      }));
    } catch (err) {
      setError(err.message || "Erro inesperado ao analisar a prescrição.");
    } finally {
      setLoadingAnalysis(false);
    }
  }

  async function submitFeedback() {
    if (!analysisResult?.analysis_id) {
      setError("Ainda não existe uma análise para associar o feedback.");
      return;
    }

    setLoadingFeedback(true);
    setError("");

    const hasUserAlternative = feedback.userAlternativeMedication.trim().length > 0;

    const payload = {
      analysis_id: analysisResult.analysis_id,
      patient_id: patient.patient_id,
      medication: prescription.medication,
      recommendation: feedback.recommendation || selectedRecommendation || null,
      decision: feedback.decision,
      comment: feedback.comment || null,
      user_alternative: hasUserAlternative
        ? {
            medication: feedback.userAlternativeMedication,
            dose: feedback.userAlternativeDose || null,
            frequency: feedback.userAlternativeFrequency || null,
            route: feedback.userAlternativeRoute || null,
          }
        : null,
      user_alternative_justification:
        feedback.userAlternativeJustification || null,
    };

    try {
      const response = await fetch(`${API_URL}/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Erro ao guardar feedback: ${response.status} ${detail}`);
      }

      const data = await response.json();
      setFeedbackResult(data);
      await loadMetrics();
    } catch (err) {
      setError(err.message || "Erro inesperado ao guardar feedback.");
    } finally {
      setLoadingFeedback(false);
    }
  }
  async function loadMetrics() {
    setLoadingMetrics(true);
    setError("");

    try {
      const response = await fetch(`${API_URL}/metrics`);

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Erro ao carregar métricas: ${response.status} ${detail}`);
      }

      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      setError(err.message || "Erro inesperado ao carregar métricas.");
    } finally {
      setLoadingMetrics(false);
    }
  }

  function formatPercentage(value) {
    if (value === null || value === undefined) return "—";
    return `${Math.round(value * 100)}%`;
  }

  function formatMetricValue(value) {
    if (value === null || value === undefined) return "—";
    return value;
  }

  async function generateLlmExplanation() {
    if (!analysisResult?.analysis_id) {
      setError("Ainda não existe uma análise para explicar.");
      return;
    }

    setLoadingLlmExplanation(true);
    setError("");

    const payload = {
      analysis_id: analysisResult.analysis_id,
      user_question: "Explica de forma objetiva os alertas e recomendações deste caso.",
    };

    try {
      const response = await fetch(`${API_URL}/explain/llm`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Erro ao gerar explicação LLM: ${response.status} ${detail}`);
      }

      const data = await response.json();
      setLlmExplanation(data);
    } catch (err) {
      setError(err.message || "Erro inesperado ao gerar explicação LLM.");
    } finally {
      setLoadingLlmExplanation(false);
    }
  }
  return (
    <main className="page">
      <header className="header">
        <div>
          <p className="eyebrow">Protótipo de Apoio à Decisão Clínica</p>
          <h1>Sistema de Recomendação e Alertas de Prescrição</h1>
          <p className="subtitle">
            Análise de prescrição, alertas de segurança clínica, recomendações
            terapêuticas e registo de feedback do profissional de saúde.
          </p>
        </div>

        <button className="secondaryButton" onClick={loadExampleCase}>
          Carregar caso de teste
        </button>
      </header>

      {error && <div className="errorBox">{error}</div>}

      <section className="grid">
        <section className="card">
          <div className="cardHeader">
            <span className="step">1</span>
            <h2>Dados do utente</h2>
          </div>
          <div className="modeSelector">
            <label className={analysisMode === "synthea" ? "modeOption active" : "modeOption"}>
              <input
                type="radio"
                name="analysisMode"
                value="synthea"
                checked={analysisMode === "synthea"}
                onChange={() => {
                  setAnalysisMode("synthea");
                  setPatient((current) => ({
                    ...EMPTY_PATIENT,
                    patient_id: current.patient_id,
                    main_problem: current.main_problem,
                  }));
                  setAnalysisResult(null);
                  setFeedbackResult(null);
                  setLlmExplanation(null);
                }}
              />
              Usar utente Synthea
            </label>

            <label className={analysisMode === "manual" ? "modeOption active" : "modeOption"}>
              <input
                type="radio"
                name="analysisMode"
                value="manual"
                checked={analysisMode === "manual"}
                onChange={() => {
                  setAnalysisMode("manual");
                  setAnalysisResult(null);
                  setFeedbackResult(null);
                  setLlmExplanation(null);
                }}
              />
              Inserção manual
            </label>
          </div>

          {analysisMode === "synthea" && (
            <p className="modeHint">
              Neste modo, o sistema constrói automaticamente idade, sexo, estado renal,
              alergias e medicação ativa a partir do índice Synthea. Indicar apenas o ID
              do utente, o problema clínico principal e a prescrição a analisar.
            </p>
          )}

          <div className="formGrid">
            <label>
              Identificador
              <input
                  value={patient.patient_id}
                  placeholder="Ex.: 1e154a40-6620-e696-d29c-ead7db18c8a2"
                  onChange={(e) => updatePatientField("patient_id", e.target.value)}
              />
            </label>

            <label>
              Idade
              <input
                  type="number"
                  value={patient.age}
                  disabled={analysisMode === "synthea"}
                  placeholder="Ex.: 27"
                  onChange={(e) => updatePatientField("age", e.target.value)}
              />
            </label>

            <label>
              Sexo
              <select
                value={patient.sex}
                disabled={analysisMode === "synthea"}
                onChange={(e) => updatePatientField("sex", e.target.value)}
              >
                <option value="">Selecionar...</option>
                <option value="F">Feminino</option>
                <option value="M">Masculino</option>
                <option value="Other">Outro</option>
              </select>
            </label>

            <label>
              Estado renal
              <select
                value={patient.renal_status}
                disabled={analysisMode === "synthea"}
                onChange={(e) =>
                  updatePatientField("renal_status", e.target.value)
                }
              >
                <option value="">Selecionar...</option>
                <option value="normal">Normal</option>
                <option value="mild_impairment">Compromisso ligeiro/moderado</option>
                <option value="severe_impairment">Compromisso grave</option>
              </select>
            </label>
          </div>

          <label>
            Problema clínico principal
            <input
              value={patient.main_problem}
              onChange={(e) =>
                updatePatientField("main_problem", e.target.value)
              }
              placeholder="Ex.: dor, infeção bacteriana, hipertensão arterial"
            />
          </label>

          <label>
            Condições clínicas
            <textarea
              value={patient.conditions}
              disabled={analysisMode === "synthea"}
              onChange={(e) => updatePatientField("conditions", e.target.value)}
              placeholder="Separar por vírgulas. Ex.: heart_failure, diabetes"
            />
          </label>

          <label>
            Alergias
            <textarea
              value={patient.allergies}
              disabled={analysisMode === "synthea"}
              onChange={(e) => updatePatientField("allergies", e.target.value)}
              placeholder="Separar por vírgulas"
            />
          </label>

          <label>
            Medicação ativa
            <textarea
              value={patient.active_medications}
              disabled={analysisMode === "synthea"}
              placeholder="Ex.: clopidogrel, simvastatin, metoprolol"
              onChange={(e) =>
                updatePatientField("active_medications", e.target.value)
              }
            />
          </label>
        </section>

        <section className="card">
          <div className="cardHeader">
            <span className="step">2</span>
            <h2>Prescrição a analisar</h2>
          </div>

          <label>
            Medicamento
            <input
              value={prescription.medication}
              placeholder="Ex.: ibuprofen, naproxen, clarithromycin"
              onChange={(e) =>
                updatePrescriptionField("medication", e.target.value)
              }
            />
          </label>

          <div className="formGrid">
            <label>
              Dose
              <input
                value={prescription.dose}
                placeholder="Ex.: 400mg"
                onChange={(e) =>
                  updatePrescriptionField("dose", e.target.value)
                }
              />
            </label>

            <label>
              Frequência
              <input
                value={prescription.frequency}
                placeholder="Ex.: 8/8h, 12/12h, 1x/dia"
                onChange={(e) =>
                  updatePrescriptionField("frequency", e.target.value)
                }
              />
            </label>

            <label>
              Via
              <input
                value={prescription.route}
                placeholder="Ex.: oral"
                onChange={(e) =>
                  updatePrescriptionField("route", e.target.value)
                }
              />
            </label>
          </div>

          <button
            className="primaryButton"
            onClick={analyzePrescription}
            disabled={loadingAnalysis}
          >
            {loadingAnalysis ? "A analisar..." : "Analisar prescrição"}
          </button>

          {analysisResult?.analysis_id && (
            <div className="analysisId">
              <strong>Analysis ID:</strong>
              <code>{analysisResult.analysis_id}</code>
            </div>
          )}
        </section>
      </section>

      <section className="card fullWidth">
        <div className="cardHeader">
          <span className="step">3</span>
          <h2>Alertas e recomendações</h2>
        </div>

        {!analysisResult && (
          <p className="emptyState">
            Submete uma prescrição para visualizar alertas e recomendações.
          </p>
        )}

        {analysisResult && (
          <>
            <div className="resultColumns">
              <div>
                <h3>Alertas</h3>

                {analysisResult.alerts.length === 0 ? (
                  <p className="emptyState">Sem alertas identificados.</p>
                ) : (
                  <div className="list">
                    {Object.entries(groupAlertsByOrigin(analysisResult.alerts)).map(
                      ([origin, alerts]) => (
                        <div key={origin} className="alertGroup">
                          <h4>{originLabel(origin)}</h4>

                          {alerts.map((alert, index) => (
                            <article
                              key={`${alert.type}-${alert.medication}-${index}`}
                              className={`alert alert-${alert.severity}`}
                            >
                              <div className="itemHeader">
                                <strong>{alert.medication}</strong>
                                <span>{severityLabel(alert.severity)}</span>
                              </div>

                              <p>{alert.description}</p>

                              <small>
                                Tipo: {alert.type}
                                {alert.rule_id ? ` · Regra: ${alert.rule_id}` : ""}
                              </small>
                            </article>
                          ))}
                        </div>
                      )
                    )}
                  </div>
                )}
              </div>

              <div>
                <h3>Recomendações</h3>

                {analysisResult.recommendations.length === 0 ? (
                  <div className="emptyState">
                    <p>
                      Não foi identificada alternativa admissível no âmbito atual do protótipo.
                    </p>

                    {analysisResult.recommendation_notes?.length > 0 && (
                      <div className="recommendationNotes">
                        {analysisResult.recommendation_notes.map((note, index) => (
                          <p key={index}>
                            <strong>{note.medication ? `${note.medication}: ` : ""}</strong>
                            {note.description}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="list">
                    {analysisResult.recommendations.map((rec) => {
                      const status = recommendationStatus(rec);
                      const reasons = visibleClinicalReasons(rec.reasons || []);

                      return (
                        <article key={rec.medication} className={status.className}>
                          <div className="itemHeader">
                            <strong>{rec.medication}</strong>
                            <span>{status.label}</span>
                          </div>

                          {reasons.length > 0 && (
                            <ul>
                              {reasons.map((reason, index) => (
                                <li key={index}>{reason}</li>
                              ))}
                            </ul>
                          )}

                        <details className="technicalDetails">
                          <summary>Detalhes técnicos</summary>
                          <p>Score interno: {rec.score_final}</p>
                          {rec.secondary_historical_score !== undefined && (
                            <p>Padrão retrospetivo secundário: {rec.secondary_historical_score}</p>
                          )}
                        </details>

                        <button
                          className="miniButton"
                          onClick={() =>
                            setFeedback((current) => ({
                              ...current,
                              recommendation: rec.medication,
                              decision: "accepted",
                              comment: `Recomendação aceite: ${rec.medication}.`,
                            }))
                          }
                        >
                          Usar no feedback
                        </button>
                          </article>
                        );
                      })}
                  </div>
                )}
              </div>
            </div>

            <div className="llmBox">
              <h3>Explicação gerada por LLM</h3>
              <p>
                Esta explicação é gerada, quando solicitada, a partir dos alertas e
                recomendações já produzidos pelo sistema. O LLM não toma decisões clínicas.
              </p>

              <button
                className="secondaryButton"
                onClick={generateLlmExplanation}
                disabled={loadingLlmExplanation}
              >
                {loadingLlmExplanation ? "A gerar explicação..." : "Gerar explicação LLM"}
              </button>

              {llmExplanation && (
                <div className="llmResult">
                  <div className="itemHeader">
                    <strong>Modelo utilizado</strong>
                    <span>{llmExplanation.model}</span>
                  </div>
                  <p>{llmExplanation.explanation}</p>
                </div>
              )}
            </div>
          </>
        )}
      </section>

      <section className="card fullWidth">
        <div className="cardHeader">
          <span className="step">4</span>
          <h2>Feedback do profissional de saúde</h2>
        </div>

        {!analysisResult ? (
          <p className="emptyState">
            O feedback só pode ser registado depois de uma análise.
          </p>
        ) : (
          <>
            <div className="formGrid">
              <label>
                Decisão
                <select
                  value={feedback.decision}
                  onChange={(e) =>
                    updateFeedbackField("decision", e.target.value)
                  }
                >
                  <option value="accepted">Aceitar recomendação</option>
                  <option value="rejected">Rejeitar recomendação</option>
                  <option value="ignored">Ignorar alerta/recomendação</option>
                </select>
              </label>

              <label>
                Recomendação avaliada
                <input
                  value={feedback.recommendation}
                  onChange={(e) =>
                    updateFeedbackField("recommendation", e.target.value)
                  }
                  placeholder="Ex.: paracetamol"
                />
              </label>
            </div>

            <label>
              Comentário
              <textarea
                value={feedback.comment}
                onChange={(e) => updateFeedbackField("comment", e.target.value)}
                placeholder="Justificação clínica ou observação do profissional"
              />
            </label>

            <div className="alternativeBox">
              <h3>Alternativa proposta pelo profissional</h3>
              <p>
                Opcional. Se for preenchida, o sistema tenta avaliar a alternativa
                com a base de conhecimento atual.
              </p>

              <div className="formGrid">
                <label>
                  Medicamento alternativo
                  <input
                    value={feedback.userAlternativeMedication}
                    onChange={(e) =>
                      updateFeedbackField(
                        "userAlternativeMedication",
                        e.target.value
                      )
                    }
                    placeholder="Ex.: naproxeno"
                  />
                </label>

                <label>
                  Dose
                  <input
                    value={feedback.userAlternativeDose}
                    onChange={(e) =>
                      updateFeedbackField("userAlternativeDose", e.target.value)
                    }
                  />
                </label>

                <label>
                  Frequência
                  <input
                    value={feedback.userAlternativeFrequency}
                    onChange={(e) =>
                      updateFeedbackField(
                        "userAlternativeFrequency",
                        e.target.value
                      )
                    }
                  />
                </label>

                <label>
                  Via
                  <input
                    value={feedback.userAlternativeRoute}
                    onChange={(e) =>
                      updateFeedbackField("userAlternativeRoute", e.target.value)
                    }
                  />
                </label>
              </div>

              <label>
                Justificação da alternativa
                <textarea
                  value={feedback.userAlternativeJustification}
                  onChange={(e) =>
                    updateFeedbackField(
                      "userAlternativeJustification",
                      e.target.value
                    )
                  }
                />
              </label>
            </div>

            <button
              className="primaryButton"
              onClick={submitFeedback}
              disabled={loadingFeedback}
            >
              {loadingFeedback ? "A guardar..." : "Guardar feedback"}
            </button>

            {feedbackResult && (
              <div className="feedbackResult">
                <h3>Feedback guardado</h3>
                <p>
                  <strong>Feedback ID:</strong>{" "}
                  <code>{feedbackResult.feedback_id}</code>
                </p>
                <p>
                  <strong>Decisão:</strong> {decisionLabel(feedback.decision)}
                </p>

                {feedbackResult.alternative_evaluation && (
                  <div className="alternativeEvaluation">
                    <h4>Avaliação da alternativa proposta</h4>
                    <p>{feedbackResult.alternative_evaluation.message}</p>

                    {feedbackResult.alternative_evaluation.alerts.length > 0 && (
                      <div className="list">
                        {feedbackResult.alternative_evaluation.alerts.map(
                          (alert, index) => (
                            <article
                              key={`${alert.medication}-${index}`}
                              className={`alert alert-${alert.severity}`}
                            >
                              <div className="itemHeader">
                                <strong>{alert.medication}</strong>
                                <span>{severityLabel(alert.severity)}</span>
                              </div>
                              <p>{alert.description}</p>
                            </article>
                          )
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </section>
      <section className="card fullWidth">
        <div className="cardHeader">
          <span className="step">5</span>
          <h2>Dashboard de métricas</h2>
        </div>

        <p className="dashboardIntro">
          Indicadores agregados de utilização do protótipo, incluindo análises realizadas,
          feedback registado, tipos de alerta e tempo médio de resposta.
        </p>

        <button
          className="secondaryButton"
          onClick={loadMetrics}
          disabled={loadingMetrics}
        >
          {loadingMetrics ? "A carregar métricas..." : "Atualizar métricas"}
        </button>

        {!metrics && (
          <p className="emptyState">
            Ainda não foram carregadas métricas. Clica em “Atualizar métricas” ou regista
            feedback para atualizar o dashboard.
          </p>
        )}

        {metrics && (
          <>
            <div className="metricGrid">
              <article className="metricCard">
                <span>Total de análises</span>
                <strong>{formatMetricValue(metrics.total_analyses)}</strong>
              </article>

              <article className="metricCard">
                <span>Total de feedbacks</span>
                <strong>{formatMetricValue(metrics.total_feedback)}</strong>
              </article>

              <article className="metricCard">
                <span>Taxa de aceitação</span>
                <strong>{formatPercentage(metrics.acceptance_rate)}</strong>
              </article>

              <article className="metricCard">
                <span>Tempo médio de resposta</span>
                <strong>
                  {metrics.average_response_time_ms
                    ? `${metrics.average_response_time_ms} ms`
                    : "—"}
                </strong>
              </article>

              <article className="metricCard">
                <span>Total de recomendações</span>
                <strong>{formatMetricValue(metrics.total_recommendations)}</strong>
              </article>

              <article className="metricCard">
                <span>Análises com recomendação</span>
                <strong>{formatMetricValue(metrics.analyses_with_recommendations)}</strong>
              </article>
            </div>

            <div className="dashboardColumns">
              <section className="dashboardPanel">
                <h3>Feedback por decisão</h3>

                <div className="metricRows">
                  <div>
                    <span>Aceites</span>
                    <strong>{metrics.feedback_by_decision?.accepted ?? 0}</strong>
                  </div>
                  <div>
                    <span>Rejeitadas</span>
                    <strong>{metrics.feedback_by_decision?.rejected ?? 0}</strong>
                  </div>
                  <div>
                    <span>Ignoradas</span>
                    <strong>{metrics.feedback_by_decision?.ignored ?? 0}</strong>
                  </div>
                </div>
              </section>

              <section className="dashboardPanel">
                <h3>Alertas por gravidade</h3>

                {Object.keys(metrics.alerts_by_severity || {}).length === 0 ? (
                  <p className="emptyState">Sem alertas registados.</p>
                ) : (
                  <div className="metricRows">
                    {Object.entries(metrics.alerts_by_severity).map(([severity, total]) => (
                      <div key={severity}>
                        <span>{severityLabel(severity)}</span>
                        <strong>{total}</strong>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section className="dashboardPanel">
                <h3>Alertas por tipo</h3>

                {Object.keys(metrics.alerts_by_type || {}).length === 0 ? (
                  <p className="emptyState">Sem tipos de alerta registados.</p>
                ) : (
                  <div className="metricRows">
                    {Object.entries(metrics.alerts_by_type).map(([type, total]) => (
                      <div key={type}>
                        <span>{type}</span>
                        <strong>{total}</strong>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>
          </>
        )}
      </section>
    </main>
  );
}