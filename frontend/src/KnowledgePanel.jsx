import { useEffect, useMemo, useState } from "react";
import "./KnowledgePanel.css";

const SEVERITY_LABELS = {
  low: "Baixa",
  moderate: "Moderada",
  high: "Elevada",
  critical: "Crítica",
};

const MATCH_LABELS = {
  drug_drug: "Medicamento com medicamento",
  class_drug: "Classe com medicamento",
  class_class: "Classe com classe",
  qt_qt: "Prolongamento do intervalo QT",
};

const ORIGIN_LABELS = {
  base: "Base original",
  base_modificada: "Base, alterada no painel",
  adicionada: "Adicionada no painel",
};

const CHANGE_LABELS = {
  rule_added: "Regra adicionada",
  rule_modified: "Regra alterada",
  rule_disabled: "Regra desativada",
  rule_removed: "Regra removida",
  rule_reenabled: "Regra reativada",
  medication_override: "Medicamento alterado",
};

const EMPTY_RULE = {
  id: "",
  match: "drug_drug",
  severity: "moderate",
  description: "",
  class_a: "",
  class_b: "",
  medication_a: "",
  medication_b: "",
  fonte: "",
  documento: "",
  autor: "",
};

export default function KnowledgePanel({ apiUrl }) {
  const [summary, setSummary] = useState(null);
  const [rules, setRules] = useState([]);
  const [catalog, setCatalog] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [changes, setChanges] = useState([]);

  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [search, setSearch] = useState("");
  const [originFilter, setOriginFilter] = useState("todas");

  const [newRule, setNewRule] = useState(EMPTY_RULE);
  const [disablingRule, setDisablingRule] = useState(null);
  const [justification, setJustification] = useState("");
  const [justificationError, setJustificationError] = useState("");
  const [formErrors, setFormErrors] = useState([]);
  const [formMessage, setFormMessage] = useState("");

  async function getJson(path) {
    const response = await fetch(`${apiUrl}${path}`);

    if (!response.ok) {
      throw new Error(`Erro ao consultar ${path}: ${response.status}`);
    }

    return response.json();
  }

  async function refreshAll() {
    setLoading(true);
    setError("");

    try {
      const [summaryData, rulesData, catalogData, documentsData, changesData] =
        await Promise.all([
          getJson("/kb/summary"),
          getJson("/kb/rules"),
          getJson("/kb/catalog"),
          getJson("/kb/provenance"),
          getJson("/kb/changes?limit=20"),
        ]);

      setSummary(summaryData);
      setRules(rulesData);
      setCatalog(catalogData);
      setDocuments(documentsData);
      setChanges(changesData);
    } catch (err) {
      setError(err.message || "Erro ao carregar a base de conhecimento.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refreshAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function submitChange(path, options) {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${apiUrl}${path}`, options);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Erro ${response.status}`);
      }

      setResult(data);

      window.setTimeout(() => {
        document
          .getElementById("kbChangeResult")
          ?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 0);

      if (data.applied) {
        await refreshAll();
      }

      return data;
    } catch (err) {
      setError(err.message || "Erro inesperado ao submeter a alteração.");
      return null;
    } finally {
      setLoading(false);
    }
  }

  function updateRuleField(field, value) {
    setNewRule((current) => ({ ...current, [field]: value }));
  }

  function buildRulePayload() {
    const payload = {
      id: newRule.id.trim(),
      match: newRule.match,
      severity: newRule.severity,
      description: newRule.description.trim(),
      fonte: newRule.fonte.trim(),
    };

    if (newRule.documento) {
      payload.documento = newRule.documento;
    }

    if (newRule.autor.trim()) {
      payload.autor = newRule.autor.trim();
    }

    if (newRule.match === "drug_drug") {
      payload.medication_a = newRule.medication_a;
      payload.medication_b = newRule.medication_b;
    }

    if (newRule.match === "class_class") {
      payload.class_a = newRule.class_a;
      payload.class_b = newRule.class_b;
    }

    if (newRule.match === "class_drug") {
      payload.class_a = newRule.class_a;
      payload.medication_b = newRule.medication_b;
    }

    return payload;
  }

  function validateRule() {
    const missing = [];

    if (!newRule.id.trim()) {
      missing.push("id");
    }

    if (newRule.description.trim().length < 20) {
      missing.push("description");
    }

    if (!newRule.fonte.trim()) {
      missing.push("fonte");
    }

    if (newRule.match === "drug_drug") {
      if (!newRule.medication_a) {
        missing.push("medication_a");
      }

      if (!newRule.medication_b) {
        missing.push("medication_b");
      }
    }

    if (newRule.match === "class_class") {
      if (!newRule.class_a) {
        missing.push("class_a");
      }

      if (!newRule.class_b) {
        missing.push("class_b");
      }
    }

    if (newRule.match === "class_drug") {
      if (!newRule.class_a) {
        missing.push("class_a");
      }

      if (!newRule.medication_b) {
        missing.push("medication_b");
      }
    }

    return missing;
  }

  async function createRule() {
    const missing = validateRule();
    setFormErrors(missing);

    if (missing.length > 0) {
      setFormMessage(
        "Preencha os campos assinalados a vermelho. A descrição clínica tem de ter pelo menos 20 caracteres."
      );
      return;
    }

    setFormMessage("");

    const data = await submitChange("/kb/rules", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRulePayload()),
    });

    if (data?.applied) {
      setNewRule(EMPTY_RULE);
      setFormErrors([]);
    }
  }

  async function confirmDisable() {
    if (justification.trim().length < 20) {
      setJustificationError(
        "A justificação clínica tem de ter pelo menos 20 caracteres."
      );
      return;
    }

    setJustificationError("");

    const data = await submitChange(`/kb/rules/${disablingRule.id}/disable`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        justificacao: justification,
        autor: newRule.autor.trim() || null,
      }),
    });

    if (data?.applied) {
      setDisablingRule(null);
      setJustification("");
    }
  }

  async function enableRule(ruleId) {
    await submitChange(`/kb/rules/${ruleId}/enable`, { method: "POST" });
  }

  async function removeRule(ruleId) {
    await submitChange(`/kb/rules/${ruleId}`, { method: "DELETE" });
  }

  async function uploadDocument(file) {
    if (!file) {
      return;
    }

    setUploading(true);
    setError("");

    try {
      const form = new FormData();
      form.append("file", file);

      const response = await fetch(`${apiUrl}/kb/provenance`, {
        method: "POST",
        body: form,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Erro ${response.status}`);
      }

      await refreshAll();
      updateRuleField("documento", data.documento);
    } catch (err) {
      setError(err.message || "Erro ao carregar o documento.");
    } finally {
      setUploading(false);
    }
  }

  const filteredRules = useMemo(() => {
    const term = search.trim().toLowerCase();

    return rules.filter((rule) => {
      if (originFilter === "geridas" && rule.origem === "base") {
        return false;
      }

      if (originFilter === "desativadas" && rule.enabled) {
        return false;
      }

      if (!term) {
        return true;
      }

      const haystack = [
        rule.id,
        rule.description,
        rule.class_a,
        rule.class_b,
        rule.medication_a,
        rule.medication_b,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return haystack.includes(term);
    });
  }, [rules, search, originFilter]);

  const medicationOptions = catalog?.medications || [];
  const classOptions = catalog?.therapeutic_classes || [];

  return (
    <section className="knowledgePanel">
      <section className="card fullWidth">
        <div className="cardHeader">
          <span className="step">A</span>
          <h2>Conhecimento em vigor</h2>
        </div>

        <p className="dashboardIntro">
          Este painel permite consultar e corrigir o conhecimento clínico que o sistema
          aplica. Cada alteração exige fonte documental e só é gravada se os cenários de
          regressão clínica continuarem a passar na totalidade.
        </p>

        {error && <div className="errorBox">{error}</div>}

        {summary && (
          <div className="metricGrid">
            <article className="metricCard">
              <span>Regras na base original</span>
              <strong>{summary.rules_base}</strong>
            </article>

            <article className="metricCard">
              <span>Regras ativas</span>
              <strong>{summary.rules_ativas}</strong>
            </article>

            <article className="metricCard">
              <span>Medicamentos</span>
              <strong>{summary.medications}</strong>
            </article>

            <article className="metricCard">
              <span>Regras adicionadas</span>
              <strong>{summary.extensions?.added_rules ?? 0}</strong>
            </article>

            <article className="metricCard">
              <span>Regras desativadas</span>
              <strong>{summary.extensions?.disabled_rules ?? 0}</strong>
            </article>

            <article className="metricCard">
              <span>Medicamentos alterados</span>
              <strong>{summary.extensions?.medication_overrides ?? 0}</strong>
            </article>
          </div>
        )}

        <button
          className="secondaryButton"
          onClick={refreshAll}
          disabled={loading}
        >
          {loading ? "A carregar..." : "Atualizar"}
        </button>

        {result && (
          <div
            id="kbChangeResult"
            className={result.applied ? "changeResult applied" : "changeResult refused"}
          >            <strong>{result.applied ? "Alteração gravada" : "Alteração recusada"}</strong>
            <p>{result.message}</p>

            {result.errors?.length > 0 && (
              <ul className="errorList">
                {result.errors.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            )}

            {result.validation?.cases?.length > 0 && (
              <div className="regressionReport">
                <p className="technicalNote">
                  Validação de segurança: {result.validation.total - result.validation.failed} de{" "}
                  {result.validation.total} cenários clínicos passaram.
                </p>

                <ul>
                  {result.validation.cases
                    .filter((item) => !item.passed)
                    .map((item) => (
                      <li key={item.id}>
                        <strong>
                          {item.id} — {item.name}
                        </strong>
                        <ul>
                          {item.failures.map((failure) => (
                            <li key={failure}>{failure}</li>
                          ))}
                        </ul>
                      </li>
                    ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </section>

      <section className="card fullWidth">
        <div className="cardHeader">
          <span className="step">B</span>
          <h2>Regras de interação</h2>
        </div>

        <div className="ruleFilters">
          <label>
            Pesquisar
            <input
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Identificador, medicamento, classe ou descrição"
            />
          </label>

          <label>
            Mostrar
            <select
              value={originFilter}
              onChange={(event) => setOriginFilter(event.target.value)}
            >
              <option value="todas">Todas as regras</option>
              <option value="geridas">Apenas alteradas no painel</option>
              <option value="desativadas">Apenas desativadas</option>
            </select>
          </label>
        </div>

        <p className="technicalNote">
          {filteredRules.length} regra(s) apresentada(s) em {rules.length}.
        </p>

        <ul className="ruleList">
          {filteredRules.map((rule) => (
            <li
              key={rule.id}
              className={rule.enabled ? "ruleItem" : "ruleItem ruleDisabled"}
            >
              <div className="itemHeader">
                <strong>{rule.id}</strong>
                <span>{SEVERITY_LABELS[rule.severity] || rule.severity}</span>
              </div>

              <p>{rule.description}</p>

              <p className="technicalNote">
                {MATCH_LABELS[rule.match] || rule.match} ·{" "}
                {ORIGIN_LABELS[rule.origem] || rule.origem} ·{" "}
                {rule.enabled ? "Ativa" : "Desativada"}
                {rule.fonte ? ` · Fonte: ${rule.fonte}` : ""}
              </p>

              {rule.documento && (
                <p className="technicalNote">
                  
                    <a href={`${apiUrl}/kb/provenance/${encodeURIComponent(rule.documento)}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Abrir documento de proveniência
                  </a>
                </p>
              )}

              {!rule.enabled && rule.justificacao_desativacao && (
                <p className="technicalNote">
                  Justificação: {rule.justificacao_desativacao}
                </p>
              )}

              <div className="outcomeButtons">
                {rule.enabled ? (
                  <button
                    className="secondaryButton"
                    disabled={loading}
                    onClick={() => {
                      setDisablingRule(rule);
                      setJustification("");
                    }}
                  >
                    Desativar
                  </button>
                ) : (
                  <button
                    className="secondaryButton"
                    disabled={loading}
                    onClick={() => enableRule(rule.id)}
                  >
                    Reativar
                  </button>
                )}

                {rule.origem === "adicionada" && (
                  <button
                    className="secondaryButton"
                    disabled={loading}
                    onClick={() => removeRule(rule.id)}
                  >
                    Remover
                  </button>
                )}
              </div>

              {disablingRule?.id === rule.id && (
                <div className="disableForm">
                  <label>
                    Justificação clínica da desativação
                    <textarea
                      className={justificationError ? "fieldError" : undefined}
                      rows={2}
                      value={justification}
                      onChange={(event) => setJustification(event.target.value)}
                      placeholder="Mínimo de 20 caracteres. Fica registada na auditoria."
                    />
                  </label>

                  {justificationError && (
                    <p className="fieldErrorText">{justificationError}</p>
                  )}

                  {result && !result.applied && (
                    <p className="fieldErrorText">{result.message}</p>
                  )}

                  <div className="outcomeButtons">
                    <button
                      className="primaryButton"
                      disabled={loading}
                      onClick={confirmDisable}
                    >
                      Confirmar desativação
                    </button>

                    <button
                      className="secondaryButton"
                      onClick={() => {
                        setDisablingRule(null);
                        setJustification("");
                      }}
                    >
                      Cancelar
                    </button>
                  </div>
                </div>
              )}
            </li>
          ))}
        </ul>
      </section>

      <section className="card fullWidth">
        <div className="cardHeader">
          <span className="step">C</span>
          <h2>Nova regra de interação</h2>
        </div>

        <div className="ruleForm">
          <label className={formErrors.includes("id") ? "fieldError" : undefined}>
            Identificador
            <input
              type="text"
              value={newRule.id}
              onChange={(event) => updateRuleField("id", event.target.value)}
              placeholder="Ex.: alopurinol_azatioprina_mielotoxicidade"
            />
          </label>

          <label>
            Tipo de correspondência
            <select
              value={newRule.match}
              onChange={(event) => {
                updateRuleField("match", event.target.value);
                setFormErrors([]);
              }}
            >
              {Object.entries(MATCH_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Severidade
            <select
              value={newRule.severity}
              onChange={(event) => updateRuleField("severity", event.target.value)}
            >
              {Object.entries(SEVERITY_LABELS).map(([value, label]) => (
                <option key={value} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>

          {(newRule.match === "class_class" || newRule.match === "class_drug") && (
            <label className={formErrors.includes("class_a") ? "fieldError" : undefined}>
              Classe terapêutica
              <select
                value={newRule.class_a}
                onChange={(event) => updateRuleField("class_a", event.target.value)}
              >
                <option value="">Selecionar</option>
                {classOptions.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
          )}

          {newRule.match === "class_class" && (
            <label className={formErrors.includes("class_b") ? "fieldError" : undefined}>
              Segunda classe terapêutica
              <select
                value={newRule.class_b}
                onChange={(event) => updateRuleField("class_b", event.target.value)}
              >
                <option value="">Selecionar</option>
                {classOptions.map((item) => (
                  <option key={item} value={item}>
                    {item}
                  </option>
                ))}
              </select>
            </label>
          )}

          {newRule.match === "drug_drug" && (
            <label
              className={formErrors.includes("medication_a") ? "fieldError" : undefined}
            >
              Primeiro medicamento
              <select
                value={newRule.medication_a}
                onChange={(event) => updateRuleField("medication_a", event.target.value)}
              >
                <option value="">Selecionar</option>
                {medicationOptions.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.display_name}
                  </option>
                ))}
              </select>
            </label>
          )}

          {(newRule.match === "drug_drug" || newRule.match === "class_drug") && (
            <label
              className={formErrors.includes("medication_b") ? "fieldError" : undefined}
            >
              {newRule.match === "drug_drug" ? "Segundo medicamento" : "Medicamento"}
              <select
                value={newRule.medication_b}
                onChange={(event) => updateRuleField("medication_b", event.target.value)}
              >
                <option value="">Selecionar</option>
                {medicationOptions.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.display_name}
                  </option>
                ))}
              </select>
            </label>
          )}

          <label
            className={
              formErrors.includes("description") ? "fullRow fieldError" : "fullRow"
            }
          >
            Descrição clínica
            <textarea
              rows={3}
              value={newRule.description}
              onChange={(event) => updateRuleField("description", event.target.value)}
              placeholder="Mínimo de 20 caracteres. É este o texto apresentado no alerta."
            />
          </label>

          <label
            className={formErrors.includes("fonte") ? "fullRow fieldError" : "fullRow"}
          >
            Fonte documental
            <input
              type="text"
              value={newRule.fonte}
              onChange={(event) => updateRuleField("fonte", event.target.value)}
              placeholder="Ex.: RCM Alopurinol Generis, secção 4.5"
            />
          </label>

          <label>
            Documento associado
            <select
              value={newRule.documento}
              onChange={(event) => updateRuleField("documento", event.target.value)}
            >
              <option value="">Nenhum</option>
              {documents.map((item) => (
                <option key={item.documento} value={item.documento}>
                  {item.documento}
                </option>
              ))}
            </select>
          </label>

          <label>
            Autor
            <input
              type="text"
              value={newRule.autor}
              onChange={(event) => updateRuleField("autor", event.target.value)}
              placeholder="Identificação do profissional"
            />
          </label>

          <label className="fullRow">
            Carregar novo documento (PDF)
            <input
              type="file"
              accept="application/pdf"
              disabled={uploading}
              onChange={(event) => uploadDocument(event.target.files?.[0])}
            />
          </label>
        </div>

        {formMessage && <p className="fieldErrorText">{formMessage}</p>}

        <button className="primaryButton" onClick={createRule} disabled={loading}>
          {loading ? "A validar..." : "Validar e gravar regra"}
        </button>
      </section>

      <section className="card fullWidth">
        <div className="cardHeader">
          <span className="step">D</span>
          <h2>Histórico de alterações</h2>
        </div>

        {changes.length === 0 ? (
          <p className="emptyState">
            Ainda não foram registadas alterações ao conhecimento clínico.
          </p>
        ) : (
          <ul className="changeList">
            {changes.map((change) => (
              <li key={change.change_id}>
                <div className="itemHeader">
                  <strong>{CHANGE_LABELS[change.change_type] || change.change_type}</strong>
                  <span>{change.target_id}</span>
                </div>

                <p className="technicalNote">
                  {new Date(change.created_at).toLocaleString("pt-PT")}
                  {change.author ? ` · ${change.author}` : ""}
                  {change.source_reference ? ` · ${change.source_reference}` : ""}
                  {change.validation?.total
                    ? ` · ${change.validation.total - change.validation.failed}/${change.validation.total} cenários validados`
                    : ""}
                </p>

                {change.justification && (
                  <p className="technicalNote">Justificação: {change.justification}</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </section>
    </section>
  );
}