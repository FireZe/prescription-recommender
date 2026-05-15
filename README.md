# Prescription Recommender

Protótipo académico de apoio à decisão clínica para identificação de potenciais erros de prescrição medicamentosa, geração de alertas de segurança, recomendação de alternativas terapêuticas e explicação contextual com LLM.

## Objetivo

O sistema analisa uma prescrição no contexto clínico do utente e identifica potenciais problemas como:

- interações medicamentosas;
- duplicação terapêutica;
- risco renal;
- contraindicações específicas;
- riscos associados à medicação ativa do utente.

O protótipo foi desenvolvido no âmbito de uma dissertação de mestrado sobre um sistema de recomendação e alertas para identificar e prevenir erros de prescrição de medicamentos.

## Componentes principais

- Backend em FastAPI;
- motor de regras clínicas determinísticas;
- sistema de recomendação baseado em conhecimento;
- scoring supervisionado offline com silver labels;
- feedback do profissional de saúde;
- dashboard de métricas;
- explicação contextual com LLM local via Ollama;
- frontend em React.

## Estrutura do projeto

```text
backend/
  app/
    main.py
    rules_engine.py
    recommender.py
    llm_explainer.py
    schemas.py
    database.py
    normalization.py
    synthea_loader.py
  data/
    knowledge_base.json
    historical_patterns.json
  scripts/
    build_synthea_context_index.py
    generate_synthea_training_examples_v2.py
    train_silver_label_model.py
    run_clinical_regression_report.py
  tests/
    clinical_regression_cases.py
    test_clinical_scenarios.py

frontend/
  src/
    App.jsx
    App.css