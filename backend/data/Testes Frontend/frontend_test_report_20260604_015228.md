# Relatório de testes do sistema — 20260604_015228

## AINE + antiagregante (risco hemorragia GI)
- **Doente:** 70a F, problema=pain, renal=normal, ativos=['clopidogrel'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa clínica:** Alerta alto AINE+antiagregante; recomendar paracetamol.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
- **Recomendações (medicamento — heurístico / combinado / final):**
    - paracetamol: 0.860 / 0.919 / 0.919
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre ibuprofeno e clopidogrel. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou paracetamol como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## AINE + anticoagulante (varfarina)
- **Doente:** 68a M, problema=pain, renal=normal, ativos=['warfarin'], condições=['atrial_fibrillation'], alergias=—
- **Prescrição:** naproxen
- **Expectativa clínica:** Alerta de hemorragia; alternativa não-AINE.
- **Alertas:**
    - [high] A associacao de um AINE com um anticoagulante aumenta o risco de hemorragia e deve ser evitada ou monitorizada. (regra: aine_anticoagulante_hemorragia)
- **Recomendações (medicamento — heurístico / combinado / final):**
    - paracetamol: 0.860 / 0.896 / 0.896
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de naproxeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre Varfarina e Naproxeno. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo. A associação de um AINE com um anticoagulante aumenta o risco de hemorragia.    3. Motivo da recomendação   O sistema identificou Paracetamol como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## AINE em insuficiência renal grave
- **Doente:** 75a F, problema=pain, renal=severe_impairment, ativos=—, condições=['renal_disease'], alergias=—
- **Prescrição:** ibuprofen
- **Expectativa clínica:** Precaução renal; preferir paracetamol.
- **Alertas:**
    - [high] Ibuprofeno requer precaução acrescida em doentes com compromisso renal grave. (regra: renal_caution)
- **Recomendações (medicamento — heurístico / combinado / final):**
    - paracetamol: 0.390 / 0.995 / 0.995
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou risco renal associado ao medicamento prescrito. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou paracetamol como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## AINE com úlcera GI ativa (contraindicação)
- **Doente:** 60a M, problema=pain, renal=normal, ativos=—, condições=['active_gi_ulcer'], alergias=—
- **Prescrição:** naproxen
- **Expectativa clínica:** Contraindicação por úlcera GI ativa.
- **Alertas:**
    - [critical] Naproxeno está contraindicado ou deve ser evitado neste contexto clínico devido à condição clínica identificada: active_gi_ulcer. (regra: —)
- **Recomendações (medicamento — heurístico / combinado / final):**
    - paracetamol: 0.860 / 0.995 / 0.995
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de Naproxeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou uma contraindicação entre Naproxeno e a condição clínica de úlcera ativa no trato gastrointestinal. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou Paracetamol como alternativa sugerida. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## Macrólido + amiodarona (risco QT)
- **Doente:** 72a M, problema=infection, renal=normal, ativos=['amiodarone'], condições=['arrhythmia'], alergias=—
- **Prescrição:** azithromycin
- **Expectativa clínica:** Alerta de risco QT combinado.
- **Alertas:**
    - [high] A associacao de dois farmacos com potencial de prolongamento do intervalo QT pode aumentar o risco de arritmias ventriculares, incluindo torsades de pointes. (regra: qt_risk_combination)
- **Recomendações:** nenhuma
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta infeção. Foi submetida prescrição de azitromicina, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma associação de fármacos com risco de prolongamento do intervalo QT, envolvendo amiodarona e azitromicina. Ambos os medicamentos têm potencial de aumentar o risco de arritmias ventriculares, incluindo torsades de pointes. Este alerta está relacionado com a prescrição submetida e com a medicação ativa do utente.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual. A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual.

## Paracetamol em inflamação após AINE (fallback sintomático)
- **Doente:** 55a F, problema=inflammation, renal=normal, ativos=['ibuprofen'], condições=['inflammation'], alergias=—
- **Prescrição:** paracetamol
- **Expectativa clínica:** Nota: paracetamol é sintomático, não substitui o AINE.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta inflamação e foi submetida prescrição de paracetamol, um analgésico e antipirético, com indicação compatível com o problema clínico principal.  2. Motivo do alerta   A análise baseia-se nos dados submetidos e na base de conhecimento atual do protótipo. Não foram identificados alertas relevantes, pois o paracetamol não apresenta interações com a medicação ativa ou com o contexto clínico consolidado.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. Ibuprofeno já consta da medicação ativa do utente e tem indicação compatível com a inflamação, pelo que não foi apresentada como nova alternativa.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não avalia ajuste de dose, suspensão ou opções terapêuticas fora da base atual.

## Alergia conhecida ao candidato
- **Doente:** 40a F, problema=pain, renal=normal, ativos=—, condições=—, alergias=['ibuprofen']
- **Prescrição:** ibuprofen
- **Expectativa clínica:** Conflito com alergia; alternativa segura.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, um AINE indicado para dor, febre e inflamação.  2. Motivo do alerta   A análise baseia-se no contexto clínico consolidado e na prescrição submetida. Não foram identificados alertas relevantes, pois o utente não tem condições ou medicações ativas que gerem interações. O ibuprofeno está indicado para dor, mas o utente tem alergia a este medicamento.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. O sistema não identificou uma alternativa terapêutica admissível dentro da sua base atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## Caso de controlo (sem fatores de risco)
- **Doente:** 30a M, problema=pain, renal=normal, ativos=—, condições=—, alergias=—
- **Prescrição:** paracetamol
- **Expectativa clínica:** Sem alerta bloqueante; prescrição adequada.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de paracetamol, um analgésico e antipirético, com indicação para dor e febre.  2. Motivo do alerta   A análise baseia-se no contexto clínico consolidado e na prescrição submetida. Não foram identificados alertas relevantes, pois o utente não tem condições, alergias ou medicações ativas, e o estado renal é normal.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. O sistema não identificou uma alternativa terapêutica admissível dentro da sua base atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## Idoso polimedicado, hipertensão
- **Doente:** 82a F, problema=hypertension, renal=mild_impairment, ativos=['furosemide', 'ramipril'], condições=['hypertension', 'heart_failure'], alergias=—
- **Prescrição:** ibuprofen
- **Expectativa clínica:** AINE em IC/HTA/diurético: cautela (nefro/retenção).
- **Alertas:**
    - [moderate] A associacao de AINEs com diureticos pode atenuar o efeito diuretico e aumentar o risco de compromisso renal. (regra: aine_diuretico_risco_renal)
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
    - [high] Associação com risco aumentado de deterioração da função renal: AINE em combinação com inibidor da ECA ou antagonista dos recetores da angiotensina II e diurético. Recomenda-se evitar a associação ou monitorizar função renal, hidratação e eletrólitos. (regra: triple_whammy)
- **Recomendações (medicamento — heurístico / combinado / final):**
    - paracetamol: 0.670 / 0.598 / 0.598
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta hipertensão e insuficiência cardíaca com compromisso renal ligeiro/moderado. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.  2. Motivo do alerta   O sistema identificou associação de AINE com inibidor da enzima de conversão da angiotensina (IECA) e diurético de ansa. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. Motivo da recomendação   O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## Infeção simples sem riscos
- **Doente:** 45a M, problema=infection, renal=normal, ativos=—, condições=—, alergias=—
- **Prescrição:** azithromycin
- **Expectativa clínica:** Sem alerta relevante esperado.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Explicação LLM (qwen3:4b-instruct):**
  > 1. Problema identificado   O utente apresenta uma infeção. Foi submetida prescrição de azitromicina, um macrólido indicado para tratamento de infecções.  2. Motivo do alerta   A análise baseia-se no contexto clínico consolidado e na base de conhecimento atual do protótipo. Não foram identificados alertas relevantes, pois o medicamento prescrito está indicado para o problema principal e não há interações com medicações ativas.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. O protótipo não identificou uma alternativa terapêutica admissível dentro da sua base atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.
