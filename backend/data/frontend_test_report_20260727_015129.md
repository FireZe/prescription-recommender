# Relatório de testes — 20260727_015129

**Total de casos:** 53 | **com alerta:** 45 | **alertas por severidade:** {'high': 33, 'moderate': 25, 'critical': 4}

## [hand-crafted] AINE + antiagregante (hemorragia)
- **Doente:** 70a F, problema=pain, renal=normal, ativos=['clopidogrel'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta alto; recomendar paracetamol.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.923/0.923
    - tramadol: 0.910/0.386/0.386
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre ibuprofeno e clopidogrel. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo. A associação de um AINE com um antiagregante plaquetário aumenta o risco de hemorragia gastrointestinal.    3. Motivo da recomendação   O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. Tramadol foi sugerido como alternativa com precaução.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + anticoagulante (hemorragia)
- **Doente:** 68a M, problema=pain, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** naproxen
- **Expectativa:** Alerta alto; alternativa não-AINE.
- **Alertas:**
    - [high] A associacao de um AINE com um anticoagulante aumenta o risco de hemorragia e deve ser evitada ou monitorizada. (regra: aine_anticoagulante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.926/0.926
    - tramadol: 0.910/0.364/0.364
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de naproxeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre Varfarina e Naproxeno. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo. A associação de um AINE com um anticoagulante aumenta o risco de hemorragia.    3. Motivo da recomendação   O sistema identificou Paracetamol como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. Tramadol foi sugerido com precaução devido à sua classe terapêutica.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Duplicação AINE+AINE
- **Doente:** 50a M, problema=pain, renal=normal, ativos=['ibuprofen'], condições=—, alergias=—
- **Prescrição:** naproxen
- **Expectativa:** Alerta de duplicação terapêutica.
- **Alertas:**
    - [high] A utilizacao concomitante de dois AINEs deve ser evitada devido ao aumento do risco de toxicidade gastrointestinal, renal e hemorragica. (regra: aine_aine_duplicacao)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.995/0.995
    - tramadol: 0.910/0.672/0.672
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de naproxeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou duplicação terapêutica de AINEs entre ibuprofeno e naproxeno. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou paracetamol e tramadol como alternativas admissíveis na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + IECA (risco renal)
- **Doente:** 65a M, problema=pain, renal=normal, ativos=['ramipril'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta moderado renal.
- **Alertas:**
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.438/0.438
    - tramadol: 0.910/0.032/0.032
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre AINE e inibidor da enzima de conversão da angiotensina (IECA) entre Ramipril e Ibuprofeno. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou Paracetamol e Tramadol como alternativas com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para estas alternativas.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + ARA (risco renal)
- **Doente:** 65a F, problema=pain, renal=normal, ativos=['losartan'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta moderado renal.
- **Alertas:**
    - [moderate] A associacao de AINEs com antagonistas dos recetores da angiotensina II pode reduzir o efeito anti-hipertensor e aumentar o risco de agravamento da funcao renal. (regra: aine_ara_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.442/0.442
    - tramadol: 0.910/0.034/0.034
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre ibuprofeno e losartan. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo. A associação de AINE com antagonista dos recetores da angiotensina II pode reduzir o efeito anti-hipertensor e aumentar o risco de agravamento do estado renal.    3. Motivo da recomendação   O sistema identificou paracetamol e tramadol como alternativas com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para estas alternativas.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + diurético de ansa
- **Doente:** 72a F, problema=pain, renal=normal, ativos=['furosemide'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta moderado renal/diurético.
- **Alertas:**
    - [moderate] A associacao de AINEs com diureticos pode atenuar o efeito diuretico e aumentar o risco de compromisso renal. (regra: aine_diuretico_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.534/0.534
    - tramadol: 0.910/0.074/0.074
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre AINE e diurético entre furosemida e ibuprofeno. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + tiazida
- **Doente:** 72a M, problema=pain, renal=normal, ativos=['hydrochlorothiazide'], condições=—, alergias=—
- **Prescrição:** naproxen
- **Expectativa:** Alerta moderado tiazida.
- **Alertas:**
    - [moderate] A associacao de AINEs com diureticos tiazidicos pode reduzir a eficacia anti-hipertensora e aumentar o risco de deterioracao da funcao renal. (regra: aine_tiazida_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.527/0.527
    - tramadol: 0.910/0.070/0.070
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de naproxeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre hidroclorotiazida e naproxeno. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou paracetamol como alternativa sugerida e tramadol como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Triple whammy (AINE+IECA+diurético)
- **Doente:** 82a F, problema=hypertension, renal=mild_impairment, ativos=['ramipril', 'furosemide'], condições=['hypertension', 'heart_failure'], alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta alto triple whammy + os moderados.
- **Alertas:**
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
    - [moderate] A associacao de AINEs com diureticos pode atenuar o efeito diuretico e aumentar o risco de compromisso renal. (regra: aine_diuretico_risco_renal)
    - [high] Associação com risco aumentado de deterioração da função renal: AINE em combinação com inibidor da ECA ou antagonista dos recetores da angiotensina II e diurético. Recomenda-se evitar a associação ou monitorizar função renal, hidratação e eletrólitos. (regra: triple_whammy)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.750/0.321/0.321
    - tramadol: 0.720/0.199/0.199
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta hipertensão e insuficiência cardíaca. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.  2. Motivo do alerta   O sistema identificou associação de AINE com inibidor da enzima de conversão da angiotensina (IECA) e diurético de ansa. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. Motivo da recomendação   O sistema identificou paracetamol como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Sinvastatina + claritromicina (CRÍTICO)
- **Doente:** 60a M, problema=infection, renal=normal, ativos=['simvastatin'], condições=—, alergias=—
- **Prescrição:** clarithromycin
- **Expectativa:** Interação crítica (miopatia/rabdomiólise).
- **Alertas:**
    - [critical] A administracao concomitante de sinvastatina e claritromicina esta contraindicada devido ao aumento do risco de miopatia e rabdomiolise. (regra: sinvastatina_claritromicina_contraindicada)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.685/0.685
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta infeção. Foi submetida prescrição de claritromicina, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou uma interação crítica entre sinvastatina e claritromicina. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou amoxicilina como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Estatina + macrólido
- **Doente:** 61a F, problema=infection, renal=normal, ativos=['atorvastatin'], condições=—, alergias=—
- **Prescrição:** azithromycin
- **Expectativa:** Alerta alto miopatia.
- **Alertas:**
    - [high] A associacao de estatinas com macrolidos pode aumentar o risco de miopatia ou rabdomiolise. A relevancia clinica depende do macrolido e da estatina. (regra: estatina_macrolido_miopatia)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.674/0.674
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta infeção. Foi submetida prescrição de azitromicina, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação estatina + macrólido entre atorvastatina e azitromicina. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou amoxicilina como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Varfarina + amiodarona
- **Doente:** 70a M, problema=arrhythmia, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** amiodarone
- **Expectativa:** Alerta alto (potenciação anticoagulante).
- **Alertas:**
    - [high] A amiodarona pode potenciar o efeito anticoagulante da varfarina, aumentando o risco de hemorragia. Recomenda-se monitorizacao rigorosa do INR. (regra: varfarina_amiodarona)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta arritmia. Foi submetida prescrição de amiodarona, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma interação entre varfarina e amiodarona, com base no contexto clínico consolidado e nas regras implementadas. A amiodarona pode potenciar o efeito anticoagulante da varfarina, aumentando o risco de hemorragia. Este alerta envolve tanto a medicação ativa do utente como a prescrição submetida.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. O protótipo não identificou uma alternativa terapêutica admissível dentro da sua base atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Macrólido + varfarina
- **Doente:** 69a F, problema=infection, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** clarithromycin
- **Expectativa:** Alerta alto.
- **Alertas:**
    - [high] A associacao de varfarina com antibioticos macrolidos pode aumentar o efeito anticoagulante e o risco de hemorragia. Deve ser considerada monitorizacao do INR. (regra: varfarina_macrolido)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.344/0.344
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta infeção. Foi submetida prescrição de claritromicina, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre varfarina e claritromicina. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou amoxicilina como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] ISRS + anticoagulante
- **Doente:** 58a F, problema=depression, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** sertraline
- **Expectativa:** Alerta moderado hemorragia.
- **Alertas:**
    - [moderate] Os ISRS podem aumentar o risco hemorragico quando associados a anticoagulantes. Deve ser ponderada monitorizacao clinica e laboratorial. (regra: sertralina_anticoagulante)
- **Recomendações (med — heurístico/combinado/final):**
    - amitriptyline: 0.910/0.088/0.088
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta depressão. Foi submetida prescrição de sertralina, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre inibidor seletivo da recaptação da serotonina (ISRS) e anticoagulante. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou amitriptilina como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] ISRS + antiagregante
- **Doente:** 58a M, problema=depression, renal=normal, ativos=['clopidogrel'], condições=—, alergias=—
- **Prescrição:** sertraline
- **Expectativa:** Alerta moderado.
- **Alertas:**
    - [moderate] Os ISRS podem aumentar o risco de hemorragia quando associados a farmacos com efeito antiagregante plaquetario. (regra: sertralina_antiagregante)
- **Recomendações (med — heurístico/combinado/final):**
    - amitriptyline: 0.910/0.126/0.126
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta depressão. Foi submetida prescrição de sertralina, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre inibidor seletivo da recaptação da serotonina (ISRS) e antiagregante plaquetário. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou amitriptilina como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] ISRS + AINE
- **Doente:** 45a F, problema=depression, renal=normal, ativos=['ibuprofen'], condições=—, alergias=—
- **Prescrição:** sertraline
- **Expectativa:** Alerta moderado.
- **Alertas:**
    - [moderate] A associacao de ISRS com AINEs pode aumentar o risco de hemorragia gastrointestinal. (regra: sertralina_aine)
- **Recomendações (med — heurístico/combinado/final):**
    - amitriptyline: 0.910/0.364/0.364
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta depressão. Foi submetida prescrição de sertralina, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação ISRS + AINE entre ibuprofeno e sertralina. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou amitriptilina como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] ISRS + tricíclico (serotoninérgico)
- **Doente:** 50a M, problema=depression, renal=normal, ativos=['amitriptyline'], condições=—, alergias=—
- **Prescrição:** sertraline
- **Expectativa:** Alerta alto (síndrome serotoninérgica).
- **Alertas:**
    - [high] A associacao de antidepressivos serotoninergicos pode aumentar o risco de síndrome serotoninérgica. (regra: serotoninergicos)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta depressão. Foi submetida prescrição de sertralina, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma associação entre amitriptilina e sertralina, ambos antidepressivos serotoninérgicos. Esta combinação pode aumentar o risco de síndrome serotoninérgica. O medicamento prescrito foi avaliado no contexto da medicação ativa do utente.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual. A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual do protótipo.

## [hand-crafted] Combinação QT (amiodarona + azitromicina)
- **Doente:** 72a M, problema=infection, renal=normal, ativos=['amiodarone'], condições=['arrhythmia'], alergias=—
- **Prescrição:** azithromycin
- **Expectativa:** Alerta alto QT.
- **Alertas:**
    - [high] A associacao de dois farmacos com potencial de prolongamento do intervalo QT pode aumentar o risco de arritmias ventriculares, incluindo torsades de pointes. (regra: qt_risk_combination)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.030/0.030
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta infeção. Foi submetida prescrição de azitromicina, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou associação de fármacos com risco de prolongamento do intervalo QT entre Amiodarona e Azitromicina. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou Amoxicilina como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Amiodarona + digoxina
- **Doente:** 74a M, problema=arrhythmia, renal=normal, ativos=['digoxin'], condições=—, alergias=—
- **Prescrição:** amiodarone
- **Expectativa:** Alerta alto (toxicidade digoxina).
- **Alertas:**
    - [high] A amiodarona pode aumentar a exposicao a digoxina e potenciar perturbacoes de conducao. Recomenda-se monitorizacao de ECG e niveis de digoxina. (regra: amiodarona_digoxina)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta arritmia. Foi submetida prescrição de amiodarona, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma interação entre amiodarona e digoxina, com base no contexto clínico consolidado. A amiodarona pode aumentar a exposição a digoxina e potenciar perturbações da condução elétrica. Ambos os medicamentos estão ativos no utente: digoxina como medicação ativa e amiodarona como prescrição submetida.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Diurético de ansa + digoxina
- **Doente:** 78a F, problema=heart_failure, renal=normal, ativos=['digoxin'], condições=—, alergias=—
- **Prescrição:** furosemide
- **Expectativa:** Alerta moderado (hipocaliemia).
- **Alertas:**
    - [moderate] A hipocaliemia induzida por diureticos pode aumentar o risco de toxicidade digitalica. Recomenda-se monitorizacao do potassio serico. (regra: diuretico_digoxina_hipocaliemia)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta insuficiência cardíaca. Foi submetida prescrição de furosemida, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma interação entre furosemida e digoxina, com base no contexto clínico consolidado. A furosemida, como diurético de ansa, pode induzir hipocaliemia, aumentando o risco de toxicidade digitalica. Este alerta envolve tanto a medicação ativa do utente quanto a prescrição submetida.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Beta-bloqueante + amiodarona (bradicardia)
- **Doente:** 70a M, problema=arrhythmia, renal=normal, ativos=['bisoprolol'], condições=—, alergias=—
- **Prescrição:** amiodarone
- **Expectativa:** Alerta alto bradicardia.
- **Alertas:**
    - [high] A associacao de um beta-bloqueante com amiodarona pode causar bradicardia sinusal grave e perturbacoes da conducao auriculo-ventricular. Combinacao nao recomendada. (regra: beta_bloqueante_amiodarona_bradicardia)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta arritmia. Foi submetida prescrição de amiodarona, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma interação entre bisoprolol e amiodarona, com base nas regras implementadas e no contexto clínico consolidado. A combinação pode causar bradicardia sinusal grave e perturbações da condução auriculo-ventricular. O medicamento prescrito foi avaliado no contexto da medicação ativa do utente.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual do protótipo. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Duplicação beta-bloqueante
- **Doente:** 66a M, problema=hypertension, renal=normal, ativos=['metoprolol'], condições=—, alergias=—
- **Prescrição:** atenolol
- **Expectativa:** Alerta moderado duplicação.
- **Alertas:**
    - [high] A utilizacao concomitante de dois beta-bloqueantes representa duplicacao terapeutica e pode potenciar efeitos bradicardicos e hipotensores. (regra: beta_bloqueante_duplicacao)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta hipertensão. Foi submetida prescrição de atenolol, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma interação de duplicação terapêutica entre metoprolol e atenolol, ambos beta-bloqueantes. A utilização concomitante de dois beta-bloqueantes representa duplicação terapêutica e pode potenciar efeitos bradicardicos e hipotensores. Este alerta envolve tanto a medicação ativa do utente como a prescrição submetida.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual. A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual.

## [hand-crafted] Contraindicação: úlcera GI ativa
- **Doente:** 60a M, problema=pain, renal=normal, ativos=—, condições=['active_gi_ulcer'], alergias=—
- **Prescrição:** naproxen
- **Expectativa:** Alerta crítico de contraindicação.
- **Alertas:**
    - [critical] Naproxeno está contraindicado ou deve ser evitado neste contexto clínico devido à condição clínica identificada: active_gi_ulcer. (regra: contraindication)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.999/0.999
    - tramadol: 0.910/0.844/0.844
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de Naproxeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou uma contraindicação entre Naproxeno e a condição clínica de úlcera ativa no trato gastrointestinal. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou Paracetamol e Tramadol como alternativas admissíveis. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para estas alternativas.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE em insuficiência renal grave
- **Doente:** 75a F, problema=pain, renal=severe_impairment, ativos=—, condições=['renal_disease'], alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta renal; preferir paracetamol.
- **Alertas:**
    - [high] Ibuprofeno requer precaução acrescida em doentes com compromisso renal grave. (regra: renal_caution)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.999/0.999
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou risco renal associado ao medicamento prescrito. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou paracetamol como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Alergia ao medicamento prescrito
- **Doente:** 40a F, problema=pain, renal=normal, ativos=—, condições=—, alergias=['ibuprofen']
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta CRÍTICO de alergia (regressão da nova regra).
- **Alertas:**
    - [critical] Ibuprofeno está registado como alergia do utente. A prescrição deve ser evitada. (regra: allergy_conflict)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.999/0.999
    - tramadol: 0.910/0.826/0.826
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou um alerta de alergia entre ibuprofeno e a alergia registada do utente. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou paracetamol e tramadol como alternativas admissíveis. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para estas alternativas.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Multi-fármaco (3 de uma vez)
- **Doente:** 70a M, problema=cardiovascular_prevention, renal=normal, ativos=['clopidogrel'], condições=—, alergias=—
- **Prescrição:** ibuprofen, warfarin, simvastatin
- **Expectativa:** Vários alertas em simultâneo.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
    - [high] A associacao de um AINE com um anticoagulante aumenta o risco de hemorragia e deve ser evitada ou monitorizada. (regra: aine_anticoagulante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.750/0.999/0.999
    - tramadol: 0.750/0.999/0.999
    - atorvastatin: 0.750/0.992/0.992
    - apixaban: 0.750/0.675/0.675
    - acenocoumarol: 0.750/0.018/0.018
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta necessidade de alívio de dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa de clopidogrel.    2. Motivo do alerta   O sistema identificou interação entre AINE e antiagregante plaquetário, aumentando risco de hemorragia gastrointestinal. Também detetou interação entre AINE e anticoagulante, elevando risco de hemorragia. Ambos os alertas estão relacionados com a prescrição submetida.    3. Motivo da recomendação   O sistema identificou paracetamol, tramadol, atorvastatina e apixabano como alternativas admissíveis. Na base de conhecimento atual do protótipo, não foi identificado o mesmo alerta para estas opções.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Controlo (sem riscos)
- **Doente:** 30a M, problema=pain, renal=normal, ativos=—, condições=—, alergias=—
- **Prescrição:** paracetamol
- **Expectativa:** Sem alerta bloqueante.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de paracetamol, um analgésico e antipirético, com indicação para dor e febre.  2. Motivo do alerta   A análise baseia-se no contexto clínico consolidado e na prescrição submetida. Não foram identificados alertas relevantes, pois o utente não tem condições, alergias ou medicações ativas, e o estado renal é normal.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. O sistema não identificou uma alternativa terapêutica admissível dentro da sua base atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Omeprazol + clopidogrel (eficácia)
- **Doente:** 68a M, problema=active_gi_ulcer, renal=normal, ativos=['clopidogrel'], condições=—, alergias=—
- **Prescrição:** omeprazole
- **Expectativa:** Alerta moderado: IBP reduz eficácia do clopidogrel.
- **Alertas:**
    - [moderate] Os inibidores da bomba de protões (ex.: omeprazol) podem reduzir a eficácia antiagregante do clopidogrel. Considerar IBP alternativo ou separação temporal. (Norma DGS Antiagregantes, p.11) (regra: ibp_clopidogrel_eficacia)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta um úlcero ativo no trato gastrointestinal. Foi submetida prescrição de omeprazol, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma interação entre clopidogrel e omeprazol, com base nas regras implementadas e no contexto clínico consolidado. O omeprazol, como inibidor da bomba de protões (IBP), pode reduzir a eficácia antiagregante do clopidogrel. Este alerta envolve tanto a medicação ativa do utente como a prescrição submetida.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual. A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual.

## [hand-crafted] Tramadol + ISRS (serotoninérgico)
- **Doente:** 55a F, problema=pain, renal=normal, ativos=['sertraline'], condições=—, alergias=—
- **Prescrição:** tramadol
- **Expectativa:** Alerta moderado serotoninérgico.
- **Alertas:**
    - [moderate] A associação de tramadol com ISRS aumenta o risco de síndrome serotoninérgica e de convulsões. (regra: tramadol_isrs_serotoninergico)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.517/0.517
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de tramadol, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou uma interação entre Sertralina e Tramadol. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo. A associação aumenta o risco de síndrome serotoninérgica e convulsões.    3. Motivo da recomendação   O sistema identificou Paracetamol como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + apixabano (DOAC, hemorragia)
- **Doente:** 70a M, problema=pain, renal=normal, ativos=['apixaban'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta de hemorragia (DOAC herda regra AINE+anticoagulante).
- **Alertas:**
    - [high] A associacao de um AINE com um anticoagulante aumenta o risco de hemorragia e deve ser evitada ou monitorizada. (regra: aine_anticoagulante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.927/0.927
    - tramadol: 0.910/0.365/0.365
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre ibuprofeno e apixabano. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo. A associação de um AINE com um anticoagulante aumenta o risco de hemorragia.    3. Motivo da recomendação   O sistema identificou paracetamol como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. Tramadol foi sugerido como alternativa com precaução.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Macrólido+varfarina → amoxicilina alternativa
- **Doente:** 69a F, problema=infection, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** clarithromycin
- **Expectativa:** Alerta varfarina+macrólido; amoxicilina deve surgir como alternativa segura.
- **Alertas:**
    - [high] A associacao de varfarina com antibioticos macrolidos pode aumentar o efeito anticoagulante e o risco de hemorragia. Deve ser considerada monitorizacao do INR. (regra: varfarina_macrolido)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.344/0.344
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta uma infeção. Foi submetida prescrição de claritromicina, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou uma interação entre varfarina e claritromicina. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou amoxicilina como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Escitalopram + amiodarona (QT)
- **Doente:** 72a M, problema=depression, renal=normal, ativos=['amiodarone'], condições=—, alergias=—
- **Prescrição:** escitalopram
- **Expectativa:** Alerta QT (escitalopram qt_risk).
- **Alertas:**
    - [high] A associacao de dois farmacos com potencial de prolongamento do intervalo QT pode aumentar o risco de arritmias ventriculares, incluindo torsades de pointes. (regra: qt_risk_combination)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta depressão e foi submetida prescrição de escitalopram. Foi identificado um alerta relevante sobre associação de fármacos com risco de prolongamento do intervalo QT.  2. Motivo do alerta   O sistema identificou uma associação entre amiodarona e escitalopram, ambos com potencial de prolongar o intervalo QT. Este risco aumenta a probabilidade de arritmias ventriculares, incluindo torsades de pointes. O alerta envolve tanto a medicação ativa do utente como a prescrição submetida.  3. Motivo da recomendação   O protótipo não identificou uma alternativa terapêutica admissível dentro da sua base de conhecimento atual.  4. Limitações   A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual. A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual do sistema.

## [hand-crafted] Metformina em insuf. renal grave
- **Doente:** 75a M, problema=diabetes, renal=severe_impairment, ativos=—, condições=['renal_disease'], alergias=—
- **Prescrição:** metformin
- **Expectativa:** Alerta renal (metformina contraindicada em TFG<30; via renal_caution).
- **Alertas:**
    - [critical] Metformina requer precaução acrescida em doentes com compromisso renal grave. (regra: renal_caution)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta diabetes e compromisso renal grave. Foi submetida prescrição de metformina, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou um alerta de risco renal associado à metformina, com base nas regras implementadas e no contexto clínico consolidado. A metformina requer precaução acrescida em utentes com compromisso renal grave.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual do protótipo. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Infeção com amoxicilina (controlo)
- **Doente:** 40a M, problema=infection, renal=normal, ativos=—, condições=—, alergias=—
- **Prescrição:** amoxicillin
- **Expectativa:** Sem alerta; 1ª linha adequada.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta uma infeção. Foi submetida prescrição de amoxicilina, que está indicada para tratamento de infecções.  2. Motivo do alerta   A análise baseia-se no contexto clínico consolidado e na prescrição submetida. Não foram identificados alertas relevantes, pois a amoxicilina é indicada para infeções e o estado renal é normal.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. O protótipo não identificou uma alternativa terapêutica admissível dentro da sua base atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Tiazida + digoxina (hipocaliemia)
- **Doente:** 76a F, problema=heart_failure, renal=normal, ativos=['digoxin'], condições=—, alergias=—
- **Prescrição:** hydrochlorothiazide
- **Expectativa:** Alerta moderado (hipocaliemia -> toxicidade digitálica).
- **Alertas:**
    - [moderate] A hipocaliemia associada a diureticos tiazidicos pode aumentar o risco de toxicidade por digoxina. (regra: tiazida_digoxina_hipocaliemia)
- **Recomendações (med — heurístico/combinado/final):**
    - furosemide: 0.710/0.711/0.711
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta insuficiência cardíaca. Foi submetida prescrição de hidroclorotiazida, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação diurético tiazídico + digoxina entre hidroclorotiazida e digoxina. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou furosemida como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Tramadol + tricíclico (serotoninérgico)
- **Doente:** 60a M, problema=pain, renal=normal, ativos=['amitriptyline'], condições=—, alergias=—
- **Prescrição:** tramadol
- **Expectativa:** Alerta moderado serotoninérgico/convulsivo.
- **Alertas:**
    - [moderate] A associação de tramadol com antidepressivos tricíclicos aumenta o risco de síndrome serotoninérgica e de convulsões. (regra: tramadol_triciclico_serotoninergico)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.526/0.526
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de tramadol, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou uma interação entre amitriptilina e tramadol. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo. A associação aumenta o risco de síndrome serotoninérgica e convulsões.    3. Motivo da recomendação   O sistema identificou paracetamol como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Duplicação de anticoagulantes
- **Doente:** 72a M, problema=atrial_fibrillation, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** apixaban
- **Expectativa:** Alerta alto (duplicação de anticoagulação).
- **Alertas:**
    - [high] A utilização concomitante de dois anticoagulantes (AVK, DOAC ou HBPM) aumenta marcadamente o risco hemorrágico e deve ser evitada, salvo indicação específica e monitorizada. (regra: anticoagulante_duplicacao)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta fibrilação atrial. Foi submetida prescrição de apixabano, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma interação com o medicamento ativo do utente, a varfarina. A utilização concomitante de dois anticoagulantes (varfarina e apixabano) aumenta significativamente o risco hemorrágico e deve ser evitada, salvo indicação específica e monitorizada.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Duplo bloqueio SRAA (IECA + ARA)
- **Doente:** 68a M, problema=hypertension, renal=normal, ativos=['enalapril'], condições=—, alergias=—
- **Prescrição:** losartan
- **Expectativa:** Alerta alto (IECA+ARA: hipercaliemia/lesão renal, desaconselhado).
- **Alertas:**
    - [high] O duplo bloqueio do sistema renina-angiotensina (IECA + ARA) aumenta o risco de hipotensão, hipercaliemia e deterioração da função renal, sendo desaconselhado pelas guidelines (Norma DGS 026/2011). (regra: ieca_ara_duplo_bloqueio)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta hipertensão. Foi submetida prescrição de losartan, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou um alerta de interação entre enalapril e losartan, devido ao duplo bloqueio do sistema renina-angiotensina (IECA + ARA). Este risco aumenta a probabilidade de hipotensão, hipercaliemia e deterioração da função renal, conforme indicado nas guidelines DGS 026/2011. O medicamento prescrito foi avaliado no contexto da medicação ativa do utente.  3. Motivo da recomendação   O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações   A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual. A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual.

## [hand-crafted] Benzodiazepina + opióide (depressão SNC)
- **Doente:** 70a M, problema=anxiety, renal=normal, ativos=['tramadol'], condições=—, alergias=—
- **Prescrição:** mexazolam
- **Expectativa:** Alerta alto (depressão SNC/respiratória).
- **Alertas:**
    - [high] A associação de uma benzodiazepina com um opióide potencia a depressão do SNC e respiratória, podendo ser fatal (FT Mexazolam 4.5; RCM Tramadol 4.4). (regra: benzodiazepina_opioide_depressao)
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta ansiedade. Foi submetida prescrição de mexazolam, tendo sido identificados alertas relevantes.  2. Motivo do alerta   O sistema identificou uma interação entre tramadol e mexazolam, ambos presentes no contexto clínico do utente. A associação de um opióide com uma benzodiazepina potencia a depressão do sistema nervoso central e pode comprometer a respiração, com risco potencial de morte.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. O protótipo não identificou uma alternativa terapêutica admissível dentro da sua base atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [synthea] Synthea a71a4da3 (56a F, 3 ativos)
- **Doente:** 56a F, problema=diabetes, renal=normal, ativos=['clopidogrel', 'metoprolol', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.998/0.998
    - tramadol: 0.910/0.576/0.576
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.  2. Motivo do alerta   O sistema identificou interação entre ibuprofeno e clopidogrel. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo. A associação de um AINE com um antiagregante plaquetário aumenta o risco de hemorragia gastrointestinal.  3. Motivo da recomendação   O sistema identificou paracetamol e tramadol como alternativas admissíveis. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para estas alternativas. Paracetamol é adequado ao problema clínico principal e pertence a uma classe diferente, mas apresenta finalidade terapêutica compatível.  4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [synthea] Synthea 8e6a93c0 (59a M, 4 ativos)
- **Doente:** 59a M, problema=pain, renal=severe_impairment, ativos=['clopidogrel', 'metoprolol', 'paracetamol', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [high] Ibuprofeno requer precaução acrescida em doentes com compromisso renal grave. (regra: renal_caution)
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
- **Recomendações:** nenhuma
- **Notas:**
    - Paracetamol já consta da medicação ativa do utente e tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo. Por esse motivo, não foi apresentado como nova alternativa.
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor e foi submetida prescrição de ibuprofeno, que originou alertas relevantes.  2. Motivo do alerta   O sistema identificou dois alertas relacionados com a prescrição submetida: um de interação entre AINE e antiagregante plaquetário, aumentando o risco de hemorragia, e outro de risco renal associado ao ibuprofeno em utente com compromisso renal grave.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para a alternativa paracetamol. Paracetamol já consta da medicação ativa do utente e tem indicação compatível com a dor, pelo que não foi apresentada como nova recomendação.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não avalia ajuste de dose, suspensão ou outras opções clínicas fora da base atual.

## [synthea] Synthea 88c95938 (23a F, 1 ativos)
- **Doente:** 23a F, problema=pain, renal=normal, ativos=['paracetamol'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Notas:**
    - Paracetamol já consta da medicação ativa do utente e tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo. Por esse motivo, não foi apresentado como nova alternativa.
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor e foi submetida prescrição de ibuprofeno, um AINE, com indicação compatível com a dor.  2. Motivo do alerta   Não foram identificados alertas relevantes na análise. A análise baseia-se nos dados submetidos e na base de conhecimento atual do protótipo.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para a alternativa Paracetamol. Uma alternativa candidata já consta da medicação ativa do utente, pelo que não foi apresentada como nova recomendação.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não avalia ajuste de dose, suspensão ou opções clínicas fora da base atual.

## [synthea] Synthea 147f7d4f (18a F, 1 ativos)
- **Doente:** 18a F, problema=pain, renal=normal, ativos=['paracetamol'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Notas:**
    - Paracetamol já consta da medicação ativa do utente e tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo. Por esse motivo, não foi apresentado como nova alternativa.
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor e foi submetida prescrição de ibuprofeno, um AINE, com indicação para alívio de dor.  2. Motivo do alerta   Não foram identificados alertas relevantes na análise. A prescrição de ibuprofeno não gerou interações clínicas com a medicação ativa do utente, que é apenas paracetamol.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo, não foi identificada uma nova alternativa admissível. O paracetamol já consta da medicação ativa do utente e tem indicação compatível com a dor, pelo que não foi apresentada como nova opção.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não avalia ajuste de dose, suspensão ou opções terapêuticas fora da base atual.

## [synthea] Synthea 119aedc5 (71a M, 3 ativos)
- **Doente:** 71a M, problema=inflammation, renal=normal, ativos=['naproxen', 'ramipril', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
    - [high] A utilizacao concomitante de dois AINEs deve ser evitada devido ao aumento do risco de toxicidade gastrointestinal, renal e hemorragica. (regra: aine_aine_duplicacao)
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.754/0.754
    - tramadol: 0.910/0.290/0.290
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor aguda. Foi submetida prescrição de Ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou duplicação terapêutica de AINEs entre Naproxeno e Ibuprofeno, bem como interação entre AINEs e IECA com Ramipril. Ambos os alertas estão relacionados com a prescrição submetida e resultam das regras implementadas na base de conhecimento atual.    3. Motivo da recomendação   O sistema identificou Paracetamol como alternativa sugerida e Tramadol com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para estas alternativas.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [synthea] Synthea f777959f (47a M, 1 ativos)
- **Doente:** 47a M, problema=pain, renal=normal, ativos=['ibuprofen'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Notas:**
    - Ibuprofeno já consta da medicação ativa do utente e tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo. Por esse motivo, não foi apresentado como nova alternativa.
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor e foi submetida prescrição de ibuprofeno, que tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo.  2. Motivo do alerta   Não foram identificados alertas relevantes, pois a prescrição de ibuprofeno está alinhada com o contexto clínico consolidado e não há interações detectadas com a medicação ativa.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. Uma alternativa candidata já consta da medicação ativa do utente, pelo que não foi apresentada como nova recomendação.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não avalia ajuste de dose, suspensão ou opções clínicas fora da base atual.

## [synthea] Synthea 690a7c16 (85a M, 6 ativos)
- **Doente:** 85a M, problema=unspecified, renal=normal, ativos=['acetylsalicylic_acid', 'atorvastatin', 'clopidogrel', 'metoprolol', 'ramipril', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
    - [moderate] A utilização concomitante de duas estatinas constitui duplicação terapêutica e aumenta o risco de miopatia. (regra: estatina_duplicacao)
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.362/0.362
    - tramadol: 0.910/0.068/0.068
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor como problema clínico principal. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.  2. Motivo do alerta   O sistema identificou interações entre ibuprofeno e acido acetilsalicilico, bem como entre ibuprofeno e ramipril. Essas interações aumentam riscos de hemorragia e deterioração renal, resultando de regras na base de conhecimento atual.  3. Motivo da recomendação   O sistema propõe paracetamol e tramadol como alternativas com precaução. Ambas são adequadas para alívio da dor, com finalidade terapêutica compatível. Na base de conhecimento atual não foi identificado o mesmo alerta para essas opções.  4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [synthea] Synthea 09807e61 (64a F, 3 ativos)
- **Doente:** 64a F, problema=pain, renal=normal, ativos=['clopidogrel', 'metoprolol', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.998/0.998
    - tramadol: 0.910/0.583/0.583
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.  2. Motivo do alerta   O sistema identificou interação entre ibuprofeno e clopidogrel. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo. A associação de um AINE com um antiagregante plaquetário aumenta o risco de hemorragia gastrointestinal.  3. Motivo da recomendação   O sistema identificou paracetamol e tramadol como alternativas admissíveis. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para estas alternativas. Paracetamol é adequado ao problema clínico principal e pertence a uma classe diferente, mas apresenta finalidade terapêutica compatível.  4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [synthea] Synthea 75089c8b (57a M, 1 ativos)
- **Doente:** 57a M, problema=hypertension, renal=normal, ativos=['hydrochlorothiazide'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [moderate] A associacao de AINEs com diureticos tiazidicos pode reduzir a eficacia anti-hipertensora e aumentar o risco de deterioracao da funcao renal. (regra: aine_tiazida_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.936/0.936
    - tramadol: 0.910/0.423/0.423
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.  2. Motivo do alerta   O sistema identificou interação entre hidroclorotiazida e ibuprofeno. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. Motivo da recomendação   O sistema identificou paracetamol como alternativa sugerida e tramadol como alternativa com precaução. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [synthea] Synthea 70e4cf5c (43a F, 1 ativos)
- **Doente:** 43a F, problema=diabetes, renal=normal, ativos=['simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, um AINE indicado para dor, febre e inflamação.  2. Motivo do alerta   A análise baseia-se no contexto clínico consolidado e na base de conhecimento atual do protótipo. Não foram identificados alertas relevantes, pois o estado renal é normal e não há interações com a medicação ativa (sinvastatina).  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. O protótipo não identificou uma alternativa terapêutica admissível dentro da sua base atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [synthea] Synthea b01206ca (58a F, 1 ativos)
- **Doente:** 58a F, problema=diabetes, renal=normal, ativos=['simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, um AINE indicado para dor, febre e inflamação.  2. Motivo do alerta   A análise baseia-se no contexto clínico consolidado e no base de conhecimento atual do protótipo. Não foram identificados alertas relevantes, pois o estado renal é normal e o utente não tem contraindicações específicas para AINEs.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. O sistema não identificou uma alternativa terapêutica admissível dentro da sua base atual.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [synthea] Synthea aa442ba0 (45a M, 1 ativos)
- **Doente:** 45a M, problema=pain, renal=normal, ativos=['paracetamol'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Notas:**
    - Paracetamol já consta da medicação ativa do utente e tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo. Por esse motivo, não foi apresentado como nova alternativa.
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor e foi submetida prescrição de ibuprofeno, um AINE, com indicação para alívio de dor.  2. Motivo do alerta   Não foram identificados alertas relevantes na análise. A análise baseia-se nos dados do utente e na base de conhecimento atual do protótipo.  3. Motivo da recomendação   Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para a alternativa Paracetamol. Paracetamol já consta da medicação ativa do utente e tem indicação compatível com a dor, pelo que não foi apresentada como nova alternativa.  4. Limitações   A análise depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não avalia ajuste de dose, suspensão ou opções clínicas fora da base atual.

## [synthea] Synthea c0e63b08 (51a F, 1 ativos)
- **Doente:** 51a F, problema=pain, renal=normal, ativos=['hydrochlorothiazide'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [moderate] A associacao de AINEs com diureticos tiazidicos pode reduzir a eficacia anti-hipertensora e aumentar o risco de deterioracao da funcao renal. (regra: aine_tiazida_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.940/0.940
    - tramadol: 0.910/0.426/0.426
- **LLM (qwen3:4b-instruct):** 1. Problema identificado   O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.    2. Motivo do alerta   O sistema identificou interação entre AINE e diurético tiazídico envolvendo hidroclorotiazida e ibuprofeno. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.    3. Motivo da recomendação   O sistema identificou paracetamol como alternativa admissível. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa. Tramadol foi sugerido com precaução devido ao risco renal.    4. Limitações   A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [synthea] Synthea 217a7c06 (76a F, 4 ativos)
- **Doente:** 76a F, problema=inflammation, renal=severe_impairment, ativos=['clopidogrel', 'hydrochlorothiazide', 'metoprolol', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [high] Ibuprofeno requer precaução acrescida em doentes com compromisso renal grave. (regra: renal_caution)
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
    - [moderate] A associacao de AINEs com diureticos tiazidicos pode reduzir a eficacia anti-hipertensora e aumentar o risco de deterioracao da funcao renal. (regra: aine_tiazida_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.539/0.539
- **LLM (qwen3:4b-instruct + fallback determinístico):** 1. Problema identificado O utente apresenta dor. Foi submetida prescrição de Ibuprofeno, avaliada no contexto da medicação ativa.  2. Motivo do alerta Foram identificados alertas relacionados com a prescrição submetida: Interação AINE + antiagregante plaquetário (Clopidogrel + Ibuprofeno); Risco renal associado ao medicamento prescrito (Ibuprofeno); Interação AINE + diurético tiazídico (Hidroclorotiazida + Ibuprofeno).  3. Motivo da recomendação Paracetamol foi identificado como alternativa sugerida na base de conhecimento atual.  4. Limitações A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual do protótipo. Não substitui validação clínica.

## [synthea] Synthea 1353b50a (95a M, 5 ativos)
- **Doente:** 95a M, problema=inflammation, renal=severe_impairment, ativos=['furosemide', 'losartan', 'metoprolol', 'naproxen', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [high] Ibuprofeno requer precaução acrescida em doentes com compromisso renal grave. (regra: renal_caution)
    - [moderate] A associacao de AINEs com diureticos pode atenuar o efeito diuretico e aumentar o risco de compromisso renal. (regra: aine_diuretico_risco_renal)
    - [moderate] A associacao de AINEs com diureticos pode atenuar o efeito diuretico e aumentar o risco de compromisso renal. (regra: aine_diuretico_risco_renal)
    - [moderate] A associacao de AINEs com antagonistas dos recetores da angiotensina II pode reduzir o efeito anti-hipertensor e aumentar o risco de agravamento da funcao renal. (regra: aine_ara_risco_renal)
    - [moderate] A associacao de AINEs com antagonistas dos recetores da angiotensina II pode reduzir o efeito anti-hipertensor e aumentar o risco de agravamento da funcao renal. (regra: aine_ara_risco_renal)
    - [high] A utilizacao concomitante de dois AINEs deve ser evitada devido ao aumento do risco de toxicidade gastrointestinal, renal e hemorragica. (regra: aine_aine_duplicacao)
    - [high] Associação com risco aumentado de deterioração da função renal: AINE em combinação com inibidor da ECA ou antagonista dos recetores da angiotensina II e diurético. Recomenda-se evitar a associação ou monitorizar função renal, hidratação e eletrólitos. (regra: triple_whammy)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.199/0.199
- **LLM (qwen3:4b-instruct + fallback determinístico):** 1. Problema identificado O utente apresenta dor. Foi submetida prescrição de Ibuprofeno, avaliada no contexto da medicação ativa.  2. Motivo do alerta Foram identificados alertas relacionados com a prescrição submetida: Associação AINE + IECA/ARA + diurético (AINE + IECA/ARA + diurético); Duplicação terapêutica de AINEs (Naproxeno + Ibuprofeno); Risco renal associado ao medicamento prescrito (Ibuprofeno); Interação AINE + ARA (Losartan + Ibuprofeno); Interação AINE + diurético (Furosemida + Ibuprofeno). Foram também identificados alertas pré-existentes na medicação ativa: Interação AINE + ARA (Losartan + Naproxeno); Interação AINE + diurético (Furosemida + Naproxeno).  3. Motivo da recomendação Paracetamol foi identificado como alternativa com precaução e requer validação clínica.  4. Limitações A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual do protótipo. Não substitui validação clínica.
