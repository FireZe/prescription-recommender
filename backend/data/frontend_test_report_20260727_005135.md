# Relatório de testes — 20260727_005135

**Total de casos:** 53 | **com alerta:** 45 | **alertas por severidade:** {'high': 33, 'moderate': 25, 'critical': 4}

## [hand-crafted] AINE + antiagregante (hemorragia)
- **Doente:** 70a F, problema=pain, renal=normal, ativos=['clopidogrel'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta alto; recomendar paracetamol.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.952/0.952
    - tramadol: 0.910/0.281/0.281
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa que inclui clopidogrel.  2. **Motivo do alerta** O sistema identificou um alerta relacionado com a prescrição submetida para AINE (ibuprofeno) em associação com antiagregante plaquetário (clopidogrel). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + anticoagulante (hemorragia)
- **Doente:** 68a M, problema=pain, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** naproxen
- **Expectativa:** Alerta alto; alternativa não-AINE.
- **Alertas:**
    - [high] A associacao de um AINE com um anticoagulante aumenta o risco de hemorragia e deve ser evitada ou monitorizada. (regra: aine_anticoagulante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.951/0.951
    - tramadol: 0.910/0.258/0.258
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa que inclui Varfarina.  2. **Motivo do alerta** O sistema identificou um alerta relacionado com a prescrição submetida para AINE (naproxeno) associado ao anticoagulante (varfarina). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Duplicação AINE+AINE
- **Doente:** 50a M, problema=pain, renal=normal, ativos=['ibuprofen'], condições=—, alergias=—
- **Prescrição:** naproxen
- **Expectativa:** Alerta de duplicação terapêutica.
- **Alertas:**
    - [high] A utilizacao concomitante de dois AINEs deve ser evitada devido ao aumento do risco de toxicidade gastrointestinal, renal e hemorragica. (regra: aine_aine_duplicacao)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.984/0.984
    - tramadol: 0.910/0.408/0.408
- **LLM (qwen2.5:3b):** 1. **Problema identificado**    Utente com dor, apresentando prescrição de Naproxeno.  2. **Motivo do alerta**    Alerta de duplicação terapêutica de AINEs (Ibuprofeno + Naproxeno), relacionado com a prescrição submetida.  3. **Motivo da recomendação**    Recomenda-se Paracetamol como alternativa, considerando adequação ao problema clínico e segurança do sistema.  4. **Limitações**    Explicação depende dos dados submetidos, regras implementadas e base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + IECA (risco renal)
- **Doente:** 65a M, problema=pain, renal=normal, ativos=['ramipril'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta moderado renal.
- **Alertas:**
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.706/0.706
    - tramadol: 0.910/0.095/0.095
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa que inclui Ramipril.  2. **Motivo do alerta** O sistema identificou interação entre AINE (ibuprofeno) e IECA (Ramipril), relacionado com a prescrição submetida. Alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + ARA (risco renal)
- **Doente:** 65a F, problema=pain, renal=normal, ativos=['losartan'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta moderado renal.
- **Alertas:**
    - [moderate] A associacao de AINEs com antagonistas dos recetores da angiotensina II pode reduzir o efeito anti-hipertensor e aumentar o risco de agravamento da funcao renal. (regra: aine_ara_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.712/0.712
    - tramadol: 0.910/0.103/0.103
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa que inclui Losartan.  2. **Motivo do alerta** O sistema identificou interação entre AINE (Losartan) e ARA (Ibuprofeno). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + diurético de ansa
- **Doente:** 72a F, problema=pain, renal=normal, ativos=['furosemide'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta moderado renal/diurético.
- **Alertas:**
    - [moderate] A associacao de AINEs com diureticos pode atenuar o efeito diuretico e aumentar o risco de compromisso renal. (regra: aine_diuretico_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.757/0.757
    - tramadol: 0.910/0.133/0.133
- **LLM (qwen2.5:3b):** 1. **Problema identificado**    Utente com dor, apresentando prescrição de ibuprofeno.  2. **Motivo do alerta**    Alerta relacionado com a prescrição submetida: A associação de AINE (ibuprofeno) e diurético (Furosemida) aumenta o risco renal.  3. **Motivo da recomendação**    Recomenda-se paracetamol como alternativa admissível, compatível com o problema clínico principal.  4. **Limitações**    A explicação depende dos dados submetidos e da base de conhecimento atual do protótipo. Não substitui validação clínica.

## [hand-crafted] AINE + tiazida
- **Doente:** 72a M, problema=pain, renal=normal, ativos=['hydrochlorothiazide'], condições=—, alergias=—
- **Prescrição:** naproxen
- **Expectativa:** Alerta moderado tiazida.
- **Alertas:**
    - [moderate] A associacao de AINEs com diureticos tiazidicos pode reduzir a eficacia anti-hipertensora e aumentar o risco de deterioracao da funcao renal. (regra: aine_tiazida_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.750/0.750
    - tramadol: 0.910/0.123/0.123
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre AINE (naproxeno) e diurético tiazídico (hidroclorotiazida). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Triple whammy (AINE+IECA+diurético)
- **Doente:** 82a F, problema=hypertension, renal=mild_impairment, ativos=['ramipril', 'furosemide'], condições=['hypertension', 'heart_failure'], alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta alto triple whammy + os moderados.
- **Alertas:**
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
    - [moderate] A associacao de AINEs com diureticos pode atenuar o efeito diuretico e aumentar o risco de compromisso renal. (regra: aine_diuretico_risco_renal)
    - [high] Associação com risco aumentado de deterioração da função renal: AINE em combinação com inibidor da ECA ou antagonista dos recetores da angiotensina II e diurético. Recomenda-se evitar a associação ou monitorizar função renal, hidratação e eletrólitos. (regra: triple_whammy)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.750/0.657/0.657
    - tramadol: 0.720/0.596/0.596
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta hipertensão e insuficiência cardíaca, submetida prescrição de ibuprofeno.  2. **Motivo do alerta** O sistema identificou associação entre AINE (ibuprofeno), inibidor da ECA (Ramipril) e diurético (Furosemida). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Sinvastatina + claritromicina (CRÍTICO)
- **Doente:** 60a M, problema=infection, renal=normal, ativos=['simvastatin'], condições=—, alergias=—
- **Prescrição:** clarithromycin
- **Expectativa:** Interação crítica (miopatia/rabdomiólise).
- **Alertas:**
    - [critical] A administracao concomitante de sinvastatina e claritromicina esta contraindicada devido ao aumento do risco de miopatia e rabdomiolise. (regra: sinvastatina_claritromicina_contraindicada)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.417/0.417
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta infeção. Foi submetida prescrição de clarithromicina, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou interação crítica entre sinvastatina e clarithromicina. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou amoxicilina como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Estatina + macrólido
- **Doente:** 61a F, problema=infection, renal=normal, ativos=['atorvastatin'], condições=—, alergias=—
- **Prescrição:** azithromycin
- **Expectativa:** Alerta alto miopatia.
- **Alertas:**
    - [high] A associacao de estatinas com macrolidos pode aumentar o risco de miopatia ou rabdomiolise. A relevancia clinica depende do macrolido e da estatina. (regra: estatina_macrolido_miopatia)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.406/0.406
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta infeção. Foi submetida prescrição de azitromicina, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre estatinas (atorvastatina) e macrolidos (azitromicina). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou amoxicilina como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Varfarina + amiodarona
- **Doente:** 70a M, problema=arrhythmia, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** amiodarone
- **Expectativa:** Alerta alto (potenciação anticoagulante).
- **Alertas:**
    - [high] A amiodarona pode potenciar o efeito anticoagulante da varfarina, aumentando o risco de hemorragia. Recomenda-se monitorizacao rigorosa do INR. (regra: varfarina_amiodarona)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. **Problema identificado**: O utente apresenta arritmia e foi submetida prescrição de amiodarona.  2. **Motivo do alerta**: A amiodarona pode potenciar o efeito anticoagulante da varfarina, aumentando risco de hemorragia. Alertas relacionados foram identificados.  3. **Motivo da recomendação**: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. **Limitações**: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Macrólido + varfarina
- **Doente:** 69a F, problema=infection, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** clarithromycin
- **Expectativa:** Alerta alto.
- **Alertas:**
    - [high] A associacao de varfarina com antibioticos macrolidos pode aumentar o efeito anticoagulante e o risco de hemorragia. Deve ser considerada monitorizacao do INR. (regra: varfarina_macrolido)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.242/0.242
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta infeção. Foi submetida prescrição de clarithromicina, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre varfarina e macrólido (clarithromicina). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou amoxicilina como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] ISRS + anticoagulante
- **Doente:** 58a F, problema=depression, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** sertraline
- **Expectativa:** Alerta moderado hemorragia.
- **Alertas:**
    - [moderate] Os ISRS podem aumentar o risco hemorragico quando associados a anticoagulantes. Deve ser ponderada monitorizacao clinica e laboratorial. (regra: sertralina_anticoagulante)
- **Recomendações (med — heurístico/combinado/final):**
    - amitriptyline: 0.910/0.096/0.096
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta depressão. Foi submetida prescrição de sertralina, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre anticoagulante (warfarin) e inibidor seletivo da recaptação da serotonina (ISRS) (sertralina). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou amitriptilina como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] ISRS + antiagregante
- **Doente:** 58a M, problema=depression, renal=normal, ativos=['clopidogrel'], condições=—, alergias=—
- **Prescrição:** sertraline
- **Expectativa:** Alerta moderado.
- **Alertas:**
    - [moderate] Os ISRS podem aumentar o risco de hemorragia quando associados a farmacos com efeito antiagregante plaquetario. (regra: sertralina_antiagregante)
- **Recomendações (med — heurístico/combinado/final):**
    - amitriptyline: 0.910/0.115/0.115
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta depressão. Foi submetida prescrição de sertralina, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre antiagregante plaquetário (clopidogrel) e inibidor seletivo da recaptação da serotonina (sertralina). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou amitriptilina como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] ISRS + AINE
- **Doente:** 45a F, problema=depression, renal=normal, ativos=['ibuprofen'], condições=—, alergias=—
- **Prescrição:** sertraline
- **Expectativa:** Alerta moderado.
- **Alertas:**
    - [moderate] A associacao de ISRS com AINEs pode aumentar o risco de hemorragia gastrointestinal. (regra: sertralina_aine)
- **Recomendações (med — heurístico/combinado/final):**
    - amitriptyline: 0.910/0.236/0.236
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta depressão. Foi submetida prescrição de sertralina, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre anti-inflamatório não esteroide (AINE) e inibidor seletivo da recaptação da serotonina (ISRS). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou amitriptilina como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] ISRS + tricíclico (serotoninérgico)
- **Doente:** 50a M, problema=depression, renal=normal, ativos=['amitriptyline'], condições=—, alergias=—
- **Prescrição:** sertraline
- **Expectativa:** Alerta alto (síndrome serotoninérgica).
- **Alertas:**
    - [high] A associacao de antidepressivos serotoninergicos pode aumentar o risco de síndrome serotoninérgica. (regra: serotoninergicos)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. **Problema identificado**: O utente apresenta depressão e foi submetida prescrição de sertralina.  2. **Motivo do alerta**: A associação de antidepressivos serotoninérgicos (Amitriptilina + Sertralina) aumenta o risco de síndrome serotoninérgica, conforme regras implementadas e contexto clínico consolidado.  3. **Motivo da recomendação**: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. **Limitações**: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Combinação QT (amiodarona + azitromicina)
- **Doente:** 72a M, problema=infection, renal=normal, ativos=['amiodarone'], condições=['arrhythmia'], alergias=—
- **Prescrição:** azithromycin
- **Expectativa:** Alerta alto QT.
- **Alertas:**
    - [high] A associacao de dois farmacos com potencial de prolongamento do intervalo QT pode aumentar o risco de arritmias ventriculares, incluindo torsades de pointes. (regra: qt_risk_combination)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.089/0.089
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta infeção e foi submetida prescrição de azitromicina, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou associação entre dois fármacos com potencial de prolongamento do intervalo QT (Amiodarona + Azitromicina). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou amoxicilina como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Amiodarona + digoxina
- **Doente:** 74a M, problema=arrhythmia, renal=normal, ativos=['digoxin'], condições=—, alergias=—
- **Prescrição:** amiodarone
- **Expectativa:** Alerta alto (toxicidade digoxina).
- **Alertas:**
    - [high] A amiodarona pode aumentar a exposicao a digoxina e potenciar perturbacoes de conducao. Recomenda-se monitorizacao de ECG e niveis de digoxina. (regra: amiodarona_digoxina)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. **Problema identificado**: O utente apresenta arritmia e foi submetida prescrição de Amiodarona.  2. **Motivo do alerta**: O sistema identificou alertas relacionados com a interação entre Digoxina e Amiodarona, incluindo aumento da exposição à Digoxina e perturbações na condução cardíaca.  3. **Motivo da recomendação**: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. **Limitações**: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Diurético de ansa + digoxina
- **Doente:** 78a F, problema=heart_failure, renal=normal, ativos=['digoxin'], condições=—, alergias=—
- **Prescrição:** furosemide
- **Expectativa:** Alerta moderado (hipocaliemia).
- **Alertas:**
    - [moderate] A hipocaliemia induzida por diureticos pode aumentar o risco de toxicidade digitalica. Recomenda-se monitorizacao do potassio serico. (regra: diuretico_digoxina_hipocaliemia)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. **Problema identificado**: Utente com insuficiência cardíaca, receita de furosemida.  2. **Motivo do alerta**: Alerta relacionado com a prescrição submetida indica risco de hipocaliemia induzida por diuretico e toxicidade digitalica.  3. **Motivo da recomendação**: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. **Limitações**: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Beta-bloqueante + amiodarona (bradicardia)
- **Doente:** 70a M, problema=arrhythmia, renal=normal, ativos=['bisoprolol'], condições=—, alergias=—
- **Prescrição:** amiodarone
- **Expectativa:** Alerta alto bradicardia.
- **Alertas:**
    - [high] A associacao de um beta-bloqueante com amiodarona pode causar bradicardia sinusal grave e perturbacoes da conducao auriculo-ventricular. Combinacao nao recomendada. (regra: beta_bloqueante_amiodarona_bradicardia)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. **Problema identificado**: O utente apresenta arritmia e foi submetida prescrição de amiodarona.  2. **Motivo do alerta**: A associação de beta-bloqueante (Bisoprolol) com amiodarona pode causar bradicardia sinusal grave e perturbações da condução auriculo-ventricular, alerta relacionado à prescrição submetida.  3. **Motivo da recomendação**: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. **Limitações**: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Duplicação beta-bloqueante
- **Doente:** 66a M, problema=hypertension, renal=normal, ativos=['metoprolol'], condições=—, alergias=—
- **Prescrição:** atenolol
- **Expectativa:** Alerta moderado duplicação.
- **Alertas:**
    - [high] A utilizacao concomitante de dois beta-bloqueantes representa duplicacao terapeutica e pode potenciar efeitos bradicardicos e hipotensores. (regra: beta_bloqueante_duplicacao)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. **Problema identificado**: O utente apresenta hipertensão e recebeu prescrição de Atenolol, tendo sido identificados alertas relevantes.  2. **Motivo do alerta**: Alerta relacionado com duplicação terapêutica de beta-bloqueantes (Metoprolol + Atenolol), potencialmente aumentando efeitos bradicardicos e hipotensores.  3. **Motivo da recomendação**: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. **Limitações**: Ausência de recomendação não significa ausência de opções clínicas; indica apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Contraindicação: úlcera GI ativa
- **Doente:** 60a M, problema=pain, renal=normal, ativos=—, condições=['active_gi_ulcer'], alergias=—
- **Prescrição:** naproxen
- **Expectativa:** Alerta crítico de contraindicação.
- **Alertas:**
    - [critical] Naproxeno está contraindicado ou deve ser evitado neste contexto clínico devido à condição clínica identificada: active_gi_ulcer. (regra: contraindication)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.997/0.997
    - tramadol: 0.910/0.919/0.919
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, com histórico de úlcera gástrica ativa.  2. **Motivo do alerta** O sistema identificou contraindicação entre Naproxeno e a condição clínica de úlcera gástrica ativa. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE em insuficiência renal grave
- **Doente:** 75a F, problema=pain, renal=severe_impairment, ativos=—, condições=['renal_disease'], alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta renal; preferir paracetamol.
- **Alertas:**
    - [high] Ibuprofeno requer precaução acrescida em doentes com compromisso renal grave. (regra: renal_caution)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.997/0.997
- **LLM (qwen2.5:3b):** 1. **Problema identificado**    Utente com compromisso renal grave (severa) e dor.  2. **Motivo do alerta**    Alerta de precaução para ibuprofeno por causa do compromisso renal grave, relacionado à prescrição submetida.  3. **Motivo da recomendação**    Recomenda-se paracetamol como alternativa terapêutica adequada ao problema principal (dor), não envolvendo alertas pré-existentes na medicação ativa.  4. **Limitações**    A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual do protótipo. Não substitui validação clínica.

## [hand-crafted] Alergia ao medicamento prescrito
- **Doente:** 40a F, problema=pain, renal=normal, ativos=—, condições=—, alergias=['ibuprofen']
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta CRÍTICO de alergia (regressão da nova regra).
- **Alertas:**
    - [critical] Ibuprofeno está registado como alergia do utente. A prescrição deve ser evitada. (regra: allergy_conflict)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.997/0.997
    - tramadol: 0.910/0.910/0.910
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa.  2. **Motivo do alerta** O sistema identificou conflito de alergia entre Ibuprofeno e a prescrição submetida. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Multi-fármaco (3 de uma vez)
- **Doente:** 70a M, problema=cardiovascular_prevention, renal=normal, ativos=['clopidogrel'], condições=—, alergias=—
- **Prescrição:** ibuprofen, warfarin, simvastatin
- **Expectativa:** Vários alertas em simultâneo.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
    - [high] A associacao de um AINE com um anticoagulante aumenta o risco de hemorragia e deve ser evitada ou monitorizada. (regra: aine_anticoagulante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.750/0.996/0.996
    - tramadol: 0.750/0.996/0.996
    - atorvastatin: 0.750/0.976/0.976
    - apixaban: 0.750/0.683/0.683
    - acenocoumarol: 0.750/0.053/0.053
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta problemas clínicos relacionados com dor e inflamação, avaliado no contexto da medicação ativa.  2. Motivo do alerta: Sistema identificou interação entre AINE (ibuprofeno) e antiagregante plaquetário (clopidogrel), associada a risco de hemorragia gastrointestinal.  3. Motivo da recomendação: Alternativa sugerida como atorvastatina considerada admissível pela base de conhecimento atual, sem identificar o mesmo alerta para esta alternativa.  4. Limitações: Explicação depende dos dados submetidos, regras implementadas e base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Controlo (sem riscos)
- **Doente:** 30a M, problema=pain, renal=normal, ativos=—, condições=—, alergias=—
- **Prescrição:** paracetamol
- **Expectativa:** Sem alerta bloqueante.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta dor, com prescrição de paracetamol.  2. Motivo do alerta: Nenhum alerta foi identificado na base de conhecimento atual do protótipo.  3. Motivo da recomendação: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Omeprazol + clopidogrel (eficácia)
- **Doente:** 68a M, problema=active_gi_ulcer, renal=normal, ativos=['clopidogrel'], condições=—, alergias=—
- **Prescrição:** omeprazole
- **Expectativa:** Alerta moderado: IBP reduz eficácia do clopidogrel.
- **Alertas:**
    - [moderate] Os inibidores da bomba de protões (ex.: omeprazol) podem reduzir a eficácia antiagregante do clopidogrel. Considerar IBP alternativo ou separação temporal. (Norma DGS Antiagregantes, p.11) (regra: ibp_clopidogrel_eficacia)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta [problema clínico] e recebeu prescrição de omeprazole.  2. Motivo do alerta: O sistema identificou alerta relacionado com a interação entre clopidogrel e omeprazole, reduzindo eficácia antiagregante.  3. Motivo da recomendação: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Tramadol + ISRS (serotoninérgico)
- **Doente:** 55a F, problema=pain, renal=normal, ativos=['sertraline'], condições=—, alergias=—
- **Prescrição:** tramadol
- **Expectativa:** Alerta moderado serotoninérgico.
- **Alertas:**
    - [moderate] A associação de tramadol com ISRS aumenta o risco de síndrome serotoninérgica e de convulsões. (regra: tramadol_isrs_serotoninergico)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.749/0.749
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre Sertralina e Tramadol (AINE). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] AINE + apixabano (DOAC, hemorragia)
- **Doente:** 70a M, problema=pain, renal=normal, ativos=['apixaban'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Alerta de hemorragia (DOAC herda regra AINE+anticoagulante).
- **Alertas:**
    - [high] A associacao de um AINE com um anticoagulante aumenta o risco de hemorragia e deve ser evitada ou monitorizada. (regra: aine_anticoagulante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.952/0.952
    - tramadol: 0.910/0.258/0.258
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa que inclui Apixabano.  2. **Motivo do alerta** O sistema identificou um alerta relacionado com a prescrição submetida para AINE (Ibuprofeno) em associação com anticoagulante (Apixabano). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Macrólido+varfarina → amoxicilina alternativa
- **Doente:** 69a F, problema=infection, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** clarithromycin
- **Expectativa:** Alerta varfarina+macrólido; amoxicilina deve surgir como alternativa segura.
- **Alertas:**
    - [high] A associacao de varfarina com antibioticos macrolidos pode aumentar o efeito anticoagulante e o risco de hemorragia. Deve ser considerada monitorizacao do INR. (regra: varfarina_macrolido)
- **Recomendações (med — heurístico/combinado/final):**
    - amoxicillin: 0.910/0.242/0.242
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta infeção. Foi submetida prescrição de clarithromicina, avaliada no contexto da medicação ativa.  2. Motivo do alerta: O sistema identificou interação entre varfarina e macrólido (clarithromicina). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. Motivo da recomendação: O sistema identificou amoxicilina como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. Limitações: A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Escitalopram + amiodarona (QT)
- **Doente:** 72a M, problema=depression, renal=normal, ativos=['amiodarone'], condições=—, alergias=—
- **Prescrição:** escitalopram
- **Expectativa:** Alerta QT (escitalopram qt_risk).
- **Alertas:**
    - [high] A associacao de dois farmacos com potencial de prolongamento do intervalo QT pode aumentar o risco de arritmias ventriculares, incluindo torsades de pointes. (regra: qt_risk_combination)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. **Problema identificado**: O utente apresenta depressão e foi submetida prescrição de escitalopram.  2. **Motivo do alerta**: O sistema identificou alertas relacionados com prolongamento do intervalo QT, causado pela associação de amiodarona (medicação ativa) e escitalopram (prescrita).  3. **Motivo da recomendação**: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. **Limitações**: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Metformina em insuf. renal grave
- **Doente:** 75a M, problema=diabetes, renal=severe_impairment, ativos=—, condições=['renal_disease'], alergias=—
- **Prescrição:** metformin
- **Expectativa:** Alerta renal (metformina contraindicada em TFG<30; via renal_caution).
- **Alertas:**
    - [critical] Metformina requer precaução acrescida em doentes com compromisso renal grave. (regra: renal_caution)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta diabetes e compromisso renal grave, com prescrição de metformina.  2. Motivo do alerta: Metformina requer precaução acrescida em doentes com compromisso renal grave (alerta "renal_caution").  3. Motivo da recomendação: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Infeção com amoxicilina (controlo)
- **Doente:** 40a M, problema=infection, renal=normal, ativos=—, condições=—, alergias=—
- **Prescrição:** amoxicillin
- **Expectativa:** Sem alerta; 1ª linha adequada.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta infeção. Foi submetida prescrição de amoxicilina.  2. Motivo do alerta: Nenhum alerta foi identificado no contexto da prescrição submetida ou na medicação ativa do utente, baseado nas regras implementadas e no dicionário atual do protótipo.  3. Motivo da recomendação: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações: A ausência de alerta ou recomendação não significa ausência de opções clínicas; apenas que o protótipo não encontrou alternativa admissível na sua base atual do conhecimento.

## [hand-crafted] Tiazida + digoxina (hipocaliemia)
- **Doente:** 76a F, problema=heart_failure, renal=normal, ativos=['digoxin'], condições=—, alergias=—
- **Prescrição:** hydrochlorothiazide
- **Expectativa:** Alerta moderado (hipocaliemia -> toxicidade digitálica).
- **Alertas:**
    - [moderate] A hipocaliemia associada a diureticos tiazidicos pode aumentar o risco de toxicidade por digoxina. (regra: tiazida_digoxina_hipocaliemia)
- **Recomendações (med — heurístico/combinado/final):**
    - furosemide: 0.710/0.783/0.783
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta insuficiência cardíaca. Foi submetida prescrição de hidroclorotiazida, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre digoxina e hidroclorotiazida. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou furosemida como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Tramadol + tricíclico (serotoninérgico)
- **Doente:** 60a M, problema=pain, renal=normal, ativos=['amitriptyline'], condições=—, alergias=—
- **Prescrição:** tramadol
- **Expectativa:** Alerta moderado serotoninérgico/convulsivo.
- **Alertas:**
    - [moderate] A associação de tramadol com antidepressivos tricíclicos aumenta o risco de síndrome serotoninérgica e de convulsões. (regra: tramadol_triciclico_serotoninergico)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.749/0.749
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta dor, avaliada no contexto de medicação ativa.  2. Motivo do alerta: O sistema identificou um alerta relacionado com a prescrição submetida entre Amitriptilina e Tramadol. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. Motivo da recomendação: O sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual não foi identificado o mesmo alerta para esta alternativa.  4. Limitações: A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [hand-crafted] Duplicação de anticoagulantes
- **Doente:** 72a M, problema=atrial_fibrillation, renal=normal, ativos=['warfarin'], condições=—, alergias=—
- **Prescrição:** apixaban
- **Expectativa:** Alerta alto (duplicação de anticoagulação).
- **Alertas:**
    - [high] A utilização concomitante de dois anticoagulantes (AVK, DOAC ou HBPM) aumenta marcadamente o risco hemorrágico e deve ser evitada, salvo indicação específica e monitorizada. (regra: anticoagulante_duplicacao)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. **Problema identificado**    O utente apresenta atrial fibrilação e foi submetida prescrição de apixaban.  2. **Motivo do alerta**    O sistema identificou alerta relacionado com a duplicação de anticoagulantes (anticoagulante AVK, DOAC ou HBPM), aumentando o risco hemorrágico e recomendando evitá-lo.  3. **Motivo da recomendação**    O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. **Limitações**    A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Duplo bloqueio SRAA (IECA + ARA)
- **Doente:** 68a M, problema=hypertension, renal=normal, ativos=['enalapril'], condições=—, alergias=—
- **Prescrição:** losartan
- **Expectativa:** Alerta alto (IECA+ARA: hipercaliemia/lesão renal, desaconselhado).
- **Alertas:**
    - [high] O duplo bloqueio do sistema renina-angiotensina (IECA + ARA) aumenta o risco de hipotensão, hipercaliemia e deterioração da função renal, sendo desaconselhado pelas guidelines (Norma DGS 026/2011). (regra: ieca_ara_duplo_bloqueio)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta hipertensão e foi submetida prescrição de Losartan.  2. Motivo do alerta: O sistema identificou duplo bloqueio do sistema renina-angiotensina (Enalapril + Losartan), com risco elevado de complicações renais e hipotensão, conforme regras implementadas.  3. Motivo da recomendação: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [hand-crafted] Benzodiazepina + opióide (depressão SNC)
- **Doente:** 70a M, problema=anxiety, renal=normal, ativos=['tramadol'], condições=—, alergias=—
- **Prescrição:** mexazolam
- **Expectativa:** Alerta alto (depressão SNC/respiratória).
- **Alertas:**
    - [high] A associação de uma benzodiazepina com um opióide potencia a depressão do SNC e respiratória, podendo ser fatal (FT Mexazolam 4.5; RCM Tramadol 4.4). (regra: benzodiazepina_opioide_depressao)
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. **Problema identificado**: Utente com ansiedade, receita de Mexazolam submetida.  2. **Motivo do alerta**: Alerta relacionado com a prescrição submetida indica risco de depressão do sistema nervoso central e respiratório associado à associação de Tramadol (medicação ativa) e Mexazolam (prescrita).  3. **Motivo da recomendação**: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. **Limitações**: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [synthea] Synthea a71a4da3 (56a F, 3 ativos)
- **Doente:** 56a F, problema=diabetes, renal=normal, ativos=['clopidogrel', 'metoprolol', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.995/0.995
    - tramadol: 0.910/0.444/0.444
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre AINE (ibuprofeno) e antiagregante plaquetário (clopidogrel). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

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
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta dor e foi submetida prescrição de ibuprofeno.  2. Motivo do alerta: O sistema identificou alertas relacionados com a prescrição submetida, incluindo risco associado ao AINE (ibuprofeno) em doentes com compromisso renal grave e interação potencial entre AINE e antiagregante plaquetário.  3. Motivo da recomendação: O protótipo não identificou uma nova alternativa admissível dentro da sua base de conhecimento atual. Uma alternativa candidata já consta da medicação ativa do utente, pelo que não foi apresentada mesmo como nova recomendação.  4. Limitações: A análise não avalia ajuste de dose, suspensão ou manutenção terapêutica fora da base atual do protótipo.

## [synthea] Synthea 88c95938 (23a F, 1 ativos)
- **Doente:** 23a F, problema=pain, renal=normal, ativos=['paracetamol'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Notas:**
    - Paracetamol já consta da medicação ativa do utente e tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo. Por esse motivo, não foi apresentado como nova alternativa.
- **LLM (qwen2.5:3b):** 1. **Problema identificado**    O utente apresenta dor e foi submetida prescrição de ibuprofeno.  2. **Motivo do alerta**    Alertas relacionados com a prescrição submetida, incluindo risco renal associado ao ibuprofeno.  3. **Motivo da recomendação**    O protótipo não identificou uma nova alternativa admissível dentro da sua base de conhecimento atual. Uma alternativa candidata já consta da medicação ativa do utente, por isso não foi apresentada como nova recomendação.  4. **Limitações**    A análise não avalia ajuste de dose, suspensão ou manutenção terapêutica fora da base atual do protótipo.

## [synthea] Synthea 147f7d4f (18a F, 1 ativos)
- **Doente:** 18a F, problema=pain, renal=normal, ativos=['paracetamol'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Notas:**
    - Paracetamol já consta da medicação ativa do utente e tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo. Por esse motivo, não foi apresentado como nova alternativa.
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta dor e foi submetida prescrição de ibuprofeno.  2. Motivo do alerta: Nenhuns alertas relevantes foram identificados relacionados com a prescrição submetida ou a medicação ativa.  3. Motivo da recomendação: O protótipo não identificou uma nova alternativa admissível dentro da sua base de conhecimento atual. Uma alternativa candidata já consta da medicação ativa do utente, pelo que não foi apresentada como nova recomendação.  4. Limitações: A análise depende dos dados submetidos e da base de conhecimento atual do protótipo. Não avaliou ajustes de dose ou outras opções clínicas fora desta base.

## [synthea] Synthea 119aedc5 (71a M, 3 ativos)
- **Doente:** 71a M, problema=inflammation, renal=normal, ativos=['naproxen', 'ramipril', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
    - [high] A utilizacao concomitante de dois AINEs deve ser evitada devido ao aumento do risco de toxicidade gastrointestinal, renal e hemorragica. (regra: aine_aine_duplicacao)
    - [moderate] A associacao de AINEs com inibidores da ECA pode reduzir o efeito anti-hipertensor e aumentar o risco de deterioracao da funcao renal. (regra: aine_ieca_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.873/0.873
    - tramadol: 0.910/0.311/0.311
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta dor e febre. Foi submetida prescrição de Ibuprofeno, avaliada no contexto da medicação ativa.  2. Motivo do alerta: O sistema identificou duplicação terapêutica entre Naproxeno e Ibuprofeno. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. Motivo da recomendação: O sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. Limitações: A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [synthea] Synthea f777959f (47a M, 1 ativos)
- **Doente:** 47a M, problema=pain, renal=normal, ativos=['ibuprofen'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Notas:**
    - Ibuprofeno já consta da medicação ativa do utente e tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo. Por esse motivo, não foi apresentado como nova alternativa.
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta dor e foi submetida prescrição de Ibuprofeno.  2. Motivo do alerta: Nenhum alerta identificado relacionado com a prescrição submetida ou medicação ativa.  3. Motivo da recomendação: O protótipo não identificou uma nova alternativa admissível dentro da sua base de conhecimento atual. Uma alternativa candidata já consta da medicação ativa do utente, pelo que não foi apresentado como nova recomendação.  4. Limitações: A análise não avalia ajuste de dose, suspensão, manutenção terapêutica ou outras opções clínicas fora da base atual.

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
    - paracetamol: 0.910/0.677/0.677
    - tramadol: 0.910/0.205/0.205
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor devido a inflamação. Foi submetida prescrição de Ibuprofeno, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre AINE (Ibuprofeno) e antiagregante plaquetário (Clopidogrel). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual.  3. **Motivo da recomendação** O sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [synthea] Synthea 09807e61 (64a F, 3 ativos)
- **Doente:** 64a F, problema=pain, renal=normal, ativos=['clopidogrel', 'metoprolol', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.995/0.995
    - tramadol: 0.910/0.455/0.455
- **LLM (qwen2.5:3b):** 1. **Problema identificado**: Utente com dor, hipertensão e histórico de doença cardíaca apresenta prescrição de ibuprofeno.  2. **Motivo do alerta**: Alerta relacionado à interação AINE + antiagregante plaquetário (Clopidogrel + Ibuprofeno), aumentando risco de hemorragia gastrointestinal.  3. **Motivo da recomendação**: Paracetamol sugerido como alternativa terapêutica, compatível com o problema clínico principal e pertencendo a uma classe diferente do ibuprofeno.  4. **Limitações**: Explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual do protótipo. Não substitui validação clínica.

## [synthea] Synthea 75089c8b (57a M, 1 ativos)
- **Doente:** 57a M, problema=hypertension, renal=normal, ativos=['hydrochlorothiazide'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [moderate] A associacao de AINEs com diureticos tiazidicos pode reduzir a eficacia anti-hipertensora e aumentar o risco de deterioracao da funcao renal. (regra: aine_tiazida_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.958/0.958
    - tramadol: 0.910/0.302/0.302
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor, avaliada no contexto de medicação ativa.  2. **Motivo do alerta** O sistema identificou interação entre AINE (ibuprofeno) e diurético tiazídico (hidroclorotiazida). O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

## [synthea] Synthea 70e4cf5c (43a F, 1 ativos)
- **Doente:** 43a F, problema=diabetes, renal=normal, ativos=['simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta dor, tendo sido submetida prescrição de ibuprofeno.  2. Motivo do alerta: Nenhum alerta foi identificado no contexto da prescrição submetida ou na medicação ativa do utente.  3. Motivo da recomendação: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações: A ausência de alertas e recomendações não significa ausência de opções clínicas; indica apenas que o protótipo não encontrou alternativas admissíveis na sua base atual.

## [synthea] Synthea b01206ca (58a F, 1 ativos)
- **Doente:** 58a F, problema=diabetes, renal=normal, ativos=['simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta dor, com histórico de anemia, diabetes e hipertensão.  2. Motivo do alerta: Nenhum alerta foi identificado relacionado à prescrição de ibuprofeno.  3. Motivo da recomendação: O protótipo não identificou uma alternativa admissível dentro da sua base de conhecimento atual.  4. Limitações: A ausência de recomendação não significa ausência de opções clínicas; significa apenas que o protótipo não identificou alternativa admissível na sua base atual.

## [synthea] Synthea aa442ba0 (45a M, 1 ativos)
- **Doente:** 45a M, problema=pain, renal=normal, ativos=['paracetamol'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:** nenhum
- **Recomendações:** nenhuma
- **Notas:**
    - Paracetamol já consta da medicação ativa do utente e tem indicação compatível com o problema clínico principal na base de conhecimento atual do protótipo. Por esse motivo, não foi apresentado como nova alternativa.
- **LLM (qwen2.5:3b):** 1. Problema identificado: O utente apresenta dor e foi submetida prescrição de ibuprofeno.  2. Motivo do alerta: Nenhum alerta identificado relacionado com a prescrição submetida ou medicação ativa.  3. Motivo da recomendação: O protótipo não identificou uma nova alternativa admissível dentro da sua base de conhecimento atual. Uma alternativa candidata já consta da medicação ativa do utente, pelo que não foi apresentado como nova recomendação.  4. Limitações: A análise não avalia ajuste de dose, suspensão, manutenção terapêutica ou outras opções clínicas fora da base atual.

## [synthea] Synthea c0e63b08 (51a F, 1 ativos)
- **Doente:** 51a F, problema=pain, renal=normal, ativos=['hydrochlorothiazide'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [moderate] A associacao de AINEs com diureticos tiazidicos pode reduzir a eficacia anti-hipertensora e aumentar o risco de deterioracao da funcao renal. (regra: aine_tiazida_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.962/0.962
    - tramadol: 0.910/0.309/0.309
- **LLM (qwen2.5:3b):** 1. **Problema identificado**    Utente com dor, hipertensão e diabetes, receita de ibuprofeno.  2. **Motivo do alerta**    Alerta relacionado com AINE (ibuprofeno) associado a diurético tiazídico (hidroclorotiazida), potencial redução da eficácia anti-hipertensiva e risco renal.  3. **Motivo da recomendação**    Paracetamol sugerido como alternativa, pertencendo a classe analgesica-antipirética compatível com o problema clínico.  4. **Limitações**    Explicação depende dos dados submetidos e base de conhecimento atual do protótipo, não substitui validação clínica.

## [synthea] Synthea 217a7c06 (76a F, 4 ativos)
- **Doente:** 76a F, problema=inflammation, renal=severe_impairment, ativos=['clopidogrel', 'hydrochlorothiazide', 'metoprolol', 'simvastatin'], condições=—, alergias=—
- **Prescrição:** ibuprofen
- **Expectativa:** Perfil realista — sonda com ibuprofeno (AINE) para revelar interações.
- **Alertas:**
    - [high] Ibuprofeno requer precaução acrescida em doentes com compromisso renal grave. (regra: renal_caution)
    - [high] A associacao de um AINE com um antiagregante plaquetario aumenta o risco de hemorragia gastrointestinal. (regra: aine_antiagregante_hemorragia)
    - [moderate] A associacao de AINEs com diureticos tiazidicos pode reduzir a eficacia anti-hipertensora e aumentar o risco de deterioracao da funcao renal. (regra: aine_tiazida_risco_renal)
- **Recomendações (med — heurístico/combinado/final):**
    - paracetamol: 0.910/0.765/0.765
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor e inflamação, submetida prescrição de Ibuprofeno.  2. **Motivo do alerta** Sistema identificou Interação AINE + antiagregante plaquetário entre Clopidogrel e Ibuprofeno. Alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** Sistema identificou Paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.

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
    - paracetamol: 0.910/0.596/0.596
- **LLM (qwen2.5:3b):** 1. **Problema identificado** O utente apresenta dor de cabeça. Foi submetida prescrição de paracetamol, avaliada no contexto da medicação ativa.  2. **Motivo do alerta** O sistema identificou associação AINE + ARA (Losartan e Ibuprofeno) entre os medicamentos envolvidos. O alerta está relacionado com a prescrição submetida e resulta das regras implementadas na base de conhecimento atual do protótipo.  3. **Motivo da recomendação** O sistema identificou paracetamol como alternativa admissível na base de conhecimento atual. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.  4. **Limitações** A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.
