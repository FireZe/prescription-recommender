from app.llm_explainer import (
    contains_cjk_characters,
    contains_garbled_or_unwanted_language,
    contains_non_latin_letters,
    contains_unwanted_language_markers,
    is_valid_llm_explanation,
)


def test_rejects_cjk_extension_a_character():
    text = """
㘎. Problema identificado
O utente apresenta inflamação.

2. Motivo do alerta
O sistema identificou alertas relacionados com a prescrição submetida.

3. Motivo da recomendação
O sistema identificou paracetamol como alternativa sugerida.

4. Limitações
A explicação depende dos dados submetidos e da base de conhecimento atual.
"""

    assert contains_cjk_characters(text)
    assert not is_valid_llm_explanation(text)


def test_rejects_garbled_joined_token():
    text = """
1. Problema identificado
O utente apresenta inflamação.

2. Motivo do alerta
O sistema identificou interação entre ibuprofeno e clopidogrel.afferentes alertas estão relacionados com a prescrição submetida.

3. Motivo da recomendação
O sistema identificou paracetamol como alternativa sugerida.

4. Limitações
A explicação depende dos dados submetidos e da base de conhecimento atual.
"""

    assert contains_garbled_or_unwanted_language(text)
    assert not is_valid_llm_explanation(text)

def test_rejects_webkit_artifact():
    text = """
1. Problema identificado
O utente apresenta inflamação.

2. Motivo do alerta
O sistema identificou interação entre ibuproWebKit e clopidogrel aumenta o risco de hemorragia.

3. Motivo da recomendação
O sistema identificou paracetamol como alternativa sugerida.

4. Limitações
A explicação depende dos dados submetidos e da base de conhecimento atual.
"""

    assert contains_garbled_or_unwanted_language(text)
    assert not is_valid_llm_explanation(text)

def test_rejects_thai_text():
    text = """
เมื่อวันที่
1. Problema identificado
O utente apresenta inflamação.

2. Motivo do alerta
O sistema identificou alertas relacionados com a prescrição submetida.

3. Motivo da recomendação
O sistema identificou paracetamol como alternativa sugerida.

4. Limitações
A explicação depende dos dados submetidos e da base de conhecimento atual.
"""

    assert contains_non_latin_letters(text)
    assert not is_valid_llm_explanation(text)


def test_rejects_english_explanation():
    text = """
1. Problema identificado
The patient presents pain and was prescribed ibuprofen.

2. Motivo do alerta
The system identified a bleeding risk with active medication.

3. Motivo da recomendação
The recommendation is paracetamol.

4. Limitações
This explanation depends on the current knowledge base.
"""

    assert contains_unwanted_language_markers(text)
    assert not is_valid_llm_explanation(text)


def test_accepts_valid_portuguese_explanation():
    text = """
1. Problema identificado
O utente apresenta dor. Foi submetida prescrição de ibuprofeno, avaliada no contexto da medicação ativa.

2. Motivo do alerta
O sistema identificou um alerta relacionado com a prescrição submetida, com base nas regras implementadas na base de conhecimento atual do protótipo.

3. Motivo da recomendação
O sistema identificou paracetamol como alternativa sugerida. Na base de conhecimento atual do protótipo não foi identificado o mesmo alerta para esta alternativa.

4. Limitações
A explicação depende dos dados submetidos, das regras implementadas e da base de conhecimento atual. Não substitui validação clínica.
"""

    assert is_valid_llm_explanation(text)