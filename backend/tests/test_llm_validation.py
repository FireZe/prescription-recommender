from app.llm_explainer import (
    contains_cjk_characters,
    contains_garbled_or_unwanted_language,
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