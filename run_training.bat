@echo off
REM ============================================================
REM  Pipeline completo: extract -> treino, tudo em sequencia.
REM  Cada passo so arranca quando o anterior termina COM SUCESSO.
REM
REM  Se NAO quiseres re-gerar os dados, comenta a linha do extract
REM  (poe REM no inicio) para ir direto aos treinos.
REM
REM  ORDEM OBRIGATORIA: extract -> supervised -> (ltr) -> meta
REM    (o meta carrega o ranking_model.joblib do supervised)
REM ============================================================

cd /d C:\Users\pedro\Uni\Pedro\2ano\DIMEI\prescription-recommender

echo.
echo [1/4] extract_mimic_training_data.py  (%TIME%)
python backend\scripts\extract_mimic_training_data.py || goto :error

echo.
echo [2/4] train_supervised_ranking_model.py  (%TIME%)
python backend\scripts\train_supervised_ranking_model.py || goto :error

echo.
echo [3/4] train_ltr_model.py  (%TIME%)
python backend\scripts\train_ltr_model.py || goto :error

echo.
echo [4/4] train_meta_learner.py  (%TIME%)
python backend\scripts\train_meta_learner.py || goto :error

echo.
echo ============================================================
echo  PIPELINE COMPLETO CONCLUIDO COM SUCESSO  (%TIME%)
echo ============================================================
goto :eof

:error
echo.
echo ************************************************************
echo  ERRO no script anterior (codigo %errorlevel%).
echo  Cadeia interrompida — os passos seguintes NAO correram.
echo ************************************************************
exit /b %errorlevel%
