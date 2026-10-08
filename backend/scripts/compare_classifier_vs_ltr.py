"""Ensaio: classificadores (HistGradientBoosting e Random Forest) vs LambdaMART
(LGBMRanker), com as mesmas features, o mesmo split e o mesmo conjunto de teste.

Baseline adicional: popularidade (frequencia global do candidato no treino).

Uso (a partir da RAIZ do repositorio):
    python -m backend.scripts.compare_classifier_vs_ltr
ou
    python backend/scripts/compare_classifier_vs_ltr.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, ndcg_score,
    balanced_accuracy_score,
)
from scipy.stats import wilcoxon, kendalltau

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "training_examples_mimic.csv"

NUMERIC = [
    "age", "age_squared", "is_elderly", "is_female", "active_medication_count",
    "condition_count", "renal_status_score", "candidate_renal_caution",
    "candidate_qt_risk", "has_anticoagulant", "has_antiplatelet", "has_diuretic",
    "has_acei_or_arb", "has_qt_risk_medication", "candidate_is_nsaid",
]
CATEG = ["candidate", "candidate_class", "main_problem"]
FEATURES = NUMERIC + CATEG
KS = (3, 5)
COMPUTE_KENDALL = True   # concordancia com a ordenacao observada (acrescenta ~1-2 min)
METRIC_ORDER = ["NDCG@3", "NDCG@5", "MAP", "MRR", "Hit@1",
                "Precision@3", "Precision@5", "Recall@3", "Recall@5",
                "HitRate@3", "HitRate@5", "Kendall"]


def per_query_metrics(df, score_cols, ks=KS):
    """Uma unica passagem pelos grupos, calculando as metricas para TODOS os
    scores em simultaneo. Devolve (metricas_por_query, cobertura_do_catalogo)."""
    keys = (["MRR", "Hit@1", "MAP", "Kendall", "n_cand"]
            + [f"{m}@{k}" for k in ks for m in ("NDCG", "Recall", "Precision", "HitRate")])
    out = {c: {k: [] for k in keys} for c in score_cols}
    top3 = {c: set() for c in score_cols}
    catalogo = set()

    for _, g in df.groupby("query_id", sort=False):
        y = g["label_class"].to_numpy(dtype=float)
        if g.shape[0] < 2 or y.sum() == 0:
            continue
        cand = g["candidate"].to_numpy()
        catalogo.update(cand.tolist())
        total_pos = y.sum()
        y_row = y.reshape(1, -1)
        n = len(y)
        y_varia = len(np.unique(y)) > 1

        for c in score_cols:
            s = g[c].to_numpy(dtype=float)
            order = np.argsort(-s)
            yr = y[order]
            top3[c].update(cand[order][:3].tolist())

            hits = np.where(yr >= 1)[0] + 1          # posicoes (1-based) dos positivos
            out[c]["MRR"].append(1.0 / hits[0])
            out[c]["Hit@1"].append(1.0 if yr[0] >= 1 else 0.0)
            out[c]["MAP"].append(float((np.arange(1, len(hits) + 1) / hits).mean()))
            out[c]["n_cand"].append(float(n))

            if COMPUTE_KENDALL and y_varia:
                tau, _ = kendalltau(s, y)
                out[c]["Kendall"].append(0.0 if np.isnan(tau) else float(tau))

            for k in ks:
                out[c][f"NDCG@{k}"].append(ndcg_score(y_row, s.reshape(1, -1), k=k))
                out[c][f"Recall@{k}"].append(yr[:k].sum() / total_pos)
                out[c][f"Precision@{k}"].append(yr[:k].sum() / min(k, n))
                out[c][f"HitRate@{k}"].append(1.0 if yr[:k].sum() > 0 else 0.0)

    pq = {c: {k: np.asarray(v, dtype=float) for k, v in d.items()}
          for c, d in out.items()}
    cobertura = {c: (len(top3[c]) / len(catalogo) if catalogo else 0.0)
                 for c in score_cols}
    return pq, cobertura

def classifier_report(name, yte, proba, t0):
    pred = (proba >= 0.5).astype(int)
    print(f"  {name:<22} Accuracy={accuracy_score(yte, pred):.4f} "
          f"BalancedAcc={balanced_accuracy_score(yte, pred):.4f} "
          f"AUC={roc_auc_score(yte, proba):.4f} "
          f"F1={f1_score(yte, pred):.4f}  [{time.time()-t0:.0f}s]")


def main():
    t0 = time.time()
    print("A carregar dataset...")
    df = pd.read_csv(DATA)

    # Query = (admissao, problema terapeutico): ordenar dentro do mesmo fim clinico
    df["query_id"] = df["query_id"].astype(str) + "|" + df["main_problem"].astype(str)
    for c in NUMERIC:
        if c not in df:
            df[c] = 0
    df = df.sort_values("query_id").reset_index(drop=True)

    split_group = df["subject_id"] if "subject_id" in df.columns else df["query_id"]
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                  .split(df, df["label_class"], split_group))
    train = df.iloc[tr].sort_values("query_id").reset_index(drop=True)
    test = df.iloc[te].copy()

    ytr = (train["label_class"].to_numpy() >= 1).astype(int)
    yte = (test["label_class"].to_numpy() >= 1).astype(int)

    print(f"Treino: {len(train):,} exemplos | Teste: {len(test):,} exemplos "
          f"| split por {'subject_id' if 'subject_id' in df.columns else 'query_id'}")

    # --- Representacao 1: categorias nativas (HistGB e LightGBM) ---
    Xtr, Xte = train[FEATURES].copy(), test[FEATURES].copy()
    for c in CATEG:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)

    # --- Representacao 2: codificacao ordinal (Random Forest) ---
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xtr_ord = train[FEATURES].copy()
    Xte_ord = test[FEATURES].copy()
    Xtr_ord[CATEG] = enc.fit_transform(train[CATEG].astype(str))
    Xte_ord[CATEG] = enc.transform(test[CATEG].astype(str))

    # ------------------------------------------------------------------
    # 1) Classificador A: HistGradientBoosting
    # ------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.0f}s] A treinar HistGradientBoosting...")
    clf_hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=True, class_weight="balanced",
        categorical_features="from_dtype", random_state=42,
    )
    clf_hgb.fit(Xtr, ytr)
    test["hgb_score"] = clf_hgb.predict_proba(Xte)[:, 1]

    # ------------------------------------------------------------------
    # 2) Classificador B: Random Forest
    # ------------------------------------------------------------------
    print(f"[{time.time()-t0:.0f}s] A treinar Random Forest (pode demorar)...")
    clf_rf = RandomForestClassifier(
        n_estimators=250, min_samples_leaf=3, class_weight="balanced",
        n_jobs=-1, random_state=42,
    )
    clf_rf.fit(Xtr_ord, ytr)
    test["rf_score"] = clf_rf.predict_proba(Xte_ord)[:, 1]

    print(f"\n[{time.time()-t0:.0f}s] Desempenho pontual (classificacao binaria):")
    classifier_report("HistGradientBoosting", yte, test["hgb_score"].to_numpy(), t0)
    classifier_report("Random Forest", yte, test["rf_score"].to_numpy(), t0)

    # ------------------------------------------------------------------
    # 3) LambdaMART
    # ------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.0f}s] A treinar o LambdaMART...")
    grp = train.groupby("query_id", sort=False).size().to_numpy()
    rk = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=149,        # <-- SUBSTITUI pelo best_iteration do train_ltr_model.py (deu 176)
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=5,     # = min_data_in_leaf
        colsample_bytree=0.8,    # = feature_fraction
        subsample=0.8,           # = bagging_fraction
        subsample_freq=5,        # = bagging_freq
        random_state=42, verbose=-1)
    rk.fit(Xtr, train["label_class"].to_numpy(), group=grp, categorical_feature=CATEG)
    test["ltr_score"] = rk.predict(Xte)

    # ------------------------------------------------------------------
    # 4) Baseline de popularidade
    # ------------------------------------------------------------------
    rate = train.groupby("candidate")["label_class"].mean()
    glob = train["label_class"].mean()
    test["pop_score"] = test["candidate"].map(rate).fillna(glob)

    # ------------------------------------------------------------------
    # 5) Metricas de ordenacao
    # ------------------------------------------------------------------
    cols = ["ltr_score", "hgb_score", "rf_score", "pop_score"]
    labels = ["LambdaMART", "HistGB", "RandomForest", "Popularidade"]
    print(f"\n[{time.time()-t0:.0f}s] A calcular metricas de ordenacao...")
    pq, cobertura = per_query_metrics(test, cols)

    sizes = test.groupby("query_id", sort=False).size()
    n_queries = len(pq["ltr_score"]["NDCG@3"])
    print(f"\nContextos de teste: {len(sizes):,} | queries avaliadas: {n_queries:,} "
          f"| mediana de candidatos/query: {sizes.median():.0f} "
          f"| queries avaliadas com >=3 candidatos: "
          f"{int((pq['ltr_score']['n_cand'] >= 3).sum()):,}")

    print(f"\n[{time.time()-t0:.0f}s] Metricas de ordenacao (teste):")
    header = f"{'Metrica':11}" + "".join(f"{lb:>15}" for lb in labels)
    print(header)
    for key in METRIC_ORDER:
        row = f"{key:11}" + "".join(f"{pq[c][key].mean():>15.4f}" for c in cols)
        print(row)
    # ------------------------------------------------------------------
    # 5b) Estratificacao por numero de candidatos na consulta
    # ------------------------------------------------------------------
    n_cand = pq["ltr_score"]["n_cand"]
    for minimo in (3, 4, 5):
        mask = n_cand >= minimo
        if mask.sum() < 100:
            continue
        print(f"\n[{time.time()-t0:.0f}s] Apenas consultas com >= {minimo} candidatos "
              f"({int(mask.sum()):,} consultas):")
        print(f"{'Metrica':11}" + "".join(f"{lb:>15}" for lb in labels))
        for key in ("NDCG@3", "NDCG@5", "MAP", "Hit@1"):
            print(f"{key:11}" + "".join(f"{pq[c][key][mask].mean():>15.4f}" for c in cols))
        for c, lb in zip(cols[1:], labels[1:]):
            a, b = pq["ltr_score"]["NDCG@3"][mask], pq[c]["NDCG@3"][mask]
            if np.allclose(a - b, 0):
                continue
            _, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            print(f"  NDCG@3 LTR vs {lb:<13} delta={(a-b).mean():+.4f}  p={p:.3g}")

    # ------------------------------------------------------------------
    # 5c) Cobertura do catalogo no top-3
    # ------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.0f}s] Cobertura do catalogo no top-3 "
          f"(fracao de farmacos distintos que chegam a aparecer no top-3):")
    for c, lb in zip(cols, labels):
        print(f"  {lb:<15} {cobertura[c]:.3f}")

    # ------------------------------------------------------------------
    # 6) Significancia estatistica (Wilcoxon emparelhado por query)
    # ------------------------------------------------------------------
    print(f"\n[{time.time()-t0:.0f}s] Wilcoxon emparelhado (LambdaMART vs cada alternativa):")
    for c, lb in zip(cols[1:], labels[1:]):
        for key in ("NDCG@3", "NDCG@5"):
            a, b = pq["ltr_score"][key], pq[c][key]
            diff = a - b
            if np.allclose(diff, 0):
                print(f"  {key} LTR vs {lb:<13} diferenca nula")
                continue
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
            print(f"  {key} LTR vs {lb:<13} delta={diff.mean():+.4f}  p={p:.3g}")

    # Intervalo de confianca bootstrap para o ganho do LTR sobre cada alternativa
    rng = np.random.default_rng(42)
    print(f"\n[{time.time()-t0:.0f}s] IC 95% bootstrap (1000 reamostragens) do ganho do LTR:")
    for c, lb in zip(cols[1:], labels[1:]):
        for key in ("NDCG@3", "NDCG@5"):
            diff = pq["ltr_score"][key] - pq[c][key]
            idx = rng.integers(0, len(diff), size=(1000, len(diff)))
            boots = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  {key} LTR - {lb:<13} {diff.mean():+.4f}  IC95% [{lo:+.4f}; {hi:+.4f}]")

    print(f"\nConcluido em {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
