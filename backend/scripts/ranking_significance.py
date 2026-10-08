"""Significância do ganho do LambdaMART sobre o baseline de popularidade.

Reaproveita o mesmo split (GroupShuffleSplit por subject_id, seed=42), treina o
LambdaMART e calcula o NDCG@3/@5 POR QUERY para o LTR e para a popularidade.
Sobre os ganhos emparelhados (LTR - popularidade) reporta:
  - teste de Wilcoxon signed-rank (p-value);
  - intervalo de confiança 95% do ganho medio por bootstrap.

Uso: python -m scripts.ranking_significance
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score
import lightgbm as lgb

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "training_examples_mimic.csv"

NUMERIC = ["age", "age_squared", "is_elderly", "is_female","active_medication_count",
    "condition_count", "renal_status_score", "candidate_renal_caution", "candidate_qt_risk",
    "has_anticoagulant", "has_antiplatelet", "has_diuretic", "has_acei_or_arb",
    "has_qt_risk_medication", "candidate_is_nsaid"]
CATEG = ["candidate", "candidate_class", "main_problem"]
FEATURES = NUMERIC + CATEG
KS = (3, 5)
N_BOOT = 2000
SEED = 42


def per_query_ndcg(df, score_col, ks=KS):
    """Devolve {k: {query_id: ndcg}} apenas para queries avaliaveis (>=2 candidatos, >=1 positivo)."""
    out = {k: {} for k in ks}
    for qid, g in df.groupby("query_id"):
        y = g["label_class"].values.astype(float)
        if g.shape[0] < 2 or y.sum() == 0:
            continue
        s = g[score_col].values.astype(float)
        for k in ks:
            out[k][qid] = ndcg_score(y.reshape(1, -1), s.reshape(1, -1), k=k)
    return out


def bootstrap_ci(gains, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    gains = np.asarray(gains, dtype=float)
    n = len(gains)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = gains[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(gains.mean()), float(lo), float(hi)


def main():
    t0 = time.time()
    print("A carregar dataset..."); df = pd.read_csv(DATA)
    df["query_id"] = df["query_id"].astype(str) + "|" + df["main_problem"].astype(str)
    for c in NUMERIC:
        if c not in df:
            df[c] = 0
    df = df.sort_values("query_id").reset_index(drop=True)
    split_group = df["subject_id"] if "subject_id" in df.columns else df["query_id"]
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED)
                  .split(df, df["label_class"], split_group))
    train = df.iloc[tr].sort_values("query_id").reset_index(drop=True)
    test = df.iloc[te].copy()

    Xtr, Xte = train[FEATURES].copy(), test[FEATURES].copy()
    for c in CATEG:
        Xtr[c] = Xtr[c].astype("category")
        Xte[c] = pd.Categorical(Xte[c], categories=Xtr[c].cat.categories)

    print(f"[{time.time()-t0:.0f}s] A treinar o LambdaMART...")
    grp = train.groupby("query_id").size().values
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05,
                        random_state=SEED, verbose=-1)
    rk.fit(Xtr, train["label_class"].values, group=grp, categorical_feature=CATEG)
    test["ltr_score"] = rk.predict(Xte)

    rate = train.groupby("candidate")["label_class"].mean()
    glob = train["label_class"].mean()
    test["pop_score"] = test["candidate"].map(rate).fillna(glob)

    nd_ltr = per_query_ndcg(test, "ltr_score")
    nd_pop = per_query_ndcg(test, "pop_score")

    print(f"\n[{time.time()-t0:.0f}s] Significancia do ganho LambdaMART - popularidade:")
    print(f"{'Metrica':10}{'LTR':>9}{'Pop.':>9}{'Ganho':>9}{'IC95%':>20}{'p (Wilcoxon)':>15}")
    for k in KS:
        qids = sorted(nd_ltr[k].keys())
        a = np.array([nd_ltr[k][q] for q in qids])
        b = np.array([nd_pop[k][q] for q in qids])
        gain = a - b
        mean_gain, lo, hi = bootstrap_ci(gain)
        if wilcoxon is not None and np.any(gain != 0):
            p = wilcoxon(a, b, zero_method="wilcox", alternative="greater").pvalue
            p_str = f"{p:.2e}"
        else:
            p_str = "n/d (instala scipy)"
        print(f"NDCG@{k:<5}{a.mean():>9.4f}{b.mean():>9.4f}{mean_gain:>9.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>20}{p_str:>15}")
    print(f"\nQueries avaliadas (>=2 candidatos, >=1 positivo): {len(nd_ltr[3]):,}")
    print(f"Concluido em {time.time()-t0:.0f}s.")


if __name__ == "__main__":
    main()
