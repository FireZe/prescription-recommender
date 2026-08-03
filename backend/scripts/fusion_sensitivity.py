"""Sensibilidade da fusão meta-learner/LTR: varia w em (w·meta + (1-w)·ltr) e mede
NDCG no conjunto de teste do MIMIC (mesmo split do treino). Uso: python scripts\\fusion_sensitivity.py"""
from pathlib import Path
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score

BASE = Path(__file__).resolve().parents[1]
df   = pd.read_csv(BASE/"data"/"training_examples_mimic.csv")
rank = joblib.load(BASE/"models"/"ranking_model.joblib")
ltr  = lgb.Booster(model_file=str(BASE/"models"/"ltr_model.txt"))
meta = joblib.load(BASE/"models"/"meta_learner.joblib")

_, test_idx = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                   .split(df, df["label_class"], df["query_id"]))
test = df.loc[test_idx].copy()

# classificador -> prob classe 1 e 0
Xc = test.reindex(columns=list(rank.feature_names_in_)).copy()
for c in Xc.columns:
    if Xc[c].dtype == object: Xc[c] = Xc[c].astype("category")
proba = rank.predict_proba(Xc); classes = list(rank.classes_)
p1 = proba[:, classes.index(1)]; p0 = proba[:, classes.index(0)]

# LTR
LTR_CATS = {"candidate", "candidate_class", "main_problem"}
Xl = test.reindex(columns=ltr.feature_name()).copy()
for c in Xl.columns:
    if c in LTR_CATS: Xl[c] = Xl[c].astype("category")
test["ltr"] = np.clip((ltr.predict(Xl) + 2) / 6, 0, 1)

# meta (proxy heurístico vetorizado, como no ml_model.py)
def col(n): return test[n].fillna(0).astype(float).values
renal, rc = col("renal_status_score"), col("candidate_renal_caution")
sseg = np.clip(1 - 0.4*(renal==2)*rc - 0.2*col("candidate_qt_risk")*col("has_qt_risk_medication"), 0, 1)
sctx = np.clip(0.5 + 0.35*col("same_therapeutic_class") - 0.1*(renal==1)*rc - 0.3*(renal==2)*rc, 0, 1)
ssim = np.clip(0.5 + 0.5*col("same_therapeutic_class"), 0, 1)
hproxy = 0.45*sseg + 0.30*sctx + 0.20*ssim
hcol = list(getattr(meta, "feature_names_in_", ["heuristic_proxy"]))[0]
metaX = pd.DataFrame({hcol: hproxy, "prob_class_1": p1, "prob_class_0": p0})[[hcol, "prob_class_1", "prob_class_0"]]
test["meta"] = meta.predict_proba(metaX)[:, list(meta.classes_).index(1)]

def mean_ndcg(colname, ks=(3,5)):
    out = {k: [] for k in ks}
    for _, g in test.groupby("query_id"):
        rel = g["label_class"].values.reshape(1,-1)
        if rel.max() > 0 and rel.shape[1] > 1:
            sc = g[colname].values.reshape(1,-1)
            for k in ks: out[k].append(ndcg_score(rel, sc, k=k))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in out.items()}

print("| w_meta | w_ltr | NDCG@3 | NDCG@5 |"); print("|--:|--:|--:|--:|")
best = (-1, -1)
for i in range(11):
    w = i/10; test["fused"] = w*test["meta"] + (1-w)*test["ltr"]
    nd = mean_ndcg("fused")
    print(f"| {w:.2f} | {1-w:.2f} | {nd[3]:.4f} | {nd[5]:.4f} |")
    if nd[3] > best[1]: best = (w, nd[3])
print(f"\nMelhor w_meta por NDCG@3: {best[0]:.2f}; configuração atual: 0,60")