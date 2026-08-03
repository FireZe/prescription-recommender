import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
n=3000; rng=np.random.default_rng(0)
X=pd.DataFrame({"age":rng.integers(20,90,n).astype(float),
    "candidate":pd.Series(rng.choice([f"d{i}" for i in range(120)],n)).astype("category"),
    "main_problem":pd.Series(rng.choice(["a","b","c"],n)).astype("category")})
y=(X["age"]>60).astype(int); g=rng.integers(0,400,n)
m=CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=50,
    categorical_features="from_dtype",random_state=42),method="isotonic",cv=3)
p=cross_val_predict(m,X,y,cv=StratifiedGroupKFold(5,shuffle=True,random_state=42),
    method="predict_proba",groups=g)
print("OK", p.shape)