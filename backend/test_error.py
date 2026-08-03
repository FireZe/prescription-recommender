import lightgbm as lgb
from app.ml_model import _aligned_row
m = lgb.Booster(model_file="models/ltr_model.txt")
feats = {"age":72,"age_squared":5184,"is_elderly":1,"is_female":0,
 "active_medication_count":3,"condition_count":2,"renal_status_score":1,
 "candidate_renal_caution":0,"candidate_qt_risk":0,"has_anticoagulant":0,
 "has_antiplatelet":0,"has_diuretic":0,"has_acei_or_arb":0,"has_qt_risk_medication":0,
 "same_therapeutic_class":1,"candidate_is_nsaid":0,"candidate":"naproxeno",
 "candidate_class":"aine","original_medication":"ibuprofeno","original_class":"aine","main_problem":"pain"}
from app.ml_model import predict_ltr_score; print(predict_ltr_score(feats))