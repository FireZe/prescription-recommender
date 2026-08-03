import lightgbm as lgb
m = lgb.Booster(model_file="backend/models/ltr_model.txt")
for name, gain in sorted(zip(m.feature_name(),
        m.feature_importance(importance_type="gain")), key=lambda x: -x[1])[:10]:
    print(f"{name:28s} {gain:,.0f}")