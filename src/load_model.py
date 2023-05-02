import xgboost as xgb

model = xgb.Booster()

model.load_model("../models/ENTER-MODEl-NAME.json")

# Can then do any addition testing on models here
