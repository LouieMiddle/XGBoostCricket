import xgboost as xgb

from matplotlib import pyplot as plt

model = xgb.Booster()

model.load_model("../models/JosButtler_RightArmSeam_stumpsX_stumpsY.json")

# model.feature_names = ['pitchY', 'stumpsX', 'stumpsY', 'pitchX']

xgb.plot_tree(model)
plt.show()
