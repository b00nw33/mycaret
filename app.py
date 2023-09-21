import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.regression import *

data = get_data('insurance')

with open("metrics.txt", "w") as outfile:
    outfile.write("Before data.info():\n")
    data.info()
    outfile.write("After data.info()")

numeric = ['age', 'bmi', 'children', 'charges']
categorical = ['smoker', 'sex', 'region']
# color = ['C0', 'C1', 'C2', 'C3']
# cols = ['age', 'bmi', 'charges', 'smoker']

reg = setup(data=data, target='charges', train_size = 0.8, session_id = 7402,
            numeric_features = numeric[:-1], categorical_features = categorical,
            transformation = True, normalize = True)

# best = compare_models(sort='RMSE')

model = create_model('gbr', fold = 10)

params = {
        'learning_rate': [0.05, 0.08, 0.1],
        'max_depth': [1,2, 3, 4, 5, 6, 7, 8],
        'subsample': [0.8, 0.9, 1, 1.1],
        'n_estimators' : [100, 200, 300, 400, 500]
    }

tuned_model = tune_model(model, optimize = 'RMSE', fold = 10,
                       custom_grid = params, n_iter = 100)

predictions = predict_model(model)
predictions.head()

plot_model(tuned_model, 'feature', scale = 4)

plot_model(model, 'error')

final_model = finalize_model(tuned_model)

save_model(final_model, 'regression_model')