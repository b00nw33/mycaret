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
color = ['C0', 'C1', 'C2', 'C3']
cols = ['age', 'bmi', 'charges', 'smoker']

reg = setup(data=data, target='charges', train_size = 0.8, session_id = 7402,
            numeric_features = numeric[:-1], categorical_features = categorical,
            transformation = True, normalize = True)

best = compare_models(sort='RMSE')