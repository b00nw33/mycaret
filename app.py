import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.regression import *

data = get_data('insurance')

with open("metrics.txt", "w") as outfile:
    outfile.write(data.head() + "\n")
    outfile.write(data.info() + "\n")
