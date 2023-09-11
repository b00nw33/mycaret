import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.regression import *

data = get_data('insurance')

with open("metrics.txt", "w") as outfile:
    outfile.write("Before writing data.head() to file:\n")
    data.head().to_csv(outfile, header=None, index=None, sep=' ', mode='a')
    outfile.write("After writing data.head() to file")