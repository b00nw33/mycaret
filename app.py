import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pycaret.datasets import get_data
from pycaret.regression import *

data = get_data('insurance')

with open("metrics.txt", "w") as outfile:
    outfile.write("Before data.info():\n")
    data.info() #.to_csv(outfile, header=None, index=None, sep=' ', mode='a')
    outfile.write("After data.info()")