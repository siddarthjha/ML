import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Sigmoid function

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

