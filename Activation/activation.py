import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# Sigmoid function

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

# Hyperbolic function

def tanh(Z):
    A = np.tanh(Z)
    return A, Z

# Rectified Linear unit

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

