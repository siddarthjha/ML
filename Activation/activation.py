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

# Leaky Rectified Linear Unit

def leaky_relu(Z):
    A = np.maximum(0.1 * Z, Z)
    return A, Z



# Plot the 4 activation functions
z = np.linspace(-10, 10, 100)

# Computes post-activation outputs
A_sigmoid, z = sigmoid(z)
A_tanh, z = tanh(z)
A_relu, z = relu(z)
A_leaky_relu, z = leaky_relu(z)


# Plot sigmoid

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(z, A_sigmoid, label="Function")
plt.plot(z, A_sigmoid * (1 - A_sigmoid), label = "Derivative") 
plt.legend(loc="upper left")
plt.xlabel("z")
plt.ylabel(r"$\frac{1}{1 + e^{-z}}$")
plt.title("Sigmoid Function", fontsize=16)
plt.show()

# Plot tanh

plt.subplot(2, 2, 2)
plt.plot(z, A_tanh, 'b', label = "Function")
plt.plot(z, 1 - np.square(A_tanh), 'r',label="Derivative") 
plt.legend(loc="upper left")
plt.xlabel("z")
plt.ylabel(r"$\frac{e^z - e^{-z}}{e^z + e^{-z}}$") 
plt.title("Hyperbolic Tangent Function", fontsize=16)
plt.show()


# plot relu

plt.subplot(2, 2, 3)
plt.plot(z, A_relu, 'g')
plt.xlabel("z")
plt.ylabel(r"$max\{0, z\}$")
plt.title("ReLU Function", fontsize=16)
plt.show()


# plot leaky relu

plt.subplot(2, 2, 4)
plt.plot(z, A_leaky_relu, 'y')
plt.xlabel("z")
plt.ylabel(r"$max\{0.1z, z\}$")
plt.title("Leaky ReLU Function", fontsize=16)
plt.tight_layout();
plt.show()
