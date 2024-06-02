import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    
    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
    
    def predict(self, pattern, steps=5):
        for _ in range(steps):
            pattern = np.sign(self.weights @ pattern)
        return pattern
    
def plot_pattern(pattern, title):
    plt.imshow(pattern.reshape(10, 10), cmap='binary')
    plt.title(title)
    plt.show()

# Definición de patrones
patterns = [
    np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
              1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
              1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
              0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
              0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
              0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
]

# Inicializar y entrenar la red
hopfield = HopfieldNetwork(size=100)
hopfield.train(patterns)

# Patrones ruidosos para probar
noisy_pattern = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                          1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# Visualización del patrón ruidoso
plot_pattern(noisy_pattern, "Patrón Ruidoso")

# Predicción con la red de Hopfield
predicted_pattern = hopfield.predict(noisy_pattern)

# Visualización del patrón predicho
plot_pattern(predicted_pattern, "Patrón Predicho")

