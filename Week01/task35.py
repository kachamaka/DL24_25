import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    x = np.arange(-10, 10, 0.2)
    y = sigmoid(x)

    plt.plot(x, y)
    plt.grid(True)
    plt.show()
