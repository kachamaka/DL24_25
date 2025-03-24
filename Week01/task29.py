import numpy as np

def create_dataset(n):
    return [(i, 2 * i) for i in range(n)]

def initialize_weights(x, y):
    return np.random.uniform(x, y)

if __name__ == '__main__':
    print(create_dataset(4))
    print(initialize_weights(0, 100))
    print(initialize_weights(0, 10))

    # General form of the model: y_pred = w * x
    # The model has 1 parameter: w
