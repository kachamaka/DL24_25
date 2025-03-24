import numpy as np
import matplotlib.pyplot as plt

def create_or_dataset():
    return np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

def create_and_dataset():
    return np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

def create_nand_dataset():
    return np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_weights(x, y):
    return np.random.uniform(x, y, size=2)

def calculate_loss(w1, w2, bias, dataset):
    loss = 0
    for (x, y, target) in dataset:
        prediction_sigmoid = sigmoid(w1 * x + w2 * y + bias)
        loss += (prediction_sigmoid - target) ** 2
    loss /= len(dataset)
    return loss
    
def experiment(dataset, learning_rate, bias, epochs=100_000, epsilon=1e-5):
    w1, w2 = initialize_weights(-1, 1)
    loss_data = []

    for epoch in range(epochs):
        loss = calculate_loss(w1, w2, bias, dataset)
        loss_plus_epsilon = calculate_loss(w1 + epsilon, w2 + epsilon, bias, dataset)
        L = (loss_plus_epsilon - loss) / epsilon

        offset = learning_rate * L
        w1 -= offset
        w2 -= offset

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Weights: {w1, w2}")
        
        loss_data.append(loss)

    return w1, w2, loss_data

def experiment_or(learning_rate, bias, epochs=100_000, epsilon=1e-5):
    dataset = create_or_dataset()
    return experiment(dataset=dataset, learning_rate=learning_rate, bias=bias, epochs=epochs, epsilon=epsilon)

def experiment_and(learning_rate, bias, epochs=100_000, epsilon=1e-5):
    dataset = create_and_dataset()
    return experiment(dataset=dataset, learning_rate=learning_rate, bias=bias, epochs=epochs, epsilon=epsilon)

def experiment_nand(learning_rate, bias, epochs=100_000, epsilon=1e-5):
    dataset = create_nand_dataset()
    return experiment(dataset=dataset, learning_rate=learning_rate, bias=bias, epochs=epochs, epsilon=epsilon)

def test_model(dataset, w1, w2, bias):
    for (x, y, target) in dataset:
        prediction_sigmoid = sigmoid(w1 * x + w2 * y + bias)
        prediction = prediction_sigmoid >= 0.5
        print(f"Input: {x, y}, Expected: {target}, Predicted: {int(prediction)}, Confidence: {prediction_sigmoid}")

def test_or_model(w1, w2, bias):
    dataset = create_or_dataset()
    test_model(dataset=dataset, w1=w1, w2=w2, bias=bias)

def test_and_model(w1, w2, bias):
    dataset = create_and_dataset()
    test_model(dataset=dataset, w1=w1, w2=w2, bias=bias)

def test_nand_model(w1, w2, bias):
    dataset = create_nand_dataset()
    test_model(dataset=dataset, w1=w1, w2=w2, bias=bias)

def test_models(w1_or, w2_or, bias_or, w1_and, w2_and, bias_and, w1_nand, w2_nand, bias_nand):
    print("\n-----------------------------------\n")
    print("OR")
    test_or_model(w1=w1_or, w2=w2_or, bias=bias_or)
    print("\n-----------------------------------\n")
    print("AND")
    test_and_model(w1=w1_and, w2=w2_and, bias=bias_and)
    print("\n-----------------------------------\n")
    print("NAND")
    test_nand_model(w1=w1_nand, w2=w2_nand, bias=bias_nand)

if __name__ == '__main__':
    bias_or = -1
    bias_and = -6
    bias_nand = 7
    epochs = 100_000

    # w1_or, w2_or, loss_data_or = experiment_or(learning_rate=1e-2, bias=bias_or, epochs=epochs)
    # w1_and, w2_and, loss_data_and = experiment_and(learning_rate=1e-1, bias=bias_and, epochs=epochs)
    w1_nand, w2_nand, loss_data_nand = experiment_nand(learning_rate=1e-1, bias=bias_nand, epochs=epochs)
    # test_models(w1_or, w2_or, bias_or, w1_and, w2_and, bias_and, w1_nand, w2_nand, bias_nand)
        
    test_nand_model(w1_nand, w2_nand, bias_nand)
    plt.plot(loss_data_nand)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # I successfully reused the functionality by only changing the dataset and tweaking the parameters a bit
