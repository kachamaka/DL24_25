import numpy as np
import matplotlib.pyplot as plt


def create_or_dataset():
    return np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

def create_and_dataset():
    return np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def initialize_weights(x, y):
    return np.random.uniform(x, y, size=2)

def calculate_loss(w1, w2, bias, dataset, use_sigmoid=True):
    loss = 0
    for (x, y, target) in dataset:
        prediction_sigmoid = sigmoid(w1 * x + w2 * y + bias) if use_sigmoid else w1 * x + w2 * y + bias
        loss += (prediction_sigmoid - target) ** 2
    loss /= len(dataset)
    return loss
    

def experiment(dataset, learning_rate, bias, epochs=100_000, use_sigmoid=True, epsilon=1e-5):
    w1, w2 = initialize_weights(-1, 1)
    loss_data = []

    for epoch in range(epochs):
        loss = calculate_loss(w1, w2, bias, dataset, use_sigmoid)
        loss_plus_epsilon = calculate_loss(w1 + epsilon, w2 + epsilon, bias, dataset, use_sigmoid)
        L = (loss_plus_epsilon - loss) / epsilon

        offset = learning_rate * L
        w1 -= offset
        w2 -= offset

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Weights: {w1, w2}")
        
        loss_data.append(loss)

    return w1, w2, loss_data

def experiment_or(learning_rate, bias, epochs=100_000, use_sigmoid=True, epsilon=1e-5):
    dataset = create_or_dataset()
    return experiment(dataset=dataset, learning_rate=learning_rate, bias=bias, epochs=epochs, use_sigmoid=use_sigmoid, epsilon=epsilon)

def experiment_and(learning_rate, bias, epochs=100_000, use_sigmoid=True, epsilon=1e-5):
    dataset = create_and_dataset()
    return experiment(dataset=dataset, learning_rate=learning_rate, bias=bias, epochs=epochs, use_sigmoid=use_sigmoid, epsilon=epsilon)

def test_model(dataset, w1, w2, bias, use_sigmoid=True):
    for (x, y, target) in dataset:
        prediction_sigmoid = sigmoid(w1 * x + w2 * y + bias) if use_sigmoid else w1 * x + w2 * y + bias
        prediction = prediction_sigmoid >= 0.5
        print(f"Input: {x, y}, Expected: {target}, Predicted: {int(prediction)}, Confidence: {prediction_sigmoid}")

def test_or_model(w1, w2, bias, use_sigmoid=True):
    dataset = create_or_dataset()
    test_model(dataset=dataset, w1=w1, w2=w2, bias=bias, use_sigmoid=use_sigmoid)

def test_and_model(w1, w2, bias, use_sigmoid=True):
    dataset = create_and_dataset()
    test_model(dataset=dataset, w1=w1, w2=w2, bias=bias, use_sigmoid=use_sigmoid)

def test_models(w1_or, w2_or, bias_or, w1_and, w2_and, bias_and, use_sigmoid=True):
    print("\n-----------------------------------\n")
    print("OR")
    test_or_model(w1=w1_or, w2=w2_or, bias=bias_or, use_sigmoid=use_sigmoid)
    print("\n-----------------------------------\n")
    print("AND")
    test_and_model(w1=w1_and, w2=w2_and, bias=bias_and, use_sigmoid=use_sigmoid)

def plot_loss(loss_data_or, loss_data_and):
    plt.plot(loss_data_or, label="OR")
    plt.plot(loss_data_and, label="AND")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    bias_or = -1
    bias_and = -6
    epochs = 100_000

    w1_or, w2_or, loss_data_or = experiment_or(learning_rate=1e-2, bias=bias_or, epochs=epochs)
    w1_and, w2_and, loss_data_and = experiment_and(learning_rate=1e-1, bias=bias_and, epochs=epochs)
    test_models(w1_or, w2_or, bias_or, w1_and, w2_and, bias_and)

    # If you want to test, uncomment all the 3 lines below and comment the 3 lines above
    # w1_or, w2_or, loss_data_or = experiment_or(learning_rate=1e-2, bias=bias_or, epochs=epochs, use_sigmoid=False)
    # w1_and, w2_and, loss_data_and = experiment_and(learning_rate=1e-1, bias=bias_and, epochs=epochs, use_sigmoid=False)
    # test_models(w1_or, w2_or, bias_or, w1_and, w2_and, bias_and, use_sigmoid=False)

    plot_loss(loss_data_or, loss_data_and)

    # Before sigmoid
    # Loss function converges to some value almost instantly and does not change at all after that

    # After sigmoid
    # Loss function eventually converges to some value smoother
    # Also sigmoid returns values between 0 and 1 which is more suitable for binary classification

