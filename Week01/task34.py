import numpy as np

def create_or_dataset():
    return np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

def create_and_dataset():
    return np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

def initialize_weights(x, y):
    return np.random.uniform(x, y, size=2)

def calculate_loss(w1, w2, bias, dataset):
    loss = 0
    for (x, y, target) in dataset:
        prediction = w1 * x + w2 * y + bias
        loss += (prediction - target) ** 2
    loss /= len(dataset)
    return loss
    

def experiment(dataset, learning_rate, bias, epochs=100_000, epsilon=1e-5):
    w1, w2 = initialize_weights(-1, 1)

    for epoch in range(epochs):
        loss = calculate_loss(w1, w2, bias, dataset)
        loss_plus_epsilon = calculate_loss(w1 + epsilon, w2 + epsilon, bias, dataset)
        L = (loss_plus_epsilon - loss) / epsilon

        offset = learning_rate * L
        w1 -= offset
        w2 -= offset

        if epoch % 1000 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}, Weights: {w1, w2}")

    return w1, w2

def experiment_or(learning_rate, bias, epochs=100_000, epsilon=1e-5):
    dataset = create_or_dataset()
    return experiment(dataset, learning_rate, bias, epochs, epsilon)

def experiment_and(learning_rate, bias, epochs=100_000, epsilon=1e-5):
    dataset = create_and_dataset()
    return experiment(dataset, learning_rate, bias, epochs, epsilon)

def test_model(dataset, w1, w2, bias):
    for (x, y, target) in dataset:
        prediction_value = w1 * x + w2 * y + bias
        prediction = prediction_value >= 0.5
        print(f"Input: {x, y}, Expected: {target}, Predicted: {int(prediction)}, Confidence: {prediction_value}")

def test_or_model(w1, w2, bias):
    dataset = create_or_dataset()
    test_model(dataset, w1, w2, bias)

def test_and_model(w1, w2, bias):
    dataset = create_and_dataset()
    test_model(dataset, w1, w2, bias)

if __name__ == '__main__':
    np.random.seed(42)

    bias_or = -1
    bias_and = -1

    w1_or, w2_or = experiment_or(learning_rate=1e-2, bias=bias_or)
    w1_and, w2_and = experiment_and(learning_rate=1e-1, bias=bias_and)

    print("\n-----------------------------------\n")

    print("OR")
    test_or_model(w1_or, w2_or, bias_or)
    print("\n-----------------------------------\n")
    print("AND")
    test_and_model(w1_and, w2_and, bias_and)


    # OR
    # Input: (np.int64(0), np.int64(0)), Expected: 0, Predicted: 0, Confidence: -1.0
    # Input: (np.int64(0), np.int64(1)), Expected: 1, Predicted: 1, Confidence: 0.9095025208926142
    # Input: (np.int64(1), np.int64(0)), Expected: 1, Predicted: 0, Confidence: -0.24284585423249316
    # Input: (np.int64(1), np.int64(1)), Expected: 1, Predicted: 1, Confidence: 1.666656666660121

    # -----------------------------------

    # AND
    # Input: (np.int64(0), np.int64(0)), Expected: 0, Predicted: 0, Confidence: -1.0
    # Input: (np.int64(0), np.int64(1)), Expected: 0, Predicted: 0, Confidence: -0.13334045761694213
    # Input: (np.int64(1), np.int64(0)), Expected: 0, Predicted: 0, Confidence: 0.13333045761179485
    # Input: (np.int64(1), np.int64(1)), Expected: 1, Predicted: 1, Confidence: 0.9999899999948527

    # Adding a bias seems to have shifted the confidence values
    # The OR models seems to have an error in the prediction
    # Even tweaking the learning rate didn't change the confidence values by a lot