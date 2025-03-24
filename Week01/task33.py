import numpy as np

def create_or_dataset():
    return np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

def create_and_dataset():
    return np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])

def initialize_weights(x, y):
    return np.random.uniform(x, y, size=2)

def calculate_loss(w1, w2, dataset):
    loss = 0
    for (x, y, target) in dataset:
        prediction = w1 * x + w2 * y
        loss += (prediction - target) ** 2
    loss /= len(dataset)
    return loss

def experiment(dataset, learning_rate, epochs=100_000, epsilon=1e-5):
    w1, w2 = initialize_weights(-1, 1)

    for epoch in range(epochs):
        loss1 = (calculate_loss(w1 + epsilon, w2, dataset) - calculate_loss(w1, w2, dataset)) / epsilon
        loss2 = (calculate_loss(w1, w2 + epsilon, dataset) - calculate_loss(w1, w2, dataset)) / epsilon

        w1 -= learning_rate * loss1
        w2 -= learning_rate * loss2

        if epoch % 1000 == 0:
            loss = calculate_loss(w1, w2, dataset)
            print(f"Epoch: {epoch}, Loss: {loss}, Weights: {w1, w2}")

    return w1, w2

def experiment_or(learning_rate, epochs=100_000, epsilon=1e-5):
    dataset = create_or_dataset()
    return experiment(dataset, learning_rate, epochs, epsilon)

def experiment_and(learning_rate, epochs=100_000, epsilon=1e-5):
    dataset = create_and_dataset()
    return experiment(dataset, learning_rate, epochs, epsilon)

def test_model(dataset, w1, w2):
    for (x, y, target) in dataset:
        prediction_value = w1 * x + w2 * y
        prediction = prediction_value >= 0.5
        print(f"Input: {x, y}, Expected: {target}, Predicted: {int(prediction)}, Confidence: {prediction_value}")

def test_or_model(w1, w2):
    dataset = create_or_dataset()
    test_model(dataset, w1, w2)

def test_and_model(w1, w2):
    dataset = create_and_dataset()
    test_model(dataset, w1, w2)

if __name__ == '__main__':
    w1_or, w2_or = experiment_or(learning_rate=1e-4)
    w1_and, w2_and = experiment_and(learning_rate=1e-3)

    print("\n-----------------------------------\n")

    print("OR")
    test_or_model(w1_or, w2_or)
    print("\n-----------------------------------\n")
    print("AND")
    test_and_model(w1_and, w2_and)


    # OR
    # Input: (np.int64(0), np.int64(0)), Expected: 0, Predicted: 0, Confidence: 0.0
    # Input: (np.int64(0), np.int64(1)), Expected: 1, Predicted: 1, Confidence: 0.5853278555404106
    # Input: (np.int64(1), np.int64(0)), Expected: 1, Predicted: 1, Confidence: 0.74799547779403
    # Input: (np.int64(1), np.int64(1)), Expected: 1, Predicted: 1, Confidence: 1.3333233333344405

    # -----------------------------------

    # AND
    # Input: (np.int64(0), np.int64(0)), Expected: 0, Predicted: 0, Confidence: 0.0
    # Input: (np.int64(0), np.int64(1)), Expected: 0, Predicted: 0, Confidence: 0.16872112184249377
    # Input: (np.int64(1), np.int64(0)), Expected: 0, Predicted: 0, Confidence: 0.4979355448245504
    # Input: (np.int64(1), np.int64(1)), Expected: 1, Predicted: 1, Confidence: 0.6666566666670442

    # Both models predict correctly, however, this was only accomplished after tweaking the learning rate.
    # The values of the confidence are very high and not very precise.
    # Also the weights get updated occasionally by tiny bit
