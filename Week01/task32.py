import numpy as np

def create_dataset(n):
    return [(i, 2 * i) for i in range(n)]

def initialize_weights(x, y):
    return np.random.uniform(x, y)

def calculate_loss(w, dataset):
    loss = 0
    for (input, target) in dataset:
        prediction = w * input
        loss += (prediction - target) ** 2
    loss /= len(dataset)
    return loss

def experiment(learning_rate, epochs=500, n=6, epsilon=1e-5):
    dataset = create_dataset(n)
    w = initialize_weights(0, 10)

    for epoch in range(epochs):
        loss_before = calculate_loss(w, dataset)
        loss_plus_epsilon = calculate_loss(w + epsilon, dataset)

        L = (loss_plus_epsilon - loss_before) / epsilon
        w -= learning_rate * L

        print(f"Epoch {epoch}: Weight = {w}")
    

if __name__ == '__main__':
    # np.random.seed(42)
    
    learning_rate = 1e-2
    experiment(learning_rate)

    # The model converges relatively close to 2 (1.9999999) which I guess is good?
