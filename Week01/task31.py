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

def experiment_with_prints(learning_rate, epochs=10, n=6, epsilon=1e-5):
    dataset = create_dataset(n)
    w = initialize_weights(0, 10)

    for epoch in range(epochs):
        loss_before = calculate_loss(w, dataset)
        print(f"Epoch {epoch + 1}")
        print(f"Loss before updating w: {loss_before}")

        loss_plus_epsilon = calculate_loss(w + epsilon, dataset)

        L = (loss_plus_epsilon - loss_before) / epsilon # Slope = delta(y) / delta(x) from notes.md where y is value of loss function

        w -= learning_rate * L

        loss_after = calculate_loss(w, dataset)
        print(f"Loss after updating w: {loss_after}")
        print(f"Updated w: {w}\n")
    
    return w

def experiment(learning_rate, epochs=10, n=6, epsilon=1e-5):
    dataset = create_dataset(n)
    w = initialize_weights(0, 10)

    for _ in range(epochs):
        loss_before = calculate_loss(w, dataset)
        loss_plus_epsilon = calculate_loss(w + epsilon, dataset)

        L = (loss_plus_epsilon - loss_before) / epsilon
        w -= learning_rate * L
    
    return w


if __name__ == '__main__':
    np.random.seed(42)
    experiment_with_prints(learning_rate=0.01)
    
    print("\n-----------------------------------\n")

    learning_rates = [1, 0.1, 0.01, 0.001]
    for learning_rate in learning_rates:
        print(f"Learning rate: {learning_rate}")
        w = experiment(learning_rate)
        print(f"Final weight: {w}")

    # Learning rate: 1
    # Final weight: 9950111893.945934

    # Learning rate: 0.1
    # Final weight: 1.9999950000859201

    # Learning rate: 0.01
    # Final weight: 3.5242375280758687

    # Learning rate: 0.001
    # Final weight: 1.5988791728501839

    # It is interesting how a learning rate of 0.1 gives the best result.
    # I think it is important what kind of gradient we are using
    # In task29, I directly calculated the derivative and used it like this, and a learning rate of 0.001 was optimal (based on that that I had 1000 epochs)
    # However, here I used the approximation of the derivative and a learning rate of 0.1 was optimal (based on that that I had 10 epochs)