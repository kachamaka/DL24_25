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

if __name__ == '__main__':
    np.random.seed(42)
    n = 6
    dataset = create_dataset(n)

    w = initialize_weights(0, 10)
    loss = calculate_loss(w, dataset)
    print(f"MSE: {loss}")

    # loss_plus_2 = calculate_loss(w + 0.001 * 2, dataset)
    # loss_plus_1 = calculate_loss(w + 0.001, dataset)
    # loss_minus_1 = calculate_loss(w - 0.001, dataset)
    # loss_minus_2 = calculate_loss(w - 0.001 * 2, dataset)
    # print(f"Loss (w + 0.001 * 2): {loss_plus_2}")
    # print(f"Loss (w + 0.001): {loss_plus_1}")
    # print(f"Loss (w - 0.001): {loss_minus_1}")
    # print(f"Loss (w - 0.001 * 2): {loss_minus_2}")

    # Loss (w + 0.001 * 2): 27.989600040224502
    # Loss (w + 0.001): 27.957573518435822
    # Loss (w - 0.001): 27.893575474858455
    # Loss (w - 0.001 * 2): 27.86160395306978

    # The loss increases as the weight increases and decreases as the weight decreases
    # This means that the gradient is positive? So maybe we want to decrese the weight since we aim to minimize the loss? 
    
    # Could you please confirm that I have this right? :)
