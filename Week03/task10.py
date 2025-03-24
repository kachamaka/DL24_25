import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

RELU = 'ReLU'
SIGMOID = 'Sigmoid'
LEAKY_RELU = 'LeakyReLU'

def get_dataloader(data):
    input = data.drop("Potability", axis=1)
    target = data["Potability"]

    input = torch.tensor(input.to_numpy(), dtype=torch.float32)
    target = torch.tensor(target.to_numpy(), dtype=torch.float32)

    dataset = TensorDataset(input, target)
    return DataLoader(dataset, batch_size=8, shuffle=True)

def create_model(input_size, hidden_layers, activation_functions):
    layers = []
    layers.append(nn.Linear(input_size, hidden_layers[0]))

    for i, activation_function_name in enumerate(activation_functions):
        activation_function = getattr(nn, activation_function_name, None)
        if activation_function:
            layers.append(activation_function())
        layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))

    layers.append(nn.Linear(hidden_layers[-1], 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)

def plot_losses(train_list_losses, test_list_losses, train_f1_scores, test_f1_scores, name):
    fig, axs = plt.subplots(1, 2, figsize=(13, 7))

    axs[0].plot(train_list_losses, label='Train Loss')
    axs[0].plot(test_list_losses, label='Test Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Average Loss')
    axs[0].set_title(f'Train vs Test Loss for {name}')
    axs[0].legend()

    axs[1].plot(train_f1_scores, label='Train F1 Score')
    axs[1].plot(test_f1_scores, label='Test F1 Score')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('F1 Score')
    axs[1].set_title(f'Train vs Test F1 Score for {name}')
    axs[1].legend()

    fig.suptitle(f'Experiment: {name}', fontsize=16)

    plt.tight_layout()
    plt.show()

def simulation(train_dataloader, test_dataloader, validation_dataloader, input_size, hidden_sizes, activation_functions, learning_rate=0.001, epochs=30):
    model = create_model(input_size, hidden_sizes, activation_functions)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    validation_losses = []
    train_f1_scores = []
    validation_f1_scores = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_targets = []
        train_predictions = []

        train_progress_bar = tqdm(train_dataloader)
        for input, target in train_progress_bar:
            optimizer.zero_grad()
            prediction = model(input).squeeze()

            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_targets.extend(target.numpy())
            train_predictions.extend((prediction.detach().numpy() > 0.5).astype(int))

        train_avg_loss = train_loss / len(train_dataloader)
        train_losses.append(train_avg_loss)

        train_f1 = f1_score(train_targets, train_predictions)
        train_f1_scores.append(train_f1)

        model.eval()
        validation_loss = 0
        validation_targets = []
        validation_predictions = []

        with torch.no_grad():
            validation_progress_bar = tqdm(validation_dataloader)
            for input, target in validation_progress_bar:
                prediction = model(input).squeeze()

                loss = criterion(prediction, target)
                validation_loss += loss.item()
                validation_targets.extend(target.numpy())
                validation_predictions.extend((prediction.detach().numpy() > 0.5).astype(int))

        validation_avg_loss = validation_loss / len(validation_dataloader)
        validation_losses.append(validation_avg_loss)

        validation_f1 = f1_score(validation_targets, validation_predictions)
        validation_f1_scores.append(validation_f1)

        train_counts = np.bincount(train_predictions, minlength=2)
        train_distribution = train_counts / len(train_predictions)

        validation_counts = np.bincount(validation_predictions, minlength=2)
        validation_distribution = validation_counts / len(validation_predictions)

        tqdm.write(f'Epoch [{epoch+1}/{epochs}]:')
        tqdm.write(f'Average training loss: {train_avg_loss}')
        tqdm.write(f'Average validation loss: {validation_avg_loss}')
        tqdm.write(f'Training metric score: {train_f1}')
        tqdm.write(f'Validation metric score: {validation_f1}')
        tqdm.write(f'Distribution of train predictions: {train_distribution}')
        tqdm.write(f'Distribution of validation predictions: {validation_distribution}')

    
    test_loss = 0
    test_targets = []
    test_predictions = []
    model.eval()
    test_progress_bar = tqdm(test_dataloader)
    for input, target in test_progress_bar:
        prediction = model(input).squeeze()

        loss = criterion(prediction, target)
        test_loss += loss.item()
        test_targets.extend(target.numpy())
        test_predictions.extend((prediction.detach().numpy() > 0.5).astype(int))

    test_avg_loss = test_loss / len(test_dataloader)
    test_f1 = f1_score(test_targets, test_predictions)
    tqdm.write(f'Average test loss: {test_avg_loss}')
    tqdm.write(f'Test metric score: {test_f1}')

    return train_losses, validation_losses, train_f1_scores, validation_f1_scores

def base(train_dataloader, test_dataloader, validation_dataloader, learning_rate, epochs, name):
    hidden_layers = [32, 16]
    activation_functions = [RELU]
    train_losses, validation_losses, train_f1_scores, validation_f1_scores = simulation(train_dataloader, test_dataloader, validation_dataloader, 9, hidden_layers, activation_functions, learning_rate, epochs)
    plot_losses(train_losses, validation_losses, train_f1_scores, validation_f1_scores, name)

def experiment1(train_dataloader, test_dataloader, validation_dataloader, learning_rate, epochs, name):
    hidden_layers = [16, 64, 32, 8]
    activation_functions = [RELU, RELU, RELU]
    train_losses, validation_losses, train_f1_scores, validation_f1_scores = simulation(train_dataloader, test_dataloader, validation_dataloader, 9, hidden_layers, activation_functions, learning_rate, epochs)
    plot_losses(train_losses, validation_losses, train_f1_scores, validation_f1_scores, name)

def experiment2(train_dataloader, test_dataloader, validation_dataloader, learning_rate, epochs, name):
    hidden_layers = [16, 32, 64, 12]
    activation_functions = [RELU, LEAKY_RELU, LEAKY_RELU]
    train_losses, validation_losses, train_f1_scores, validation_f1_scores = simulation(train_dataloader, test_dataloader, validation_dataloader, 9, hidden_layers, activation_functions, learning_rate, epochs)
    plot_losses(train_losses, validation_losses, train_f1_scores, validation_f1_scores, name)

def experiment3(train_dataloader, test_dataloader, validation_dataloader, learning_rate, epochs, name):
    hidden_layers = [64, 32, 16]
    activation_functions = [RELU, LEAKY_RELU]
    train_losses, validation_losses, train_f1_scores, validation_f1_scores = simulation(train_dataloader, test_dataloader, validation_dataloader, 9, hidden_layers, activation_functions, learning_rate, epochs)
    plot_losses(train_losses, validation_losses, train_f1_scores, validation_f1_scores, name)

def experiment4(train_dataloader, test_dataloader, validation_dataloader, learning_rate, epochs, name):
    hidden_layers = [32, 32, 32, 16]
    activation_functions = [LEAKY_RELU, RELU, LEAKY_RELU]
    train_losses, validation_losses, train_f1_scores, validation_f1_scores = simulation(train_dataloader, test_dataloader, validation_dataloader, 9, hidden_layers, activation_functions, learning_rate, epochs)
    plot_losses(train_losses, validation_losses, train_f1_scores, validation_f1_scores, name)

def experiment5(train_dataloader, test_dataloader, validation_dataloader, learning_rate, epochs, name):
    hidden_layers = [128, 64, 32, 16, 8]
    activation_functions = [RELU, RELU, LEAKY_RELU, LEAKY_RELU]
    train_losses, validation_losses, train_f1_scores, validation_f1_scores = simulation(train_dataloader, test_dataloader, validation_dataloader, 9, hidden_layers, activation_functions, learning_rate, epochs)
    plot_losses(train_losses, validation_losses, train_f1_scores, validation_f1_scores, name)


def main():
    train_df = pd.read_csv("../../DATA/water_train.csv")
    test_df = pd.read_csv("../../DATA/water_test.csv")
    test_df, val_df = train_test_split(test_df, test_size=0.5)

    train_dataloader = get_dataloader(train_df)
    test_dataloader = get_dataloader(test_df)
    validation_dataloader = get_dataloader(val_df)

    base(train_dataloader, test_dataloader, validation_dataloader, 0.001, 30, 'Base')
    experiment1(train_dataloader, test_dataloader, validation_dataloader, 0.001, 45, 'Experiment 1')
    experiment2(train_dataloader, test_dataloader, validation_dataloader, 0.001, 50, 'Experiment 2')
    experiment3(train_dataloader, test_dataloader, validation_dataloader, 0.001, 60, 'Experiment 3')
    experiment4(train_dataloader, test_dataloader, validation_dataloader, 0.001, 76, 'Experiment 4')
    experiment5(train_dataloader, test_dataloader, validation_dataloader, 0.001, 92, 'Experiment 5')

if __name__ == "__main__":
    main()
