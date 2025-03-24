import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from tqdm import tqdm


def load_data() -> DataLoader:
    data = pd.read_csv("../../DATA/ds_salaries.csv")

    features = ["experience_level", "employment_type", "remote_ratio", "company_size"]
    target = "salary_in_usd"

    subset_data = data[features + [target]].copy()
    subset_data["experience_level"] = LabelEncoder().fit_transform(subset_data["experience_level"])
    subset_data["employment_type"] = LabelEncoder().fit_transform(subset_data["employment_type"])
    subset_data["company_size"] = LabelEncoder().fit_transform(subset_data["company_size"])

    X = subset_data[features]
    y = subset_data[target]
    
    # scaler = StandardScaler() # I dont know which scaler is better
    scaler = MinMaxScaler() # This one gives lower loss
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.values.reshape(-1, 1))

    # print(X, y)
    # exit(42)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    return dataloader


def train_model(model, dataloader, optimizer, criterion):
    average_loss = 0
    progress_bar = tqdm(dataloader, desc="Training Progress")
    for input, target in progress_bar:
        optimizer.zero_grad() 
        prediction = model(input) 
        loss = criterion(prediction, target)  
        loss.backward() 
        optimizer.step() 
        average_loss += loss.item() 
        progress_bar.set_postfix(loss=loss.item())
    return average_loss / len(dataloader) 


SIGMOID = "nn_with_sigmoid"
RELU = "nn_with_relu"
LEAKY_RELU = "nn_with_leakyrelu"

def main() -> None:
    activation_functions = {
        SIGMOID: nn.Sigmoid(), 
        RELU: nn.ReLU(), 
        LEAKY_RELU: nn.LeakyReLU()
    }
    
    epochs = 20
    learning_rate = 0.001

    dataloader = load_data()

    losses = {SIGMOID: [], RELU: [], LEAKY_RELU: []}
    num = 32

    for name, func in activation_functions.items():
        print(f"Training model: {name}")
        model = nn.Sequential(
            nn.Linear(4, num),
            func,
            nn.Linear(num, 1)
        )

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        average_losses = []
        for epoch in range(epochs):
            average_loss = train_model(model, dataloader, optimizer, criterion)
            average_losses.append(average_loss)
            print(f"Epoch [{epoch + 1}/{epochs}]: Average loss = {average_loss}")

        losses[name] = average_losses
        print("\n\n")


    final_losses = {activation_function: losses[activation_function][-1] for activation_function in activation_functions}
    best_activation = min(final_losses, key=final_losses.get)

    print(f"Lowest loss of {final_losses[best_activation]} was achieved by model {best_activation}.")


    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    all_losses = [loss for loss in losses.values()]
    min_loss = min([min(loss) for loss in all_losses])
    max_loss = max([max(loss) for loss in all_losses])

    for i, (name, loss) in enumerate(losses.items()):
        axes[i].plot(range(1, epochs + 1), loss)
        axes[i].set_title(f'{name}')
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel('Average Loss')
        axes[i].set_ylim(min_loss, max_loss) 
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

    # I don't know why the values are veery different with the ones in the example

if __name__ == "__main__":
    main()
