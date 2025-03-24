import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from tqdm import tqdm

torch.manual_seed(42)

class WaterDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path).to_numpy().astype(np.float32)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input = self.data[idx, :-1]
        target = self.data[idx, -1]
        return input, target

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(9, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 8)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc3 = nn.Linear(8, 1)

        self.init_weights()

    def forward(self, x):
        x = F.elu(self.bn1(self.fc1(x)))
        x = F.elu(self.bn2(self.fc2(x)))
        x = F.sigmoid(self.fc3(x))
        return x
    
    def init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='sigmoid')

        init.uniform_(self.fc1.bias)
        init.uniform_(self.fc2.bias)
        init.uniform_(self.fc3.bias)

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train_model(dataloader_train: DataLoader, optimizer: optim.Optimizer, net: Net, num_epochs: int, create_plot=False):
    # reset_weights(net)
    net.train()
    criterion = nn.BCELoss()
    
    train_losses = []

    progress_bar = tqdm(dataloader_train)
    for epoch in range(num_epochs):
        train_loss = 0.0
        for input, target in progress_bar:
            optimizer.zero_grad()
            prediction = net(input)
            loss = criterion(prediction, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_loss = train_loss / len(dataloader_train)
        train_losses.append(avg_loss)
    
    print(f"Average loss: {sum(train_losses)/len(train_losses)}")

    if create_plot:
        plt.plot(train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss per epoch with stable gradients')
        plt.show()
    
    return train_losses

def evaluate_model(net, dataloader_test):
    net.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for input, target in dataloader_test:
            prediction = net(input)
            preds = (prediction > 0.5).float() 
            all_predictions.extend(preds)
            all_targets.extend(target)
    
    f1 = f1_score(all_targets, all_predictions)
    return f1

def main():
    batch_size = 512
    learning_rate = 0.001

    train_dataset = WaterDataset('../../DATA/water_train.csv')
    test_dataset = WaterDataset('../../DATA/water_test.csv')

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size)

    net = Net()

    optimizers = {
        'SGD': optim.SGD(net.parameters(), lr=learning_rate),
        'RMSprop': optim.RMSprop(net.parameters(), lr=learning_rate),
        'Adam': optim.Adam(net.parameters(), lr=learning_rate),
        'AdamW': optim.AdamW(net.parameters(), lr=learning_rate)
    }

    for optimizer_name, optimizer in optimizers.items():
        print(f"Using the {optimizer_name} optimizer:")
        train_model(dataloader_train, optimizer, net, num_epochs=10, create_plot=False)

    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    train_model(dataloader_train, optimizer, net, num_epochs=1000, create_plot=True)

    f1 = evaluate_model(net, dataloader_test)
    print(f"F1 score on test set: {f1}")

    # Which of the following statements is true about batch normalization?
    # A. Adding batch normalization doesn't impact the number of parameters the model has to learn. 
    # B. Batch normalization normalizes a layer's inputs to a standard normal distribution and passes these normalized values further. 
    # C. Batch normalization effectively learns the optimal input distribution for each layer it precedes.

    # Answer: B. 

    # For some reason I get a bit different graphic and my loss is also different than what you have in the example.
    # Do you have an idea why that is?


if __name__ == "__main__":
    main()
