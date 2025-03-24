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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

torch.manual_seed(42)

TRAIN_DATA_PATH = '../../DATA/water_train.csv'
TEST_DATA_PATH = '../../DATA/water_test.csv'

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
        self.fc1 = nn.Linear(9, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.3)

        self.init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return torch.sigmoid(x)
    
    def init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.fc3.weight, nonlinearity='leaky_relu')
        init.kaiming_uniform_(self.fc4.weight, nonlinearity='sigmoid')

        init.uniform_(self.fc1.bias)
        init.uniform_(self.fc2.bias)
        init.uniform_(self.fc3.bias)
        init.uniform_(self.fc4.bias)

def train_model(dataloader_train: DataLoader, dataloader_validation: DataLoader, optimizer: optim.Optimizer, net: Net, num_epochs: int, create_plot=False):
    criterion = nn.BCELoss()
    
    train_losses = []
    validation_losses = []

    train_progress_bar = tqdm(dataloader_train)
    validation_progress_bar = tqdm(dataloader_validation)
    for epoch in range(num_epochs):
        net.train()
        train_loss = 0.0
        for input, target in train_progress_bar:
            optimizer.zero_grad()
            prediction = net(input)
            loss = criterion(prediction, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)

        net.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for input, target in validation_progress_bar:
                prediction = net(input)
                loss = criterion(prediction, target.unsqueeze(1))
                validation_loss += loss.item()

            avg_validation_loss = validation_loss / len(dataloader_validation)
            validation_losses.append(avg_validation_loss)
    
    print(f"Average train loss: {sum(train_losses)/len(train_losses)}")
    print(f"Average validation loss: {sum(validation_losses)/len(validation_losses)}")

    if create_plot:
        plt.plot(train_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Epoch')
        plt.legend()
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
    batch_size = 256
    learning_rate = 0.001
    num_epochs = 300
    wd = 0.001

    train_dataset = WaterDataset(TRAIN_DATA_PATH)
    test_dataset = WaterDataset(TEST_DATA_PATH)
    validation_dataset = WaterDataset(TEST_DATA_PATH)

    test_data, val_data = train_test_split(test_dataset.data, test_size=0.5)

    test_dataset.data = test_data
    validation_dataset.data = val_data

    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size)
    dataloader_validation = DataLoader(validation_dataset, batch_size=batch_size)

    net = Net()

    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=wd)
    train_model(dataloader_train, dataloader_validation, optimizer, net, num_epochs=num_epochs, create_plot=True)

    f1 = evaluate_model(net, dataloader_test)
    print(f"F1 score on test set: {f1}")

    # With this configuration I managed to get my F1 score a bit higher than what I had in last task


if __name__ == "__main__":
    main()
