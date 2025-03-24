import os
import random

import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim
import torch
from torch.utils import data
from torch import nn
from tqdm import tqdm

TRAIN_DATASET_PATH = "../../DATA/clouds/clouds_train"
TEST_DATASET_PATH = "../../DATA/clouds/clouds_test"

def load_dataset(dataset_path):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAutocontrast(),
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
    ])

    return ImageFolder(dataset_path, transform=train_transforms)

def load_dataloader(dataset_path, shuffle=True, batch_size=1):
    train_dataset = load_dataset(dataset_path)
    return data.DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size)


class CloudCNN(nn.Module):
    def __init__(self, num_classes):
        super(CloudCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Linear(64 * 32 * 32, num_classes)
        self.flatten = nn.Flatten()
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def train_model(train_loader, net, optimizer, num_epochs):
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(num_epochs):
        batch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for input, target in progress_bar:
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        
        average_batch_loss = batch_loss / len(train_loader)
        epoch_losses.append(average_batch_loss)

        progress_bar.write(f"Average training loss per batch: {average_batch_loss}")

    average_loss_per_epoch = sum(epoch_losses) / num_epochs
    progress_bar.write(f"Average training loss per epoch: {average_loss_per_epoch}")


    plt.plot(epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.show()


def main():
    learning_rate = 0.001
    num_epochs = 20

    train_dataset = load_dataset(TRAIN_DATASET_PATH)
    image = train_dataset[0][0]
    image = image.permute(1, 2, 0)
    plt.imshow(image)
    plt.show()

    train_dataloader = load_dataloader(TRAIN_DATASET_PATH, shuffle=True, batch_size=16)
    num_classes = len(train_dataset.classes)
    net = CloudCNN(num_classes=num_classes)

    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    
    train_model(train_dataloader, net, optimizer, num_epochs)

if __name__ == "__main__":
    main()
