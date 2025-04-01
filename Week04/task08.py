import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import time
import pprint

TRAIN_DATASET_PATH = "../../DATA/clouds/clouds_train"
TEST_DATASET_PATH = "../../DATA/clouds/clouds_test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_TRANSFORMATIONS = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomAutocontrast(),
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])

TEST_TRANSFORMATIONS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])


def load_train_dataset():
    return ImageFolder(TRAIN_DATASET_PATH, transform=TRAIN_TRANSFORMATIONS)

def load_test_dataset():
    return ImageFolder(TEST_DATASET_PATH, transform=TEST_TRANSFORMATIONS)

def load_dataloader(dataset, batch_size=1, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class CloudCNN(nn.Module):
    def __init__(self, num_classes):
        super(CloudCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64 * 32 * 32, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def train_model(dataloader_train: DataLoader, dataloader_test: DataLoader, dataloader_validation: DataLoader, idx_to_class: set, net: CloudCNN, optimizer: optim.Optimizer, num_epochs):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    validation_losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        net.train()
        batch_loss = 0.0
        training_bar = tqdm(dataloader_train, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for input, target in training_bar:
            input, target = input.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            prediction = net(input)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        
        average_batch_loss = batch_loss / len(dataloader_train)
        training_bar.write(f"Average training loss per batch: {average_batch_loss}")
        train_losses.append(average_batch_loss)

        # validation loss
        net.eval()
        val_loss = 0.0
        validation_bar = tqdm(dataloader_validation, desc=f"Validation [{epoch+1}/{num_epochs}]")
        with torch.no_grad():
            for input, target in validation_bar:
                input, target = input.to(DEVICE), target.to(DEVICE)
                prediction = net(input)
                loss = criterion(prediction, target)
                val_loss += loss.item()

        average_val_loss = val_loss / len(dataloader_validation)
        validation_losses.append(average_val_loss)


    end_time = time.time()
    train_time = end_time - start_time


    # test loss
    net.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for input, target in dataloader_test:
            input, target = input.to(DEVICE), target.to(DEVICE)
            prediction = net(input)
            preds = torch.argmax(prediction, dim=1)
            all_predictions.extend(preds.tolist())
            all_targets.extend(target.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division=0)

    print()
    print("Summary statistics:")
    print(f"Average training loss per epoch: {sum(train_losses)/num_epochs}")
    print(f"Precison: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Total time taken to train the model in seconds: {train_time}")
    print()    
    print("Per class F1 score:")

    _, _, f1_per_class, _ = precision_recall_fscore_support(all_targets, all_predictions, average=None, zero_division=0)
    f1_scores = {idx_to_class[i]: round(float(f1), 4) for i, f1 in enumerate(f1_per_class)}
    pprint.pprint(f1_scores)

    
    plt.plot(train_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Losses per epoch')
    plt.legend()
    plt.show()


def main():
    learning_rate = 1e-3
    num_epochs = 50
    batch_size = 128

    train_dataset = load_train_dataset()
    idx_to_class = {idx: cls for cls, idx in train_dataset.class_to_idx.items()}
    train_dataloader = load_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    num_classes = len(train_dataset.classes)

    net = CloudCNN(num_classes=num_classes).to(DEVICE)

    test_dataset = load_test_dataset()

    test_size = len(test_dataset) // 2
    val_size = len(test_dataset) - test_size

    test_dataset, validation_dataset = random_split(test_dataset, [test_size, val_size])

    validation_dataloader = load_dataloader(validation_dataset, batch_size=batch_size)
    test_dataloader = load_dataloader(test_dataset, batch_size=batch_size)

    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_model(train_dataloader, test_dataloader, validation_dataloader, idx_to_class, net, optimizer, num_epochs)

    # Summary statistics:
    # Average training loss per epoch: 2.8902506279945372
    # Precison: 0.7365928383014252
    # Recall: 0.7205722333569644
    # F1: 0.7108158686049285
    # Total time taken to train the model in seconds: 83.0619466304779

    # Per class F1 score:
    # {'cirriform clouds': 0.6609,
    # 'clear sky': 0.9394,
    # 'cumulonimbus clouds': 0.6667,
    # 'cumulus clouds': 0.7027,
    # 'high cumuliform clouds': 0.6154,
    # 'stratiform clouds': 0.8235,
    # 'stratocumulus clouds': 0.5672}

if __name__ == "__main__":
    
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    main()