import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import time
import pprint

TRAIN_DATASET_PATH = "../../DATA/clouds/clouds_train"
TEST_DATASET_PATH = "../../DATA/clouds/clouds_test"

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_train_dataset():
    return ImageFolder(TRAIN_DATASET_PATH, transform=TRAIN_TRANSFORMATIONS)

def load_test_dataset():
    return ImageFolder(TEST_DATASET_PATH, transform=TEST_TRANSFORMATIONS)

def load_dataloader(dataset, batch_size=1, shuffle=False):
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


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
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64 * 32 * 32, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def train_model(dataloader_train: DataLoader, dataloader_test: DataLoader, idx_to_class: set, net: CloudCNN, optimizer: optim.Optimizer, num_epochs):
    net.train()
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        batch_loss = 0.0
        progress_bar = tqdm(dataloader_train, desc=f"Epoch [{epoch+1}/{num_epochs}]")
        for input, target in progress_bar:
            input, target = input.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = net(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
        
        average_batch_loss = batch_loss / len(dataloader_train)
        progress_bar.write(f"Average training loss per batch: {average_batch_loss}")

        epoch_losses.append(average_batch_loss)

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

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro')

    print()
    print("Summary statistics:")
    print(f"Average training loss per epoch: {sum(epoch_losses)/num_epochs}")
    print(f"Precison: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Total time taken to train the model in seconds: {train_time}")
    print()    
    print("Per class F1 score:")
    
    _, _, f1_per_class, _ = precision_recall_fscore_support(all_targets, all_predictions, average=None, zero_division=0)
    f1_scores = {idx_to_class[i]: round(float(f1), 4) for i, f1 in enumerate(f1_per_class)}
    pprint.pprint(f1_scores)


    plt.plot(epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per epoch')
    plt.show()


def main():
    learning_rate = 0.001
    num_epochs = 20

    train_dataset = load_train_dataset()
    idx_to_class = {idx: cls for cls, idx in train_dataset.class_to_idx.items()}
    # image = train_dataset[0][0]
    # image = image.permute(1, 2, 0)
    # plt.imshow(image)
    # plt.show()

    train_dataloader = load_dataloader(train_dataset, batch_size=16, shuffle=True)
    num_classes = len(train_dataset.classes)
    net = CloudCNN(num_classes=num_classes).to(DEVICE)

    test_dataset = load_test_dataset()
    test_dataloader = load_dataloader(test_dataset, batch_size=16)
    
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    train_model(train_dataloader, test_dataloader, idx_to_class, net, optimizer, num_epochs)

    # Summary statistics:
    # Average training loss per epoch: 1.309337333738804
    # Precison: 0.5615199123542589
    # Recall: 0.5779986648762411
    # F1: 0.5420014220904692
    # Total time taken to train the model in seconds: 73.08404755592346

    # Per class F1 score:
    # {'cirriform clouds': 0.1942,
    # 'clear sky': 0.6882,
    # 'cumulonimbus clouds': 0.2857,
    # 'cumulus clouds': 0.5766,
    # 'high cumuliform clouds': 0.6192,
    # 'stratiform clouds': 0.8831,
    # 'stratocumulus clouds': 0.547}

if __name__ == "__main__":
    main()
