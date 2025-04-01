import os
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import nn
from tqdm import tqdm

TRAIN_DATASET_PATH = "../../DATA/omniglot_train"
TEST_DATASET_PATH = "../../DATA/omniglot_test"

TRANSFORMATIONS = transforms.Compose([
    transforms.Grayscale(),  # For handrwritten characters we don't need RGB
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataloader(dataset, batch_size=1, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class OmniglotDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.samples = []
        self.num_alphabets = 0

        alphabet_idx = 0
        character_idx = 0
        for alphabet in sorted(os.listdir(dataset_path)):
            self.num_alphabets += 1
            alphabet_path = os.path.join(dataset_path, alphabet)
            for character in sorted(os.listdir(alphabet_path)):
                char_path = os.path.join(alphabet_path, character)
                if os.path.isdir(char_path):
                    for img_file in sorted(os.listdir(char_path)):
                        if img_file.endswith('.png'):
                            self.samples.append((os.path.join(char_path, img_file), alphabet_idx, character_idx))

                    character_idx += 1                

            alphabet_idx += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, alphabet_idx, character_label = self.samples[idx]

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        alphabet_one_hot = torch.zeros(self.num_alphabets)
        alphabet_one_hot[alphabet_idx] = 1.0

        return image, alphabet_one_hot, character_label


class ImageNet(nn.Module):
    def __init__(self, output):
        super(ImageNet, self).__init__()
        self.feature_extractor = nn.ModuleList([ # why do we use ModuleList instead of nn.Sequential? I saw that the former does not define forward pass automatically
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 32 -> 16
            nn.Flatten(),
        ])
        self.classifier = nn.Linear(128 * 16 * 16, output)

    def forward(self, x):
        for layer in self.feature_extractor:
            x = layer(x)
        x = self.classifier(x)
        return x


class AlphabetNet(nn.Module):
    def __init__(self, input, output):
        super(AlphabetNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input, 64),
            nn.ReLU(),
            nn.Linear(64, output)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Net(nn.Module):
    def __init__(self, num_alphabets):
        super(Net, self).__init__()
        self.net_output = 64
        self.alphabet_net_output = 64
        self.image_net = ImageNet(output=self.net_output)
        self.alphabet_net = AlphabetNet(input=num_alphabets, output=self.alphabet_net_output)
        self.classifier = nn.Linear(self.net_output + self.alphabet_net_output, 964).to(DEVICE)

    def forward(self, image, label):
        image_pred = self.image_net(image)
        alphabet_pred = self.alphabet_net(label)
        combined = torch.cat((image_pred, alphabet_pred), dim=1)
        return self.classifier(combined)


def train(dataloader_train: DataLoader, dataloader_validation: DataLoader, net: Net, optimizer: optim.Optimizer, num_epochs):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(dataloader_train)
        for input, alphabet_one_hot, target in progress_bar:
            input, alphabet_one_hot, target = input.to(DEVICE), alphabet_one_hot.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = net(input, alphabet_one_hot)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        validation_loss = 0.0
        progress_bar = tqdm(dataloader_validation)
        for input, alphabet_one_hot, target in progress_bar:
            input, alphabet_one_hot, target = input.to(DEVICE), alphabet_one_hot.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = net(input, alphabet_one_hot)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            validation_loss += loss.item()
        
        progress_bar.write(f"Epoch [{epoch+1}/{num_epochs}]:")

        average_train_loss = train_loss / len(dataloader_train)
        progress_bar.write(f"Average training loss: {average_train_loss}")
        train_losses.append(average_train_loss)

        average_validation_loss = validation_loss / len(dataloader_validation)
        progress_bar.write(f"Average validation loss: {average_validation_loss}")
        train_losses.append(average_validation_loss)

        # metrics

    
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()


def check_dataset_balance(dataset_path):
    alphabet_counts = {}

    for alphabet in sorted(os.listdir(dataset_path)):
        alphabet_path = os.path.join(dataset_path, alphabet)
        if not os.path.isdir(alphabet_path):
            continue
        
        character_counts = []
        for character in sorted(os.listdir(alphabet_path)):
            char_path = os.path.join(alphabet_path, character)
            if os.path.isdir(char_path):
                num_images = len([f for f in os.listdir(char_path) if f.endswith('.png')])
                character_counts.append(num_images)
        
        if character_counts:
            total_images = sum(character_counts)
            alphabet_counts[alphabet] = sum(character_counts)
            
            # Calculate average image count per character in the alphabet
            avg_images = total_images / len(character_counts)
            imbalance_threshold = avg_images * 0.2
            imbalanced_characters = [count for count in character_counts if abs(count - avg_images) > imbalance_threshold]
            
            print(f"Alphabet: {alphabet}")
            print(f"  Total images: {total_images}")
            print(f"  Average images per character: {avg_images:.2f}")
            if imbalanced_characters:
                print(f"  Imbalanced characters detected: {len(imbalanced_characters)}")
            else:
                print("  All characters have a roughly equal number of images.")

    # Compute the overall average image count per alphabet
    if alphabet_counts:
        avg_alphabet_image = sum(alphabet_counts.values()) / len(alphabet_counts)
        print("\nOverall Dataset Statistics:")
        print(f"  Average images per alphabet: {avg_alphabet_image:.2f}")


def main():
    check_dataset_balance(TRAIN_DATASET_PATH)
    check_dataset_balance(TEST_DATASET_PATH)
    return
    learning_rate = 0.001
    num_epochs = 10

    dataset_train = OmniglotDataset(TRAIN_DATASET_PATH, transform=TRANSFORMATIONS)
    dataset_validation = OmniglotDataset(TEST_DATASET_PATH, transform=TRANSFORMATIONS)

    dataloader_train = load_dataloader(dataset_train, batch_size=32, shuffle=True)
    dataloader_validation = load_dataloader(dataset_validation, batch_size=32, shuffle=False)

    net = Net(num_alphabets=dataset_train.num_alphabets).to(DEVICE)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

    train(dataloader_train, dataloader_validation, net, optimizer, num_epochs)

    # https://github.com/SimeonHristov99/DL_24-25/blob/main/DATA/omniglot_train/Arcadian/.DS_Store
    # I had problems because of this file, could you please remove it :) 


if __name__ == "__main__":
    main()