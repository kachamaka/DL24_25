import os
import torch

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

TRAIN_DATASET_PATH = "../../DATA/omniglot_train"
TEST_DATASET_PATH = "../../DATA/omniglot_test"

TRANSFORMATIONS = transforms.Compose([
    # transforms.PILToTensor(), Why not use this? I tried using it but I got True and False instead of 1. and 0.
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
])

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


def main():

    dataset_train = OmniglotDataset(TRAIN_DATASET_PATH, transform=TRANSFORMATIONS)

    num_instances = len(dataset_train)
    print(f"Number of instances: {num_instances}")
    
    last_item = dataset_train[-1]
    image, _, _ = last_item
    print(f"Last item: {last_item}")
    
    print(f"Shape of the last image: {image.shape}")


if __name__ == "__main__":
    main()