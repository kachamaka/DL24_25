import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class WaterDataset(Dataset):
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path).to_numpy().astype(np.float64)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input = self.data[idx, :-1]
        target = self.data[idx, -1]
        return input, target

if __name__ == "__main__":
    dataset = WaterDataset("../../DATA/water_train.csv")
    
    print(f"Number of instances: {len(dataset)}")
    
    fifth_item = dataset[4]
    print(f"Fifth item: {fifth_item}")
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for input, target in dataloader:
        print(input, target)
        break
