import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, TensorDataset


def main() -> None:
    data = np.random.rand(12, 9)
    tensor = torch.tensor(data, dtype=torch.float64)

    features = tensor[:, :-1] 
    target = tensor[:, -1]

    dataset = TensorDataset(features, target)

    last_sample, last_label = dataset[-1]

    print("Last sample:", last_sample)
    print("Last label:", last_label)


if __name__ == "__main__":
    main()
