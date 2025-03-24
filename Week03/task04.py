import torch
import torch.nn as nn

def main() -> None:
    y = [2]
    scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

    criterion = nn.CrossEntropyLoss()

    loss = criterion(scores, torch.tensor(y)).to(torch.float64)
    print(loss)  


if __name__ == "__main__":
    main()
