import torch
import torch.nn as nn

def main() -> None:
    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.Linear(4, 1)
    )

    temperature_observation = [2, 3, 6, 7, 9, 3, 2, 1]

    input = torch.tensor([temperature_observation], dtype=torch.float32)
    logit = model(input)

    print(logit)

if __name__ == "__main__":
    main()
