import torch
import torch.nn as nn

def main() -> None:
    model = nn.Sequential(
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    temperature_observation = [3, 4, 6, 2, 3, 6, 8, 9]

    input = torch.tensor([temperature_observation], dtype=torch.float32)
    output = model(input)

    print(output.item())
    
    # Which of the following is false about the output returned by your binary classifier?

    # A. We can use a threshold of 0.5 to determine if the output belongs to one class or the other. 
    # B. It can return any float value. 
    # C. It is produced from an untrained model so it is not yet meaningful.

    # Answer: B.


if __name__ == "__main__":
    main()
