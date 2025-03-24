import torch.nn as nn

def count_parameters(model: nn.Module) -> None:
    sum = 0
    for p in model.parameters():
        sum += p.numel()

    return sum

def main() -> None:
    network1 = nn.Sequential(
        nn.Linear(8, 6),  # 8*6 + 6 = 54
        nn.Linear(6, 4),  # 6*4 + 4 = 28
        nn.Linear(4, 2)   # 4*2 + 2 = 10
        # Total: 92
    )

    network2 = nn.Sequential(
        nn.Linear(8, 12),  # 8*12 + 12 = 108
        nn.Linear(12, 6),  # 12*6 + 6 = 78
        nn.Linear(6, 4),   # 6*4 + 4 = 28
        nn.Linear(4, 2)    # 4*2 + 2 = 10
        # Total: 224
    )

    network1_parameters = count_parameters(network1)    
    network2_parameters = count_parameters(network2)    

    print(f"Number of parameters in network 1: {network1_parameters}")
    print(f"Number of parameters in network 2: {network2_parameters}")


if __name__ == "__main__":
    main()
