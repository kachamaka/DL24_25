import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def freeze_layer(layer: nn.Linear) -> None:
    for param in layer.parameters():
        param.requires_grad = False

def freeze_layers(model: nn.Module) -> None:
    freeze_layer(model[0])
    freeze_layer(model[1])


def print_layer_params(layer: nn.Linear, layer_name: str):
    print(f"{layer_name}:")
    print(layer.weight.data[:5])
    print(layer.bias.data[:5])


def main() -> None:
    model = nn.Sequential(
        nn.Linear(4, 16),
        nn.Linear(16, 8),
        nn.Linear(8, 12),
        nn.Linear(12, 1)
    )

    for layer in model:
        nn.init.uniform_(layer.weight, a=-1.0, b=1.0)
        nn.init.uniform_(layer.bias, a=-1.0, b=1.0)

    freeze_layers(model)

    for i, _ in enumerate(model):
        print_layer_params(model[i], f"Layer {i+1}")


if __name__ == "__main__":
    main()
