import torch

def main() -> None:
    temperatures = torch.tensor([[72, 75, 78], [70, 73, 76]])

    print("Temperatures:", temperatures)
    print("Shape of temperatures:", temperatures.shape)
    print("Data type of temperatures:", temperatures.dtype)

    corrected_temperatures = temperatures + 2

    print("Corrected temperatures:", corrected_temperatures)
    

if __name__ == "__main__":
    main()
