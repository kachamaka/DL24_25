import numpy as np

def main():
    learning_rates = np.random.uniform(0.0001, 0.01, 10)
    momentums = np.random.uniform(0.85, 0.99, 10)

    print(list(zip(learning_rates, momentums)))

    
if __name__ == "__main__":
    main()
