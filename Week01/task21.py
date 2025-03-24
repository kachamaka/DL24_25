import numpy as np

if __name__ == '__main__':
    np.random.seed(123)

    n = np.random.rand()
    print(f"Random float: {n}")

    dice_rolls = np.random.randint(1, 7, size=2)
    print(f"Random integer 1:", dice_rolls[0])
    print(f"Random integer 2:", dice_rolls[1])

    step = 50
    print(f"Before throw step = {step}")

    dice = np.random.randint(1, 7)
    print(f"After throw dice = {dice}")

    if dice in [1, 2]:
        step -= 1
    elif dice == 6:
        dice = np.random.randint(1, 7)
        step += dice
    else:
        step += 1

    print(f"After throw step = {step}")
