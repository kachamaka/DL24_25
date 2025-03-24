import numpy as np

if __name__ == '__main__':
    np.random.seed(123)

    step = 0
    steps = [step]

    for _ in range(100):
        dice = np.random.randint(1, 7)
        if dice in [1, 2]:
            step -= 1
        elif dice == 6:
            dice = np.random.randint(1, 7)
            step += dice
        else:
            step += 1

        if step < 0:
            step = 0
        if step > 102:
            step = 102

        steps.append(step)

    print(steps)
