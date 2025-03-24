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
        steps.append(step)

    print(steps)

    # Answer the following question in a comment: Do you notice anything unexpected in the output?.
    
    # We could go out of bounds of the possible floors of Empire State Building which is 102.