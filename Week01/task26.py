import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(123)

    all_walks = []

    for _ in range(5):
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

            steps.append(step)

        all_walks.append(steps)

    np_all_walks = np.array(all_walks)

    for walk in np_all_walks:
        plt.plot(walk)
        
    plt.title("Random walks")
    plt.xlabel("Throw")
    plt.show()
