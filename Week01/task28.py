import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(123)

    all_walks = []

    for _ in range(500):
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

            if np.random.rand() <= 0.005:
                step = 0

            if step < 0:
                step = 0

            steps.append(step)

        all_walks.append(steps)

    np_all_walks = np.array(all_walks)
    end_points = np_all_walks[:, -1]

    odds = np.sum(end_points >= 60) / len(end_points)

    plt.hist(end_points)
    plt.title("Random walks")
    plt.xlabel("End step")
    plt.show()

    print(f"Odds of reaching 60 steps high: {odds:.2%}")

    # Histogram looks different due to something wrong with my code in task27.py
    # Under current data, odds of reaching 60 steps high is 60.80%
